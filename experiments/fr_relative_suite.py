"""Frozen 52-shot, six-control IFWI experiment. No automatic restart/expansion.

All PDE fields are charged by solver callbacks. Initial-state construction,
warmup, evaluation, checkpoints and analysis have separate wall-clock records.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import traceback
import uuid

import numpy as np
import torch

from experiments.dense_encoding_suite import (build_plan as old_plan, DenseCosts,
    _dense_data, gpu_execution_guard, ACTIVE_RUN)
from experiments.dense_encoding_physics import (build_dense_problem, DenseAcquisition,
    train_dense_step, evaluate_dense)
from experiments.dense_checkpoint import save_dense_checkpoint, load_dense_checkpoint
from experiments.encoding_suite_data import load_assets
from experiments.encoding_training_step import configure_runtime, _sync, CommittedUpdateError, UncertainOptimizerError
from experiments.residual_ifwi_experiment import model_from_config
from experiments.residual_ifwi_report import velocity_metrics
from experiments.fourier_modules import default_fourier
from ifwi_research.suite_control import SuiteControl, StopRequested

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent if ROOT.parent.name == '.worktrees' else ROOT
OLD_DENSE = WORKSPACE/'preparations/dense_gated_gpu_20261112_1752'
ARMS = tuple(dict(name=n, grouping=g, fields=k, model=m, preconditioned=p)
    for n,g,k,m,p in (
        ('full_raw','identity',52,'plain',False),
        ('full_preconditioned','identity',52,'plain',True),
        ('global1','global',1,'plain',True),
        ('interleaved2','interleaved',2,'plain',True),
        ('interleaved4','interleaved',4,'plain',True),
        ('interleaved4_relative_fr','interleaved',4,'relative_fr',True)))
THRESHOLDS = (500.,475.,450.,425.)
SOURCES = ('experiments/fr_relative_suite.py','experiments/fr_relative_preflight.py',
    'experiments/relative_velocity.py','experiments/shifted_interleaved.py',
    'experiments/fourier_modules.py','experiments/dense_encoding_suite.py','experiments/batched_encoding_suite.py',
    'experiments/dense_encoding_physics.py','experiments/dense_checkpoint.py',
    'experiments/encoding_checkpoint.py','experiments/encoding_suite_data.py',
    'experiments/encoding_training_step.py','experiments/encoding_batch_correctness.py',
    'experiments/residual_ifwi_experiment.py','experiments/residual_ifwi_report.py',
    'ifwi_research/suite_control.py','ifwi_research/batched_sources.py',
    'ifwi_research/encoding_illumination.py','core_speed.py','rnn_fd.py',
    'ifwi_modules.py','generator.py','experiments/fr_relative_pipeline.py','experiments/fr_relative_analysis.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_target(stage,target,resume_root):
    if stage == 'formal':
        if target not in (500,1000): raise ValueError('Only the fixed 500 or explicit 1000 boundary is allowed')
        if target == 1000 and resume_root is None: raise ValueError('1000 requires explicit verified 500-state resume')
        if target == 500 and resume_root is not None: raise ValueError('Fresh 500 stage only; no automatic failed-run restart')
    elif target != 500 or resume_root is not None:
        raise ValueError('Target/resume options apply only to the formal stage')


def budget(stage,target=500,resume=False):
    fields = sum(a['fields'] for a in ARMS)
    if stage == 'preflight':
        return dict(forward_fields=34,backward_fields=34,optimizer_updates=0)
    if stage == 'timing':
        return dict(forward_fields=fields*12,backward_fields=fields*12,
                    optimizer_updates=len(ARMS)*12)
    updates = target-500 if resume else target
    warm = 0 if resume else 2
    evaluations = updates//100
    train = fields*2*updates; warm_fields = fields*2*warm
    evaluation = 52*(len(ARMS)*2*evaluations+(0 if resume else 1))
    return dict(training_forward_fields=train,warmup_forward_fields=warm_fields,
        evaluation_forward_fields=evaluation,forward_fields=train+warm_fields+evaluation,
        backward_fields=train+warm_fields,optimizer_updates=len(ARMS)*2*(updates+warm))


def threshold_crossings(rows,thresholds=THRESHOLDS):
    result=[]
    for threshold in thresholds:
        hit=next((row for row in rows if row['rmse_mps']<=threshold),None)
        result.append(dict(threshold_rmse_mps=threshold,status='REACHED' if hit else 'NOT_REACHED',
            first_observed_update=None if hit is None else hit['update'],
            training_seconds=None if hit is None else hit['training_seconds'],
            elapsed_seconds=None if hit is None else hit['elapsed_seconds']))
    return result


def failure_update_outcome(error,frame,updates):
    """Separate returned updates, known commits and an uncertain in-step state.

    Never checkpoint/retry an unvalidated post-commit or partially mutated Adam.
    This does not infer a commit from completed physical forward/backward work.
    """
    confirmed=dict(updates)
    outcome=dict(last_returned_update=frame.get('completed'),
        attempted_update=frame.get('attempted_update'),optimizer_commit_state='NO_UNRETURNED_COMMIT_REPORTED',
        optimizer_updates_confirmed=confirmed,optimizer_updates_uncertain=0)
    if isinstance(error,CommittedUpdateError):
        phase='warmup' if frame.get('phase')=='warmup' else 'training'
        confirmed[phase]+=1
        outcome.update(optimizer_commit_state='KNOWN_COMMITTED_POSTCHECK_FAILED',
            known_committed_update_at_failure=error.completed_updates,unsafe_state=True)
    elif isinstance(error,UncertainOptimizerError) or (frame.get('in_step') and not isinstance(error,Exception)):
        outcome.update(optimizer_commit_state='UNKNOWN',optimizer_updates_uncertain=1,
            attempted_update=getattr(error,'attempted_update',frame.get('attempted_update')),unsafe_state=True)
    return outcome


def build_plan(assets,dense_manifest):
    if assets['config']['model']['omega_0']!=30 or assets['config']['model']['std_kmps']!=1:
        raise ValueError('This frozen protocol requires omega30 and sv=1000m/s')
    previous=old_plan(assets,52,13)
    config=copy.deepcopy(previous['config']);config['training'].update(epochs=1000)
    source={name:sha(ROOT/name) for name in SOURCES}
    common=dict(schema='fr_relative_52_v1',config=config,physical=previous['physical'],
        data_key=previous['data_key'],input_manifest=assets['manifest'],dense_data_manifest=copy.deepcopy(dense_manifest),
        arms=list(ARMS),conditions=['clean','noisy'],
        source_sha256=source,initial_checkpoint=assets['manifest'].get('initial_checkpoint'),
        fourier=default_fourier(),omega=30.,relative_scale='v0/rms(v0)',sv_mps=1000.,
        objective='sum squared coded residual/(M*52*T*R)',
        encoding_control='full_preconditioned',illumination='existing sqrt(S*A), water=.1, gradient norm match',
        shift='(index+shift)%B only relabels fixed groups for N52 B2/4; fresh signs each update',
        model_metric_interval=10,waveform_metric_interval=100,checkpoint_interval=100,
        thresholds_rmse_mps=list(THRESHOLDS),threshold_policy='first observed crossing; no interpolation/selection/stopping',
        update_limit=1000,requested_first_stage=500,seed=3,warmup_updates=2,
        initial_waveform='one shared v0 solve, copied with provenance to all twelve exact-initial-velocity cases',
        continuation='fresh500; 1000 requires explicit complete500 resume; no automatic restart')
    return dict(**common,signature=digest(common),status='FROZEN_BEFORE_GPU',
        budgets={s:budget(s) for s in ('preflight','timing','formal')})


def verify_plan(plan):
    signature=plan['signature']
    common={k:v for k,v in plan.items() if k not in ('signature','status','budgets')}
    if digest(common)!=signature: raise ValueError('Plan signature mismatch')
    for name,expected in plan['source_sha256'].items():
        if sha(ROOT/name)!=expected: raise ValueError('Frozen source changed: '+name)


def verify_assets(assets,plan):
    if assets['manifest']!=plan['input_manifest']:
        raise ValueError('Loaded input assets manifest differs from frozen plan')


def release_guard():
    lock=gpu_execution_guard(ACTIVE_RUN)
    start=read(OLD_DENSE/'PIPELINE_LAUNCH.json');end=read(OLD_DENSE/'PIPELINE_EXIT.json')
    if (start['run_id']!=end['run_id'] or end['status']!='COMPLETED' or end['exit_code']!=0):
        raise RuntimeError('Old dense pipeline does not have matching terminal proof')
    return lock


def make_model(assets,arm,device):
    torch.manual_seed(3);np.random.seed(3);random.seed(3)
    cfg=assets['config'];spec=cfg['model']
    if arm['model']=='plain':
        model=model_from_config(assets['background'],cfg)
        model.load_state_dict(assets['canonical_checkpoint']['model'],strict=True)
    else:
        from experiments.relative_velocity import RelativeVelocityParameterization
        model=RelativeVelocityParameterization(assets['background'],cfg['data']['grid_spacing_m'],
            spec['neurons'],omega_0=30.,mean_kmps=spec['mean_kmps'],std_kmps=spec['std_kmps'],
            parameterization='residual',fourier=default_fourier(),zero_residual_head=True)
    model=model.to(device)
    with torch.no_grad():
        if not torch.equal(model().cpu(),torch.as_tensor(assets['background'],dtype=torch.float32)[None]):
            raise ValueError('Initial physical velocity differs from v0')
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8)
    return model,opt


def acquisition(problem,arm):
    if arm['grouping']=='interleaved':
        from experiments.shifted_interleaved import ShiftedInterleavedAcquisition
        return ShiftedInterleavedAcquisition(problem,batch_size=13,fields=arm['fields'],seed=3)
    return DenseAcquisition(problem,batch_size=13,grouping=arm['grouping'],fields=arm['fields'],seed=3)


def metadata(plan,condition,arm):
    return dict(protocol='fr_relative_52_v1',signature=plan['signature'],condition=condition,arm=arm)


def optimizer_bytes(opt):
    return sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor))


def verify_previous_stage(path,stage,plan):
    report=read(Path(path)/'result.json');launch=read(Path(path)/'control/LAUNCH.json');end=read(Path(path)/'control/EXIT.json')
    if (report.get('status')!='COMPLETED' or report.get('stage')!=stage or report.get('signature')!=plan['signature']
        or launch['run_id']!=end['run_id'] or end['status']!='COMPLETED' or end['exit_code']!=0
        or report['cost_totals']!=end['cost_totals']):
        raise ValueError('Missing/mismatched prerequisite stage '+stage)
    expected=budget(stage,report.get('target_updates',500),bool(report.get('resume_root')))
    assert_cost(report['cost_totals'],expected)
    return report


def assert_cost(actual,expected):
    for kind in ('forward','backward'):
        for state in ('attempt','complete'):
            if actual[kind+'_fields_'+state]!=expected[kind+'_fields']:
                raise ValueError('Actual physics cost differs from frozen budget: '+kind+' '+state)


def execute(stage,output,assets,plan,data_directory,*,target=500,preflight=None,timing=None,resume_root=None):
    validate_target(stage,target,resume_root);verify_plan(plan);verify_assets(assets,plan)
    if stage in ('timing','formal'): verify_previous_stage(preflight,'preflight',plan)
    if stage=='formal': verify_previous_stage(timing,'timing',plan)
    if resume_root:
        previous=verify_previous_stage(resume_root,'formal',plan)
        if previous['target_updates']!=500: raise ValueError('Explicit resume source must be complete500')
    lock=release_guard()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    control=SuiteControl(output,uuid.uuid4().hex,lock);control.acquire()
    costs=DenseCosts(control);started=time.perf_counter();terminal=False;frame={};cases=[]
    updates=dict(warmup=0,training=0);runtime=None;shared_initial=None
    def checkpoint():
        p=frame['directory']/f"update_{frame['completed']:04d}.pt"
        if p.exists():
            if frame.get('checkpoint')!=str(p): raise FileExistsError('Unexpected checkpoint; no overwrite')
            return
        tick=time.perf_counter()
        save_dense_checkpoint(frame['model'],frame['optimizer'],p,frame['completed'],frame['metadata'])
        frame['checkpoint']=str(p)
        control.record('checkpoint',condition=frame['condition'],arm=frame['arm']['name'],
            completed_updates=frame['completed'],path=str(p),sha256=sha(p),wall_seconds=time.perf_counter()-tick)
    try:
        runtime=configure_runtime('cuda:0')
        control.write_json('runtime.json',runtime);control.write_json('plan.json',plan)
        if read(Path(data_directory)/'data_manifest.json')!=plan['dense_data_manifest']:
            raise ValueError('Frozen dense observations manifest differs')
        data=_dense_data(data_directory,plan);assets={**assets,**data}
        control.write_json('budget.json',budget(stage,target,bool(resume_root)))
        if stage=='preflight':
            from experiments.fr_relative_preflight import run_preflight
            result=run_preflight(control,costs,assets,plan,'cuda:0')
        else:
            for condition in (('noisy',) if stage=='timing' else ('clean','noisy')):
                control.check_stop();tick=time.perf_counter()
                problem=build_dense_problem(assets['truth'],plan['config'],'cuda:0',observed=data[condition],wavelet=assets['wavelet'])
                physical=DenseAcquisition(problem,batch_size=13)
                control.record('condition_construction',condition=condition,wall_seconds=time.perf_counter()-tick)
                for arm in ARMS:
                    control.check_stop();case_tick=time.perf_counter()
                    directory=output/condition/arm['name'];directory.mkdir(parents=True)
                    tick=time.perf_counter()
                    acq=physical if arm['grouping']=='identity' else acquisition(problem,arm)
                    if acq is not physical: acq._identity=physical
                    torch.cuda.reset_peak_memory_stats();model,opt=make_model(assets,arm,'cuda:0')
                    meta=metadata(plan,condition,arm)
                    costs.set_context(condition=condition,arm=arm['name'])
                    frame=dict(model=model,optimizer=opt,directory=directory,completed=0,phase='warmup',
                        metadata=meta,condition=condition,arm=arm,checkpoint=None)
                    control.record('case_construction',condition=condition,arm=arm['name'],wall_seconds=time.perf_counter()-tick,
                        trainable_parameters=sum(p.numel() for p in model.parameters()),
                        persistent_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()))
                    if resume_root:
                        if read(Path(resume_root)/'runtime.json')!=runtime: raise ValueError('Resume runtime differs')
                        tick=time.perf_counter()
                        p=Path(resume_root)/condition/arm['name']/'update_0500.pt'
                        frame['completed']=load_dense_checkpoint(model,opt,p,meta)
                        if frame['completed']!=500: raise ValueError('Resume count differs from500')
                        control.record('restore',path=str(p),sha256=sha(p),wall_seconds=time.perf_counter()-tick)
                    else:
                        for k in range(2):
                            frame.update(in_step=True,attempted_update=k+1)
                            row=train_dense_step(model,opt,acq,k,record_cost=costs.record,check_stop=control.check_stop,
                                phase='warmup',arm_name=arm['name'],illumination_preconditioning=arm['preconditioned'])
                            frame['in_step']=False
                            updates['warmup']+=1
                            frame['completed']=k+1
                            control.record('warmup_update',condition=condition,arm=arm['name'],result=row)
                        if stage=='formal':
                            tick=time.perf_counter();model,opt=make_model(assets,arm,'cuda:0')
                            frame.update(model=model,optimizer=opt,completed=0)
                            control.record('fresh_reset_after_warmup',condition=condition,arm=arm['name'],wall_seconds=time.perf_counter()-tick)
                        else: frame['completed']=2
                    frame['phase']='measured'
                    samples=[];curve=[];train_seconds=0.;eval_seconds=0.;elapsed_offset=0.
                    if resume_root:
                        old_dir=Path(resume_root)/condition/arm['name']
                        curve=[read(p) for p in sorted(old_dir.glob('metrics_*.json'))]
                        old_case=read(old_dir/'case.json')
                        train_seconds=old_case['training_seconds'];eval_seconds=old_case['evaluation_seconds']
                        elapsed_offset=old_case['case_elapsed_seconds']
                    def metrics_at(k,waveform):
                        nonlocal eval_seconds,shared_initial
                        tick=time.perf_counter()
                        if waveform:
                            if k==0 and shared_initial is not None:
                                metrics,pred,vel=shared_initial
                                control.record('initial_waveform_reused',condition=condition,arm=arm['name'],
                                    physical_F=0,physical_B=0,source=str(output/'shared_initial/prediction.npy'))
                            else:
                                metrics,pred,vel=evaluate_dense(model,acq,assets,record_cost=costs.record,
                                    check_stop=control.check_stop,label=f"{condition}/{arm['name']}/{k}")
                                if k==0:
                                    common_dir=output/'shared_initial';common_dir.mkdir()
                                    np.save(common_dir/'prediction.npy',pred);np.save(common_dir/'velocity.npy',vel)
                                    control.write_json('shared_initial/manifest.json',dict(metrics=metrics,
                                        prediction_sha256=sha(common_dir/'prediction.npy'),velocity_sha256=sha(common_dir/'velocity.npy')))
                                    shared_initial=(metrics,pred,vel)
                            np.save(directory/f'prediction_{k:04d}.npy',pred)
                        else:
                            with torch.no_grad(): vel=model()[0].detach().cpu().numpy().copy()
                            metrics=velocity_metrics(vel,assets['truth'])
                        np.save(directory/f'velocity_{k:04d}.npy',vel)
                        eval_seconds+=time.perf_counter()-tick
                        entry=dict(update=k,training_seconds=train_seconds,elapsed_seconds=elapsed_offset+time.perf_counter()-case_tick,
                            evaluation_seconds=eval_seconds,**metrics)
                        curve.append(entry);control.write_json(directory.relative_to(output)/f'metrics_{k:04d}.json',entry)
                        control.record('metrics',condition=condition,arm=arm['name'],result=entry)
                    if stage=='formal' and not resume_root: metrics_at(0,True)
                    limit=12 if stage=='timing' else target
                    while frame['completed']<limit:
                        k=frame['completed']
                        frame.update(in_step=True,attempted_update=k+1)
                        row=train_dense_step(model,opt,acq,k,record_cost=costs.record,check_stop=control.check_stop,
                            phase='training',arm_name=arm['name'],illumination_preconditioning=arm['preconditioned'])
                        frame['in_step']=False
                        if row['completed_updates']!=k+1: raise ValueError('Unexpected optimizer commit count')
                        frame['completed']=k+1;updates['training']+=1;samples.append(row);train_seconds+=row['wall_seconds']
                        control.record('update',condition=condition,arm=arm['name'],result=row,
                            cumulative_training_seconds=train_seconds,case_elapsed_seconds=elapsed_offset+time.perf_counter()-case_tick)
                        if stage=='formal' and frame['completed']%10==0:
                            metrics_at(frame['completed'],frame['completed']%100==0)
                        if stage=='formal' and frame['completed']%100==0: checkpoint()
                        control.write_json('status.json',dict(status='RUNNING',stage=stage,condition=condition,arm=arm['name'],
                            completed=frame['completed'],target=limit,cost_totals=costs.totals,optimizer_updates=updates,
                            training_seconds=train_seconds,stage_elapsed_seconds=time.perf_counter()-started))
                        control.check_stop()
                    checkpoint()
                    if stage=='timing':
                        # No PDE or Adam work: independently construct and strictly restore model/Adam/RNG.
                        tick=time.perf_counter();clone,clone_opt=make_model(assets,arm,'cuda:0')
                        if load_dense_checkpoint(clone,clone_opt,frame['checkpoint'],meta)!=12:
                            raise ValueError('Timing checkpoint recovery count differs')
                        with torch.no_grad():
                            if not torch.equal(clone(),model()): raise ValueError('Timing recovered velocity differs')
                        control.record('exact_checkpoint_recovery',condition=condition,arm=arm['name'],
                            wall_seconds=time.perf_counter()-tick,physical_F=0,physical_B=0,optimizer_updates=0)
                        del clone,clone_opt
                    result_case=dict(condition=condition,arm=arm['name'],completed_updates=frame['completed'],
                        measured_updates=len(samples),warmup_updates=0 if resume_root else 2,training_seconds=train_seconds,
                        evaluation_seconds=eval_seconds,case_elapsed_seconds=elapsed_offset+time.perf_counter()-case_tick,
                        incremental_case_elapsed_seconds=time.perf_counter()-case_tick,
                        wall_seconds_median=statistics.median(s['wall_seconds'] for s in samples),
                        cuda_ms_median=statistics.median(s['cuda_ms'] for s in samples),
                        trainable_parameters=sum(p.numel() for p in model.parameters()),optimizer_tensor_bytes=optimizer_bytes(opt),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        checkpoint=frame['checkpoint'],thresholds=threshold_crossings(curve))
                    cases.append(result_case)
                    control.write_json(directory.relative_to(output)/'case.json',result_case)
                    control.write_json('partial.json',dict(cases=cases,cost_totals=costs.totals))
                    frame.clear();del model,opt,acq;gc.collect();torch.cuda.empty_cache()
                del physical,problem;gc.collect();torch.cuda.empty_cache()
            result=dict(cases=cases)
        assert_cost(costs.totals,budget(stage,target,bool(resume_root)))
        if updates['warmup']+updates['training']!=budget(stage,target,bool(resume_root))['optimizer_updates']:
            raise ValueError('Actual Adam updates differ from frozen budget')
        report=dict(result,status='COMPLETED',stage=stage,signature=plan['signature'],target_updates=target,
            resume_root=None if resume_root is None else str(resume_root),cost_totals=costs.totals,
            cost_by_phase=costs.by_phase,optimizer_updates=updates,stage_wall_seconds=time.perf_counter()-started)
        control.write_json('result.json',report)
        control.finish('COMPLETED',0,stage=stage,cost_totals=costs.totals);terminal=True;return 0
    except StopRequested:
        if frame and stage=='formal' and frame.get('phase')=='measured': checkpoint()
        control.finish('STOPPED_BY_USER',130,cost_totals=costs.totals,checkpoint=frame.get('checkpoint'));terminal=True;return 130
    except BaseException as error:
        outcome=failure_update_outcome(error,frame,updates)
        control.write_json('failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            last_safe_checkpoint=frame.get('checkpoint'),**outcome,
            cost_totals=costs.totals,optimizer_updates=outcome['optimizer_updates_confirmed'],automatic_restart=False))
        if hasattr(error,'report'): control.write_json('failed_check_report.json',error.report)
        control.finish('FAILED',1,error=str(error),cost_totals=costs.totals,
            optimizer_updates=outcome['optimizer_updates_confirmed'],
            optimizer_updates_uncertain=outcome['optimizer_updates_uncertain']);terminal=True;return 1
    finally:
        if terminal: control.release()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','preflight','timing','formal'),default='prepare')
    for name in ('output','dataset-index','reference-config','initial-checkpoint','dense-data'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan',type=Path);parser.add_argument('--preflight',type=Path);parser.add_argument('--timing',type=Path)
    parser.add_argument('--target',type=int,default=500);parser.add_argument('--resume-root',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    assets=load_assets(args.dataset_index,args.reference_config,args.initial_checkpoint)
    if args.plan: plan=read(args.plan);verify_plan(plan)
    else: plan=build_plan(assets,read(args.dense_data/'data_manifest.json'))
    verify_assets(assets,plan)
    _dense_data(args.dense_data,plan)
    if args.stage=='prepare' or not args.execute:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False),encoding='utf-8')
        print('FROZEN: no GPU work; first formal stage500, explicit continuation upper limit1000',flush=True);return 0
    return execute(args.stage,args.output,assets,plan,args.dense_data,target=args.target,
        preflight=args.preflight,timing=args.timing,resume_root=args.resume_root)


if __name__=='__main__': raise SystemExit(main())

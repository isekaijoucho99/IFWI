"""Controlled batch-encoding correctness, timing and fixed-update comparisons.

Only one stage owns the explicit parent-directory lock at a time. Physical
work is charged by the solver callbacks, never inferred from GPU batch calls.
The CLI does not invoke Git, query PIDs, or automatically resume an old run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
from types import SimpleNamespace
import uuid

import numpy as np

from ifwi_research.suite_control import SuiteControl, StopRequested, MetadataPublishError

ROOT=Path(__file__).resolve().parents[1]
MILESTONES=(200,500,1000)
CHECKPOINT_INTERVAL=25
STATUS_INTERVAL=10
CONDITIONS=('clean','noisy')
COUNTERS=('forward_fields_attempt','forward_fields_complete',
          'backward_fields_attempt','backward_fields_complete')
PHYSICS_SOURCES=('core_speed.py','rnn_fd.py','ifwi_modules.py','generator.py',
    'ifwi_research/batched_sources.py','ifwi_research/encoding_illumination.py',
    'experiments/encoding_suite_data.py','experiments/encoding_batch_correctness.py',
    'experiments/encoding_training_step.py','experiments/encoding_checkpoint.py','experiments/progressive_siren.py',
    'experiments/residual_ifwi_experiment.py')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():return {name:_sha(ROOT/name)for name in PHYSICS_SOURCES}


def _array(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value)


def default_dependencies():
    import torch
    from core_speed import SourceEncoder
    from experiments.encoding_suite_data import ARMS,load_assets,make_model,build_problem
    from experiments.encoding_batch_correctness import run_correctness
    from experiments.encoding_training_step import configure_runtime,train_step,evaluate
    def save_checkpoint(*args):
        from experiments.encoding_checkpoint import save_checkpoint as save
        return save(*args)
    def load_checkpoint(*args):
        from experiments.encoding_checkpoint import load_checkpoint as load
        return load(*args)
    def reset_peak(device):
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    def memory_peak(device):
        return dict(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))if str(device).startswith('cuda')else dict(peak_allocated_bytes=None,peak_reserved_bytes=None)
    def cleanup(device):
        gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return SimpleNamespace(arms=ARMS,load_assets=load_assets,make_model=make_model,
        build_problem=build_problem,run_correctness=run_correctness,
        source_fingerprints=source_fingerprints,configure_runtime=configure_runtime,
        make_encoder=lambda problem,arm:SourceEncoder(problem,arm['fields'],3,grouping=arm['grouping']),
        train_step=train_step,evaluate=evaluate,save_checkpoint=save_checkpoint,
        load_checkpoint=load_checkpoint,reset_peak=reset_peak,memory_peak=memory_peak,
        cleanup=cleanup)


class Costs:
    def __init__(self,control):
        self.control=control;self.totals={key:0 for key in COUNTERS}
        self.context={};self.record_error=None
    def set_context(self,**values):self.context=values
    def record(self,event,**details):
        row=dict(self.context);row.update(details)
        if any(row.get(key,0)for key in COUNTERS if key.endswith('_attempt')):
            self.control.check_stop()
        for key in COUNTERS:
            value=row.get(key,0)
            if isinstance(value,bool)or not isinstance(value,(int,np.integer))or value<0:
                raise ValueError('Physical field counters must be nonnegative integers')
            self.totals[key]+=int(value)
        try:return self.control.record(event,**row)
        except BaseException as error:
            self.record_error=dict(event=event,error_type=type(error).__name__,error=str(error))
            raise


def _signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def build_protocol(assets,runtime,dependencies,code_commit):
    common=dict(data=assets['manifest'],configuration=assets['config'],
        measured_snr_db=assets.get('measured_snr_db'),runtime=runtime,
        physics_source_sha256=dependencies.source_fingerprints(),arms=list(dependencies.arms),
        conditions=list(CONDITIONS),milestones=list(MILESTONES),encoding_seed=3,
        objective='sum squared coded residual/(original13shots*time*receivers*models)',
        illumination='sqrt(sum(S_code)*sum(A_code)); common relative water=.1 and RMS match',
        progressive='low omega6 plus canonical omega30; cosine gate0..200',
        initialization='cached background authoritative (index smooth400); cfg initialization.sigma15 is legacy metadata, not a reconstruction instruction; common canonical zero-head residual, empty Adam',
        selection='fixed200/500/1000; truth metrics never choose arms/checkpoints')
    return dict(schema='batched_encoding_protocol_v1',common=common,
        protocol_signature=_signature(common),code_commit_recorded=code_commit,
        orchestrator_sha256=_sha(__file__),
        support_source_sha256={'ifwi_research/suite_control.py':_sha(ROOT/'ifwi_research/suite_control.py')},
        signature_scope='actual physics/model/training sources, data and runtime; Git label and orchestration metadata are recorded separately',
        checkpoint_interval=CHECKPOINT_INTERVAL,status_interval=STATUS_INTERVAL)


def _initial_checks(assets,dependencies,device):
    expected=np.asarray(assets['background'],dtype=np.float32)[None]
    checked=[]
    for arm in dependencies.arms:
        model,optimizer=dependencies.make_model(assets,arm,device)
        actual=_array(model())
        if actual.shape!=expected.shape or not np.array_equal(actual,expected):
            raise ValueError('Different initial physical velocity: '+arm['name'])
        checked.append(arm['name']);del model,optimizer
    dependencies.cleanup(device)
    return checked


def _envelope(control,protocol,stage,**values):
    return dict(schema='batched_encoding_stage_result_v1',stage=stage,
        run_id=control.run_id,protocol_signature=protocol['protocol_signature'],
        code_commit_recorded=protocol['code_commit_recorded'],**values)


def _correctness(control,costs,assets,protocol,dependencies,device,initial,context=None):
    control.check_stop()
    problem=dependencies.build_problem(assets,'noisy',device)
    model,optimizer=dependencies.make_model(assets,dependencies.arms[0],device)
    velocity=model()
    if hasattr(velocity,'detach'):velocity=velocity.detach()
    costs.set_context(phase='correctness',condition='noisy',arm='oracle',update=0)
    try:
        result=dependencies.run_correctness(problem,velocity,record_cost=costs.record,update=1)
    except BaseException as error:
        if hasattr(error,'report'):
            report=dict(error.report)
            report.update(_envelope(control,protocol,'correctness',
                initial_velocity_equal_arms=initial))
            control.write_json('correctness.json',report)
        raise
    report=dict(result)
    report.update(_envelope(control,protocol,'correctness',
        initial_velocity_equal_arms=initial))
    control.write_json('correctness.json',report)
    if not report.get('passed'):raise RuntimeError('Correctness gate failed; no training')
    if report.get('cost_totals')!=costs.totals:
        raise ValueError('Correctness report and append-only cost events disagree')
    control.check_stop()
    return report


def _append_json(path,row):
    with Path(path).open('a',encoding='utf-8')as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')


def _save_array(path,value):
    with Path(path).open('xb')as stream:np.save(stream,_array(value),allow_pickle=False)


def _text_snapshot(control,text):
    """Same bounded metadata-only publication policy as SuiteControl JSON."""
    target=control.output/'progress.md'
    temp=target.with_name('.progress.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x',encoding='utf-8')as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    delays=(.01,.02,.04,.08,.10)
    for index in range(6):
        try:os.replace(temp,target);return
        except OSError as error:
            code=getattr(error,'winerror',None)
            if code not in(5,32,33):raise
            if index==5:raise MetadataPublishError(temp,target,6,code)from error
            time.sleep(delays[index])


def _publish_progress(control,stage,cases,costs,*,status='RUNNING',**details):
    rows=[dict(condition=item['condition'],arm=item['arm']['name'],
        completed_updates=item['completed_updates'],latest_checkpoint=item.get('checkpoint'),
        evaluation_updates=[x['completed_updates']for x in item.get('evaluations',[])])for item in cases]
    state=dict(stage=stage,status=status,run_id=control.run_id,cases=rows,
        cost_totals=dict(costs.totals),**details)
    control.write_json('status.json',state)
    lines=['# Batch encoding suite progress','',f"Run: {control.run_id}; stage: {stage}; status: {status}", '',
        '|Condition|Arm|Completed updates|','|---|---|---:|']
    lines.extend(f"|{r['condition']}|{r['arm']}|{r['completed_updates']}|"for r in rows)
    lines.extend(['',f"Actual completed physical fields: F={costs.totals['forward_fields_complete']}, B={costs.totals['backward_fields_complete']}",
        '', '[Status](status.json) | [Ledger](ledger.jsonl) | [Terminal receipt](control/EXIT.json)',
        '', 'These are current progress records, not a final result or a speed claim.'])
    _text_snapshot(control,'\n'.join(lines)+'\n')


def _case_metadata(protocol,stage,condition,arm):
    return dict(schema='batched_encoding_case_checkpoint_v1',
        protocol_signature=protocol['protocol_signature'],stage=stage,
        condition=condition,arm=dict(arm),encoding_seed=3)


def _activate(control,dependencies,assets,protocol,stage,case,problem,device,context):
    model,optimizer=dependencies.make_model(assets,case['arm'],device)
    parameter_count=sum(parameter.numel()for parameter in model.parameters()if parameter.requires_grad)
    if case.get('trainable_parameters',parameter_count)!=parameter_count:
        raise ValueError('Restored model trainable parameter count differs')
    case['trainable_parameters']=int(parameter_count)
    metadata=_case_metadata(protocol,stage,case['condition'],case['arm'])
    completed=0
    if case.get('checkpoint'):
        completed=dependencies.load_checkpoint(model,optimizer,Path(case['checkpoint']),metadata)
    if completed!=case['completed_updates']:
        raise ValueError('Restored case update differs from its checkpoint index')
    if hasattr(model,'set_progress'):model.set_progress(completed)
    encoder=dependencies.make_encoder(problem,case['arm'])
    active=dict(model=model,optimizer=optimizer,encoder=encoder,case=case,metadata=metadata)
    def save_safe():
        n=case['completed_updates']
        if case.get('checkpoint_update')!=n:
            path=Path(case['directory'])/f'checkpoint_{n:04d}.pt'
            dependencies.save_checkpoint(model,optimizer,path,n,metadata)
            case.update(checkpoint=str(path),checkpoint_update=n)
            control.write_json(Path(case['relative'])/'latest.json',
                dict(completed_updates=n,path=path.name,metadata=metadata))
            control.record('safe_checkpoint_saved',stage=stage,condition=case['condition'],
                arm=case['arm']['name'],update=n,path=str(path))
        context['last_safe_checkpoint']=case.get('checkpoint')
        return case.get('checkpoint')
    active['save_safe']=save_safe
    context.update(save_safe_checkpoint=save_safe,active_case=dict(condition=case['condition'],
        arm=case['arm']['name']),last_safe_checkpoint=case.get('checkpoint'))
    return active


def _commit_row(control,costs,active,row,phase):
    case=active['case'];expected=case['completed_updates']+1
    if row.get('completed_updates')!=expected:
        raise ValueError('train_step returned an unexpected completed update')
    case['completed_updates']=expected
    history=dict(row,condition=case['condition'],arm=case['arm']['name'],phase=phase)
    _append_json(Path(case['directory'])/'history.jsonl',history)
    control.record('update_committed',phase=phase,condition=case['condition'],
        arm=case['arm']['name'],update=expected,result=row)


def _deactivate(context,dependencies,device,active):
    context['save_safe_checkpoint']=None
    # Remove closure references before the next model is allocated.
    active.clear();dependencies.cleanup(device)


def _new_case(control,condition,arm):
    relative=Path('cases')/condition/arm['name'];directory=control.output/relative
    directory.mkdir(parents=True,exist_ok=False)
    return dict(condition=condition,arm=dict(arm),completed_updates=0,
        relative=str(relative),directory=str(directory),checkpoint=None,
        checkpoint_update=None,evaluations=[])


def _timing(control,costs,assets,protocol,dependencies,device,initial,context):
    problem=dependencies.build_problem(assets,'noisy',device);cases=[];results=[]
    context['cases']=cases
    for arm in dependencies.arms:
        control.check_stop();case=_new_case(control,'noisy',arm);cases.append(case)
        dependencies.reset_peak(device)
        active=_activate(control,dependencies,assets,protocol,'timing',case,problem,device,context)
        samples=[];warmups=[]
        for index in range(12):
            phase='timing_warmup'if index<2 else'timing_measured'
            control.check_stop();costs.set_context(phase=phase,condition='noisy',arm=arm['name'],update=index+1)
            row=dependencies.train_step(active['model'],active['optimizer'],active['encoder'],index,
                record_cost=costs.record,check_stop=control.check_stop,phase=phase,arm_name=arm['name'])
            _commit_row(control,costs,active,row,phase)
            if not math.isfinite(float(row['wall_seconds']))or float(row['wall_seconds'])<=0:
                raise ValueError('Invalid synchronized wall-time sample')
            (warmups if index<2 else samples).append(row)
            control.check_stop()
        active['save_safe']()
        wall=[float(row['wall_seconds'])for row in samples]
        cuda=[float(row['cuda_ms'])for row in samples if row.get('cuda_ms')is not None]
        if str(device).startswith('cuda')and(len(cuda)!=10 or not all(math.isfinite(v)and v>=0 for v in cuda)):
            raise ValueError('Ten CUDA event samples are required on GPU')
        results.append(dict(condition='noisy',arm=arm['name'],fields=arm['fields'],
            trainable_parameters=case['trainable_parameters'],
            completed_updates=12,warmup_updates=2,warmup_samples=warmups,samples=samples,
            wall_seconds_mean=statistics.mean(wall),wall_seconds_median=statistics.median(wall),
            cuda_ms_mean=statistics.mean(cuda)if cuda else None,
            cuda_ms_median=statistics.median(cuda)if cuda else None,
            checkpoint=case['checkpoint'],**dependencies.memory_peak(device)))
        _publish_progress(control,'timing',cases,costs)
        _deactivate(context,dependencies,device,active)
    expected=12*sum(arm['fields']for arm in dependencies.arms)
    if costs.totals!={key:expected for key in COUNTERS}:
        raise ValueError('Timing actual cost differs from two warmup plus ten measured updates')
    report=_envelope(control,protocol,'timing',passed=True,cases=results,
        cost_totals=dict(costs.totals),initial_velocity_equal_arms=initial,
        timing_scope='10 synchronized per-update wall/CUDA samples after2 warmups; preparation/evaluation/checkpoint outside these samples',
        speed_claim='Measured results only; no assumed batch speedup')
    control.write_json('timing.json',report);return report


def _validate_preflight(paths,protocol):
    if len(paths)!=2:raise ValueError('Exactly correctness and timing preflight receipts are required')
    found={};records=[]
    for supplied in paths:
        supplied=Path(supplied).resolve();directory=supplied if supplied.is_dir()else supplied.parent
        summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        stage=summary.get('stage')
        if stage not in('correctness','timing')or stage in found:
            raise ValueError('Preflight stages are missing, duplicated or invalid')
        report_path=directory/(stage+'.json');report=json.loads(report_path.read_text(encoding='utf-8'))
        old_protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
        receipt_path=directory/'control'/'EXIT.json';receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        signature=protocol['protocol_signature']
        if old_protocol.get('protocol_signature')!=_signature(old_protocol['common']):
            raise ValueError('Preflight protocol contents do not match its signature')
        if any(obj.get('protocol_signature')!=signature for obj in(summary,report,old_protocol,receipt)):
            raise ValueError('Preflight physics/data/model/runtime signature differs')
        run_id=summary.get('run_id')
        if(not isinstance(run_id,str)or not run_id.strip()or
            summary.get('status')!='COMPLETED'or summary.get('exit_code')!=0 or
            receipt.get('status')!='COMPLETED'or receipt.get('exit_code')!=0 or
            receipt.get('stage')!=stage or report.get('stage')!=stage or report.get('passed')is not True or
            len({summary.get('run_id'),report.get('run_id'),receipt.get('run_id')})!=1):
            raise ValueError('Preflight did not finish successfully with matching identity')
        arms=protocol['common']['arms'];expected_fields=48 if stage=='correctness'else 12*sum(arm['fields']for arm in arms)
        expected_cost={key:expected_fields for key in COUNTERS}
        if any(obj.get('cost_totals')!=expected_cost for obj in(summary,report,receipt)):
            raise ValueError('Preflight actual physical field totals are incomplete')
        if report.get('initial_velocity_equal_arms')!=[arm['name']for arm in arms]:
            raise ValueError('Preflight did not verify every canonical initial velocity')
        if stage=='timing':
            cases=report.get('cases',[])
            if len(cases)!=len(arms)or {case.get('arm')for case in cases}!={arm['name']for arm in arms}:
                raise ValueError('Preflight timing arms are incomplete')
            for case in cases:
                arm=next(arm for arm in arms if arm['name']==case['arm'])
                samples=case.get('samples',[]);warmups=case.get('warmup_samples',[])
                if(case.get('completed_updates')!=12 or case.get('warmup_updates')!=2 or
                    case.get('fields')!=arm['fields']or case.get('condition')!='noisy'or
                    [row.get('completed_updates')for row in warmups]!=[1,2]or
                    [row.get('completed_updates')for row in samples]!=list(range(3,13))):
                    raise ValueError('Preflight timing requires2 warmups and10 measured samples per arm')
                if any(not math.isfinite(float(row['wall_seconds']))or float(row['wall_seconds'])<=0 for row in samples):
                    raise ValueError('Preflight timing has invalid synchronized wall samples')
                if str(protocol['common']['runtime']['device']).startswith('cuda')and any(
                    row.get('cuda_ms')is None or not math.isfinite(float(row['cuda_ms']))or float(row['cuda_ms'])<0 for row in samples):
                    raise ValueError('Preflight timing has invalid CUDA samples')
        found[stage]=report
        records.append(dict(stage=stage,directory=str(directory),
            report_sha256=_sha(report_path),exit_sha256=_sha(receipt_path),
            protocol_sha256=_sha(directory/'protocol.json')))
    if set(found)!={'correctness','timing'}:raise ValueError('Both preflight stages are required')
    return records


def _train(control,costs,assets,protocol,dependencies,device,initial,*,preflight,context):
    records=_validate_preflight(preflight,protocol)
    control.write_json('preflight_validation.json',dict(passed=True,receipts=records,
        protocol_signature=protocol['protocol_signature']))
    problems={condition:dependencies.build_problem(assets,condition,device)for condition in CONDITIONS}
    control.check_stop()
    model,optimizer=dependencies.make_model(assets,dependencies.arms[0],device)
    costs.set_context(phase='initial_shared_validation',condition='shared',arm='all',update=0)
    metrics,prediction,velocity=dependencies.evaluate(model,problems['clean'],assets,
        record_cost=costs.record,check_stop=control.check_stop,label='initial_shared')
    folder=control.output/'initial_shared';folder.mkdir()
    _save_array(folder/'prediction.npy',prediction);_save_array(folder/'velocity.npy',velocity)
    control.write_json('initial_shared/metrics.json',dict(metrics,
        shared_all_cases=True,extra_clean_supervision=False,completed_updates=0))
    del model,optimizer,prediction,velocity;dependencies.cleanup(device)
    cases=[_new_case(control,condition,arm)for condition in CONDITIONS for arm in dependencies.arms]
    context['cases']=cases
    for milestone in MILESTONES:
        for case in cases:
            control.check_stop();condition=case['condition'];arm=case['arm']
            active=_activate(control,dependencies,assets,protocol,'train',case,problems[condition],device,context)
            while case['completed_updates']<milestone:
                control.check_stop();n=case['completed_updates']
                costs.set_context(phase='training',condition=condition,arm=arm['name'],update=n+1)
                row=dependencies.train_step(active['model'],active['optimizer'],active['encoder'],n,
                    record_cost=costs.record,check_stop=control.check_stop,phase='training',arm_name=arm['name'])
                _commit_row(control,costs,active,row,'training')
                n=case['completed_updates']
                if n%CHECKPOINT_INTERVAL==0 and n!=milestone:active['save_safe']()
                if n%STATUS_INTERVAL==0:_publish_progress(control,'train',cases,costs,current_milestone=milestone)
                control.check_stop()
            costs.set_context(phase='validation',condition=condition,arm=arm['name'],update=milestone)
            label=condition+'_'+arm['name']+f'_update{milestone:04d}'
            metrics,prediction,velocity=dependencies.evaluate(active['model'],problems[condition],assets,
                record_cost=costs.record,check_stop=control.check_stop,label=label)
            relative=Path(case['relative'])/f'validation_{milestone:04d}'
            folder=control.output/relative;folder.mkdir()
            _save_array(folder/'prediction.npy',prediction);_save_array(folder/'velocity.npy',velocity)
            metrics=dict(metrics,completed_updates=milestone,condition=condition,arm=arm['name'])
            control.write_json(relative/'metrics.json',metrics)
            case['evaluations'].append(metrics)
            active['save_safe']()
            del prediction,velocity
            _publish_progress(control,'train',cases,costs,current_milestone=milestone)
            _deactivate(context,dependencies,device,active)
            control.check_stop()
        control.write_json('train_partial.json',_envelope(control,protocol,'train',
            completed_milestone=milestone,cases=cases,cost_totals=dict(costs.totals)))
    final=MILESTONES[-1]
    if len(cases)!=2*len(dependencies.arms)or any(c['completed_updates']!=final or len(c['evaluations'])!=len(MILESTONES)for c in cases):
        raise ValueError('All fixed case endpoints have not completed')
    training_fields=len(CONDITIONS)*sum(a['fields']for a in dependencies.arms)*final
    evaluation_fields=13*(1+len(cases)*len(MILESTONES))
    expected=dict(forward_fields_attempt=training_fields+evaluation_fields,
        forward_fields_complete=training_fields+evaluation_fields,
        backward_fields_attempt=training_fields,backward_fields_complete=training_fields)
    if costs.totals!=expected:raise ValueError('Actual complete training/validation cost differs from fixed protocol')
    report=_envelope(control,protocol,'train',passed=True,cases=cases,
        cost_totals=dict(costs.totals),training_fields_each_direction=training_fields,
        validation_forward_fields=evaluation_fields,initial_velocity_equal_arms=initial,
        preflight=records,selection='fixed endpoints; no truth-based selection')
    control.write_json('train.json',report);return report


def run_suite(*,stage,output,dataset_index,reference_config,initial_checkpoint=None,
              code_commit,preflight=None,device='cuda:0',run_id=None,dependencies=None):
    if stage not in ('correctness','timing','train'):raise ValueError('Unknown suite stage')
    if initial_checkpoint is None:raise ValueError('initial-checkpoint is required for the canonical zero-update frame')
    if stage=='train'and not preflight:raise ValueError('train requires successful correctness/timing preflight receipts')
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    output.mkdir(exist_ok=False)
    control=SuiteControl(output,run_id or uuid.uuid4().hex,output.parent/'ENCODING_SUITE_GPU.lock')
    control.acquire();costs=Costs(control);finished=False;started=time.perf_counter()
    protocol=None;context={}
    try:
        dependencies=dependencies or default_dependencies()
        control.check_stop()
        assets=dependencies.load_assets(dataset_index,reference_config,initial_checkpoint)
        runtime=dependencies.configure_runtime(device)
        protocol=build_protocol(assets,runtime,dependencies,code_commit)
        control.write_json('protocol.json',protocol)
        if stage=='train':_validate_preflight(preflight,protocol)
        initial=_initial_checks(assets,dependencies,device)
        control.record('initial_velocity_all_arms_exact',arms=initial)
        function={'correctness':_correctness,'timing':_timing,'train':_train}[stage]
        report=function(control,costs,assets,protocol,dependencies,device,initial,
            context=context,**({'preflight':preflight}if stage=='train'else{}))
        summary=_envelope(control,protocol,stage,status='COMPLETED',exit_code=0,
            cost_totals=dict(costs.totals),stage_wall_seconds=time.perf_counter()-started,
            stage_result_file=stage+'.json')
        control.write_json('summary.json',summary)
        _publish_progress(control,stage,context.get('cases',[]),costs,status='COMPLETED')
        control.finish('COMPLETED',0,stage=stage,protocol_signature=protocol['protocol_signature'],
            cost_totals=dict(costs.totals));finished=True
        return summary
    except StopRequested as error:
        # Training adds its current safe frame to context; non-training has no Adam state.
        if context.get('save_safe_checkpoint'):context['save_safe_checkpoint']()
        summary=dict(stage=stage,status='STOPPED_BY_USER',exit_code=130,
            cost_totals=dict(costs.totals),stage_wall_seconds=time.perf_counter()-started,
            reason=str(error),last_safe_checkpoint=context.get('last_safe_checkpoint'))
        control.write_json('summary.json',summary)
        _publish_progress(control,stage,context.get('cases',[]),costs,status='STOPPED_BY_USER')
        control.finish('STOPPED_BY_USER',130,stage=stage,cost_totals=dict(costs.totals),
            last_safe_checkpoint=context.get('last_safe_checkpoint'));finished=True
        return summary
    except BaseException as error:
        failure=dict(stage=stage,status='FAILED',exit_code=1,error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc(),cost_totals=dict(costs.totals),
            cost_record_error=costs.record_error,last_safe_checkpoint=context.get('last_safe_checkpoint'),
            committed_but_failed_update=getattr(error,'completed_updates',None),
            optimizer_step_outcome_uncertain=bool(getattr(error,'optimizer_step_started',False)or
                (type(error).__name__=='UncertainOptimizerError'and hasattr(error,'attempted_update'))),
            attempted_optimizer_update=getattr(error,'attempted_update',None),
            stage_wall_seconds=time.perf_counter()-started)
        control.write_json('failure.json',failure);control.write_json('summary.json',failure)
        _publish_progress(control,stage,context.get('cases',[]),costs,status='FAILED')
        control.finish('FAILED',1,stage=stage,cost_totals=dict(costs.totals),error=str(error));finished=True
        return failure
    finally:
        if finished:control.release()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('correctness','timing','train'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-index',type=Path,required=True)
    parser.add_argument('--reference-config',type=Path,required=True)
    parser.add_argument('--initial-checkpoint',type=Path,required=True)
    parser.add_argument('--code-commit',required=True)
    parser.add_argument('--preflight',type=Path,nargs='+')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--run-id')
    args=vars(parser.parse_args(argv))
    result=run_suite(**args);print(json.dumps(result,allow_nan=False),flush=True)
    return int(result['exit_code'])


if __name__=='__main__':raise SystemExit(main())

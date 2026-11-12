"""Prepare dense GPU checks; execution requires an explicit flag and release proof.

No1000-update mode. The default only writes a CPU-prepared plan. Dense records
must be synthesized separately, never relabeled from the old13-shot cache.
"""
from __future__ import annotations
import argparse,copy,gc,hashlib,json,random,statistics,time,uuid
from pathlib import Path
import numpy as np
import torch
from experiments.encoding_suite_data import load_assets
from experiments.residual_ifwi_experiment import model_from_config
from experiments.encoding_training_step import configure_runtime,_sync
from experiments.dense_encoding_physics import (generate_sources,build_dense_problem,
    DenseAcquisition,train_dense_step,evaluate_dense)
from experiments.dense_checkpoint import save_dense_checkpoint
from experiments.gated_siren import GatedSirenVelocity
from ifwi_research.masked_adam import MaskedAdam
from ifwi_research.suite_control import SuiteControl,StopRequested
from experiments.batched_encoding_suite import Costs,COUNTERS

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent.parent if ROOT.parent.name=='.worktrees'else ROOT
ACTIVE_RUN=WORKSPACE/'preparations/batched_encoding_20261112/gpu_train_01'
ARMS=tuple(dict(name=n,grouping=g,fields=b,model=m)for n,g,b,m in (
    ('full','identity',None,'plain'),('global1','global',1,'plain'),
    ('global2','global',2,'plain'),('interleaved2','interleaved',2,'plain'),
    ('stratified2','stratified',2,'plain'),
    ('stratified2_static','stratified',2,'static'),('stratified2_gated','stratified',2,'gated')))


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(assets,nshots=52,batch_size=13):
    if nshots not in(48,52)or type(batch_size)is not int or not 1<=batch_size<=nshots:
        raise ValueError('Choose48/52 shots and a positive batch_size <=N')
    cfg=copy.deepcopy(assets['config']);nz,nx=assets['truth'].shape
    xs=generate_sources(nx,cfg['data']['grid_spacing_m'],nshots,75.,10)
    cfg['acquisition'].update(num_shots=nshots,source_x_indices=xs,source_spacing_m=75.,source_margin_cells=10)
    cfg['training'].update(epochs=200,shot_batch_size=batch_size)
    physical=dict(geometry=cfg['acquisition'],forward=cfg['forward'],dx_m=cfg['data']['grid_spacing_m'],
        input_array_sha256={n:hashlib.sha256(np.asarray(assets[n]).tobytes()).hexdigest()for n in('truth','background','wavelet')})
    sources=('core_speed.py','rnn_fd.py','ifwi_modules.py','ifwi_research/batched_sources.py',
        'ifwi_research/masked_adam.py','experiments/gated_siren.py','experiments/dense_checkpoint.py',
        'experiments/dense_encoding_physics.py','experiments/dense_encoding_suite.py')
    common=dict(config=cfg,physical=physical,arms=list(ARMS),source_sha256={n:_sha(ROOT/n)for n in sources},
        initial_checkpoint=assets.get('manifest',{}).get('initial_checkpoint'),
        objective='sum coded squared residual/(M*N*T*R)',gate='32+floor(96*clip(k/200,0,1)); hard0/1',
        batch_limit_evidence='unmeasured configurable limit; GPU measurement required')
    return dict(**common,signature=_hash(common),data_key=_hash(physical),status='PREPARED_NOT_RUN',
        dense_observations_required=True,quick_updates=200,timing_measured_updates=10,warmup_updates=2,
        full_batch_fields=[min(batch_size,nshots-i)for i in range(0,nshots,batch_size)],
        historical_memory_basis='fa2cdc5 full13 peak allocated10.123GB; dense data/cache overhead and maximal batch remain unmeasured')


def make_dense_model(assets,arm,device):
    torch.manual_seed(3);np.random.seed(3);random.seed(3)
    base=model_from_config(assets['background'],assets['config'])
    if assets.get('canonical_checkpoint')is not None:
        base.load_state_dict(assets['canonical_checkpoint']['model'],strict=True)
    if arm['model']=='plain':model=base
    else:
        cfg=assets['config'];m=cfg['model']
        model=GatedSirenVelocity(assets['background'],cfg['data']['grid_spacing_m'],m['neurons'],
            mean_kmps=m['mean_kmps'],std_kmps=m['std_kmps'],initial_channels=128 if arm['model']=='static'else 32)
        model.net.load_state_dict(base.net.state_dict(),strict=True)
    model=model.to(device)
    with torch.no_grad():
        if not torch.equal(model().cpu(),torch.as_tensor(assets['background'],dtype=torch.float32)[None]):
            raise ValueError('Initial physical velocity differs')
    optimizer=(torch.optim.Adam if arm['model']=='plain'else MaskedAdam)(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8)
    return model,optimizer


def gpu_execution_guard(active_run):
    active=Path(active_run).resolve();lock=active.parent/'ENCODING_SUITE_GPU.lock'
    if lock.exists():raise RuntimeError('GPU suite lock exists; no concurrent GPU work')
    try:
        start=json.loads((active/'control/LAUNCH.json').read_text())
        end=json.loads((active/'control/EXIT.json').read_text())
    except (OSError,ValueError)as error:raise RuntimeError('Explicit matching terminal proof required')from error
    valid=(end.get('status')=='COMPLETED'and end.get('exit_code')==0 or
        end.get('status')=='STOPPED_BY_USER'and end.get('exit_code')==130 or
        end.get('status')=='FAILED'and type(end.get('exit_code'))is int and end['exit_code']!=0)
    if not start.get('run_id')or start.get('run_id')!=end.get('run_id')or not valid:
        raise RuntimeError('Terminal proof identity/status mismatch')
    return lock


def _problem(assets,plan,observed,device):
    return build_dense_problem(assets['truth'],plan['config'],device,observed=observed,wavelet=assets['wavelet'])


def _dense_data(directory,plan):
    directory=Path(directory);meta=json.loads((directory/'data_manifest.json').read_text())
    if meta['data_key']!=plan['data_key']:raise ValueError('Dense data geometry/physical inputs differ')
    result={}
    for name in('clean','noisy'):
        path=directory/(name+'.npy')
        if _sha(path)!=meta[name+'_sha256']:raise ValueError('Dense record hash mismatch')
        result[name]=np.load(path,allow_pickle=False)
    return result


def _synthesize(control,costs,assets,plan,device):
    n=plan['config']['acquisition']['num_shots'];fd=plan['config']['forward'];nx=assets['truth'].shape[1]
    shape=(1,n,fd['nt'],len(range(0,nx,plan['config']['acquisition']['receiver_stride'])))
    problem=_problem(assets,plan,np.zeros(shape,np.float32),device)
    acquisition=DenseAcquisition(problem,plan['config']['training']['shot_batch_size'])
    records=[];vp=torch.as_tensor(assets['truth'],dtype=torch.float32,device=device)[None]
    with torch.no_grad():
        for batch in acquisition.batches(0):
            control.check_stop();k=batch['observed'].shape[1];_sync(vp.device);tick=time.perf_counter()
            costs.record('data_F_attempt',phase='data_generation',forward_fields_attempt=k)
            prediction=batch['propagators'][0]['solver'](vp,problem['wavelet'])[2]
            _sync(vp.device)
            costs.record('data_F_complete',phase='data_generation',forward_fields_complete=k,wall_seconds=time.perf_counter()-tick)
            if prediction.shape!=batch['observed'].shape or not torch.isfinite(prediction).all():
                raise ValueError('Invalid generated records: shape or finiteness')
            records.append(prediction.cpu().numpy().copy())
    clean=np.concatenate(records,axis=1)
    if clean.shape!=shape or not np.isfinite(clean).all()or np.mean(clean.astype(float)**2)<=0:
        raise ValueError('Invalid generated records: full geometry or zero signal energy')
    noise=np.random.RandomState(3).normal(size=clean.shape)
    scale=np.sqrt(np.mean(clean.astype(np.float64)**2)*10**1.2/np.mean(noise**2))
    with np.errstate(over='ignore'):
        noisy=(clean.astype(np.float64)+noise*scale).astype(np.float32)
    if not np.isfinite(noisy).all():raise ValueError('Invalid generated noisy records: float32 overflow')
    np.save(control.output/'clean.npy',clean);np.save(control.output/'noisy.npy',noisy)
    meta=dict(data_key=plan['data_key'],noise_seed=3,requested_snr_db=-12.,
        measured_snr_db=float(10*np.log10(np.mean(clean.astype(float)**2)/np.mean((noisy.astype(float)-clean)**2))),
        clean_sha256=_sha(control.output/'clean.npy'),noisy_sha256=_sha(control.output/'noisy.npy'))
    control.write_json('data_manifest.json',meta)
    return meta


class DenseCosts(Costs):
    def __init__(self,control):
        super().__init__(control);self.by_phase={}
    def record(self,event,**details):
        row=super().record(event,**details)
        totals=self.by_phase.setdefault(details.get('phase','other'),{key:0 for key in COUNTERS})
        for key in COUNTERS:totals[key]+=int(details.get(key,0))
        return row


def execute(*,stage,output,assets,plan,dense_data=None,active_run=ACTIVE_RUN,device='cuda:0'):
    lock=gpu_execution_guard(active_run) # Before CUDA configuration or output creation.
    if stage not in('synthesize','timing','quick'):raise ValueError('No execution mode for this stage')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    control=SuiteControl(output,uuid.uuid4().hex,lock);control.acquire();costs=DenseCosts(control)
    frame={};cases=[];started=time.perf_counter();finished=False
    optimizer_updates=dict(warmup=0,training=0)
    def save_safe():
        if not frame:return
        path=frame['directory']/f"{frame['phase']}_{frame['completed']:04d}.pt"
        if not path.exists():
            save_dense_checkpoint(frame['model'],frame['optimizer'],path,frame['completed'],frame['metadata'])
        frame['checkpoint']=str(path)
    try:
        runtime=configure_runtime(device);control.write_json('plan.json',plan);control.write_json('runtime.json',runtime)
        if stage=='synthesize':result=_synthesize(control,costs,assets,plan,device)
        else:
            if dense_data is None:raise ValueError('Explicit dense data directory required')
            data=_dense_data(dense_data,plan);budget=10 if stage=='timing'else 200
            for condition in(('noisy',)if stage=='timing'else('clean','noisy')):
                problem=_problem(assets,plan,data[condition],device)
                for arm in ARMS:
                    control.check_stop();directory=output/condition/arm['name'];directory.mkdir(parents=True)
                    acquisition=DenseAcquisition(problem,plan['config']['training']['shot_batch_size'],arm['grouping'],arm['fields'],3)
                    torch.cuda.reset_peak_memory_stats(device);samples=[]
                    # Quick training starts fresh after warmup; timing continues the same model.
                    model,opt=make_dense_model(assets,arm,device)
                    metadata=dict(signature=plan['signature'],condition=condition,arm=arm,stage=stage)
                    frame.update(model=model,optimizer=opt,directory=directory,completed=0,phase='warmup',metadata=metadata,checkpoint=None)
                    for k in range(2):
                        row=train_dense_step(model,opt,acquisition,k,record_cost=costs.record,check_stop=control.check_stop,phase='warmup',arm_name=arm['name'])
                        if row['completed_updates']!=k+1:raise ValueError('Unexpected committed update')
                        optimizer_updates['warmup']+=1
                        frame['completed']=k+1;control.record('warmup_update',condition=condition,arm=arm['name'],result=row)
                    if stage=='quick':model,opt=make_dense_model(assets,arm,device)
                    frame.update(model=model,optimizer=opt,completed=2 if stage=='timing'else 0,phase='measured',checkpoint=None)
                    if stage=='quick':
                        metrics,pred,vel=evaluate_dense(model,acquisition,dict(data,truth=assets['truth']),record_cost=costs.record,check_stop=control.check_stop,label=arm['name']+'_initial')
                        control.write_json(directory.relative_to(output)/'metrics_0000.json',metrics)
                    for index in range(budget):
                        row=train_dense_step(model,opt,acquisition,frame['completed'],record_cost=costs.record,check_stop=control.check_stop,phase='training',arm_name=arm['name'])
                        if row['completed_updates']!=frame['completed']+1:raise ValueError('Unexpected committed update')
                        optimizer_updates['training']+=1
                        frame['completed']=row['completed_updates'];samples.append(row)
                        control.record('update',condition=condition,arm=arm['name'],result=row)
                        if stage=='quick'and frame['completed']%25==0:save_safe()
                        if(index+1)%10==0:control.write_json('status.json',dict(status='RUNNING',condition=condition,arm=arm['name'],completed=frame['completed'],cost_totals=costs.totals))
                        control.check_stop()
                    save_safe()
                    if stage=='quick':
                        metrics,pred,vel=evaluate_dense(model,acquisition,dict(data,truth=assets['truth']),record_cost=costs.record,check_stop=control.check_stop,label=arm['name']+'_200')
                        control.write_json(directory.relative_to(output)/'metrics_0200.json',metrics)
                        np.save(directory/'velocity_0200.npy',vel);np.save(directory/'prediction_0200.npy',pred)
                    cases.append(dict(arm=arm['name'],condition=condition,completed_updates=frame['completed'],measured_updates=budget,
                        warmup_updates=2,wall_seconds_median=statistics.median(x['wall_seconds']for x in samples),
                        samples=samples,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),trainable_parameters=sum(p.numel()for p in model.parameters()),
                        optimizer_tensor_bytes=sum(v.numel()*v.element_size()for state in opt.state.values()for v in state.values()if isinstance(v,torch.Tensor)),checkpoint=frame['checkpoint']))
                    control.write_json('partial.json',dict(cases=cases,cost_totals=costs.totals))
                    frame.clear();del model,opt,acquisition;gc.collect();torch.cuda.empty_cache()
            result=dict(cases=cases)
        n=plan['config']['acquisition']['num_shots']
        total_fields=n if stage=='synthesize'else (n+11)*(12 if stage=='timing'else 404)
        evaluation_fields=28*n if stage=='quick'else 0
        expected=dict(forward_fields_attempt=total_fields+evaluation_fields,forward_fields_complete=total_fields+evaluation_fields,
            backward_fields_attempt=0 if stage=='synthesize'else total_fields,backward_fields_complete=0 if stage=='synthesize'else total_fields)
        if costs.totals!=expected:raise ValueError('Actual cost differs from fixed execution budget')
        report=dict(result,status='COMPLETED',stage=stage,code_signature=plan['signature'],cost_totals=costs.totals,
            cost_by_phase=costs.by_phase,optimizer_updates=optimizer_updates,stage_wall_seconds=time.perf_counter()-started)
        control.write_json('result.json',report);control.finish('COMPLETED',0,stage=stage,cost_totals=costs.totals);finished=True
        return 0
    except StopRequested:
        save_safe();control.finish('STOPPED_BY_USER',130,checkpoint=frame.get('checkpoint'),cost_totals=costs.totals);finished=True;return 130
    except BaseException as error:
        control.write_json('failure.json',dict(error_type=type(error).__name__,error=str(error),last_safe_checkpoint=frame.get('checkpoint'),cost_totals=costs.totals))
        control.finish('FAILED',1,error=str(error),cost_totals=costs.totals);finished=True;return 1
    finally:
        if finished:control.release()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','synthesize','timing','quick'),default='prepare')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-index',type=Path,required=True)
    parser.add_argument('--reference-config',type=Path,required=True)
    parser.add_argument('--initial-checkpoint',type=Path,required=True)
    parser.add_argument('--shots',type=int,choices=(48,52),default=52)
    parser.add_argument('--batch-size',type=int,default=13)
    parser.add_argument('--dense-data',type=Path)
    parser.add_argument('--active-run',type=Path,default=ACTIVE_RUN)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if args.execute and args.stage!='prepare':gpu_execution_guard(args.active_run)
    assets=load_assets(args.dataset_index,args.reference_config,args.initial_checkpoint)
    plan=build_plan(assets,args.shots,args.batch_size)
    if not args.execute or args.stage=='prepare':
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False),encoding='utf-8')
        print('PREPARED_NOT_RUN: no GPU work, no dense records generated',flush=True);return 0
    return execute(stage=args.stage,output=args.output,assets=assets,plan=plan,
        dense_data=args.dense_data,active_run=args.active_run,device=args.device)


if __name__=='__main__':raise SystemExit(main())

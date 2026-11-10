"""Runnable damped L2 natural-gradient comparison on the original residual IFWI.

No PDE Hessian, no learned prior, no truth-based selection and no auto retry.
Existing residual_ifwi.py / Adam semantics are unchanged. New CLI is explicit.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import uuid
import numpy as np
import torch
from experiments import residual_ifwi_experiment as base
from experiments import natural_gradient as ng
from experiments.residual_ifwi_stochastic import sample_shot_indices, select_shot_problem

SCHEMA='ifwi_damped_natural_gradient_v1'
ROOT=Path(__file__).resolve().parents[1]


def validate_config(config):
    if not isinstance(config,dict) or set(config)!={'base','optimizer','evaluation_interval'}:
        raise ValueError('Expected base, optimizer and evaluation_interval')
    base.validate_config(config['base'])
    opt=config['optimizer']
    if not isinstance(opt,dict) or set(opt)!={'kind','damping','step_size','cg_max_iter','cg_rtol'}:
        raise ValueError('Unknown/missing optimizer settings')
    if opt['kind'] not in ('adam','natural_gradient'):
        raise ValueError('Unknown optimizer kind')
    for key in ('damping','step_size','cg_rtol'):
        base._positive(opt[key],key)
    base._positive(opt['cg_max_iter'],'cg_max_iter',integer=True)
    base._positive(config['evaluation_interval'],'evaluation_interval',integer=True)
    return config


def source_hashes():
    names=['natural_gradient_ifwi.py','experiments/natural_gradient.py',
           'experiments/natural_gradient_experiment.py','experiments/residual_ifwi_experiment.py',
           'experiments/residual_ifwi_stochastic.py','experiments/residual_ifwi_data.py']
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names}


def resume_key(config):
    value=copy.deepcopy(config)
    value['base']['training'].pop('epochs')
    return value


def branch_key(config):
    value=copy.deepcopy(config)
    value.pop('experiment_name')
    for key in ('epochs','checkpoint_interval','full_eval_interval'):
        value['training'].pop(key,None)
    return value


def sync(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


class Ledger:
    """Append-only work counts; completed backward means autograd returned, not a hook."""
    def __init__(self,out,previous=None):
        self.path=out/'work.jsonl'
        self.counts=copy.deepcopy(previous) if previous is not None else {
            'forward_shots_attempted':0,'forward_shots_completed':0,
            'backward_shots_attempted':0,'backward_shots_completed':0,
            'updates_attempted':0,'updates_completed':0,'network_matvecs':0,'inner_pde_calls':0}
        self.phase='preparation'
    def add(self,key,n=1):
        self.counts[key]+=n
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                   phase=self.phase,event=key,increment=n,counts=self.counts),allow_nan=False)+'\n')
    def simulate(self,solver,velocity,wavelet):
        # The original solver stores acquisition xs on the rnn2D module.
        shots=int(solver.xs.shape[1])
        self.add('forward_shots_attempted',shots)
        result=base._simulate(solver,velocity,wavelet)
        self.add('forward_shots_completed',shots)
        return result


def evaluate(model,problem,ledger):
    with torch.no_grad():
        velocity=model()
        base.check_velocity(velocity,problem['config'])
        predictions=[ledger.simulate(i['solver'],velocity,problem['wavelet']) for i in problem['propagators']]
        predicted=torch.cat(predictions,dim=1)
        loss=base.waveform_mse(predicted,problem['observed'])
    return float(loss),predicted.detach().cpu().numpy(),velocity.detach().cpu().numpy()[0]


def update(model,problem,config,ledger,optimizer):
    indices=list(range(problem['observed'].shape[1]))
    count=config['base']['training'].get('shots_per_update')
    if count is not None:
        indices=sample_shot_indices(len(indices),count)
        problem=select_shot_problem(problem,indices)
    items=problem['propagators'];total=problem['observed'].shape[1]
    with torch.no_grad():before=model().detach().cpu().numpy()[0].copy()
    base.check_velocity(model(),problem['config'])
    ledger.add('updates_attempted')
    if config['optimizer']['kind']=='natural_gradient':
        opt=config['optimizer']
        def observer(event,batch):
            if event=='network_matvec':ledger.add('network_matvecs')
            else:
                shots=items[batch]['last']-items[batch]['first']
                ledger.add('backward_shots_'+event.split('_',1)[1],shots)
        result=ng.train_update_ng(ng.ResidualSirenView(model),problem,
            simulate=ledger.simulate,objective=base.waveform_mse,
            scale=1000*model.std_kmps,offset=model.fixed_init_mps if model.parameterization=='residual' else 1000*model.mean_kmps,
            damping=opt['damping'],step_size=opt['step_size'],cg_max_iter=opt['cg_max_iter'],cg_rtol=opt['cg_rtol'],
            validate_candidate=lambda v:base.check_velocity(v,problem['config']),observer=observer)
        result['data_mse']=result.pop('loss')
    else:
        optimizer.zero_grad(set_to_none=True);loss_value=0.
        for item in items:
            v=model();base.check_velocity(v,problem['config'])
            pred=ledger.simulate(item['solver'],v,problem['wavelet'])
            loss=base.waveform_mse(pred,problem['observed'][:,item['first']:item['last']],total)
            base.require_finite(loss,'loss')
            ledger.add('backward_shots_attempted',item['last']-item['first'])
            loss.backward()
            ledger.add('backward_shots_completed',item['last']-item['first'])
            loss_value+=float(loss.detach())
            del loss,pred,v
        for p in model.parameters():
            if p.grad is None:raise RuntimeError('Disconnected network parameter')
            base.require_finite(p.grad,'gradient')
        optimizer.step()
        for p in model.parameters():base.require_finite(p,'updated parameter')
        with torch.no_grad():base.check_velocity(model(),problem['config'])
        result=dict(data_mse=loss_value)
    # Physical work is never rolled back; only numerical state is rolled back on failure.
    return dict(**result,shot_indices=indices,velocity_before_mps=before)


def run_experiment(config,output_parent,device='cpu',resume=None,initialize_from=None):
    config=copy.deepcopy(validate_config(config))
    if resume and initialize_from:raise ValueError('resume and initialize_from are mutually exclusive')
    if torch.device(device).type not in ('cpu','cuda'):raise ValueError('Only CPU/CUDA qualified')
    if torch.device(device).type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    sources=base.original_source_hashes();runner_sources=source_hashes()
    truth,background,provenance=base.load_velocity_data(config['base'],ROOT)
    saved=None;start=0;history=[];previous=None
    if resume or initialize_from:
        # Only load trusted own checkpoints. torch pickle is not a sandbox.
        saved=torch.load(resume or initialize_from,map_location='cpu',weights_only=False)
        if saved.get('original_source_hashes')!=sources or saved.get('provenance')!=provenance:
            raise ValueError('Checkpoint source/data provenance differs')
        if resume:
            if saved.get('schema')!=SCHEMA or resume_key(saved['config'])!=resume_key(config):
                raise ValueError('Resume configuration/schema differs')
            if saved['runner_sources']!=runner_sources:raise ValueError('Resume runner code differs')
            history=copy.deepcopy(saved['history']);previous=saved['counts']
        else:
            if saved.get('schema') not in (SCHEMA,base.SCHEMA):raise ValueError('Unsupported branch checkpoint')
            oldcfg=saved['config']['base'] if saved['schema']==SCHEMA else saved['config']
            if branch_key(oldcfg)!=branch_key(config['base']):raise ValueError('Branch configuration differs')
        start=int(saved['completed_updates'])
        if start>config['base']['training']['epochs']:raise ValueError('Total epochs must not precede completed updates')
        if config['optimizer']['kind']=='adam' and saved.get('optimizer') is None:
            raise ValueError('Cannot silently reset Adam when branching/resuming')
    out=Path(output_parent).resolve()/f"{config['optimizer']['kind']}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir()
    ledger=Ledger(out,previous);carry=copy.deepcopy(ledger.counts)
    started=time.perf_counter();completed=start;model=optimizer=None;state_ready=False
    def status(state,**kw):base._json(out/'status.json',dict(state=state,completed_updates=completed,counts=ledger.counts,**kw))
    def checkpoint(path):
        if model is None or not state_ready:return
        payload=dict(schema=SCHEMA,config=config,model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),
            completed_updates=completed,history=history,rng=base._rng_state(),counts=ledger.counts,provenance=provenance,
            original_source_hashes=sources,runner_sources=runner_sources)
        temporary=path.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,path)
    base._json(out/'config.json',config)
    base._json(out/'provenance.json',provenance)
    base._json(out/'source_hashes.json',dict(core=sources,runner=runner_sources))
    base._json(out/'lineage.json',dict(resume=str(resume) if resume else None,initialize_from=str(initialize_from) if initialize_from else None,
        starting_update=start,previous_counts=carry,branch_previous_work='Not included in branch counts; inspect source checkpoint/run.'))
    base._json(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,device=str(device),threads=torch.get_num_threads()))
    status('preparing')
    try:
        base._seed(config['base']['seed'])
        problem=base.build_forward_problem(truth,config['base'],device,simulate=ledger.simulate)
        mc=config['base']['model']
        model=base.VelocityParameterization(background,config['base']['data']['grid_spacing_m'],mc['neurons'],mc['omega_0'],mc['mean_kmps'],mc['std_kmps'],config['base']['parameterization']).to(device)
        if config['optimizer']['kind']=='adam':optimizer=torch.optim.Adam(model.parameters(),lr=config['base']['training']['learning_rate'])
        if saved:
            model.load_state_dict(saved['model'],strict=True)
            if optimizer is not None:
                if saved.get('optimizer') is None:raise ValueError('Cannot silently reset Adam when branching/resuming')
                optimizer.load_state_dict(saved['optimizer'])
            base._restore_rng(saved['rng'])
        state_ready=True
        checkpoint(out/'checkpoint.pt')
        ledger.phase='initial_evaluation'
        initial_loss,pred,velocity=evaluate(model,problem,ledger)
        initial=dict(**base.velocity_metrics(velocity,truth),data_mse=initial_loss)
        for name,array in [('truth_mps',truth),('background_mps',background),('run_start_velocity',velocity),('observed',problem['observed'].cpu().numpy()),('wavelet',problem['wavelet'].cpu().numpy())]:np.save(out/(name+'.npy'),array)
        base._json(out/'acquisition.json',problem['geometry'])
        if torch.device(device).type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        sync(device);train_started=time.perf_counter()
        for k in range(start+1,config['base']['training']['epochs']+1):
            ledger.phase='training';t=time.perf_counter()
            model_before=copy.deepcopy(model.state_dict());opt_before=copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None;rng_before=base._rng_state()
            try:
                diag=update(model,problem,config,ledger,optimizer)
            except BaseException:
                model.load_state_dict(model_before)
                if optimizer is not None:optimizer.load_state_dict(opt_before)
                base._restore_rng(rng_before)
                raise
            completed=k
            ledger.add('updates_completed')
            sync(device)
            vbefore=diag.pop('velocity_before_mps')
            entry=dict(update=k,evaluated_updates=k-1,**diag,**base.velocity_metrics(vbefore,truth),step_seconds=time.perf_counter()-t)
            history.append(entry)
            with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
            status('running',updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            if k%config['evaluation_interval']==0 or k==config['base']['training']['epochs']:
                ledger.phase='evaluation';loss,pred,velocity=evaluate(model,problem,ledger)
                row=dict(update=k,**base.velocity_metrics(velocity,truth),data_mse=loss)
                with (out/'evaluation.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if k%config['base']['training']['checkpoint_interval']==0 or k==config['base']['training']['epochs']:
                checkpoint(out/'checkpoint.pt');checkpoint(out/'checkpoints'/f'update_{k:06d}.pt')
            print(f'{config["optimizer"]["kind"]} {k}: preupdate MSE={entry["data_mse"]:.8g}',flush=True)
        if start==config['base']['training']['epochs']:
            ledger.phase='finalization_evaluation'
            loss,pred,velocity=evaluate(model,problem,ledger)
        sync(device);elapsed=time.perf_counter()-train_started
        np.save(out/'final_velocity.npy',velocity);np.save(out/'final_prediction.npy',pred)
        observed=problem['observed'].detach().cpu().numpy()
        final=dict(**base.velocity_metrics(velocity,truth),data_mse=loss,data_mse_fp64=float(np.mean((pred.astype(np.float64)-observed.astype(np.float64))**2)))
        base._json(out/'metrics.json',dict(initial=initial,final=final,completed_updates=completed,starting_update=start,
            counts=ledger.counts,run_added_counts={k:ledger.counts[k]-carry[k] for k in carry},training_and_evaluation_seconds=elapsed,
            total_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else 0,
            selection='Fixed final update; truth is evaluation only; no clipping.',inner_metric='sum pixel L2 on normalized velocity, not physical Gauss-Newton'))
        status('completed');checkpoint(out/'checkpoint.pt')
        return out
    except BaseException as error:
        # Do not hide a checkpoint-writing failure with a false successful stop.
        try:checkpoint(out/'interrupted.pt')
        finally:status('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
        raise

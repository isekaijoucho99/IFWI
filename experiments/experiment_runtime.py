"""Training state and atomic, reproducible checkpoints for experiment runner v2."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from improved_modules.optimizers import create_improved_optimizer

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def atomic_save(path, value):
    path=Path(path); temporary=path.with_suffix('.tmp')
    torch.save(value,temporary); os.replace(temporary,path)


def source_hashes():
    files=[ROOT/n for n in ('ifwi_modules.py','rnn_fd.py','generator.py','plot_functions.py')]
    files+=list((ROOT/'experiments').rglob('*.py'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def array_hash(tensor):
    a=tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def contract(config, data):
    c=copy.deepcopy(config)
    for key in ('experiment_name','description','evaluation'): c.pop(key,None)
    for key in ('max_iterations','log_interval'): c.get('training',{}).pop(key,None)
    return {'config':c,'truth':array_hash(data['vp_true']),'observed':array_hash(data['shots']),
            'wavelet':array_hash(data['wavelet']),'params':data['params'],'sources':source_hashes()}


def comparison_contract(config,data):
    """Items held fixed for an ablation; intentionally excludes treatment choices."""
    model=config['model']
    return {'data':config.get('data',{}),'truth':array_hash(data['vp_true']),
            'observed':array_hash(data['shots']),'wavelet':array_hash(data['wavelet']),
            'params':data['params'],'seed':config.get('seed',42),
            'budget':config['training']['max_iterations'],
            'clip_grad':config['training'].get('clip_grad'),
            'backbone':{k:model.get(k) for k in ('neuron','omega_0','activation','outermost_linear','dropout')},
            'evaluation':{k:config.get('evaluation',{}).get(k,default) for k,default in [('depth_threshold',.5),('corner_size',.25)]},
            'sources':source_hashes()}


def validate_resume_contract(saved, expected, actual_shots, allow_execution_change=False):
    if saved == expected:
        return
    if not allow_execution_change:
        raise ValueError('Resume contract mismatch (config/data/source)')
    previous, current = copy.deepcopy(saved), copy.deepcopy(expected)
    for item in (previous, current):
        cfg = item['config']
        if cfg['model'].get('network_type', 'vanilla') != 'vanilla' or cfg['model'].get('dropout', False) or cfg.get('spatiotemporal', {}).get('enabled'):
            raise ValueError('Execution-change resume requires deterministic vanilla IFWI')
        cfg.get('training', {}).pop('shot_batch_size', None)
        data = cfg.setdefault('data', {})
        data.setdefault('num_shots', actual_shots)
    allowed = {'experiments/run_experiment.py', 'experiments/experiment_runtime.py',
        'experiments/run_ablation_suite.py', 'experiments/run_parameter_sweep.py',
        'experiments/parameter_sweep.py', 'experiments/summarize_parameter_sweep.py'}
    # Core solver, network, objective and optimizer files must match byte-for-byte.
    required = {'ifwi_modules.py', 'rnn_fd.py', 'generator.py',
        'experiments/improved_modules/losses.py', 'experiments/improved_modules/networks.py',
        'experiments/improved_modules/optimizers.py'}
    for name in required:
        if name not in saved['sources'] or saved['sources'][name] != expected['sources'].get(name):
            raise ValueError('Resume contract changed core source: ' + name)
    for name, digest in saved['sources'].items():
        if name not in allowed and expected['sources'].get(name) != digest:
            raise ValueError('Resume contract changed source: ' + name)
    previous.pop('sources'); current.pop('sources')
    if previous != current:
        raise ValueError('Resume contract changed model/data/objective/optimizer')


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.vel_net.state_dict().items()}


def velocity(model):
    previous=model.vel_net.training
    try:
        model.vel_net.eval()
        with torch.no_grad():
            v=(model.vel_net(model.coords)[0].squeeze(-1)*model.std+model.mean)*1000
            model.finite(v,'predicted velocity')
            return v.squeeze(0).cpu().numpy()
    finally: model.vel_net.train(previous)


def evaluate_loss(model,data,alpha):
    previous=model.vel_net.training
    try:
        model.vel_net.eval()
        # Coordinate TV requires autograd; no backward is called during evaluation.
        with torch.set_grad_enabled(bool(alpha)):
            _,loss,parts,_=model.objective(data['wavelet'],data['shots'],alpha)
            return float(loss.detach()),{k:float(v.detach()) for k,v in parts.items()}
    finally: model.vel_net.train(previous)


def train_loop(model,data,config,output_dir,resume=None,stop_after=None,allow_execution_change=False):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pth').exists() or (out/'loss_history.csv').exists():
        raise ValueError('Refuse to overwrite training output')
    tcfg=config['training']; total=int(tcfg['max_iterations']); interval=int(tcfg['log_interval'])
    if total<1 or interval<1: raise ValueError('Iterations and log_interval must be positive')
    alpha=float(tcfg.get('alpha',0))
    if not np.isfinite(alpha) or alpha<0: raise ValueError('Invalid alpha')
    opt,scheduler,_=create_improved_optimizer(model.vel_net,config['optimizer'])
    expected=contract(config,data)
    initial_velocity = velocity(model)
    completed=0; history=[]; best_state=None; best_loss=float('inf'); best_update=0
    selection_metric = getattr(model, 'selection_metric', 'total_loss')
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        if ck.get('schema')!=2: raise ValueError('Unsupported legacy checkpoint schema')
        validate_resume_contract(ck['contract'], expected, data['shots'].shape[1], allow_execution_change)
        if ck.get('selection_metric', 'total_loss') != selection_metric:
            raise ValueError('Resume selection metric mismatch')
        completed=ck['completed_updates']
        if total<=completed: raise ValueError('Total budget must exceed completed updates')
        model.vel_net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        if scheduler: scheduler.load_state_dict(ck['scheduler'])
        history=copy.deepcopy(ck['history']); best_state=copy.deepcopy(ck['best_state'])
        best_loss=ck['best_loss']; best_update=ck['best_update']; restore_rng(ck['rng'])
        if allow_execution_change:
            model.set_training_step(completed-1)
            observed_loss, _ = evaluate_loss(model, data, alpha)
            saved_loss = history[completed-1]['loss_after_update']
            if saved_loss is None or not np.isclose(observed_loss, saved_loss, rtol=2e-5, atol=1e-7):
                raise ValueError('Checkpoint objective differs from saved result')
            write_json(out/'resume_provenance.json', dict(checkpoint=str(Path(resume).resolve()),
                checkpoint_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest(),
                completed_updates=completed, original_contract=ck['contract'], current_contract=expected,
                saved_loss=saved_loss, reproduced_loss=observed_loss))
        atomic_save(out/'best.pth',{'schema':2,'completed_updates':best_update,
            'model':best_state,'loss_after_update':history[best_update-1]['loss_after_update'],
            'selection_metric':selection_metric, 'selection_score':best_loss,
            'objective_step':best_update-1})
    if scheduler and total>scheduler.max_epochs: raise ValueError('Budget exceeds fixed scheduler horizon')
    limit=min(total,stop_after) if stop_after is not None else total
    if limit<=completed: raise ValueError('stop_after must exceed completed updates')
    np.save(out/'initial_velocity.npy',initial_velocity)
    np.save(out/'v_true.npy',data['vp_true'].squeeze().cpu().numpy())
    write_json(out/'comparison_contract.json',comparison_contract(config,data))
    write_json(out/'status.json',{'state':'running','completed_updates':completed})
    resumed_from=completed
    timing=[]; start=time.perf_counter()
    if str(model.device).startswith('cuda'): torch.cuda.reset_peak_memory_stats(model.device)
    fields=['completed_updates','loss_before_update','data_loss_before_update','prior_loss_before_update',
            'loss_after_update','data_loss_after_update','data_mse_before_update','data_mse_after_update',
            'gradient_norm','learning_rates','cutoff_hz']
    try:
        with (out/'loss_history.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(history)
            while completed<limit:
                step_start=time.perf_counter()
                if scheduler: scheduler.step(completed)
                model.set_training_step(completed)
                before,losses=model.train_one_epoch(opt,wavelet=data['wavelet'],shots=data['shots'],trade_off=alpha)
                completed+=1
                row=dict(completed_updates=completed,loss_before_update=losses[0],
                    data_loss_before_update=losses[1],prior_loss_before_update=losses[2],
                    loss_after_update=None,data_loss_after_update=None,gradient_norm=model.last_gradient_norm,
                    data_mse_before_update=losses[3],data_mse_after_update=None,
                    learning_rates=json.dumps([g['lr'] for g in opt.param_groups]),
                    cutoff_hz=model.attention.current_cutoff_hz if model.attention is not None else None)
                save=completed%interval==0 or completed==limit
                if save:
                    if model.attention is not None:
                        spatial = model.attention.last_spatial_weights.detach().cpu().numpy()
                        temporal = model.attention.last_time_weights.detach().mean(dim=(0, 3)).cpu().numpy()
                        after = velocity(model)
                        np.savez_compressed(out/f'attention_step{completed:06d}.npz',
                            spatial_weights=spatial, temporal_weights_shot_time=temporal,
                            velocity_change=after-before.detach().squeeze(0).cpu().numpy(),
                            completed_updates=completed, objective_step=completed-1)
                    value,parts=evaluate_loss(model,data,alpha)
                    row.update(loss_after_update=value,data_loss_after_update=parts['data_loss'],
                               data_mse_after_update=parts['data_mse'])
                    score = parts['data_mse'] if selection_metric == 'data_mse' else value
                    if score<best_loss:
                        best_loss=score; best_update=completed; best_state=cpu_state(model)
                        atomic_save(out/'best.pth',{'schema':2,'completed_updates':best_update,
                            'model':best_state,'loss_after_update':value,
                            'selection_metric':selection_metric, 'selection_score':best_loss,
                            'objective_step':completed-1})
                history.append(row); writer.writerow(row); stream.flush()
                if save:
                    ck={'schema':2,'completed_updates':completed,'model':cpu_state(model),
                        'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,
                        'rng':rng_state(),'contract':expected,'config':config,'history':history,
                        'best_state':best_state,'best_loss':best_loss,'best_update':best_update}
                    ck.update(selection_metric=selection_metric, selection_score=best_loss,
                              objective_step=completed-1)
                    atomic_save(out/'last.pth',ck)
                    write_json(out/'status.json',{'state':'running','completed_updates':completed})
                if str(model.device).startswith('cuda'): torch.cuda.synchronize(model.device)
                seconds=time.perf_counter()-step_start; timing.append(seconds)
                print(f"Update {completed}/{total} loss_before={losses[0]:.6e} grad={model.last_gradient_norm:.4e} seconds={seconds:.2f}",flush=True)
        last=velocity(model); np.save(out/'last_velocity.npy',last)
        last_state=cpu_state(model)
        model.vel_net.load_state_dict(best_state)
        np.save(out/'best_velocity.npy',velocity(model)); model.vel_net.load_state_dict(last_state)
        summary={'completed_updates':completed,'best_update':best_update,'best_loss':best_loss,
            'selection_metric':selection_metric, 'selection_score':best_loss,
            'total_seconds':time.perf_counter()-start,'step_seconds':timing,
            'resumed_from_updates':resumed_from,'executed_updates':completed-resumed_from,
            'timing_scope':'this invocation only',
            'peak_cuda_bytes':torch.cuda.max_memory_allocated(model.device) if str(model.device).startswith('cuda') else 0,
            'parameter_count':sum(p.numel() for p in model.vel_net.parameters()),
            'last_velocity_min_mps':float(last.min()),'last_velocity_max_mps':float(last.max())}
        write_json(out/'training_summary.json',summary)
        write_json(out/'status.json',{'state':'trained' if completed==total else 'paused','completed_updates':completed})
        return dict(summary,history=history,best_state=best_state)
    except BaseException as exc:
        write_json(out/'status.json',{'state':'failed','completed_updates':completed,'error':str(exc)})
        raise

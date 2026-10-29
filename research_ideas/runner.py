"""Independent research runner using the unchanged original RNN-FD solver.

One method per run. Truth is used to synthesize observations and report metrics,
never to route, select a step, train a gate, or stop. New checkpoints are local,
trusted Python/PyTorch objects; never load an untrusted pickle with --resume.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from experiments import residual_ifwi_experiment as e
from experiments.residual_ifwi_data import load_velocity_data
from experiments.residual_ifwi_report import velocity_metrics
from experiments.frequency_continuation import energy_scales,stage_cutoff,lowpass
from experiments.lr_schedule import lr_factor
from core_models import smooth400_background,slowness_gaussian_background,linear_depth_background
from core_runner import atomic_json,atomic_torch,source_hashes,check_spec
from robust_observations import make_observations,array_hash
from .physics import OriginalPhysics
from .safety import (rng_state,restore_rng,parameter_vjp,assign_gradients,
                     combine_gradients,safeguarded_step)
from .wavelet import WaveletResidual
from .interfaces import OrderedInterfaceResidual
from .routing import CompressedShotRouter,compress_traces
from .gradient_attention import GradientAttention,fit_cross_group
from .dfm import FrozenDFMPrior

ROOT=Path(__file__).resolve().parents[1]
DEFAULTS={
 'baseline':{},
 'routing':dict(top_k=3,samples=2,refresh_every=20,exploration=.2,diversity=.2),
 'wavelet':dict(hidden=16,strength=.5,high_start=0.,ramp_updates=200),
 'interfaces':dict(interfaces=3,width_m=60.,amplitude_mps=300.,min_thickness_m=None,
                   hidden=16,warmup=100,ramp_updates=100,curvature_weight=0.,ratio=.1),
 'gradient_attention':dict(width=16,strength=.5,warmup=100,fit_every=5,gate_lr=1e-3,
                           penalty=.05,ratio=.1),
 'dfm':dict(checkpoint=None,warmup=100,every=5,allow_smoke=False,ratio=.1,lift_time=.8,
            steps=4,direct_every=20,blend=.5,batch_size=4),
}
RECIPE_KEYS={'method','options','steps','eval_every','safeguard','max_backtracks','armijo','base_spec'}


def _integer(x,name,minimum=1):
    if type(x) is not int or x<minimum: raise ValueError(f'{name} must be an integer >= {minimum}')


def validate(spec,recipe,smoke):
    if not isinstance(recipe,dict) or set(recipe)-RECIPE_KEYS:
        raise ValueError('Unknown research recipe keys')
    method=recipe.get('method')
    if method not in DEFAULTS: raise ValueError(f'method must be one of {list(DEFAULTS)}')
    check_spec(spec); cfg=spec['experiment']; e.validate_config(cfg)
    for key in ('speedup','accel','diffusion_prior','target'):
        if spec.get(key): raise ValueError(f'{key} is not supported by this isolated runner')
    if spec['control']!={'kind':'zero_output'} or cfg['parameterization']!='residual':
        raise ValueError('Research baseline requires residual SIREN with zero_output initialization')
    if spec.get('model') not in (None,{},dict(kind='siren')):
        raise ValueError('Custom representations and coordinate warps require a separate integration')
    if any(cfg['model'].get(k) for k in ('fourier','trainable_omega','low_rank')):
        raise ValueError('First protocol is matched plain P-IFWI-MS, not a stacked FR/low-rank model')
    if cfg['initialization'].get('pretrain') or cfg['training'].get('optimizer') not in (None,{'name':'adam'}):
        raise ValueError('Only zero-output initialization and Adam are supported')
    for key in ('shots_per_update','full_eval_interval','shot_sampling','gradient_preconditioner','time_gain','report_estimates'):
        if cfg['training'].get(key): raise ValueError(f'Unsupported training option: {key}')
    opts=copy.deepcopy(DEFAULTS[method]); supplied=recipe.get('options',{})
    if not isinstance(supplied,dict) or set(supplied)-set(opts): raise ValueError('Unknown method options')
    opts.update(supplied)
    steps=recipe.get('steps',cfg['training']['epochs']); interval=recipe.get('eval_every',50)
    _integer(steps,'steps'); _integer(interval,'eval_every')
    for key in ('warmup',):
        if key in opts: _integer(opts[key],key,0)
    for key in ('ramp_updates','fit_every','refresh_every','every','direct_every','batch_size','steps'):
        if key in opts: _integer(opts[key],key)
    if method=='dfm' and (opts['checkpoint'] is None or (opts['allow_smoke'] and not smoke)):
        raise ValueError('DFM needs trained weights; smoke weights need explicit --smoke')
    if method=='wavelet' and not 0<=opts['high_start']<=1: raise ValueError('high_start must be in [0,1]')
    for key in ('curvature_weight','penalty','ratio','gate_lr'):
        if key in opts and (not math.isfinite(opts[key]) or opts[key]<0): raise ValueError(f'Invalid {key}')
    guarded=recipe.get('safeguard',method in ('dfm','gradient_attention'))
    if type(guarded) is not bool: raise ValueError('safeguard must be a boolean')
    _integer(recipe.get('max_backtracks',6),'max_backtracks',0)
    if not 0<recipe.get('armijo',1e-4)<1: raise ValueError('Invalid Armijo factor')
    return method,opts,steps,interval,guarded


def implementation_hashes():
    return {**source_hashes(),**{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT/'research_ideas').glob('*.py'))}}


def run(spec,recipe,out,device='cpu',resume=None,smoke=False):
    method,opts,steps,interval,guarded=validate(spec,recipe,smoke)
    out=Path(out)
    if out.exists(): raise FileExistsError(f'Use a new output directory: {out}')
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(1); started=time.perf_counter()
    cfg=copy.deepcopy(spec['experiment']); seed=cfg['seed']; e._seed(seed)
    cfg['training']['shot_batch_size']=1
    truth,background,provenance=load_velocity_data(cfg,ROOT)
    bg=spec.get('background',{}); spacing=cfg['data']['grid_spacing_m']
    if bg.get('kind')=='smooth400': background=smooth400_background(ROOT/'data',truth.shape)
    elif bg.get('kind')=='slowness_gaussian':
        background=slowness_gaussian_background(truth,spacing,bg['slowness_sigma_m'],bg['velocity_sigma_m'])
    elif bg.get('kind')=='linear_depth':
        background=linear_depth_background(truth.shape,spacing,bg['top_mps'],bg['gradient_per_s'])
    problem=e.build_forward_problem(truth,cfg,str(device))
    observed,_,noise_info=make_observations(problem['observed'].cpu().numpy(),
        spec['robustness']['noise_sigma'],spec['robustness']['noise_seed'])
    problem['observed']=torch.as_tensor(observed,device=device)
    schedule=cfg['training'].get('frequency_schedule')
    if schedule:
        problem['frequency_scales']=energy_scales(problem['observed'],schedule,cfg['forward']['dt_s'])
    physics=OriginalPhysics(problem); physics.counts['observation_forward_shots']=physics.shots
    model=e.model_from_config(background,cfg)
    if method=='wavelet': model=WaveletResidual(model,hidden=opts['hidden'],strength=opts['strength'])
    elif method=='interfaces':
        model=OrderedInterfaceResidual(model,spacing,**{k:opts[k] for k in
              ('interfaces','width_m','amplitude_mps','min_thickness_m','hidden')})
    model.to(device)
    learning_rate=cfg['training']['learning_rate']
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,
        betas=tuple(cfg['training'].get('adam_betas',(.9,.999))))
    router=gate=gate_optimizer=prior=None
    if method=='routing':
        router=CompressedShotRouter(compress_traces(problem['observed']),seed=seed,
            **{k:opts[k] for k in ('top_k','samples','exploration','diversity')})
    if method=='gradient_attention':
        gate=GradientAttention(opts['width'],opts['strength']).to(device)
        gate_optimizer=torch.optim.Adam(gate.parameters(),lr=opts['gate_lr'])
    if method=='dfm':
        prior=FrozenDFMPrior(opts['checkpoint'],device,spacing,seed=seed,
            **{k:opts[k] for k in ('allow_smoke','ratio','lift_time','steps','direct_every','blend','batch_size')})
    if prior is not None:
        from .train_dfm import model_digest
        known={m['sha256'] for key in ('training_models','validation_models') for m in prior.meta.get(key,[])}
        if model_digest(truth) in known:
            raise ValueError('Inversion truth appears in prior training or validation models')
    identity=dict(spec=spec,recipe={k:v for k,v in recipe.items() if k not in ('steps','base_spec')},
                  truth=array_hash(truth),background=array_hash(background),noise=noise_info['observed_hash'],
                  sources=implementation_hashes(),device=str(device),torch=str(torch.__version__),
                  prior_sha=None if prior is None else prior.sha256)
    history=[]; evaluations=[]; completed=0; last_cutoff='uninitialized'; last_refresh=0
    best_loss=math.inf; best_velocity=None; past_seconds=0.; past_physics_seconds=0.
    if resume is not None:
        saved=torch.load(resume,map_location=device,weights_only=False)
        if saved.get('schema')!='ifwi_five_v1' or saved['identity']!=identity:
            raise ValueError('Resume source/config/data/device/prior identity mismatch')
        if steps<=saved['completed']: raise ValueError('New budget must exceed the saved update count')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        if router: router.load_state_dict(saved['router'])
        if gate:
            gate.load_state_dict(saved['gate']); gate_optimizer.load_state_dict(saved['gate_optimizer'])
        if prior: prior.load_state_dict(saved['prior'])
        completed=saved['completed']; history=saved['history']; evaluations=saved['evaluations']
        last_cutoff=saved['last_cutoff']; last_refresh=saved['last_refresh']
        best_loss=saved['best_loss']; best_velocity=saved['best_velocity']
        # Rebuilding observations on resume is a real extra cost, not silently erased.
        for k,v in saved['costs'].items(): physics.counts[k]+=v
        past_seconds=saved['seconds']; past_physics_seconds=saved['physics_seconds']
        restore_rng(saved['rng'])
    out.mkdir(parents=True)
    atomic_json(out/'run.json',dict(spec=spec,recipe=recipe,options=opts,identity=identity,
                original_source_commit='226be2262377a0bbde4f813205ceb7cbcd69f335',
                solver='unchanged rnn_fd.py',shot_batch_size=1,smoke=smoke))
    np.save(out/'initial_velocity.npy',background)
    def seconds():
        if device.type=='cuda': torch.cuda.synchronize(device)
        return past_seconds+time.perf_counter()-started
    def checkpoint():
        atomic_torch(out/'checkpoint.pt',dict(schema='ifwi_five_v1',identity=identity,
            completed=completed,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),
            router=None if router is None else router.state_dict(),gate=None if gate is None else gate.state_dict(),
            gate_optimizer=None if gate_optimizer is None else gate_optimizer.state_dict(),
            prior=None if prior is None else prior.state_dict(),history=history,evaluations=evaluations,
            costs=physics.counts,seconds=seconds(),physics_seconds=past_physics_seconds+physics.seconds,
            last_cutoff=last_cutoff,last_refresh=last_refresh,best_loss=best_loss,best_velocity=best_velocity))
    for update in range(completed+1,steps+1):
        cutoff=stage_cutoff(schedule,update)
        factor=lr_factor(cfg['training'].get('lr_schedule'),schedule,update)
        for group in optimizer.param_groups: group['lr']=learning_rate*factor
        if method=='wavelet':
            model.gate.high_fraction.fill_(opts['high_start']+(1-opts['high_start'])*min(1.,(update-1)/opts['ramp_updates']))
        if method=='interfaces': model.activation.fill_(min(1.,max(0.,(update-opts['warmup'])/opts['ramp_updates'])))
        if router and (update==1 or update-last_refresh>=opts['refresh_every'] or cutoff!=last_cutoff):
            _,pred=physics.value(model(),cutoff=cutoff,return_records=True)
            router.refresh(lowpass(problem['observed'],cutoff,cfg['forward']['dt_s']),
                           lowpass(pred,cutoff,cfg['forward']['dt_s']))
            last_refresh=update
        last_cutoff=cutoff
        weights=None if router is None else router.sample()
        v=model(); report=physics.value_and_gradient(v,weights,cutoff)
        use_prior=prior is not None and opts['ratio']>0 and update>opts['warmup'] and (update-opts['warmup']-1)%opts['every']==0
        use_gate=gate is not None and update>opts['warmup']
        gp=parameter_vjp(v,model,report['gradient'],retain_graph=use_prior or use_gate)
        proposal=gp; guidance={}; external=None
        if use_prior:
            external=copy.deepcopy(prior.state_dict())
            auxiliary=parameter_vjp(v,model,prior.gradient(v.detach(),update))
            proposal,guidance=combine_gradients(gp,auxiliary,opts['ratio'])
        if use_gate:
            external=(copy.deepcopy(gate.state_dict()),copy.deepcopy(gate_optimizer.state_dict()))
            if (update-opts['warmup']-1)%opts['fit_every']==0:
                guidance['fit_loss']=fit_cross_group(gate,gate_optimizer,v,*report['groups'],penalty=opts['penalty'])
            mask=gate(v,report['gradient']).detach()
            gmod=parameter_vjp(v,model,mask*report['gradient'])
            proposal,geometry=combine_gradients(gp,[a-b for a,b in zip(gmod,gp)],opts['ratio'])
            guidance.update(geometry,gate_min=float(mask.min()),gate_max=float(mask.max()))
        if method=='interfaces' and opts['curvature_weight']:
            params=tuple(model.parameters()); reg=opts['curvature_weight']*model.regularizer()
            gr=torch.autograd.grad(reg,params,allow_unused=True)
            auxiliary=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gr)]
            proposal,guidance=combine_gradients(gp,auxiliary,opts['ratio'])
        before_calls=physics.counts['forward_shots']
        if guarded:
            acceptance=safeguarded_step(model,optimizer,gp,proposal,
                lambda:physics.value(model(),weights,cutoff),before_loss=report['loss'],
                max_backtracks=recipe.get('max_backtracks',6),armijo=recipe.get('armijo',1e-4))
        else:
            state=copy.deepcopy(model.state_dict()); opt_state=copy.deepcopy(optimizer.state_dict())
            assign_gradients(model,proposal); optimizer.step()
            try:
                with torch.no_grad(): e.check_velocity(model(),cfg)
                acceptance=dict(accepted=True,alpha=1.,reason='unguarded',evaluations=0)
            except (ValueError,FloatingPointError):
                model.load_state_dict(state); optimizer.load_state_dict(opt_state)
                acceptance=dict(accepted=False,alpha=0.,reason='velocity_bounds',evaluations=0)
        # Keep measured cost even when restoring state after a rejected proposal.
        if not acceptance['accepted'] and external is not None:
            if use_prior:
                actual_calls=prior.calls; prior.load_state_dict(external); prior.calls=actual_calls
            if use_gate:
                gate.load_state_dict(external[0]); gate_optimizer.load_state_dict(external[1])
        completed=update
        history.append(dict(update=update,cutoff_hz=cutoff,loss_before=report['loss'],lr_factor=factor,
            active_shots=physics.shots if weights is None else int((weights>0).sum()),
            acceptance=acceptance,guidance=guidance,costs=dict(physics.counts),
            acceptance_forward_shots=physics.counts['forward_shots']-before_calls,seconds=seconds()))
        if update%interval==0 or update==steps:
            with torch.no_grad(): velocity=model().cpu().numpy()[0].copy()
            loss=physics.value(model())  # All shots, full band, same evaluation definition.
            metrics=velocity_metrics(velocity,truth)
            evaluations.append(dict(update=update,full_band_loss=loss,velocity=metrics,
                                    costs=dict(physics.counts),seconds=seconds()))
            if loss<best_loss: best_loss=loss; best_velocity=velocity.copy()
            np.save(out/'final_velocity.npy',velocity)
            if method=='interfaces': np.save(out/'interfaces_m.npy',model.heights().detach().cpu().numpy())
            atomic_json(out/'history.json',history); atomic_json(out/'evaluations.json',evaluations)
            checkpoint()
    np.save(out/'lowest_misfit_velocity.npy',best_velocity)
    final=dict(method=method,completed=completed,velocity=evaluations[-1]['velocity'],
        full_band_loss=evaluations[-1]['full_band_loss'],costs=physics.counts,
        seconds=seconds(),physics_seconds=past_physics_seconds+physics.seconds,
        prior_calls=0 if prior is None else prior.calls,
        rejected_updates=sum(not x['acceptance']['accepted'] for x in history),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        qualification='smoke only' if smoke else 'experimental; not a validated reconstruction claim')
    atomic_json(out/'metrics.json',final)
    return out


def smoke_spec(directory):
    """Small SAME-solver test; not a replacement wave equation or a benchmark."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    truth=np.linspace(2300,3000,9,dtype=np.float32)[:,None]+np.zeros((9,11),np.float32)
    truth[5:7,4:8]+=250; source=directory/'smoke_truth.csv'; np.savetxt(source,truth,delimiter=',')
    cfg=e.default_config(); cfg['data'].update(model_file=str(source.resolve()),downsample=1,csv_header='none')
    cfg['model']['neurons']=[2,8,8,1]; cfg['initialization']['sigma']=1.
    cfg['acquisition'].update(num_shots=3,source_x_indices=[2,5,8])
    cfg['forward'].update(dt_s=.001,nt=48,frequency_hz=60.,npad=3)
    cfg['training'].update(epochs=3,checkpoint_interval=1,shot_batch_size=1)
    return dict(name='research_smoke',experiment=cfg,control=dict(kind='zero_output'),
                robustness=dict(condition='smoke',noise_sigma=0.,noise_seed=3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path); p.add_argument('--spec',type=Path)
    p.add_argument('--method',choices=list(DEFAULTS)); p.add_argument('--smoke',action='store_true')
    p.add_argument('--steps',type=int); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu'); p.add_argument('--resume',type=Path)
    p.add_argument('--prior',type=Path); p.add_argument('--seed',type=int)
    args=p.parse_args(); recipe=json.loads(args.recipe.read_text()) if args.recipe else dict(method=args.method or 'baseline')
    if args.method: recipe['method']=args.method
    if args.smoke:
        spec=smoke_spec(args.out.parent/(args.out.name+'_input')); recipe.update(steps=args.steps or 3,eval_every=1)
        options=dict(recipe.get('options',{})); method=recipe['method']
        if method=='routing': options.update(top_k=1,samples=1,refresh_every=2)
        if method=='interfaces': options.update(interfaces=2,width_m=30.,warmup=0,ramp_updates=1)
        if method=='gradient_attention': options.update(warmup=0)
        if method=='dfm': options.update(warmup=0,every=1,allow_smoke=True)
        recipe['options']=options
    else:
        path=args.spec or Path(recipe.get('base_spec','experiments/configs/diffusion_prior/g2_pms/01_pms_baseline.json'))
        spec=json.loads(path.read_text())
    if args.steps is not None: recipe['steps']=args.steps
    if args.seed is not None: spec['experiment']['seed']=args.seed
    if args.prior: recipe.setdefault('options',{})['checkpoint']=str(args.prior)
    print(run(spec,recipe,args.out,args.device,args.resume,args.smoke))

if __name__=='__main__': main()

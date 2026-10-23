"""Spec-driven runner for the paper's experiments (docs/paper_supplement.md).

Started as the robustness-batch runner (2313716 model, FD and Adam update; previous commit).
Snapshots include config, source hashes, optimizer and RNG. Resume into a new
attempt directory; retain previous attempt. Per-state times include required
preparation, updates, diagnostic and periodic checkpoint costs, excluding idle
time between attempts. No truth-based stopping or inversion model selection.

Supplement additions, all off unless the spec asks for them (an unchanged E2/E3/E4 spec runs
the same arithmetic as before):
* training.frequency_schedule (multiscale, energy-normalized low-pass stages; E2-MS),
  training.lr_schedule (warmup / cosine decay) and training.optimizer Adam betas/eps;
* spec.model.kind: "siren" (default), "grid" (grid FWI, Adam on dv in m/s) or
  "fourier_features" (input Fourier features + ReLU MLP, zero head);
* spec.model.coordinate_warp: {"kind": "traveltime", "axes": "z" | "xz"} feeds the SIREN
  traveltime-through-v0 coordinates instead of (x, z) (core_models.traveltime_coordinates);
* spec.background: {"kind": "linear_depth", "top_mps", "gradient_per_s"} replaces the
  Gaussian background with a depth-only one;
* control prefit_background with keep_optimizer_state: start the inversion with the
  pretraining Adam moments instead of a reset;
* --shared-dir for the shared clean/noisy observations, --epochs for short checks.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[_key]='1'
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import torch
from robust_observations import shared_observations, array_hash
from experiments import residual_ifwi_experiment as e
from experiments.residual_ifwi_data import load_velocity_data
from experiments.residual_ifwi_report import velocity_metrics, save_plots
from experiments.trainable_omega import omega_values
from experiments.muon_optimizer import build_optimizer
from experiments.frequency_continuation import energy_scales, stage_cutoff
from experiments.lr_schedule import lr_factor
from core_controls import small_output, prefit_background, training_state, restore_training_state, sync, layer_snapshot, layer_changes
from core_models import FourierFeatureVelocity, GridVelocity, linear_depth_background, smooth400_background, slowness_gaussian_background, traveltime_coordinates, warp_coordinates, warp_report, WARP_AXES
from core_speed import CoarseView, SourceEncoder, coarse_problem, validate_speedup
from core_accel import validate_accel, StepProblems, FunctionSpaceOptimizer, function_space_update, coordinate_scale

ROOT=Path(__file__).resolve().parent
SCHEMA='paper_supplement_v1'
SPEC_KEYS={'name','experiment','control','robustness','model','background','protocol','speedup','target','accel'}
EXTENDABLE_SCHEMAS=(SCHEMA,'robustness_overthrust_noise_v1')


def budget_free(spec):
    spec=copy.deepcopy(spec); training=spec['experiment']['training']
    training.pop('epochs',None); training.pop('checkpoint_interval',None)
    return spec


SOURCE_FILES=('core_runner.py','core_controls.py','core_models.py','core_speed.py','core_accel.py','robust_observations.py',
              'ifwi_modules.py','rnn_fd.py','generator.py','plot_functions.py')


def atomic_json(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    replace(tmp,path)


def replace(src,dst):
    for attempt in range(6):
        try: os.replace(src,dst); return
        except PermissionError:
            if attempt==5: raise
            time.sleep(.1*(attempt+1))


def atomic_torch(path,payload):
    tmp=path.with_suffix('.pt.tmp'); torch.save(payload,tmp); replace(tmp,path)


def source_hashes():
    # Explicit files: the repository root also holds .venv/ and outputs/, which must not be hashed.
    files=[ROOT/name for name in SOURCE_FILES]+sorted((ROOT/'experiments').glob('*.py'))
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def git_head():
    try:
        import subprocess
        return subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    except Exception:
        return None


def check_spec(spec):
    unknown=set(spec)-SPEC_KEYS
    if unknown or not {'name','experiment','control','robustness'}<=set(spec):
        raise ValueError(f'Spec needs name, experiment, control, robustness; unknown keys {sorted(unknown)}')
    training=spec['experiment']['training']
    for key in ('time_gain','report_estimates','shots_per_update'):
        if training.get(key) is not None:
            raise ValueError(f'core_runner does not support training.{key}; use residual_ifwi.py')
    kind=spec.get('model',{}).get('kind','siren')
    if kind not in ('siren','grid','fourier_features'):
        raise ValueError('spec.model.kind must be siren, grid or fourier_features')
    if kind!='siren' and spec['experiment']['parameterization']!='residual':
        raise ValueError('Grid and Fourier-feature models are residual (v0 + update)')
    warp=spec.get('model',{}).get('coordinate_warp')
    if kind=='siren' and set(spec.get('model',{}))-{'kind','coordinate_warp'}:
        raise ValueError('spec.model for siren takes only kind and coordinate_warp')
    if warp is not None and (kind!='siren' or spec['experiment']['model'].get('low_rank') is not None
                             or warp.get('kind')!='traveltime' or set(warp)!={'kind','axes'} or warp['axes'] not in WARP_AXES):
        raise ValueError(f'coordinate_warp is {{kind: traveltime, axes: {WARP_AXES}}} on a 2-D SIREN')
    if kind=='grid' and spec['control']['kind']!='zero_output':
        raise ValueError('Grid FWI starts at dv = 0 (control zero_output)')
    validate_speedup(spec.get('speedup'),spec['experiment'])
    validate_accel(spec.get('accel'))
    if (spec.get('accel') or {}).get('function_space') and spec['control'].get('keep_optimizer_state'):
        raise ValueError('function_space cannot reuse prefit Adam moments')
    if spec.get('accel') and (spec.get('speedup') or {}).get('coarse_lowpass'):
        raise ValueError('spec.accel cannot be combined with coarse low-pass stages')
    background=spec.get('background')
    if background is not None and not (
            (background.get('kind')=='linear_depth' and set(background)=={'kind','top_mps','gradient_per_s'})
            or (background.get('kind')=='smooth400' and set(background)=={'kind'})
            or (background.get('kind')=='slowness_gaussian' and set(background)=={'kind','slowness_sigma_m','velocity_sigma_m'})):
        raise ValueError('spec.background must be {kind: linear_depth, top_mps, gradient_per_s}, {kind: smooth400} '
                         'or {kind: slowness_gaussian, slowness_sigma_m, velocity_sigma_m}')
    target=spec.get('target')
    if target is not None:
        plateau=target.get('plateau')
        if set(target)-{'stop_rmse','plateau'} or (plateau is not None and (set(plateau)!={'window','rel'}
                or int(plateau['window'])<1 or not 0<plateau['rel']<1)):
            raise ValueError('spec.target is {stop_rmse?, plateau?: {window >= 1, rel in (0, 1)}}')


def build_model(spec, cfg, background):
    cm=cfg['model']; kind=spec.get('model',{}).get('kind','siren')
    if kind=='grid':
        return GridVelocity(background)
    if kind=='fourier_features':
        options=spec['model']
        return FourierFeatureVelocity(background,cfg['data']['grid_spacing_m'],options['scale_cpkm'],
            options.get('features',128),tuple(options.get('hidden',(128,128,128,128))),options.get('seed',cfg['seed']))
    model=e.VelocityParameterization(background,cfg['data']['grid_spacing_m'],cm['neurons'],cm['omega_0'],
        cm['mean_kmps'],cm['std_kmps'],cfg['parameterization'],cm.get('fourier'),cm.get('trainable_omega') is not None)
    warp=spec.get('model',{}).get('coordinate_warp')
    if warp is not None:
        coords=traveltime_coordinates(background,cfg['data']['grid_spacing_m'],warp['axes'])
        warp_coordinates(model,coords)
        model.coordinate_warp_report={**warp,**warp_report(background,cfg['data']['grid_spacing_m'],coords)}
    return model


def run(spec, out, device='cuda:0', resume=None, save_figures=True, shared_dir=None, extend=False, stop_rmse=None,
        stop_plateau=None):
    check_spec(spec)
    # Time-to-target protocol: stop at the first update whose velocity RMSE reaches the target.
    if stop_rmse is None: stop_rmse=(spec.get('target') or {}).get('stop_rmse')
    # Convergence: stop when the best RMSE so far improved by less than rel over the last window updates.
    if stop_plateau is None: stop_plateau=(spec.get('target') or {}).get('plateau')
    reached=None
    cfg=copy.deepcopy(spec['experiment']); control=spec['control']
    e.validate_config(cfg)
    schedule=cfg['training'].get('frequency_schedule'); lr_setting=cfg['training'].get('lr_schedule')
    shared_dir=Path(shared_dir) if shared_dir else ROOT/'outputs'/'shared'
    if cfg['training'].get('shots_per_update') is not None:
        raise ValueError('This core protocol uses every shot in each update')
    if control['kind'] not in ('zero_output','prefit_background','small_output'):
        raise ValueError('Unknown initialization control')
    if control.get('keep_optimizer_state') and control['kind']!='prefit_background':
        raise ValueError('keep_optimizer_state applies only to prefit_background')
    if (control['kind']=='prefit_background') != (cfg['parameterization']=='absolute'):
        raise ValueError('Prefit applies only to direct absolute IFWI')
    torch.set_num_threads(1)
    out=Path(out); out.mkdir(parents=True,exist_ok=False); (out/'checkpoints').mkdir()
    log=(out/'progress.log').open('w',encoding='utf-8',buffering=1)
    def emit(message):
        message=time.strftime('%Y-%m-%d %H:%M:%S')+' '+message
        print(message,flush=True); print(message,file=log,flush=True)
    sources=source_hashes(); e.original_source_hashes()
    atomic_json(out/'config.json',cfg); atomic_json(out/'run_spec.json',spec)
    atomic_json(out/'source_hashes.json',sources)
    atomic_json(out/'environment.json',{'python':platform.python_version(),'torch':str(torch.__version__),
        'numpy':np.__version__,'cuda':torch.version.cuda,'device':device,'pid':os.getpid(),
        'gpu':torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
        'base_commit':git_head(),'threads':1,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32})
    atomic_json(out/'status.json',{'state':'preparing','pid':os.getpid(),'completed_updates':0})
    if device.startswith('cuda'): torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter(); phase='preparing'; model=opt=None; completed=0
    timing={'preparation_seconds':0.,'training_seconds':0.,'diagnostic_seconds':0.,'checkpoint_seconds':0.,'prefit_seconds':0.}
    history=[]; evaluations=[]; train_started=None; train_prior=0.; saved=None
    rollback=None; rollback_history_length=0
    def checkpoint(name):
        sync(model)
        timing['training_seconds']=train_prior+time.perf_counter()-train_started if train_started else timing['training_seconds']
        payload=training_state(model,opt,completed,timing)
        payload.update(schema=SCHEMA,phase='inversion',spec=spec,sources=sources,provenance=provenance,
                       history=history,evaluations=evaluations,control_report=control_report)
        write_started=time.perf_counter()
        atomic_torch(out/name,payload)
        timing['checkpoint_seconds']+=time.perf_counter()-write_started
        if train_started: timing['training_seconds']=train_prior+time.perf_counter()-train_started
        stat=(out/name).stat()
        atomic_json(out/(name+'.timing.json'),{'checkpoint_bytes':stat.st_size,
            'checkpoint_mtime_ns':stat.st_mtime_ns,'timing':timing})
    try:
        e._seed(cfg['seed'])
        truth,background,provenance=load_velocity_data(cfg,ROOT)
        if spec.get('background') is not None:
            override=spec['background']
            if override['kind']=='smooth400':
                background=smooth400_background(ROOT/'data',truth.shape)
                note='the earliest IFWI background (pretrain_marmousi.py): smooth400 CSV, Gaussian sigma 50, stride 4'
            elif override['kind']=='slowness_gaussian':
                background=slowness_gaussian_background(truth,cfg['data']['grid_spacing_m'],override['slowness_sigma_m'],override['velocity_sigma_m'])
                note='Gaussian smoothing of the true slowness, then of the velocity'
            else:
                background=linear_depth_background(truth.shape,cfg['data']['grid_spacing_m'],override['top_mps'],override['gradient_per_s'])
                note='replaces the Gaussian background; no truth used'
            provenance={**provenance,'background_override':{**override,'array_hash':array_hash(background),'note':note}}
        atomic_json(out/'initial_model_provenance.json',provenance)
        if resume:
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            if extend:
                # Longer budget for a finished run: only epochs/checkpoint_interval may change; source edits
                # since the original run are recorded (the core arithmetic is unchanged, see the first-5 check).
                if saved.get('schema') not in EXTENDABLE_SCHEMAS or budget_free(saved['spec'])!=budget_free(spec) or saved['provenance']!=provenance:
                    raise ValueError('Extend: schema/config/data identity mismatch beyond the update budget')
                if saved['phase']!='inversion' or saved['completed_updates']>=spec['experiment']['training']['epochs']:
                    raise ValueError('Extend needs an inversion checkpoint and a larger budget')
                changed=sorted(k for k in set(saved['sources'])|set(sources) if saved['sources'].get(k)!=sources.get(k))
                atomic_json(out/'extend.json',{'from_checkpoint':str(Path(resume).resolve()),'original_schema':saved['schema'],
                    'from_updates':saved['completed_updates'],'to_updates':spec['experiment']['training']['epochs'],
                    'source_files_changed_since_original':changed})
            elif saved.get('schema')!=SCHEMA or saved['spec']!=spec or saved['sources']!=sources or saved['provenance']!=provenance:
                raise ValueError('Resume source/config/data identity mismatch')
            timing_path=Path(str(resume)+'.timing.json')
            if saved['phase']=='inversion' and timing_path.exists():
                side=json.loads(timing_path.read_text(encoding='utf-8')); stat=Path(resume).stat()
                if side['checkpoint_bytes']==stat.st_size and side['checkpoint_mtime_ns']==stat.st_mtime_ns:
                    saved['timing']=side['timing']
            atomic_json(out/'resume.json',{'checkpoint':str(Path(resume).resolve()),'phase':saved['phase']})
        emit(f"PREPARE {spec['name']}: grid={truth.shape}, all {cfg['acquisition']['num_shots']} shots/update, seed={cfg['seed']}")
        problem=e.build_forward_problem(truth,cfg,device)
        robustness=spec['robustness']
        identity={'truth_hash':array_hash(truth),'background_hash':array_hash(background),
                  'acquisition':cfg['acquisition'],'forward':cfg['forward'],
                  'grid_spacing_m':cfg['data']['grid_spacing_m'],
                  'shot_batch_size':cfg['training']['shot_batch_size']}
        shared=shared_dir/robustness['condition']
        observed,observation_info=shared_observations(problem['observed'].cpu().numpy(),
            robustness['noise_sigma'],robustness['noise_seed'],shared,identity)
        problem['observed']=torch.from_numpy(observed).to(device)
        atomic_json(out/'observation_manifest.json',observation_info)
        atomic_json(out/'shared_observations.json',{'directory':str(shared.resolve()),
            'clean':'clean_observed.npy','noise':'noise.npy','observed':'observed.npy'})
        emit(f"OBSERVATIONS sigma={robustness['noise_sigma']} dtype={observed.dtype} shared_hash={observation_info['observed_hash']}")
        if schedule is not None:
            # Energy normalization from the observations this run trains on (noisy ones for noise conditions).
            problem['frequency_scales']=energy_scales(problem['observed'],schedule,cfg['forward']['dt_s'])
            atomic_json(out/'frequency_scales.json',{str(k):v for k,v in problem['frequency_scales'].items()})
        model=build_model(spec,cfg,background).to(device)
        accel=spec.get('accel') or {}
        def make_optimizer(model):
            fs=accel.get('function_space')
            if fs is None: return build_optimizer(model,cfg)
            return FunctionSpaceOptimizer(model,**fs)
        base_coords=model.coords.detach().clone() if getattr(model,'coords',None) is not None else None
        if accel.get('coordinate_anneal') is not None and base_coords is None:
            raise ValueError('coordinate_anneal needs a coordinate network')
        if getattr(model,'coordinate_warp_report',None) is not None:
            atomic_json(out/'coordinate_warp.json',model.coordinate_warp_report)
            emit(f"WARP traveltime axes={model.coordinate_warp_report['axes']} stretch_z={model.coordinate_warp_report['stretch_z']['min']:.2f}-{model.coordinate_warp_report['stretch_z']['max']:.2f}")
        speedup=spec.get("speedup") or {}; coarse={}; encoders={}; coarse_model=None
        if speedup.get('coarse_lowpass'):
            option=speedup['coarse_lowpass']
            coarse=coarse_problem(problem,background,option['factor'],option['source_depth_index'],option['receiver_depth_index'],schedule)
            coarse_model=CoarseView(model,option['factor'])
            atomic_json(out/'coarse_lowpass.json',{**option,'source_scale':coarse['source_scale'],'grid':[coarse['geometry']['nz'],coarse['geometry']['nx']],
                'dt_s':coarse['config']['forward']['dt_s'],'nt':coarse['config']['forward']['nt']})
            emit(f"COARSE low-pass stages: grid={coarse['geometry']['nz']}x{coarse['geometry']['nx']} source scale={coarse['source_scale']:.4f}")
        if speedup.get('source_encoding'):
            option=speedup['source_encoding']
            encoders={'fine':SourceEncoder(problem,option['supershots'],option['seed'])}
            if coarse: encoders['coarse']=SourceEncoder(coarse,option['supershots'],option['seed'])
        step_problems=StepProblems(problem,accel,(speedup.get('source_encoding') or {}).get('seed',cfg['seed'])) if accel else None
        if accel: atomic_json(out/'accel.json',accel)
        control_report={}; prefit_optimizer_state=None
        if saved and saved['phase']=='inversion':
            opt=make_optimizer(model); base_lrs=[g['lr'] for g in opt.param_groups]
            completed,timing=restore_training_state(saved,model,opt)
            history=saved['history']; evaluations=saved['evaluations']; control_report=saved['control_report']
        else:
            if control['kind']=='small_output': control_report=small_output(model,control['rms_mps'])
            elif control['kind']=='prefit_background':
                phase='prefit'
                def save_prefit(payload):
                    payload.update(schema=SCHEMA,spec=spec,sources=sources,provenance=provenance,
                                   preparation_elapsed_seconds=time.perf_counter()-started+(saved.get('preparation_elapsed_seconds',0.) if saved else 0.))
                    atomic_torch(out/'prefit_checkpoint.pt',payload)
                control_report=prefit_background(model,control['updates'],control['learning_rate'],emit,save_prefit,saved,
                                                 keep_state=bool(control.get('keep_optimizer_state')))
                prefit_optimizer_state=control_report.pop('optimizer_state',None)
                timing['prefit_seconds']=control_report['seconds']
                if saved: timing['preparation_seconds']+=saved.get('preparation_elapsed_seconds',saved['seconds'])
                atomic_torch(out/'prefit_selected_model.pt',{'model':model.state_dict(),'control_report':control_report})
            opt=make_optimizer(model); base_lrs=[g['lr'] for g in opt.param_groups]
            if prefit_optimizer_state is not None:
                # Continue with the pretraining moments; only the learning rate changes to the inversion one.
                opt.load_state_dict(prefit_optimizer_state)
                for group,base in zip(opt.param_groups,base_lrs): group['lr']=base
                control_report['inversion_started_with_prefit_adam_moments']=True
        phase='preparing'
        atomic_json(out/'initialization_control.json',control_report)
        with torch.no_grad(): initial=model().cpu().numpy()[0].copy()
        post=initial
        for name,values in [('true_velocity',truth),('initial_velocity',background),('run_start_velocity',initial),
                            ('observed',problem['observed'].cpu().numpy()),('wavelet',problem['wavelet'].cpu().numpy())]:
            np.save(out/(name+'.npy'),values)
        atomic_json(out/'acquisition.json',problem['geometry'])
        initial_mse=e.evaluate_data_mse(model,problem)
        initial_metrics={**velocity_metrics(initial,truth),'data_mse':initial_mse}
        atomic_json(out/'initial_checks.json',{'completed_updates':completed,'metrics':initial_metrics,
            'background_metrics':velocity_metrics(background,truth),'velocity_min_mps':float(initial.min()),
            'velocity_max_mps':float(initial.max()),'rms_vs_background_mps':float(np.sqrt(np.mean((initial-background).astype('float64')**2))),
            'parameters':sum(p.numel() for p in model.parameters()),'fixed_background_requires_grad':model.fixed_init_mps.requires_grad,
            'ssim_window':7,'ssim_data_range_mps':float(truth.max()-truth.min()),'evaluation_region':'entire grid; no mask'})
        # B.T B is small (128x128); save every eigenvalue, including the tied-column null mode.
        spectra=[]
        for index,wrapped in enumerate(model.net.linear if getattr(model,'net',None) is not None else []):
            layer=getattr(wrapped,'layer',wrapped)
            if hasattr(layer,'bases'):
                b=layer.bases.detach().double(); eigen=torch.linalg.eigvalsh(b.T@b).cpu().numpy()
                spectra.append({'layer':index,'B_shape':list(b.shape),'eigenvalues_BtB':eigen.tolist(),
                                'rank_threshold_relative':1e-10,'rank':int((eigen>eigen[-1]*1e-10).sum())})
        atomic_json(out/'fourier_spectrum.json',spectra)
        sync(model); timing['preparation_seconds']+=time.perf_counter()-started
        if not evaluations or evaluations[-1]['evaluated_updates']!=completed:
            evaluations.append({'evaluated_updates':completed,'update':completed,**initial_metrics,
                'cumulative_seconds':timing['preparation_seconds']+timing['training_seconds']})
        emit(f"INITIAL update={completed}: RMSE={initial_metrics['rmse_mps']:.3f} m/s SSIM={initial_metrics['ssim']:.5f} data MSE={initial_mse:.7g}; preparation={timing['preparation_seconds']:.2f}s")
        # Save step zero for ALL methods, and an independently recoverable latest checkpoint.
        initial_checkpoint_started=time.perf_counter()
        checkpoint('checkpoints/update_%04d.pt'%completed); checkpoint('checkpoint.pt')
        np.save(out/'checkpoints'/('velocity_%04d.npy'%completed),initial)
        timing['preparation_seconds']+=time.perf_counter()-initial_checkpoint_started
        sync(model); train_prior=timing['training_seconds']; train_started=time.perf_counter(); phase='inversion'
        with (out/'history.jsonl').open('w',encoding='utf-8',buffering=1) as stream, (out/'mechanisms.jsonl').open('w',encoding='utf-8',buffering=1) as mechanisms:
            for row in history: stream.write(json.dumps(row,allow_nan=False)+'\n')
            for update in range(completed+1,cfg['training']['epochs']+1):
                sync(model); step_started=time.perf_counter()
                before_time=timing['preparation_seconds']+train_prior+step_started-train_started
                rollback=training_state(model,opt,completed,timing); rollback_history_length=len(history)
                diagnostic=(update<=5 or update%50==0)
                layered=getattr(model,'net',None) is not None
                max_hz=stage_cutoff(schedule,update)
                if lr_setting is not None:
                    factor=lr_factor(lr_setting,schedule,update)
                    for group,base in zip(opt.param_groups,base_lrs): group['lr']=base*factor
                ds=time.perf_counter(); old=layer_snapshot(model) if diagnostic and layered else None
                sync(model); diagnostic_seconds=time.perf_counter()-ds
                compute_start=time.perf_counter()
                use_coarse=bool(coarse) and max_hz is not None
                step_problem,step_model=(coarse,coarse_model) if use_coarse else (problem,model)
                accel_info=None
                if accel:
                    if accel.get('coordinate_anneal') is not None:
                        scale=coordinate_scale(accel['coordinate_anneal'],update)
                        with torch.no_grad(): model.coords.copy_(base_coords*scale)
                    fixed=(speedup.get('source_encoding') or {}).get('supershots')
                    step_problem,accel_info=step_problems(update,fixed)
                    if accel.get('coordinate_anneal') is not None: accel_info['coordinate_scale']=scale
                elif encoders: step_problem=encoders['coarse' if use_coarse else 'fine'].problem(update)
                if use_coarse:
                    with torch.no_grad(): fine_before=model().cpu().numpy()[0].copy()
                if isinstance(opt,FunctionSpaceOptimizer): report=function_space_update(step_model,step_problem,opt,max_hz)
                else: report=e.train_update(step_model,step_problem,opt,max_hz)
                if accel_info is not None: report['accel']=accel_info
                if use_coarse:
                    # Full-band misfit on the coarse grid is meaningless (dispersion); log the stage objective.
                    report['velocity_before_mps']=fine_before; report['data_mse']=report['objective']
                if speedup: report['speedup']={'coarse':use_coarse,'encoded':bool(encoders)}
                sync(model); compute_seconds=time.perf_counter()-compute_start
                completed=update
                pre=report.pop('velocity_before_mps')
                ds=time.perf_counter()
                with torch.no_grad(): post=model().cpu().numpy()[0].copy()
                pre_metrics=velocity_metrics(pre,truth); post_metrics=velocity_metrics(post,truth)
                if diagnostic:
                    diag={'update':update,'interval_updates':1,'layer_changes':layer_changes(model,old) if layered else [],
                        'gradient_norms':{n:float(p.grad.detach().norm()) for n,p in model.named_parameters() if p.grad is not None},
                        'omega':omega_values(model)}
                    mechanisms.write(json.dumps(diag,allow_nan=False)+'\n')
                sync(model); diagnostic_seconds+=time.perf_counter()-ds
                timing['diagnostic_seconds']+=diagnostic_seconds
                elapsed=train_prior+time.perf_counter()-train_started
                row={'update':update,'evaluated_updates':update-1,**report,**pre_metrics,
                    'cumulative_seconds':before_time,'after_evaluated_updates':update,
                    'after_metrics':post_metrics,'after_cumulative_seconds':timing['preparation_seconds']+elapsed,
                    'shots_per_update':cfg['acquisition']['num_shots'],'cumulative_shots':update*cfg['acquisition']['num_shots'],
                    'velocity_delta_rms_mps':float(np.sqrt(np.mean((post-pre).astype('float64')**2))),
                    'compute_update_seconds':compute_seconds,'diagnostic_seconds':diagnostic_seconds,
                    'step_seconds':time.perf_counter()-step_started,
                    **({'lr_factor':factor} if lr_setting is not None else {})}
                history.append(row); rollback=None
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                atomic_json(out/'status.json',{'state':'running','pid':os.getpid(),'completed_updates':completed,
                    'total_updates':cfg['training']['epochs'],'last_metrics':post_metrics,
                    'elapsed_training_seconds':elapsed,'preparation_seconds':timing['preparation_seconds'],
                    'remaining_seconds_estimate':(cfg['training']['epochs']-completed)*np.mean([x['step_seconds'] for x in history[-30:]])})
                emit(f"{spec['name']} {update}/{cfg['training']['epochs']}: RMSE(after)={post_metrics['rmse_mps']:.3f} SSIM={post_metrics['ssim']:.5f} MSE(before)={report['data_mse']:.7g} step={row['step_seconds']:.2f}s elapsed={elapsed/60:.1f}min")
                if stop_rmse is not None and post_metrics['rmse_mps']<=stop_rmse:
                    reached={'update':update,'stop_rmse':stop_rmse,'rmse_mps':post_metrics['rmse_mps'],
                             'ssim':post_metrics['ssim'],'after_cumulative_seconds':row['after_cumulative_seconds']}
                if stop_plateau is not None and not reached and len(history)>int(stop_plateau['window']):
                    curve=[h['after_metrics']['rmse_mps'] for h in history]
                    best_now,best_then=min(curve),min(curve[:-int(stop_plateau['window'])])
                    if best_then-best_now<stop_plateau['rel']*best_then:
                        reached={'update':update,'plateau':stop_plateau,'best_rmse_mps':best_now,'rmse_mps':post_metrics['rmse_mps'],
                                 'ssim':post_metrics['ssim'],'after_cumulative_seconds':row['after_cumulative_seconds']}
                if update%cfg['training']['checkpoint_interval']==0 or update==cfg['training']['epochs'] or reached:
                    es=time.perf_counter(); mse=e.evaluate_data_mse(model,problem); sync(model)
                    timing['diagnostic_seconds']+=time.perf_counter()-es
                    evaluations.append({'evaluated_updates':update,'update':update,**post_metrics,'data_mse':mse,
                        'cumulative_seconds':timing['preparation_seconds']+train_prior+time.perf_counter()-train_started})
                    atomic_json(out/'evaluation_history.json',evaluations)
                    checkpoint('checkpoints/update_%04d.pt'%update); checkpoint('checkpoint.pt')
                    np.save(out/'checkpoints'/('velocity_%04d.npy'%update),post)
                if reached:
                    np.save(out/'target_velocity.npy',post); atomic_json(out/'target_reached.json',reached)
                    if 'plateau' in reached:
                        emit(f"CONVERGED at update {update}: best RMSE {reached['best_rmse_mps']:.3f} improved < {stop_plateau['rel']} over {stop_plateau['window']} updates")
                    else:
                        emit(f"TARGET reached at update {update}: RMSE={post_metrics['rmse_mps']:.3f} <= {stop_rmse}")
                    break
        sync(model); timing['training_seconds']=train_prior+time.perf_counter()-train_started
        phase='finalizing'
        final=evaluations[-1]; np.save(out/'final_velocity.npy',post if history else initial)
        atomic_json(out/'history.json',history); atomic_json(out/'evaluation_history.json',evaluations)
        atomic_json(out/'metrics.json',{'initial_this_attempt':initial_metrics,'final':final,'timing':timing,
            'total_active_seconds':timing['preparation_seconds']+timing['training_seconds'],
            'completed_updates':completed,'cumulative_shots':completed*cfg['acquisition']['num_shots'],
            'parameters':sum(p.numel() for p in model.parameters()),
            'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30 if device.startswith('cuda') else 0.,
            'selection':'last update, no truth-based stopping','timing_includes_diagnostics':True,
            'prefit_seconds_already_in_preparation':True})
        with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(evaluations[0])); writer.writeheader(); writer.writerows(evaluations)
        if save_figures: save_plots(out,truth,initial,post if history else initial,evaluations,cfg['data']['grid_spacing_m'])
        atomic_json(out/'status.json',{'state':'completed','pid':os.getpid(),'completed_updates':completed,'final':final,'timing':timing})
        emit(f"COMPLETED {spec['name']}: RMSE={final['rmse_mps']:.3f} m/s; active time={(timing['preparation_seconds']+timing['training_seconds'])/3600:.3f} h")
        return out
    except BaseException as error:
        if rollback is not None:
            completed,_=restore_training_state(rollback,model,opt)
            del history[rollback_history_length:]
        if model is not None and opt is not None and phase=='inversion':
            try: checkpoint('interrupted.pt')
            except Exception: traceback.print_exc()
        atomic_json(out/'status.json',{'state':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            'pid':os.getpid(),'phase':phase,'completed_updates':completed,'error':repr(error)})
        emit('FAILED; queue must stop for diagnosis: '+repr(error)); raise
    finally: log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--spec'); parser.add_argument('--out',required=True)
    parser.add_argument('--resume'); parser.add_argument('--device',default='cuda:0'); parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--extend',help='checkpoint (.pt) to continue to --epochs updates (spec read from it)')
    parser.add_argument('--stop-rmse',type=float,help='stop at the first update whose velocity RMSE is at or below this value')
    parser.add_argument('--stop-plateau',nargs=2,type=float,metavar=('WINDOW','REL'),
                        help='stop when the best RMSE improved by less than REL (relative) over the last WINDOW updates')
    parser.add_argument('--shared-dir',help='shared observations root (default outputs/shared in the repository)')
    parser.add_argument('--epochs',type=int,help='override the update budget, e.g. 5 for a first-step check')
    args=parser.parse_args()
    plateau=None if args.stop_plateau is None else {'window':int(args.stop_plateau[0]),'rel':args.stop_plateau[1]}
    if args.extend:
        if args.epochs is None or args.resume:
            parser.error('--extend needs --epochs and no --resume')
        spec=torch.load(args.extend,map_location='cpu',weights_only=False)['spec']
        spec['experiment']['training']['epochs']=args.epochs
        run(spec,args.out,args.device,args.extend,not args.no_plots,args.shared_dir,extend=True,stop_rmse=args.stop_rmse,
            stop_plateau=plateau)
        raise SystemExit(0)
    if not args.spec:
        parser.error('--spec is required unless --extend is given')
    spec=json.loads(Path(args.spec).read_text(encoding='utf-8'))
    if args.epochs is not None:
        spec['experiment']['training']['epochs']=args.epochs
        spec['experiment']['training']['checkpoint_interval']=min(spec['experiment']['training']['checkpoint_interval'],args.epochs)
    run(spec,args.out,args.device,args.resume,not args.no_plots,args.shared_dir,stop_rmse=args.stop_rmse,stop_plateau=plateau)

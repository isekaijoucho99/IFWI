"""Sequential-plan runner using unchanged 2313716 model, FD and Adam update.

Snapshots include config, source hashes, optimizer and RNG. Resume into a new
attempt directory; retain previous attempt. Per-state times include required
preparation, updates, diagnostic and periodic checkpoint costs, excluding idle
time between attempts. No truth-based stopping or inversion model selection.
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
from experiments.trainable_omega import parameter_groups, omega_values
from core_controls import small_output, prefit_background, training_state, restore_training_state, sync, layer_snapshot, layer_changes

ROOT=Path(__file__).resolve().parent
SCHEMA='robustness_overthrust_noise_v1'


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
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*.py'))}


def run(spec, out, device='cuda:0', resume=None, save_figures=True):
    cfg=copy.deepcopy(spec['experiment']); control=spec['control']
    e.validate_config(cfg)
    if cfg['training'].get('shots_per_update') is not None:
        raise ValueError('This core protocol uses every shot in each update')
    if control['kind'] not in ('zero_output','prefit_background','small_output'):
        raise ValueError('Unknown initialization control')
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
        'base_commit':'231371619edc71d1b3ab7ca386954f73b56304a7','threads':1,
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
        atomic_json(out/'initial_model_provenance.json',provenance)
        if resume:
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            if saved.get('schema')!=SCHEMA or saved['spec']!=spec or saved['sources']!=sources or saved['provenance']!=provenance:
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
        shared=ROOT.parent/'shared'/robustness['condition']
        observed,observation_info=shared_observations(problem['observed'].cpu().numpy(),
            robustness['noise_sigma'],robustness['noise_seed'],shared,identity)
        problem['observed']=torch.from_numpy(observed).to(device)
        atomic_json(out/'observation_manifest.json',observation_info)
        atomic_json(out/'shared_observations.json',{'directory':str(shared.resolve()),
            'clean':'clean_observed.npy','noise':'noise.npy','observed':'observed.npy'})
        emit(f"OBSERVATIONS sigma={robustness['noise_sigma']} dtype={observed.dtype} shared_hash={observation_info['observed_hash']}")
        cm=cfg['model']; omega=cm.get('trainable_omega')
        model=e.VelocityParameterization(background,cfg['data']['grid_spacing_m'],cm['neurons'],cm['omega_0'],
            cm['mean_kmps'],cm['std_kmps'],cfg['parameterization'],cm.get('fourier'),omega is not None).to(device)
        control_report={}
        if saved and saved['phase']=='inversion':
            opt=torch.optim.Adam(parameter_groups(model,cfg['training']['learning_rate'],omega))
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
                control_report=prefit_background(model,control['updates'],control['learning_rate'],emit,save_prefit,saved)
                timing['prefit_seconds']=control_report['seconds']
                if saved: timing['preparation_seconds']+=saved.get('preparation_elapsed_seconds',saved['seconds'])
                atomic_torch(out/'prefit_selected_model.pt',{'model':model.state_dict(),'control_report':control_report})
            opt=torch.optim.Adam(parameter_groups(model,cfg['training']['learning_rate'],omega))
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
        for index,wrapped in enumerate(model.net.linear):
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
                ds=time.perf_counter(); old=layer_snapshot(model) if diagnostic else None
                sync(model); diagnostic_seconds=time.perf_counter()-ds
                compute_start=time.perf_counter()
                report=e.train_update(model,problem,opt)
                sync(model); compute_seconds=time.perf_counter()-compute_start
                completed=update
                pre=report.pop('velocity_before_mps')
                ds=time.perf_counter()
                with torch.no_grad(): post=model().cpu().numpy()[0].copy()
                pre_metrics=velocity_metrics(pre,truth); post_metrics=velocity_metrics(post,truth)
                if diagnostic:
                    diag={'update':update,'interval_updates':1,'layer_changes':layer_changes(model,old),
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
                    'step_seconds':time.perf_counter()-step_started}
                history.append(row); rollback=None
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                atomic_json(out/'status.json',{'state':'running','pid':os.getpid(),'completed_updates':completed,
                    'total_updates':cfg['training']['epochs'],'last_metrics':post_metrics,
                    'elapsed_training_seconds':elapsed,'preparation_seconds':timing['preparation_seconds'],
                    'remaining_seconds_estimate':(cfg['training']['epochs']-completed)*np.mean([x['step_seconds'] for x in history[-30:]])})
                emit(f"{spec['name']} {update}/{cfg['training']['epochs']}: RMSE(after)={post_metrics['rmse_mps']:.3f} SSIM={post_metrics['ssim']:.5f} MSE(before)={report['data_mse']:.7g} step={row['step_seconds']:.2f}s elapsed={elapsed/60:.1f}min")
                if update%cfg['training']['checkpoint_interval']==0 or update==cfg['training']['epochs']:
                    es=time.perf_counter(); mse=e.evaluate_data_mse(model,problem); sync(model)
                    timing['diagnostic_seconds']+=time.perf_counter()-es
                    evaluations.append({'evaluated_updates':update,'update':update,**post_metrics,'data_mse':mse,
                        'cumulative_seconds':timing['preparation_seconds']+train_prior+time.perf_counter()-train_started})
                    atomic_json(out/'evaluation_history.json',evaluations)
                    checkpoint('checkpoints/update_%04d.pt'%update); checkpoint('checkpoint.pt')
                    np.save(out/'checkpoints'/('velocity_%04d.npy'%update),post)
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
    parser=argparse.ArgumentParser(); parser.add_argument('--spec',required=True); parser.add_argument('--out',required=True)
    parser.add_argument('--resume'); parser.add_argument('--device',default='cuda:0'); parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    run(json.loads(Path(args.spec).read_text(encoding='utf-8')),args.out,args.device,args.resume,not args.no_plots)

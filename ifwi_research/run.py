"""Additive research runner for a checked-out IFWI repository.

Example: python -m ifwi_research.run --baseline <P-MS spec.json>
 --research experiments/configs/five_ideas/00_baseline.json --out outputs/new_run

A fixed update budget is mandatory. Test truth never selects an iterate or stops
training. Source/data/config/prior identities must match for exact resume.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .engine import ResearchConfig,ResearchEngine
from .dfm import FrozenDFMPrior
from .repository import require_repository,load_problem


def read_research_config(path,baseline_learning_rate=1e-4,baseline_seed=3):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    fields={f.name for f in dataclasses.fields(ResearchConfig)}
    unknown=set(data)-fields-{'prior','interface_reference'}
    if unknown: raise ValueError(f'Unknown research settings: {sorted(unknown)}')
    prior=data.pop('prior',None); interface=data.pop('interface_reference',None)
    data.setdefault('learning_rate',baseline_learning_rate); data.setdefault('seed',baseline_seed)
    cfg=ResearchConfig(**data).validate()
    if cfg.method!='dfm' and prior is not None: raise ValueError('Prior settings require dfm method')
    if cfg.method!='interface' and interface is not None: raise ValueError('Interface reference requires interface method')
    return cfg,prior,interface


def source_identity(root):
    paths=list((root/'ifwi_research').glob('*.py'))
    paths+=list((root/'experiments').glob('*.py'))
    paths+=[root/x for x in ('core_runner.py','core_controls.py','core_models.py','core_speed.py','core_accel.py','ifwi_modules.py','rnn_fd.py','generator.py','plot_functions.py','robust_observations.py')]
    return {str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def tensor_hash(value):
    a=np.ascontiguousarray(value.detach().cpu().numpy()) if isinstance(value,torch.Tensor) else np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def atomic_save(payload,path):
    temp=path.with_suffix('.tmp'); torch.save(payload,temp); temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True); parser.add_argument('--research',required=True)
    parser.add_argument('--out',required=True); parser.add_argument('--device',default='cpu')
    parser.add_argument('--epochs',type=int); parser.add_argument('--resume')
    parser.add_argument('--snr-db',type=float,help='Seeded AWGN; signal power computed from all clean recorded samples')
    args=parser.parse_args(); started=time.perf_counter()
    root=require_repository(Path(__file__).resolve().parents[1])
    spec=json.loads(Path(args.baseline).read_text(encoding='utf-8'))
    cfg,prior_options,interface_path=read_research_config(args.research,spec['experiment']['training']['learning_rate'],spec['experiment']['seed'])
    # Both the base INR and the research component use the same declared seed.
    spec['experiment']['seed']=cfg.seed
    if args.epochs is not None:
        if args.epochs<1: parser.error('--epochs must be positive')
        spec['experiment']['training']['epochs']=args.epochs
    epochs=spec['experiment']['training']['epochs']
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    base,background,objective,truth,provenance=load_problem(spec,root,args.device,args.snr_db)
    spacing=objective.problem['config']['data']['grid_spacing_m']
    prior=None
    if cfg.method=='dfm':
        if not isinstance(prior_options,dict) or 'checkpoint' not in prior_options:
            raise ValueError('DFM requires a trained prior checkpoint in research JSON')
        options=dict(prior_options); path=Path(options.pop('checkpoint'))
        if not path.is_absolute(): path=root/path
        prior=FrozenDFMPrior(path,spacing,args.device,**options)
        # Exact model identity check is a floor, not proof against related/cropped model leakage.
        from .train_dfm import array_hash
        if array_hash(truth) in set(prior.meta.get('train_model_hashes',[])+prior.meta.get('val_model_hashes',[])):
            raise ValueError('Test velocity model appears in prior training/validation dataset')
    phi=None
    if interface_path:
        path=Path(interface_path); path=path if path.is_absolute() else root/path
        phi=torch.from_numpy(np.load(path,allow_pickle=False)).float()
    engine=ResearchEngine(base,background,objective,spacing,cfg,prior=prior,phi_reference=phi)
    from experiments.frequency_continuation import stage_cutoff
    from experiments.lr_schedule import lr_factor
    schedule=spec['experiment']['training'].get('frequency_schedule')
    lr_schedule=spec['experiment']['training'].get('lr_schedule')
    identity={'baseline':spec,'research':dataclasses.asdict(cfg),'prior_options':prior_options,
              'source_hashes':source_identity(root),'initial_hash':tensor_hash(background),
              'observations_hash':tensor_hash(objective.problem['observed']),
              'interface_reference_hash':tensor_hash(phi) if phi is not None else None,
              'truth_evaluation_hash':tensor_hash(truth),'snr_db':args.snr_db,
              'device':args.device,'torch':str(torch.__version__)}
    previous_elapsed=0.
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=args.device,weights_only=True)
        if checkpoint['identity']!=identity: raise ValueError('Resume source/data/config/runtime identity mismatch')
        engine.load_state_dict(checkpoint['engine']); previous_elapsed=checkpoint['elapsed_seconds']
        if engine.completed>=epochs: raise ValueError('Checkpoint already reached fixed update budget')
    interval=spec['experiment']['training']['checkpoint_interval']
    def sync():
        if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
    def elapsed(): sync(); return previous_elapsed+time.perf_counter()-started
    sync()
    if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(args.device)
    prep_seconds=time.perf_counter()-started
    (out/'identity.json').write_text(json.dumps(identity,indent=2,allow_nan=False),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2,default=str),encoding='utf-8')
    history=(out/'history.jsonl').open('w',encoding='utf-8',buffering=1)
    try:
        while engine.completed<epochs:
            index=engine.completed+1
            cutoff=stage_cutoff(schedule,index)
            report=engine.step(cutoff,lr_multiplier=lr_factor(lr_schedule,schedule,index))
            report['elapsed_seconds']=elapsed()
            history.write(json.dumps(report,allow_nan=False)+'\n')
            if index%interval==0 or index==epochs:
                payload={'identity':identity,'engine':engine.state_dict(),'elapsed_seconds':elapsed()}
                atomic_save(payload,out/'checkpoint.pt')
                with torch.no_grad(): np.save(out/f'velocity_{index:05d}.npy',engine.model()[0].cpu().numpy())
                print(f"update={index} objective={report['physical_objective']:.6g} elapsed={report['elapsed_seconds']:.1f}s",flush=True)
        with torch.no_grad(): final=engine.model().detach()
        # Only now evaluate against the synthetic truth. Not an optimizer/stopping input.
        diff=final[0].cpu().numpy()-truth
        evaluation={'rmse_mps':float(np.sqrt(np.mean(diff.astype('float64')**2))),
                    'mae_mps':float(np.abs(diff).mean())}
        final_physical=objective.evaluate(final,gradient=False).value
        summary={'method':cfg.method,'completed_updates':engine.completed,'evaluation_only':evaluation,
                 'full_band_mse':final_physical,'physical_cost':objective.counters,
                 'elapsed_seconds':elapsed(),'preparation_seconds_this_attempt':prep_seconds,
                 'prior_seconds':engine.prior_seconds,'prior_calls':getattr(prior,'calls',0),
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved(args.device) if args.device.startswith('cuda') else None,
                 'note':'Timing includes preparation, guards, prior and final evaluation; offline prior training excluded and must be reported separately'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    except BaseException as exc:
        (out/'FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\nLast complete checkpoint, if any, is the only resumable state.\n',encoding='utf-8')
        raise
    finally: history.close()

if __name__=='__main__': main()

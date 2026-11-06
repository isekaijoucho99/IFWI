"""Explicit conditional-DFM first-stage entry: candidate utility + actual coupling.

CPU prototype only. --diagnostic-checkpoint accepts UNQUALIFIED training exports
without changing their status; --checkpoint keeps the strict qualified loader.
No weights are substituted. --synthetic-smoke performs plumbing/oracle checks only.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .conditional_validation import (DiagnosticConditionalDFM,evaluate_candidates,
    compare_coupling,restore_snapshot)
from .conditional_dfm import FrozenConditionalDFM
from .engine import ResearchConfig,ResearchEngine
from .illumination import NativeSensitivity
from .illumination_prior import IlluminationSoftPrior,SoftPriorConfig
from .repository import RepositoryObjective,load_problem
from .run import source_identity,tensor_hash


def sensitivity_for_objective(objective,estimate,*,probes=2,seed=3,cutoff=None):
    """Rebuild the EXACT snapshot operator; never infer masks from residuals."""
    # Accessing identity checks uniform float32 order-2 solver metadata first.
    identity=objective.operator_identity
    solvers=[item['solver'] for item in objective.problem['propagators']]; fd=solvers[0].fd
    geometry={'nz':estimate.shape[-2],'nx':estimate.shape[-1]}
    for name in ('xs','zs','xr','zr'):
        geometry[name]=torch.cat([getattr(s,name).detach().cpu() for s in solvers],dim=1)
    estimator=NativeSensitivity(geometry=geometry,wavelet=objective.problem['wavelet'].detach().cpu(),
        dt_s=fd.dt,spacing_m=fd.dz,npad=fd.npad,free_surface=fd.freeSurface,
        pml_vmax_mps=fd.vmax,pml_reflection=fd.log_para,shot_batch_size=1)
    if estimator.operator_identity!=identity: raise ValueError('Snapshot sensitivity operator mismatch')
    result=estimator.estimate(estimate,probes=probes,seed=seed,cutoff=cutoff)
    return result


def preflight_snapshot_spec(spec,root):
    """Reject truth-assisted initialization and oversized work BEFORE PDE solves."""
    if (spec.get('background') or {}).get('kind')!='linear_depth':
        raise ValueError('Data-only snapshot validation requires independent linear_depth initialization; truth-smoothed backgrounds are not allowed')
    cfg=spec['experiment']
    if not 1<=cfg['forward']['nt']<=256:
        raise ValueError('CPU diagnostic requires nt <=256 before constructing the physical problem')
    if not 1<=cfg['acquisition']['num_shots']<=16:
        raise ValueError('CPU diagnostic requires at most 16 shots')
    from experiments.residual_ifwi_data import load_velocity_data
    grid,_,_=load_velocity_data(cfg,root)
    if grid.size>4096:
        raise ValueError('CPU diagnostic requires <=4096 grid cells before constructing the physical problem')


def synthetic_snapshot():
    """Small native RNN-FD fixture; no checkpoint or geological quality claim."""
    from experiments import residual_ifwi_experiment as e
    cfg=e.default_config(); cfg['model']['neurons']=[2,8,8,1]
    cfg['forward'].update(nt=48,dt_s=.001,frequency_hz=30.,npad=3)
    cfg['data']['grid_spacing_m']=20.
    cfg['acquisition'].update(num_shots=3,source_x_indices=[3,8,13],receiver_stride=1)
    cfg['training']['shot_batch_size']=1
    initial=np.full((10,18),2000.,dtype=np.float32); truth=initial.copy(); truth[4:7,5:12]+=150.
    raw=e.build_forward_problem(truth,cfg,'cpu')
    torch.manual_seed(3)
    background=torch.from_numpy(initial)[None]
    engine=ResearchEngine(e.model_from_config(initial,cfg),background,RepositoryObjective(raw),20.,ResearchConfig())
    return engine,background,torch.from_numpy(truth)[None]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True); p.add_argument('--baseline'); p.add_argument('--snapshot')
    weights=p.add_mutually_exclusive_group()
    weights.add_argument('--checkpoint'); weights.add_argument('--diagnostic-checkpoint')
    p.add_argument('--synthetic-smoke',action='store_true'); p.add_argument('--oracle',action='store_true')
    p.add_argument('--condition-background',help='Optional data-only HxW .npy; default is the frozen snapshot estimate')
    p.add_argument('--condition-provenance',default='snapshot_estimate_identity_background_v1')
    p.add_argument('--steps',type=int,default=3); p.add_argument('--draws',type=int,default=4)
    p.add_argument('--ratio',type=float,default=.025); p.add_argument('--nfe',type=int,default=1)
    p.add_argument('--seed',type=int,default=3); p.add_argument('--probes',type=int,default=2)
    p.add_argument('--refresh-every',type=int,default=5)
    args=p.parse_args(argv)
    if args.steps<1 or not 1<=args.draws<=64: p.error('steps must be positive; draws must be 1..64')
    if not 0<=args.ratio<1: p.error('ratio must lie in [0,1)')
    if not args.condition_provenance.strip(): p.error('Nonempty condition provenance required')
    if args.condition_background and args.condition_provenance=='snapshot_estimate_identity_background_v1':
        p.error('An external background requires explicit --condition-provenance')
    if Path(args.out).exists(): p.error('--out must be a new directory')
    path=args.checkpoint or args.diagnostic_checkpoint
    if path and not Path(path).is_file(): p.error('Requested checkpoint does not exist; no fallback weights')
    torch.set_num_threads(1); root=Path(__file__).resolve().parents[1]
    cutoff=None; snapshot_hash=None
    if args.synthetic_smoke:
        if args.baseline or args.snapshot: p.error('synthetic-smoke cannot load a real baseline/snapshot')
        engine,initial,truth=synthetic_snapshot()
        provenance={'condition_initialization':'independent constant 2000 m/s synthetic fixture'}
    else:
        if not args.baseline or not args.snapshot: p.error('--baseline and --snapshot are required outside synthetic smoke')
        spec=json.loads(Path(args.baseline).read_text())
        preflight_snapshot_spec(spec,root)
        saved=torch.load(args.snapshot,map_location='cpu',weights_only=True)
        cfg=ResearchConfig(**saved['engine']['config']).validate()
        model,initial,obj,target,provenance=load_problem(spec,root,'cpu',saved['identity'].get('snr_db'))
        engine=ResearchEngine(model,initial,obj,spec['experiment']['data']['grid_spacing_m'],cfg)
        restore_snapshot(engine,saved,spec,initial)
        import hashlib
        snapshot_hash=hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest()
        truth=torch.as_tensor(target,dtype=initial.dtype)[None]
        from experiments.frequency_continuation import stage_cutoff
        schedule=spec['experiment']['training'].get('frequency_schedule')
        cutoffs=[stage_cutoff(schedule,engine.completed+i+1) for i in range(args.steps)]
        if any(c!=cutoffs[0] for c in cutoffs):
            p.error('This diagnostic must stay within one frequency stage; reduce --steps')
        cutoff=cutoffs[0]
    estimate=engine.model().detach().clone()
    background=(torch.as_tensor(np.load(args.condition_background,allow_pickle=False),dtype=estimate.dtype)[None]
                if args.condition_background else estimate.clone())
    spacing=engine.objective.problem['config']['data']['grid_spacing_m']
    path=args.checkpoint or args.diagnostic_checkpoint
    source=(DiagnosticConditionalDFM(path,spacing,nfe=args.nfe,seed=args.seed) if args.diagnostic_checkpoint else
            FrozenConditionalDFM(path,spacing,nfe=args.nfe,seed=args.seed) if path else None)
    if source is not None:
        from .train_dfm import array_hash
        training=source.meta['training']
        if array_hash(truth[0].numpy()) in set(training['train_model_hashes']+training['val_model_hashes']):
            raise ValueError('Evaluation model overlaps training/validation; no held-out comparison')
    sensitivity=sensitivity_for_objective(engine.objective,estimate,probes=args.probes,seed=args.seed,cutoff=cutoff)
    if source is None:
        # Oracle-only diagnostics still need a wrapper's spatial/normalization contract.
        from .conditional_validation import _OracleSource
        class Context:
            scale_mps=1000.
            spacing_m=spacing
        source_for_context=_OracleSource(estimate,Context())
    else: source_for_context=source
    prior=IlluminationSoftPrior(source_for_context,background,sensitivity['illumination'].to(estimate),
        spacing_m=spacing,cutoff=cutoff,operator_identity=sensitivity['operator_identity'],
        config=SoftPriorConfig(refresh_every=args.refresh_every))
    candidates=(evaluate_candidates(prior,estimate,engine.objective,
        sample_ids=range(engine.completed+1,engine.completed+1+args.draws),truth=truth) if source is not None else
        {'status':'not_run_missing_checkpoint','qualification_changed':False})
    artifacts={}
    comparison=compare_coupling(engine,prior,steps=args.steps,ratio=args.ratio,
        oracle_truth=truth if args.oracle else None,evaluation_truth=truth,include_dfm=source is not None,artifacts=artifacts)
    summary=dict(schema='conditional_dfm_first_stage_v1',synthetic_smoke=args.synthetic_smoke,
        scientific_qualification=False,qualification_changed=False,candidates=candidates,coupling=comparison,
        checkpoint_hash=getattr(source,'checkpoint_hash',None),checkpoint_meta=getattr(source,'meta',None),
        diagnostic_checkpoint=bool(args.diagnostic_checkpoint),snapshot_sha256=snapshot_hash,
        inference_sampler=(dict(seed=source.seed,nfe=source.nfe,max_cells=source.max_cells,
            inference_identity=source.inference_identity) if source is not None else None),
        effective_prior_config=asdict(prior.config),
        lr_policy='fixed_base_learning_rate_no_schedule_multiplier',
        initialization_provenance=provenance,
        independent_initialization=(spec['background'] if not args.synthetic_smoke else {'kind':'constant','mps':2000.}),
        source_hashes=source_identity(root),condition_provenance=args.condition_provenance,
        condition_hashes=dict(background=tensor_hash(background),estimate=tensor_hash(estimate),anchor_mask=tensor_hash(prior.anchor_mask)),
        sensitivity=sensitivity['diagnostics'],operator_identity=sensitivity['operator_identity'],
        limitations=['CPU prototype caps apply; no production-size or GPU benchmark.',
            'Conditions must match the training recipe; provenance is a declaration, not proof.',
            'The existing benchmark operator, including its PML bound, is inherited unchanged.',
            'Truth metrics never select ordinary candidates, steps, or stopping. Oracle is a separate truth-fed control.'])
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    # These exact arrays can be used as the paired data-only training conditions.
    for name,value in dict(background=background,anchor=estimate,mask=prior.anchor_mask,illumination=prior.illumination).items():
        np.save(out/f'{name}.npy',value[0].cpu().numpy())
    for name,payload in artifacts.items():
        np.save(out/f'{name}_velocity.npy',payload['velocity'][0].cpu().numpy())
        torch.save({'diagnostic_only':True,'engine':payload['engine'],
                    'source_schema':comparison[name]['source_schema']},out/f'{name}_state.pt')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(dict(out=str(out),candidates=candidates.get('status','evaluated'),
        scientific_qualification=False),indent=2))
    return summary


if __name__=='__main__': main()

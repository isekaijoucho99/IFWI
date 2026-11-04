"""CPU-only sensitivity diagnostic; no truth, observations or fake DFM weights.

A supplied qualified checkpoint may generate one UNACCEPTED soft candidate, but
this diagnostic never applies it or claims geological inversion improvement.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .illumination import NativeSensitivity
from .conditional_dfm import FrozenConditionalDFM
from .illumination_prior import IlluminationSoftPrior


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--checkpoint',help='Compatible, independently qualified conditional checkpoint; no fallback')
    parser.add_argument('--probes',type=int,default=2)
    parser.add_argument('--seed',type=int,default=3)
    parser.add_argument('--cutoff-hz',type=float)
    parser.add_argument('--mute-before-s',type=float,default=0.,help='Explicit data-domain time gate, never inferred from residuals')
    args=parser.parse_args(); started=time.perf_counter()
    torch.set_num_threads(1)
    # Reject absent/legacy/unqualified requested weights before any PDE solve.
    source=FrozenConditionalDFM(args.checkpoint,20.,'cpu',seed=args.seed) if args.checkpoint else None
    if not math.isfinite(args.mute_before_s) or not 0<=args.mute_before_s<.064:
        parser.error('--mute-before-s must lie in [0,0.064)')
    nz,nx,nt=12,20,64; dt=.001
    geometry=dict(nz=nz,nx=nx,xs=torch.tensor([[2,10,17]]),zs=torch.ones(1,3,dtype=torch.long),
                  xr=torch.arange(nx).reshape(1,1,nx).expand(1,3,nx).clone(),
                  zr=torch.ones(1,3,nx,dtype=torch.long))
    t=torch.arange(nt)*dt; a=math.pi*30*(t-.025)
    wavelet=(1-2*a.square())*torch.exp(-a.square())
    # Chosen diagnostic background, not a hidden synthetic target or smooth truth.
    background=(2000+5*torch.arange(nz)[:,None]).expand(nz,nx).float()[None].clone()
    estimator=NativeSensitivity(geometry=geometry,wavelet=wavelet,dt_s=dt,spacing_m=20.,
                                npad=3,free_surface=True,pml_vmax_mps=3000.,shot_batch_size=1)
    data_mask=None
    if args.mute_before_s>0:
        data_mask=(t>=args.mute_before_s).float()[None,None,:,None].expand(1,3,nt,nx).clone()
    result=estimator.estimate(background,probes=args.probes,seed=args.seed,cutoff=args.cutoff_hz,data_mask=data_mask)
    arrays={'background_mps':background.numpy(),'raw_diagonal':result['raw_diagonal'].numpy(),
            'relative_sensitivity':result['illumination'].numpy()}
    summary={'is_geological_benchmark':False,'physical_inversion_updates':0,'dfm_training_executed':False,
             'sensitivity_status':result['status'],'operator_identity':result['operator_identity'],
             'diagnostics':result['diagnostics'],'prior_status':'not_run_missing_qualified_checkpoint',
             'network_evaluations':0,'posterior_uncertainty':False,
             'context':'Explicit diagnostic background and physical PML bound, no truth/observed input'}
    if source is not None:
        prior=IlluminationSoftPrior(source,background,result['illumination'],spacing_m=20.,
                                    cutoff=args.cutoff_hz,operator_identity=result['operator_identity'])
        arrays['candidate_unaccepted_mps']=prior.proposal(background,1).cpu().numpy()
        summary.update(prior_status='candidate_only_not_applied',network_evaluations=source.calls,
                       candidate_physical_acceptance='not_tested_no_inversion_update')
    summary['elapsed_seconds']=time.perf_counter()-started
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'sensitivity.npz',**arrays)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__': main()

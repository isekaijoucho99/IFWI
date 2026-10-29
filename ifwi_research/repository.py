"""Lazy adapter to the existing IFWI RNN-FD; original files are never modified.

Mandatory before production: run verification/five_ideas/test_repository_bridge.py
inside the target checkout. Standalone toy tests are NOT this integration test.
"""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import math
import numpy as np
import torch
from .engine import Evaluation
from .routing import common_contrast_basis


def require_repository(root: str | Path) -> Path:
    root=Path(root).resolve()
    required=['core_runner.py','core_speed.py','rnn_fd.py','experiments/residual_ifwi_experiment.py']
    if not all((root/name).is_file() for name in required):
        raise RuntimeError('Install this additive package into the IFWI checkout at the documented base commit; original RNN-FD files are missing')
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    return root


class RepositoryObjective:
    def __init__(self,problem):
        from experiments import residual_ifwi_experiment as e
        from experiments.frequency_continuation import energy_scales
        self.e=e; self.problem=problem; self.shots=problem['observed'].shape[1]
        cfg=problem['config']
        problem['frequency_scales']=energy_scales(problem['observed'],cfg['training'].get('frequency_schedule'),cfg['forward']['dt_s'])
        self.encoder=None
        self.counters={'forward_solves':0,'backward_solves':0,'forward_time_steps':0}

    def valid(self,velocity):
        try: self.e.check_velocity(velocity,self.problem['config']); return True
        except (ValueError,FloatingPointError): return False

    def _ensure_encoder(self):
        if self.encoder is None:
            from core_speed import SourceEncoder
            geometry=self.problem['geometry_tensors']
            for name in ('xr','zr'):
                values=geometry[name]
                if not torch.equal(values,values[:,:1].expand_as(values)):
                    raise ValueError('Common/contrast routing requires identical receivers for all shots')
            self.encoder=SourceEncoder(self.problem,1,seed=3)

    def evaluate(self,velocity,plan=None,cutoff=None,gradient=True):
        if not self.valid(velocity): raise FloatingPointError('Invalid physical trial model / CFL violation')
        if cutoff is not None and cutoff not in self.problem['frequency_scales']:
            raise ValueError('Cutoff must be registered in the baseline frequency schedule')
        obs=self.problem['observed']; global_s=self.shots
        if plan is None:
            jobs=[(it['solver'],obs[:,it['first']:it['last']],1.,it['last']-it['first'],None)
                  for it in self.problem['propagators']]
        else:
            self._ensure_encoder()
            codes=common_contrast_basis(global_s,device=obs.device,dtype=obs.dtype)
            jobs=[]
            for index,weight in zip(plan.indices.tolist(),plan.weights.tolist()):
                record=torch.einsum('s,bstr->btr',codes[index],obs)[:,None]
                jobs.append((self.encoder.propagators[0]['solver'],record,weight,1,codes[index]))
        value=0.; total_grad=torch.zeros_like(velocity) if gradient else None; energies=[]
        for solver,observations,weight,count,code in jobs:
            if code is not None: solver.fd.weights=code.to(dtype=torch.float32,device=obs.device)
            with torch.enable_grad() if gradient else torch.no_grad():
                field=velocity.detach().requires_grad_(gradient)
                prediction=self.e._simulate(solver,field,self.problem['wavelet'])
                loss=self.e.training_objective(prediction,observations,global_s if plan is None else 1,
                                                cutoff,self.problem)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite RNN-FD objective')
                scalar=float(loss.detach()); value+=weight*scalar; energies.append(scalar)
                self.counters['forward_solves']+=count
                self.counters['forward_time_steps']+=count*self.problem['config']['forward']['nt']
                if gradient:
                    total_grad.add_(torch.autograd.grad(loss,field)[0],alpha=weight)
                    self.counters['backward_solves']+=count
        return Evaluation(value,total_grad,plan.indices.clone() if plan is not None else None,
                          torch.tensor(energies,dtype=torch.float64) if plan is not None else None)


def load_problem(spec: dict,root: Path,device: str='cpu',snr_db: float | None=None):
    root=require_repository(root)
    from experiments import residual_ifwi_experiment as e
    from experiments.residual_ifwi_data import load_velocity_data
    from core_models import smooth400_background,slowness_gaussian_background,linear_depth_background
    from core_runner import build_model,check_spec
    check_spec(spec)
    if spec.get('diffusion_prior') or spec.get('accel') or spec.get('speedup') or spec.get('target'):
        raise ValueError('Use an unmodified single-factor P-MS baseline spec without prior/accel/speedup/truth-based target')
    cfg=copy.deepcopy(spec['experiment']); train=cfg['training']
    if cfg['parameterization']!='residual' or spec['control']!={'kind':'zero_output'}:
        raise ValueError('This comparison harness currently supports residual zero_output initialization only')
    if cfg['initialization'].get('pretrain') or cfg['model'].get('low_rank') or cfg['model'].get('trainable_omega'):
        raise ValueError('Pretraining/low-rank/trainable-omega parameter groups require a separate validated adapter')
    if train.get('optimizer') or train.get('shots_per_update') or train.get('time_gain'):
        raise ValueError('Keep the baseline standard Adam and full shots; routing is configured separately')
    if spec.get('robustness',{}).get('noise_sigma',0)!=0:
        raise ValueError('Use --snr-db for explicit seeded noise; legacy noise_sigma is not silently reinterpreted')
    # Same loss, microbatch one: no simultaneous wavefield graphs, including on 16GB GPUs.
    train['shot_batch_size']=1
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    truth,background,provenance=load_velocity_data(cfg,root)
    override=spec.get('background')
    if override:
        if override['kind']=='smooth400': background=smooth400_background(root/'data',truth.shape)
        elif override['kind']=='slowness_gaussian':
            background=slowness_gaussian_background(truth,cfg['data']['grid_spacing_m'],override['slowness_sigma_m'],override['velocity_sigma_m'])
        elif override['kind']=='linear_depth':
            background=linear_depth_background(truth.shape,cfg['data']['grid_spacing_m'],override['top_mps'],override['gradient_per_s'])
        else: raise ValueError('Unsupported background override')
    model=build_model(spec,cfg,background).to(device)
    problem=e.build_forward_problem(truth,cfg,device)
    if snr_db is not None:
        if not math.isfinite(snr_db): raise ValueError('SNR must be finite dB')
        rng=torch.Generator().manual_seed(cfg['seed']+811)
        noise=torch.randn(problem['observed'].shape,generator=rng).to(problem['observed'])
        scale=problem['observed'].double().square().mean().sqrt()/noise.double().square().mean().sqrt()/10**(snr_db/20)
        problem['observed']=(problem['observed']+noise*scale.to(noise)).detach()
    return model,torch.as_tensor(background,dtype=torch.float32,device=device)[None],RepositoryObjective(problem),truth,provenance

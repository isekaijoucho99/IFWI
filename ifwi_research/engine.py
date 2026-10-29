"""Separate experiment engine: unchanged physical solver, one research idea at a time.

The backend returns a velocity-space gradient. The network VJP is evaluated
once; no wavefield graph is retained across source batches. No truth is accepted.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, dataclass
import math
import time
import torch
from torch import nn
from .routing import CodePlan, ContrastRouter
from .safety import guarded_optimizer_step, protect_physical_gradient, rng_state, restore_rng


@dataclass
class Evaluation:
    value: float
    gradient: torch.Tensor | None
    indices: torch.Tensor | None = None
    energies: torch.Tensor | None = None


@dataclass(frozen=True)
class ResearchConfig:
    method: str = 'baseline'
    learning_rate: float = 1e-4
    seed: int = 3
    gate_strength: float = .2
    attention_block: int = 10
    routing_samples: int = 2
    routing_topk: int = 4
    exploration: float = .2
    audit_every: int = 25
    prior_ratio: float = .025
    prior_start: int = 201
    prior_every: int = 5
    guard_every: int = 0
    max_trials: int = 4
    interface_depth_fraction: float = .5
    interface_epsilon_cells: float = 3.
    interface_displacement_m: float = 150.
    interface_regularization: float = 0.

    def validate(self):
        if self.method not in ('baseline','routing','wavelet','interface','gradient_attention','dfm'):
            raise ValueError('Exactly one supported method is required, not a stack of ideas')
        for name in ('learning_rate','interface_displacement_m'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f'{name} must be finite positive')
        for name in ('attention_block','routing_samples','routing_topk','audit_every','prior_start','prior_every','max_trials'):
            value=getattr(self,name)
            if type(value) is not int or value<1: raise ValueError(f'{name} must be a positive integer')
        if self.attention_block<2: raise ValueError('attention_block >=2 is required for gate learning')
        if type(self.guard_every) is not int or self.guard_every<0: raise ValueError('Invalid guard_every')
        if type(self.seed) is not int or self.seed<0: raise ValueError('Invalid seed')
        for name in ('gate_strength','prior_ratio'):
            if not math.isfinite(getattr(self,name)) or not 0<=getattr(self,name)<1:
                raise ValueError(f'{name} must be in [0,1)')
        if not 0<self.exploration<=1: raise ValueError('Positive routing exploration is mandatory')
        if not 0<self.interface_depth_fraction<1 or not math.isfinite(self.interface_epsilon_cells) or self.interface_epsilon_cells<1:
            raise ValueError('Invalid physical interface initialization')
        if not math.isfinite(self.interface_regularization) or self.interface_regularization<0:
            raise ValueError('Invalid interface_regularization')
        return self


class ResearchEngine:
    """A small, independently checkpointed experiment engine, not core_runner replacement."""
    def __init__(self, base: nn.Module, background: torch.Tensor, objective,
                 spacing_m: float, config: ResearchConfig, *, prior=None, phi_reference=None):
        self.config=config.validate()
        if not math.isfinite(spacing_m) or spacing_m<=0: raise ValueError('Invalid spacing_m')
        if background.ndim!=3 or background.shape[0]!=1: raise ValueError('Require [1,H,W] background')
        self.objective,self.prior,self.completed=objective,prior,0
        if config.method=='dfm' and prior is None:
            raise ValueError('DFM needs a trained frozen prior checkpoint')
        if config.method!='dfm' and prior is not None: raise ValueError('Prior only allowed in dfm experiment')
        if config.method=='wavelet':
            from .wavelets import WaveletVelocity
            model=WaveletVelocity(base,background,strength=config.gate_strength)
        elif config.method=='gradient_attention':
            from .gradient_attention import GradientAttentionVelocity
            model=GradientAttentionVelocity(base,background,strength=config.gate_strength)
        elif config.method=='interface':
            from .interfaces import InterfaceVelocity
            model=InterfaceVelocity(base,tuple(background.shape[-2:]),spacing_m,
                (background.shape[-2]-1)*spacing_m*config.interface_depth_fraction,
                config.interface_epsilon_cells*spacing_m,displacement_m=config.interface_displacement_m,
                phi_reference=phi_reference)
        else: model=base
        self.model=model.to(device=background.device,dtype=background.dtype)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config.learning_rate)
        self.router=(ContrastRouter(objective.shots,config.routing_samples,config.routing_topk,
                                    config.exploration,config.seed) if config.method=='routing' else None)
        self.prior_seconds=0.

    def _sync(self):
        device=next(self.model.parameters()).device
        if device.type=='cuda': torch.cuda.synchronize(device)

    def state_dict(self):
        return {'schema':'ifwi_five_ideas_engine_v1','config':asdict(self.config),
                'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                'router':self.router.state_dict() if self.router else None,
                'completed':self.completed,'rng':rng_state(),
                'counters':copy.deepcopy(self.objective.counters),'prior_seconds':self.prior_seconds,
                'prior_calls':getattr(self.prior,'calls',0),
                'prior_hash':getattr(self.prior,'checkpoint_hash',None)}

    def load_state_dict(self, saved):
        if saved.get('schema')!='ifwi_five_ideas_engine_v1' or saved['config']!=asdict(self.config):
            raise ValueError('Engine configuration/schema mismatch; not an exact resume')
        if saved['prior_hash']!=getattr(self.prior,'checkpoint_hash',None):
            raise ValueError('Prior weight identity changed')
        if (saved['router'] is None)!=(self.router is None): raise ValueError('Router state mismatch')
        self.model.load_state_dict(saved['model']); self.optimizer.load_state_dict(saved['optimizer'])
        if self.router is not None: self.router.load_state_dict(saved['router'])
        self.completed=int(saved['completed']); self.objective.counters=copy.deepcopy(saved['counters'])
        self.prior_seconds=float(saved['prior_seconds'])
        if self.prior is not None: self.prior.calls=int(saved['prior_calls'])
        restore_rng(saved['rng'])

    def step(self, cutoff: float | None=None, *, lr_multiplier: float=1.):
        if not math.isfinite(lr_multiplier) or lr_multiplier<=0: raise ValueError('Invalid LR multiplier')
        update=self.completed+1; cfg=self.config
        if self.router is not None:
            plan=self.router.plan(full=(update==1 or update%cfg.audit_every==0))
        else: plan=None
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups: group['lr']=cfg.learning_rate*lr_multiplier
        velocity=self.model()
        if not self.objective.valid(velocity.detach()): raise FloatingPointError('Invalid physical start model')
        # Physical differentiation never traverses the trainable attention context.
        measured=self.objective.evaluate(velocity.detach(),plan=plan,cutoff=cutoff,gradient=True)
        params=[p for p in self.model.parameters() if p.requires_grad]
        active_prior=(self.prior is not None and update>=cfg.prior_start and (update-cfg.prior_start)%cfg.prior_every==0)
        regularize=cfg.method=='interface' and cfg.interface_regularization>0
        grads=torch.autograd.grad(velocity,params,measured.gradient,
                                 retain_graph=active_prior or regularize,allow_unused=True)
        physical=[torch.zeros_like(p) if g is None else g for p,g in zip(params,grads)]
        composed=physical; prior_report=None
        if active_prior:
            self._sync(); begin=time.perf_counter()
            target=self.prior.proposal(velocity.detach(),update)
            loss=self.prior.penalty(velocity,target)
            proposal=torch.autograd.grad(loss,params,allow_unused=True)
            proposal=[torch.zeros_like(p) if g is None else g for p,g in zip(params,proposal)]
            composed,prior_report=protect_physical_gradient(physical,proposal,cfg.prior_ratio)
            self._sync(); self.prior_seconds+=time.perf_counter()-begin
        if regularize:
            reg=self.model.regularization()*cfg.interface_regularization
            rg=torch.autograd.grad(reg,params,allow_unused=True)
            composed=[g+(torch.zeros_like(p) if h is None else h) for p,g,h in zip(params,composed,rg)]
        for p,g in zip(params,composed):
            if not torch.isfinite(g).all(): raise FloatingPointError('Nonfinite parameter gradient')
            p.grad=g.detach().clone()
        del velocity,grads,composed
        guard=active_prior or (cfg.guard_every>0 and update>=cfg.prior_start and (update-cfg.prior_start)%cfg.guard_every==0)
        # Routed + guarded is supported on the SAME selected-code objective,
        # not a guarantee for the unsampled physical shots. Full audits remain separate.
        if guard:
            result=guarded_optimizer_step(self.model,self.optimizer,
                lambda:self.objective.evaluate(self.model(),plan=plan,cutoff=cutoff,gradient=False).value,
                measured.value,max_trials=cfg.max_trials,
                valid_candidate=lambda:self.objective.valid(self.model()))
        else:
            state=copy.deepcopy(self.model.state_dict()); opt=copy.deepcopy(self.optimizer.state_dict())
            try:
                self.optimizer.step()
                with torch.no_grad():
                    if not self.objective.valid(self.model()): raise FloatingPointError('Invalid updated physical model')
            except BaseException:
                self.model.load_state_dict(state); self.optimizer.load_state_dict(opt); raise
            result={'accepted':True,'trials':1,'forward_calls':0,'step_scale':1.}
        if self.router is not None:
            self.router.update_scores(measured.indices,measured.energies)
        if cfg.method=='gradient_attention' and update%cfg.attention_block==0:
            self.model.refresh(measured.gradient)
        self.completed=update
        return {'update':update,'physical_objective':measured.value,'cutoff_hz':cutoff,
                'objective_kind':'importance_weighted_codes' if plan is not None else 'all_shots',
                'code_indices':plan.indices.tolist() if plan is not None else None,
                'full_code_audit':bool(plan is not None and plan.draw_count==0),
                'prior':prior_report,'prior_seconds':self.prior_seconds,
                'prior_calls':getattr(self.prior,'calls',0),'step':result,
                'counters':copy.deepcopy(self.objective.counters)}

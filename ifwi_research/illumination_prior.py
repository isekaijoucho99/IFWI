"""Cached sensitivity-weighted soft targets, not a posterior or hard completion.

A validated FrozenConditionalDFM is the intended source. A custom source is an
explicit caller-owned implementation of the same generate interface; its schema
is always reported, so analytic test fixtures are not labelled geological DFM.
This module never creates, trains, qualifies or substitutes missing weights.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import torch
from .safety import InvalidProposalError


@dataclass(frozen=True)
class SoftPriorConfig:
    weak_weight: float=1.
    strong_weight: float=0.
    weak_threshold: float=.25
    strong_threshold: float=.75
    velocity_weight: float=1.
    edge_weight: float=.2
    edge_length_m: float=20.
    refresh_every: int=5
    max_target_rms_change_mps: float=1000.
    max_cells: int=4096

    def validate(self):
        for name in ('weak_weight','strong_weight','velocity_weight','edge_weight'):
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value<0:
                raise ValueError(f'{name} must be finite nonnegative')
        if not 0<=self.strong_weight<=self.weak_weight<=1:
            raise ValueError('Require 0 <= strong_weight <= weak_weight <= 1')
        if not 0<=self.weak_threshold<self.strong_threshold<=1:
            raise ValueError('Ordered sensitivity thresholds in [0,1] required')
        for name in ('edge_length_m','max_target_rms_change_mps'):
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite positive')
        if type(self.refresh_every) is not int or self.refresh_every<1:
            raise ValueError('refresh_every must be a positive integer')
        if type(self.max_cells) is not int or not 1<=self.max_cells<=4096:
            raise ValueError('CPU prototype max_cells must be 1..4096')
        return self


def _tensor_digest(value):
    a=value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def _source_identity(source):
    # Custom implementations own their full identity contract. The DFM adapter
    # includes seed, NFE and normalization via its dynamic inference_identity.
    return (getattr(source,'checkpoint_hash',None),getattr(source,'inference_identity',None))


class IlluminationSoftPrior:
    """Frozen spatial weights + cached generated target through existing prior API.

    background must come from an independently chosen or data-only estimate.
    High sensitivity anchors remain estimates, never trusted truth. Zero prior
    weight means zero DIRECT penalty gradient there; shared INR parameters can
    still alter those pixels through other gradients. No projection is imposed.
    A fixed frequency stage is mandatory; reconstruct the wrapper with a new
    sensitivity map to change stage instead of silently reusing stale context.
    """
    def __init__(self,source,background,illumination,*,spacing_m,
                 config=SoftPriorConfig(),cutoff=None,operator_identity=None):
        self.config=config.validate()
        if isinstance(spacing_m,bool) or not math.isfinite(spacing_m) or spacing_m<=0:
            raise ValueError('Finite positive spacing required')
        if cutoff is not None and (isinstance(cutoff,bool) or not math.isfinite(cutoff) or cutoff<=0):
            raise ValueError('Invalid frequency context')
        if (not isinstance(background,torch.Tensor) or background.ndim!=3 or background.shape[0]!=1
                or min(background.shape[-2:])<2 or background.numel()>config.max_cells
                or not background.is_floating_point() or not torch.isfinite(background).all()
                or (background<=0).any()):
            raise ValueError('Finite positive [1,H,W] background within grid cap required')
        if (not isinstance(illumination,torch.Tensor) or illumination.shape!=background.shape
                or not torch.isfinite(illumination).all() or (illumination<0).any()
                or (illumination>1).any()):
            raise ValueError('Matching finite sensitivity in [0,1] required')
        configured=config.weak_weight>0 and (config.velocity_weight>0 or config.edge_weight>0)
        if configured and not (illumination>0).any():
            raise ValueError('All-zero sensitivity is inconclusive; no automatic prior activation')
        if operator_identity is not None and (not isinstance(operator_identity,str) or not operator_identity):
            raise ValueError('Invalid sensitivity operator identity')
        self.operator_identity=operator_identity
        self.background=background.detach().clone()
        self.illumination=illumination.detach().to(background).clone()
        self.spacing_m=float(spacing_m); self.cutoff=cutoff; self.source=source
        ramp=((self.illumination-config.weak_threshold)/(config.strong_threshold-config.weak_threshold)).clamp(0,1)
        self.weights=(config.weak_weight+(config.strong_weight-config.weak_weight)*ramp).detach()
        self.anchor_mask=(self.illumination>=config.strong_threshold).to(background).detach()
        self.enabled=bool(configured and (self.weights>0).any())
        self.scale_mps=1.
        if self.enabled:
            if source is None or not callable(getattr(source,'generate',None)):
                raise ValueError('Nonzero soft prior needs an explicitly supplied compatible frozen source')
            if not math.isclose(float(source.spacing_m),self.spacing_m,rel_tol=1e-7):
                raise ValueError('Source physical spacing mismatch')
            self.scale_mps=float(source.scale_mps)
            if not math.isfinite(self.scale_mps) or self.scale_mps<=0:
                raise ValueError('Finite positive source normalization required')
            if not isinstance(getattr(source,'checkpoint_hash',None),str) or not source.checkpoint_hash:
                raise ValueError('Stable source identity required for caching')
            if not isinstance(getattr(source,'meta',None),dict) or not isinstance(source.meta.get('schema'),str):
                raise ValueError('Source schema declaration required')
            if any(p.requires_grad for p in source.model.parameters()):
                raise ValueError('Source parameters must be frozen')
            if source.model.training: raise ValueError('Frozen source must be in eval mode')
        self._source_hash=_source_identity(source)
        identity={'schema':'illumination_soft_prior_v1','source':self._source_hash,
                  'background':_tensor_digest(self.background),'illumination':_tensor_digest(self.illumination),
                  'spacing_m':self.spacing_m,'cutoff':cutoff,'operator_identity':operator_identity,'config':asdict(config)}
        self.checkpoint_hash=hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
        self._target=None; self._refreshed_at=None; self._refreshes=0; self._cache_hits=0

    @property
    def calls(self):
        return getattr(self.source,'calls',0)

    @calls.setter
    def calls(self,value):
        if self.source is not None: self.source.calls=value

    def prepare_context(self,cutoff,*,operator_identity=None):
        if cutoff!=self.cutoff:
            raise ValueError('Prior frequency context changed; recompute sensitivity and rebuild the prior')
        if self.operator_identity is None or operator_identity!=self.operator_identity:
            raise ValueError('Sensitivity operator identity missing or changed; recompute its map')
        if _source_identity(self.source)!=self._source_hash:
            raise ValueError('Prior source identity changed')

    def _grid(self,value,label):
        if (not isinstance(value,torch.Tensor) or value.shape!=self.background.shape
                or value.device!=self.background.device or value.dtype!=self.background.dtype
                or not torch.isfinite(value).all()):
            raise ValueError(f'{label} must share the finite background grid, dtype and device')

    def _validate_candidate(self,candidate,velocity,label):
        # Grid/type mismatches are caller/source contract errors, never recovery.
        if (not isinstance(candidate,torch.Tensor) or candidate.shape!=self.background.shape
                or candidate.device!=self.background.device or candidate.dtype!=self.background.dtype):
            raise ValueError(f'{label} must share the background grid, dtype and device')
        if not torch.isfinite(candidate).all():
            raise InvalidProposalError(f'{label} contains nonfinite values',reason='nonfinite')
        if not ((candidate>=100)&(candidate<=15000)).all():
            raise InvalidProposalError(f'{label} outside physical velocity range [100,15000] m/s',
                                       reason='physical_range')
        change=(candidate.double()-velocity.double()).square().mean().sqrt()
        if change>self.config.max_target_rms_change_mps:
            raise InvalidProposalError(f'{label} exceeds declared RMS change budget',reason='rms_budget')

    @torch.no_grad()
    def proposal(self,velocity,update):
        self._grid(velocity,'Velocity')
        if type(update) is not int or update<1: raise ValueError('One-based update required')
        if not self.enabled: return velocity.detach().clone()
        if _source_identity(self.source)!=self._source_hash:
            raise ValueError('Prior source identity changed')
        if self._refreshed_at is not None and update<self._refreshed_at:
            raise ValueError('Cannot rewind cached proposal without loading its saved state')
        if self._target is not None and update-self._refreshed_at<self.config.refresh_every:
            self._validate_candidate(self._target,velocity,'Cached candidate')
            self._cache_hits+=1
            return self._target.clone()
        candidate=self.source.generate(self.background,velocity.detach().clone(),self.anchor_mask,
                                       sample_id=update)
        # Device/dtype conversion does not change physical units or resize grids.
        if isinstance(candidate,torch.Tensor): candidate=candidate.detach().to(velocity)
        self._validate_candidate(candidate,velocity,'Generated candidate')
        self._target=candidate.clone(); self._refreshed_at=update; self._refreshes+=1
        return self._target.clone()

    def penalty(self,velocity,target):
        self._grid(velocity,'Velocity'); self._grid(target,'Target')
        if not self.enabled: return velocity.sum()*0.
        residual=(velocity-target.detach())/self.scale_mps
        loss=.5*self.config.velocity_weight*(self.weights*residual.square()).mean()
        for axis in (-2,-1):
            # Min-endpoint weights prevent the edge operator directly pushing a
            # strong pixel whose own weight is zero across a weak/strong border.
            a=[slice(None)]*3; b=a.copy(); a[axis]=slice(1,None); b[axis]=slice(None,-1)
            a,b=tuple(a),tuple(b)
            edge_weight=torch.minimum(self.weights[a],self.weights[b])
            difference=(residual[a]-residual[b])*(self.config.edge_length_m/self.spacing_m)
            loss=loss+.5*self.config.edge_weight*(edge_weight*difference.square()).mean()
        return loss

    def diagnostics(self):
        return {'enabled':self.enabled,'source_schema':getattr(self.source,'meta',{}).get('schema'),
                'refreshes':self._refreshes,'cache_hits':self._cache_hits,'network_evaluations':self.calls,
                'target_refreshed_at':self._refreshed_at,'cutoff_hz':self.cutoff,'operator_identity':self.operator_identity,
                'mean_weight':float(self.weights.mean()),'estimated_anchor_fraction':float(self.anchor_mask.mean()),
                'uncertainty_calibrated':False,'hard_constraints':False}

    def state_dict(self):
        return {'schema':'illumination_soft_prior_v1','identity':self.checkpoint_hash,
                'target':None if self._target is None else self._target.clone(),
                'refreshed_at':self._refreshed_at,'refreshes':self._refreshes,
                'cache_hits':self._cache_hits,'source_calls':self.calls}

    def load_state_dict(self,state):
        if state.get('schema')!='illumination_soft_prior_v1' or state.get('identity')!=self.checkpoint_hash:
            raise ValueError('Soft prior identity changed; not an exact resume')
        target=state.get('target'); update=state.get('refreshed_at')
        if (target is None)!=(update is None): raise ValueError('Inconsistent cached target state')
        if target is not None:
            self._grid(target,'Cached target')
            if (target<=0).any(): raise ValueError('Cached target must contain positive velocities')
            if type(update) is not int or update<1: raise ValueError('Invalid cached update')
        for name in ('refreshes','cache_hits','source_calls'):
            if type(state.get(name)) is not int or state[name]<0: raise ValueError('Invalid cached counters')
        self._target=None if target is None else target.detach().clone()
        self._refreshed_at=update; self._refreshes=state['refreshes']; self._cache_hits=state['cache_hits']
        self.calls=state['source_calls']

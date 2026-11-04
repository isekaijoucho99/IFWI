"""Official DFM numerical core with an IFWI spatial proposal adapter.

Based on the two-time map and group-wise drift objective in arXiv:2605.17244.
This is a purpose-built small CNN prototype, NOT the paper's image model.
The physical prior below is a plug-in proposal, NOT a posterior sampler or
an exact gradient of a globally fixed probabilistic regularizer.
"""
from __future__ import annotations
import hashlib
import math
import warnings
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ._vendor.dfm_sinkhorn import (
    UPSTREAM_COMMIT, compute_drift_grouped_sinkhorn, _adaptive_matching_loss,
)


def smooth(x: torch.Tensor, kernel: int=9) -> torch.Tensor:
    if type(kernel) is not int or kernel<1 or kernel%2!=1 or x.ndim!=4:
        raise ValueError('Odd positive kernel and BCHW tensor required')
    p=kernel//2
    return F.avg_pool2d(F.pad(x,(p,p,p,p),mode='replicate'),kernel,stride=1)


class TwoTimeTransport(nn.Module):
    """T(x,t,r,c) = x + (r-t) u(x,t,r,c); scalar t/r shared by a group."""
    def __init__(self, width: int=32):
        super().__init__()
        if type(width) is not int or width<4: raise ValueError('width must be integer >=4')
        self.width=width
        self.net=nn.Sequential(nn.Conv2d(4,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,1,3,padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: float, r: float, condition: torch.Tensor) -> torch.Tensor:
        if not math.isfinite(t) or not math.isfinite(r) or not 0<=t<=r<=1:
            raise ValueError('Require 0 <= t <= r <= 1, shared within a group')
        if x.ndim!=4 or x.shape[1]!=1 or condition.shape!=(1,1,*x.shape[-2:]):
            raise ValueError('B1HW samples and one SHARED 11HW condition required')
        if r==t: return x
        c=condition.expand_as(x)
        features=torch.cat((x,c,torch.full_like(x,t),torch.full_like(x,r-t)),dim=1)
        return x+(r-t)*self.net(features)


@torch.no_grad()
def drift_field(negative: torch.Tensor, positive: torch.Tensor, temperature: float=1.,
                *, exclude_self: bool=False, sinkhorn_iters: int=20) -> torch.Tensor:
    """Pinned official balanced-Sinkhorn field for ONE spatial condition group.

    Upstream uses Euclidean feature distances without RMS rescaling. Self-plan
    diagonals are included. Grouping and spatial normalization belong outside
    this operator; there is no independent row-softmax approximation here.
    """
    if negative.ndim<2 or positive.shape!=negative.shape or len(negative)<2:
        raise ValueError('Matching positive and negative samples, at least two per group, required')
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('temperature must be finite and positive')
    if type(sinkhorn_iters) is not int or sinkhorn_iters<1:
        raise ValueError('sinkhorn_iters must be a positive integer')
    if exclude_self:
        raise ValueError('Official Sinkhorn includes self-plan diagonals; exclude_self is unsupported')
    if not torch.isfinite(negative).all() or not torch.isfinite(positive).all():
        raise ValueError('Nonfinite drift samples')
    q=negative.detach().flatten(1)[None]
    p=positive.detach().to(q).flatten(1)[None]
    return compute_drift_grouped_sinkhorn(q,p,kernel_temp_pos=temperature,
        kernel_temp_neg=temperature,num_sinkhorn_iters=sinkhorn_iters)[0].reshape_as(negative)


def grouped_drift_loss(predicted: torch.Tensor, target_marginal: torch.Tensor,
                       temperature: float=1., *, exclude_self: bool=False,
                       sinkhorn_iters: int=20, norm_eps: float=1e-4, norm_p: float=0.) -> torch.Tensor:
    if not math.isfinite(norm_eps) or norm_eps<=0 or not math.isfinite(norm_p) or not 0<=norm_p<=1:
        raise ValueError('Invalid adaptive loss settings')
    field=drift_field(predicted,target_marginal,temperature,exclude_self=exclude_self,
                      sinkhorn_iters=sinkhorn_iters)
    target=predicted.detach()+field
    return _adaptive_matching_loss(predicted,target,norm_eps,norm_p)


def conditional_group_loss(model: TwoTimeTransport, clean: torch.Tensor, *,
                           t: float, r: float, group_size: int, generator: torch.Generator,
                           kernel: int=9, temperature: float=1., sinkhorn_iters: int=20) -> torch.Tensor:
    """One condition, one time pair, many noise draws; no cross-condition drift.

    clean is one independently sampled normalized geological patch. The
    empirical high-dimensional conditional target can have a single endpoint,
    while multiple source draws construct the group (DFM Appendix D).
    """
    if clean.ndim!=4 or clean.shape[:2]!=(1,1) or group_size<2 or not t<r:
        raise ValueError('One clean patch, group_size >=2 and t<r required')
    shape=(group_size,1,*clean.shape[-2:])
    # CPU RNG makes this sampling independent of a caller's global/CUDA RNG.
    noise=torch.randn(shape,generator=generator,dtype=clean.dtype).to(clean.device)
    xt=(1-t)*noise+t*clean.detach()
    xr=(1-r)*noise+r*clean.detach()
    condition=smooth(clean.detach(),kernel)
    return grouped_drift_loss(model(xt,t,r,condition),xr,temperature,sinkhorn_iters=sinkhorn_iters)


def sample_time_pair(generator: torch.Generator) -> tuple[float,float]:
    """Official ordered logit-normal law with a local CPU RNG for reproducibility."""
    t=float(torch.sigmoid(torch.randn((),generator=generator)*2.5-1.))
    r=float(torch.sigmoid(torch.randn((),generator=generator)*2.5+1.))
    return min(t,r),max(t,r)


@torch.no_grad()
def sample_transport(model: nn.Module, state: torch.Tensor, condition: torch.Tensor, *,
                     start_t: float=0., nfe: int=4) -> torch.Tensor:
    """Compose official two-time intervals while keeping external condition fixed."""
    if not math.isfinite(start_t) or not 0<=start_t<1 or type(nfe) is not int or nfe<1:
        raise ValueError('Invalid transport start_t/nfe')
    times=torch.linspace(start_t,1.,nfe+1,device=state.device,dtype=state.dtype).tolist()
    for t,r in zip(times[:-1],times[1:]):
        state=model(state,t,r,condition)
        if not torch.isfinite(state).all(): raise FloatingPointError('Nonfinite DFM proposal')
    return state


def patch_origins(length: int, patch: int, stride: int) -> list[int]:
    if min(length,patch,stride)<1 or stride>patch: raise ValueError('Invalid patch/stride')
    if length<=patch: return [0]
    return sorted(set(range(0,length-patch+1,stride))|{length-patch})


class FrozenDFMPrior:
    """Loads trained small-CNN weights; cannot silently use random weights.

    The default candidate starts from independent Gaussian noise and uses the
    current patch's coarse structure as condition throughout all intervals.
    Clean-model training conditions can still differ from FWI estimates;
    validate that condition shift independently of clean-conditional generation.
    """
    def __init__(self, checkpoint: str | Path, spacing_m: float, device='cpu', *,
                 start_t: float=0., nfe: int=4, seed: int=3):
        path=Path(checkpoint)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        meta=payload.get('meta',{})
        if meta.get('schema')=='ifwi_small_dfm_v1':
            raise ValueError('Legacy DFM weights used a different drift objective; retrain a schema-v2 prior')
        if meta.get('schema')!='ifwi_small_dfm_v2' or type(meta.get('trained_steps')) is not int or meta['trained_steps']<1:
            raise ValueError('A trained ifwi_small_dfm_v2 checkpoint is required')
        if meta.get('upstream_commit')!=UPSTREAM_COMMIT or meta.get('drift_operator')!='grouped_sinkhorn_euclidean':
            raise ValueError('DFM checkpoint upstream/objective identity mismatch; retrain')
        if meta.get('toy_only',False): raise ValueError('Toy checkpoints cannot be used for geological inversion')
        if not math.isclose(float(meta['spacing_m']),spacing_m,rel_tol=1e-8):
            raise ValueError('Prior physical grid spacing mismatch; image resizing is not a scale bridge')
        if not math.isfinite(start_t) or not 0<=start_t<1 or type(nfe) is not int or nfe<1:
            raise ValueError('Invalid start_t/nfe')
        if start_t>0:
            warnings.warn('Warm-start DFM is an IFWI ablation; official generation starts from noise at t=0',RuntimeWarning)
        self.meta,self.start_t,self.nfe,self.seed=meta,start_t,nfe,seed
        self.model=TwoTimeTransport(int(meta['width'])).to(device).eval()
        self.model.load_state_dict(payload['model'])
        for p in self.model.parameters(): p.requires_grad_(False)
        self.mean=float(meta['mean_mps']); self.scale=float(meta['scale_mps'])
        self.kernel=int(meta['condition_kernel']); self.patch=int(meta['patch_size'])
        if (not math.isfinite(self.mean) or not math.isfinite(self.scale) or self.scale<=0
                or self.patch<2 or self.kernel<1 or self.kernel%2!=1):
            raise ValueError('Invalid normalization/patch/kernel metadata')
        self.checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        self.calls=0

    @torch.no_grad()
    def proposal(self, velocity: torch.Tensor, update: int) -> torch.Tensor:
        if velocity.ndim!=3 or velocity.shape[0]!=1 or update<1:
            raise ValueError('Require velocity [1,H,W] and one-based update')
        x=(velocity[:,None].detach()-self.mean)/self.scale
        h,w=x.shape[-2:]; side=self.patch
        padded=F.pad(x,(0,max(0,side-w),0,max(0,side-h)),mode='replicate')
        total=torch.zeros_like(padded); count=torch.zeros_like(padded)
        stride=max(1,side//2); counter=0
        for z in patch_origins(padded.shape[-2],side,stride):
            for col in patch_origins(padded.shape[-1],side,stride):
                patch=padded[...,z:z+side,col:col+side]
                condition=smooth(patch,self.kernel)
                rng=torch.Generator().manual_seed((self.seed*1000003+update*9176+counter)%2**63)
                noise=torch.randn(patch.shape,generator=rng,dtype=patch.dtype).to(patch.device)
                state=(1-self.start_t)*noise+self.start_t*patch
                state=sample_transport(self.model,state,condition,start_t=self.start_t,nfe=self.nfe)
                self.calls+=self.nfe
                total[...,z:z+side,col:col+side]+=state
                count[...,z:z+side,col:col+side]+=1
                counter+=1
        return ((total/count)[...,:h,:w]*self.scale+self.mean)[:,0]

    def penalty(self, velocity: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if velocity.shape!=target.shape or velocity.ndim!=3 or velocity.shape[0]!=1:
            raise ValueError('Prior and velocity must share the same [1,H,W] grid')
        if not torch.isfinite(target).all(): raise ValueError('Nonfinite prior target')
        # Autograd supplies the TRUE transpose of the replicate-padded filter.
        diff=(velocity-target.detach())[:,None]/self.scale
        high=diff-smooth(diff,self.kernel)
        return .5*high.square().mean()

"""Two-time, grouped drift-supervised transport and a frozen IFWI prior.

Algorithmic reference: Ma et al., arXiv:2605.17244, equations 13 and 17.
The small convolutional architecture and seismic adaptation are our prototype,
not a reproduction of the authors' image models or a pretrained seismic model.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def drift_field(predicted,positive,temperature=.2):
    if predicted.ndim<2 or positive.shape[1:]!=predicted.shape[1:] or min(len(predicted),len(positive))<2:
        raise ValueError('A drift group requires >=2 compatible positive and predicted particles')
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Kernel temperature must be positive')
    if not torch.isfinite(predicted).all() or not torch.isfinite(positive).all():
        raise ValueError('Non-finite drift particle')
    q=predicted.detach().flatten(1); p=positive.detach().flatten(1)
    d=float(q.shape[1]); tau=2*temperature**2*d
    # Self mass is included: identical empirical p and q then give zero drift.
    attraction=torch.softmax(-torch.cdist(q,p).square()/tau,dim=1)@p
    repulsion=torch.softmax(-torch.cdist(q,q).square()/tau,dim=1)@q
    return (attraction-repulsion).reshape_as(predicted)


def drift_loss(predicted,positive,temperature=.2):
    target=predicted.detach()+drift_field(predicted,positive,temperature)
    return .5*(predicted-target).square().mean()


class TwoTimeTransport(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        if type(width) is not int or width<4: raise ValueError('width must be an integer >=4')
        self.width=width
        self.velocity=nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,1,3,padding=1))
        nn.init.zeros_(self.velocity[-1].weight); nn.init.zeros_(self.velocity[-1].bias)

    def forward(self,x,t,r):
        if x.ndim!=4 or x.shape[1]!=1: raise ValueError('Transport expects [B,1,H,W]')
        def times(v):
            v=torch.as_tensor(v,device=x.device,dtype=x.dtype)
            if v.numel()==1: v=v.expand(x.shape[0])
            if v.numel()!=x.shape[0]: raise ValueError('One time pair per sample is required')
            return v.reshape(-1,1,1,1)
        t,r=times(t),times(r)
        if not torch.isfinite(t).all() or not torch.isfinite(r).all() or (t<0).any() or (r<t).any() or (r>1).any():
            raise ValueError('Require 0 <= t <= r <= 1')
        features=torch.cat((x,t.expand(-1,1,*x.shape[-2:]),r.expand(-1,1,*x.shape[-2:])),1)
        return x+(r-t)*self.velocity(features)


def grouped_dfm_loss(model,noise,data,times_t,times_r,group_size=8,temperature=.2):
    if type(group_size) is not int or group_size<2 or noise.shape!=data.shape or len(data)%group_size:
        raise ValueError('Equal batches divisible by group_size >=2 are required')
    groups=len(data)//group_size
    if times_t.numel()!=groups or times_r.numel()!=groups:
        raise ValueError('Distinct time pairs need separate particle groups')
    losses=[]
    for i in range(groups):
        s=slice(i*group_size,(i+1)*group_size); t=times_t[i]; r=times_r[i]
        xt=(1-t)*noise[s]+t*data[s]; xr=(1-r)*noise[s]+r*data[s]
        losses.append(drift_loss(model(xt,t,r),xr,temperature))
    return torch.stack(losses).mean()


def _starts(length,patch):
    stride=max(1,patch//2)
    return sorted(set(range(0,max(1,length-patch+1),stride))|{max(0,length-patch)})


@torch.no_grad()
def patch_transport(model,x,*,t=.8,steps=1,patch_size=32,batch_size=4):
    """Overlap-averaged transport at native spacing; no resizing of velocity images."""
    if x.ndim!=4 or x.shape[:2]!=(1,1) or not 0<=t<1:
        raise ValueError('Expected one scalar image and t in [0,1)')
    if any(type(n) is not int or n<1 for n in (steps,patch_size,batch_size)):
        raise ValueError('Patch/batch/step counts must be positive integers')
    h,w=x.shape[-2:]; ph,pw=max(h,patch_size),max(w,patch_size)
    current=F.pad(x,(0,pw-w,0,ph-h),mode='replicate'); calls=0
    positions=[(z,k) for z in _starts(ph,patch_size) for k in _starts(pw,patch_size)]
    for step in range(steps):
        start=t+(1-t)*step/steps; end=t+(1-t)*(step+1)/steps
        summed=torch.zeros_like(current); cover=torch.zeros_like(current)
        for first in range(0,len(positions),batch_size):
            places=positions[first:first+batch_size]
            batch=torch.cat([current[...,z:z+patch_size,k:k+patch_size] for z,k in places])
            predicted=model(batch,start,end); calls+=1
            if not torch.isfinite(predicted).all(): raise FloatingPointError('Non-finite DFM transport')
            for piece,(z,k) in zip(predicted,places):
                summed[...,z:z+patch_size,k:k+patch_size]+=piece[None]
                cover[...,z:z+patch_size,k:k+patch_size]+=1
        if (cover==0).any(): raise RuntimeError('Patch coverage has gaps')
        current=summed/cover
    return current[...,:h,:w],calls


class FrozenDFMPrior:
    """Two modes of ONE frozen DFM produce an auxiliary velocity-space gradient."""
    def __init__(self,path,device,spacing_m,*,allow_smoke=False,ratio=.1,lift_time=.8,
                 steps=4,direct_every=20,blend=.5,seed=3,batch_size=4):
        path=Path(path)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        meta=payload.get('metadata',{})
        if payload.get('schema')!='ifwi_dfm_v1' or meta.get('trained_steps',0)<1:
            raise ValueError('A trained ifwi_dfm_v1 checkpoint is required')
        if meta.get('smoke_only',True) and not allow_smoke:
            raise ValueError('Smoke/unqualified DFM checkpoint cannot be used for a formal inversion')
        if (not math.isfinite(float(meta.get('center_mps',math.nan)))
                or not math.isfinite(float(meta.get('scale_mps',math.nan))) or meta['scale_mps']<=0
                or type(meta.get('patch_size')) is not int or meta['patch_size']<4):
            raise ValueError('Invalid prior physical normalization or patch size')
        if not math.isclose(float(meta.get('spacing_m',-1)),spacing_m,rel_tol=1e-8):
            raise ValueError('Prior and PDE physical sampling spacings differ')
        if not 0<=blend<=1 or not 0<lift_time<1 or not 0<=ratio<=1:
            raise ValueError('Invalid DFM guidance controls')
        if any(type(n) is not int or n<1 for n in (steps,direct_every,batch_size)):
            raise ValueError('Positive DFM counts are required')
        self.model=TwoTimeTransport(meta['width']).to(device)
        self.model.load_state_dict(payload['model']); self.model.eval().requires_grad_(False)
        self.meta=meta; self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        self.ratio=ratio; self.t=lift_time; self.steps=steps; self.period=direct_every
        self.blend=blend; self.seed=int(seed); self.batch_size=batch_size
        self.cached=None; self.cached_update=None; self.calls=0

    @torch.no_grad()
    def gradient(self,velocity,update):
        scale=float(self.meta['scale_mps']); center=float(self.meta['center_mps'])
        x=(velocity[:,None]-center)/scale
        generator=torch.Generator(device=velocity.device).manual_seed(self.seed+104729*int(update))
        lifted=self.t*x+(1-self.t)*torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)
        args=dict(t=self.t,patch_size=int(self.meta['patch_size']),batch_size=self.batch_size)
        if self.blend>0 and (self.cached is None or update-self.cached_update>=self.period):
            target,calls=patch_transport(self.model,lifted,steps=1,**args)
            self.cached=target.detach(); self.cached_update=int(update); self.calls+=calls
        target=torch.zeros_like(x)
        if self.blend>0: target=target+self.blend*self.cached.to(x)
        if self.blend<1:
            local,calls=patch_transport(self.model,lifted,steps=self.steps,**args); self.calls+=calls
            target=target+(1-self.blend)*local
        # Gradient of .5*mean((N(v)-target)^2); target is stopped.
        return ((x-target)/(scale*x.numel()))[:,0]

    def state_dict(self):
        return dict(cached=None if self.cached is None else self.cached.cpu(),
                    cached_update=self.cached_update,calls=self.calls,sha256=self.sha256)

    def load_state_dict(self,state):
        if state['sha256']!=self.sha256: raise ValueError('Prior checkpoint identity changed')
        self.cached=state['cached']; self.cached_update=state['cached_update']; self.calls=int(state['calls'])

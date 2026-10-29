"""Exact Haar residual decomposition with bounded LL-conditioned detail gates."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def haar2d(x):
    if x.ndim!=4 or min(x.shape[-2:])<2 or not torch.isfinite(x).all():
        raise ValueError('Haar input must be finite [B,C,H,W], H,W >= 2')
    h,w=x.shape[-2:]
    x=F.pad(x,(0,w%2,0,h%2),mode='replicate')
    a,b,c,d=x[...,::2,::2],x[...,::2,1::2],x[...,1::2,::2],x[...,1::2,1::2]
    return ((a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2),(h,w)


def inverse_haar2d(coefficients, size):
    if len(coefficients)!=4 or any(x.shape!=coefficients[0].shape for x in coefficients):
        raise ValueError('Four equal-shaped Haar bands are required')
    ll,lh,hl,hh=coefficients
    a=(ll+lh+hl+hh)/2; b=(ll-lh+hl-hh)/2
    c=(ll+lh-hl-hh)/2; d=(ll-lh-hl+hh)/2
    top=torch.stack((a,b),dim=-1).flatten(-2)
    bottom=torch.stack((c,d),dim=-1).flatten(-2)
    out=torch.stack((top,bottom),dim=-2).flatten(-3,-2)
    h,w=size
    if not 0<h<=out.shape[-2] or not 0<w<=out.shape[-1]: raise ValueError('Invalid crop shape')
    return out[...,:h,:w]


class WaveletAttention(nn.Module):
    def __init__(self,hidden=16,strength=.5):
        super().__init__()
        if type(hidden) is not int or hidden<1 or not math.isfinite(strength) or not 0<=strength<1:
            raise ValueError('Invalid wavelet gate settings')
        self.strength=float(strength)
        self.net=nn.Sequential(nn.Conv2d(4,hidden,3,padding=1),nn.SiLU(),nn.Conv2d(hidden,3,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('high_fraction',torch.tensor(1.))
        self.last_gain=None

    def forward(self,residual):
        if residual.ndim!=4 or residual.shape[1]!=1: raise ValueError('Use a scalar velocity residual')
        (ll,lh,hl,hh),size=haar2d(residual)
        features=torch.cat((ll,lh.abs(),hl.abs(),hh.abs()),1)
        scale=features.detach().square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-8)
        gains=self.high_fraction*(1+self.strength*torch.tanh(self.net(features/scale)))
        self.last_gain=gains.detach()
        return inverse_haar2d((ll,lh*gains[:,0:1],hl*gains[:,1:2],hh*gains[:,2:3]),size)


class WaveletResidual(nn.Module):
    def __init__(self,base,**kwargs):
        super().__init__(); self.base=base; self.gate=WaveletAttention(**kwargs)
    @property
    def fixed_init_mps(self): return self.base.fixed_init_mps
    def forward(self):
        residual=self.base()-self.fixed_init_mps
        return self.fixed_init_mps+self.gate(residual[:,None])[:,0]

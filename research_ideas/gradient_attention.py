"""Bounded physics-gradient attention with cross-shot self-supervision.

No second-order wavefield graph is constructed. Input gradient statistics are
stopped, and gate supervision uses disjoint shot groups, never velocity truth.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class GradientAttention(nn.Module):
    def __init__(self,width=16,strength=.5):
        super().__init__()
        if type(width) is not int or width<1 or not math.isfinite(strength) or not 0<=strength<1:
            raise ValueError('Require a positive width and strength in [0,1)')
        self.strength=strength
        self.net=nn.Sequential(nn.Conv2d(4,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,1,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,velocity,gradient):
        if velocity.ndim!=3 or velocity.shape!=gradient.shape:
            raise ValueError('Velocity and gradient must have matching [B,H,W] shapes')
        v=velocity.detach()[:,None]; g=gradient.detach()[:,None]
        if not torch.isfinite(v).all() or not torch.isfinite(g).all(): raise ValueError('Non-finite gate feature')
        vrms=(v-v.mean()).square().mean().sqrt().clamp_min(1.)
        grms=g.square().mean().sqrt().clamp_min(1e-20)
        dz=F.pad(v[...,1:,:]-v[...,:-1,:],(0,0,0,1))
        dx=F.pad(v[...,:,1:]-v[...,:,:-1],(0,1,0,0))
        edge=(dx.abs()+dz.abs())/vrms
        features=torch.cat(((v-v.mean())/vrms,g/grms,g.abs()/grms,edge),1)
        return (1+self.strength*torch.tanh(self.net(features)))[:,0]


def fit_cross_group(gate,optimizer,velocity,group_a,group_b,penalty=.05):
    """Train a source-only gate to agree with the OTHER group's gradient."""
    if group_a.shape!=group_b.shape or group_a.shape!=velocity.shape:
        raise ValueError('Cross-group shapes differ')
    a,b=group_a.detach(),group_b.detach()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise ValueError('Non-finite group gradient')
    if float(a.norm())==0 or float(b.norm())==0: return 0.
    optimizer.zero_grad(set_to_none=True)
    ma,mb=gate(velocity,a),gate(velocity,b)
    def agreement(source,target):
        return 1-F.cosine_similarity(source.flatten()[None],target.flatten()[None],eps=1e-20).mean()
    loss=.5*(agreement(ma*a,b)+agreement(mb*b,a))+penalty*((ma-1).square().mean()+(mb-1).square().mean())
    loss.backward(); optimizer.step()
    return float(loss.detach())

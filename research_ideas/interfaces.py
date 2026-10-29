"""Ordered learnable stratigraphic horizons; all geometry is in physical metres.

This is a graph-interface model, not a general salt-body/topology representation.
"""
from __future__ import annotations
import math
import torch
from torch import nn


def level_set_blend(background,body,phi_m,width_m):
    if not math.isfinite(width_m) or width_m<=0: raise ValueError('Interface width must be positive metres')
    mask=torch.sigmoid(phi_m/width_m)
    return (1-mask)*background+mask*body


class OrderedInterfaceResidual(nn.Module):
    def __init__(self,base,spacing_m,interfaces=3,width_m=60.,amplitude_mps=300.,
                 min_thickness_m=None,hidden=16):
        super().__init__()
        self.base=base
        if type(interfaces) is not int or interfaces<1 or not math.isfinite(spacing_m) or spacing_m<=0:
            raise ValueError('Positive interface count and spacing are required')
        h,w=base.fixed_init_mps.shape[-2:]; depth=(h-1)*spacing_m
        minimum=spacing_m if min_thickness_m is None else float(min_thickness_m)
        if h<3 or w<3 or minimum<=0 or (interfaces+1)*minimum>=depth:
            raise ValueError('Minimum layer thickness does not fit the physical depth')
        if not math.isfinite(width_m) or width_m<spacing_m or not math.isfinite(amplitude_mps):
            raise ValueError('Width must be at least one cell; amplitude must be finite')
        self.spacing_m=float(spacing_m); self.count=interfaces; self.minimum=minimum; self.depth=depth
        self.horizon=nn.Sequential(nn.Linear(1,hidden),nn.Tanh(),nn.Linear(hidden,interfaces+1))
        nn.init.zeros_(self.horizon[-1].weight); nn.init.zeros_(self.horizon[-1].bias)
        self.register_buffer('x',torch.linspace(-1,1,w)[:,None])
        self.register_buffer('z',torch.arange(h,dtype=torch.float32)*spacing_m)
        self.register_buffer('width_m',torch.tensor(float(width_m)))
        self.register_buffer('activation',torch.tensor(1.))
        self.register_buffer('anchors',torch.arange(1,interfaces+1,dtype=torch.float32)[:,None].expand(-1,w)*depth/(interfaces+1))
        self.amplitudes=nn.Parameter(torch.full((interfaces,),float(amplitude_mps)))

    @property
    def fixed_init_mps(self): return self.base.fixed_init_mps

    def heights(self):
        fractions=torch.softmax(self.horizon(self.x),dim=-1).T
        thickness=self.minimum+(self.depth-(self.count+1)*self.minimum)*fractions
        return thickness.cumsum(0)[:-1]

    def regularizer(self):
        h=self.heights(); curvature=(h[:,2:]-2*h[:,1:-1]+h[:,:-2])/self.spacing_m
        return self.activation*curvature.square().mean()

    def forward(self):
        h=self.heights(); z=self.z[None,:,None]; width=self.width_m.clamp_min(self.spacing_m)
        current=torch.sigmoid((z-h[:,None,:])/width)
        anchor=torch.sigmoid((z-self.anchors[:,None,:])/width)
        correction=(self.amplitudes[:,None,None]*(current-anchor)).sum(0)
        return self.base()+self.activation*correction[None]

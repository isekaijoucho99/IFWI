"""Physics-gradient-guided gates on blockwise velocity increments.

Refreshing cached gradients must not alter v at fixed trainable parameters.
We reanchor the current field and the base-network output at every refresh.
Keep each context fixed for >=2 updates so the gate can actually learn.
"""
from __future__ import annotations
import math
import torch
from torch import nn


class GradientAttentionVelocity(nn.Module):
    def __init__(self, base: nn.Module, background: torch.Tensor,
                 strength: float=.2, width: int=16, decay: float=.9):
        super().__init__()
        if background.ndim!=3 or background.shape[0]!=1:
            raise ValueError('background must be [1,H,W]')
        if not math.isfinite(strength) or not 0<=strength<1 or not 0<=decay<1:
            raise ValueError('Invalid bounded gate strength or EMA decay')
        self.base,self.strength,self.decay=base,float(strength),float(decay)
        with torch.no_grad(): initial=base().detach().clone()
        self.register_buffer('raw_anchor',initial.clone())
        self.register_buffer('value_anchor',initial.clone())
        self.register_buffer('mean_gradient',torch.zeros_like(initial))
        self.register_buffer('energy_gradient',torch.zeros_like(initial))
        self.register_buffer('gradient_context',torch.zeros(1,4,*initial.shape[-2:],device=initial.device,dtype=initial.dtype))
        self.register_buffer('refresh_count',torch.tensor(0,device=initial.device,dtype=torch.long))
        self.gates=nn.Sequential(nn.Conv2d(4,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,1,1)).to(initial)
        nn.init.zeros_(self.gates[-1].weight); nn.init.zeros_(self.gates[-1].bias)

    def forward(self) -> torch.Tensor:
        base=self.base()
        if self.strength==0: return base  # exact disabled-path parity, no cancellation roundoff
        gate=1+self.strength*torch.tanh(self.gates(self.gradient_context))[:,0]
        return self.value_anchor+gate*(base-self.raw_anchor)

    @torch.no_grad()
    def refresh(self, velocity_gradient: torch.Tensor) -> None:
        """Use lagged *unmodified* physical gradient; all context is stop-gradient."""
        g=velocity_gradient.detach().to(self.raw_anchor)
        if g.shape!=self.raw_anchor.shape or not torch.isfinite(g).all():
            raise ValueError('Finite gradient on the original velocity grid required')
        current=self().detach().clone(); raw=self.base().detach().clone()
        # Normalize before EMA to avoid dependence on waveform scaling/units.
        rms=g.double().square().mean().sqrt().to(g.dtype)
        norm=g/rms if rms>0 else torch.zeros_like(g)
        self.mean_gradient.lerp_(norm,1-self.decay)
        self.energy_gradient.lerp_(norm.square(),1-self.decay)
        depth=torch.linspace(0,1,g.shape[-2],device=g.device,dtype=g.dtype)[None,:,None].expand_as(g)
        self.gradient_context.copy_(torch.stack((norm,self.mean_gradient,
                                                self.energy_gradient.clamp_min(0).sqrt(),depth),dim=1))
        self.value_anchor.copy_(current); self.raw_anchor.copy_(raw)
        self.refresh_count.add_(1)

"""Invertible Haar details with bounded learned cross-band gates.

This is a small wavelet-attention hypothesis, NOT a Wave-ViT reproduction.
Velocities have shape [1,H,W] and units m/s. All four bands are retained.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def haar_forward(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """BCHW -> B,C,4,ceil(H/2),ceil(W/2), using replicate padding on odd axes."""
    if x.ndim != 4 or not x.is_floating_point() or min(x.shape) < 1:
        raise ValueError('Expected nonempty floating BCHW tensor')
    h, w = x.shape[-2:]
    x = F.pad(x, (0,w % 2,0,h % 2), mode='replicate')
    a,b,c,d = x[...,::2,::2],x[...,::2,1::2],x[...,1::2,::2],x[...,1::2,1::2]
    bands = torch.stack((a+b+c+d, -a-b+c+d, -a+b-c+d, a-b-c+d), dim=2)*0.5
    return bands, (h,w)


def haar_inverse(bands: torch.Tensor, size: tuple[int,int]) -> torch.Tensor:
    """Exact inverse followed by crop; gradients include padding's adjoint."""
    if bands.ndim != 5 or bands.shape[2] != 4 or len(size) != 2:
        raise ValueError('Expected B,C,4,H,W coefficients and (original_H,original_W)')
    bh,bw = bands.shape[-2:]
    if size[0] not in (2*bh,2*bh-1) or size[1] not in (2*bw,2*bw-1):
        raise ValueError('Original size is inconsistent with coefficient lattice')
    ll,lh,hl,hh = bands.unbind(2)
    a,b,c,d = (ll-lh-hl+hh)*.5,(ll-lh+hl-hh)*.5,(ll+lh-hl-hh)*.5,(ll+lh+hl+hh)*.5
    out = bands.new_empty(bands.shape[0],bands.shape[1],2*bh,2*bw)
    out[...,::2,::2],out[...,::2,1::2] = a,b
    out[...,1::2,::2],out[...,1::2,1::2] = c,d
    return out[...,:size[0],:size[1]]


class WaveletVelocity(nn.Module):
    """v0 + IDWT(LL, bounded_attention(LL,LH,HL,HH) * details).

    The gate output layer is zero-initialized, so the wrapper starts as the
    original velocity model (up to DWT floating-point rounding).
    """
    def __init__(self, base: nn.Module, background: torch.Tensor,
                 strength: float = .2, width: int = 16):
        super().__init__()
        if not math.isfinite(strength) or not 0 <= strength < 1 or width < 1:
            raise ValueError('strength must be in [0,1), width positive')
        if background.ndim != 3 or background.shape[0] != 1:
            raise ValueError('background must be [1,H,W]')
        self.base, self.strength = base, float(strength)
        self.register_buffer('background',background.detach().clone())
        self.gates = nn.Sequential(nn.Conv2d(4,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,3,1))
        nn.init.zeros_(self.gates[-1].weight); nn.init.zeros_(self.gates[-1].bias)
        self.gates.to(background)

    def forward(self) -> torch.Tensor:
        velocity = self.base()
        if self.strength == 0:
            return velocity
        bands,size = haar_forward((velocity-self.background)[:,None])
        channels = bands[:,0]
        scale = channels.detach().square().mean(dim=(-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        gate = 1+self.strength*torch.tanh(self.gates(channels/scale))
        attended = torch.cat((channels[:,:1],channels[:,1:]*gate),dim=1)
        return self.background+haar_inverse(attended[:,None],size)[:,0]

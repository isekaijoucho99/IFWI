"""An explicitly learnable interface correction with a matched initial field.

The reference level set is independent prior information, NEVER a test label.
Its displacement is in metres. The background/residual network is unchanged.
"""
from __future__ import annotations
import math
import torch
from torch import nn


class InterfaceVelocity(nn.Module):
    """v = v_base + contrast * [sigmoid(phi/eps)-sigmoid(phi_ref/eps)].

    This anchored correction is NOT a full salt/background blend. It tests
    whether explicit interface movement helps, without changing v at step 0.
    """
    def __init__(self, base: nn.Module, shape: tuple[int,int], spacing_m: float,
                 depth_m: float, epsilon_m: float, displacement_m: float = 150.,
                 max_contrast_mps: float = 1500., initial_contrast_mps: float = 300.,
                 width: int = 32, phi_reference: torch.Tensor | None = None):
        super().__init__()
        h,w=shape
        if min(h,w)<3 or width<1:
            raise ValueError('Interface grid must have at least 3 points per axis')
        for v in (spacing_m,epsilon_m,displacement_m,max_contrast_mps):
            if not math.isfinite(v) or v<=0: raise ValueError('Physical scales must be finite positive')
        if not math.isfinite(depth_m) or not 0 <= depth_m <= (h-1)*spacing_m:
            raise ValueError('Initial interface depth must be within the model')
        if not math.isfinite(initial_contrast_mps) or abs(initial_contrast_mps)>=max_contrast_mps:
            raise ValueError('Initial contrast must lie strictly inside its bounds')
        self.base,self.spacing_m,self.displacement_m = base,float(spacing_m),float(displacement_m)
        self.max_contrast_mps=float(max_contrast_mps)
        z,x=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij')
        self.register_buffer('coordinates',torch.stack((x,z),-1))
        reference=(torch.arange(h)[:,None].expand(h,w)*spacing_m-depth_m).float()
        if phi_reference is not None:
            if phi_reference.shape!=(h,w) or not torch.isfinite(phi_reference).all():
                raise ValueError('phi_reference must be finite HxW signed-distance-like values in metres')
            reference=phi_reference.detach().float().clone()
        self.register_buffer('phi_reference',reference)
        self.register_buffer('epsilon',torch.tensor(float(epsilon_m)))
        self.shape_net=nn.Sequential(nn.Linear(2,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.shape_net[-1].weight); nn.init.zeros_(self.shape_net[-1].bias)
        self.raw_contrast=nn.Parameter(torch.tensor(math.atanh(initial_contrast_mps/max_contrast_mps)))
        self.set_epsilon(epsilon_m)

    @torch.no_grad()
    def set_epsilon(self, epsilon_m: float) -> None:
        if not math.isfinite(epsilon_m) or epsilon_m<self.spacing_m:
            raise ValueError('epsilon must be at least one grid spacing; sub-grid sharpening is unsupported')
        self.epsilon.fill_(epsilon_m)

    def level_set(self) -> torch.Tensor:
        shift=self.displacement_m*torch.tanh(self.shape_net(self.coordinates).squeeze(-1))
        return self.phi_reference+shift

    def forward(self) -> torch.Tensor:
        gate=torch.sigmoid(self.level_set()/self.epsilon)
        reference=torch.sigmoid(self.phi_reference/self.epsilon)
        jump=self.max_contrast_mps*torch.tanh(self.raw_contrast)
        return self.base()+jump*(gate-reference)[None]

    def eikonal_loss(self) -> torch.Tensor:
        phi=self.level_set()
        dz=(phi[1:,:-1]-phi[:-1,:-1])/self.spacing_m
        dx=(phi[:-1,1:]-phi[:-1,:-1])/self.spacing_m
        return ((dx.square()+dz.square()+1e-12).sqrt()-1).square().mean()

    def regularization(self) -> torch.Tensor:
        phi=self.level_set()
        dxx=(phi[:,2:]-2*phi[:,1:-1]+phi[:,:-2])/self.spacing_m
        dzz=(phi[2:]-2*phi[1:-1]+phi[:-2])/self.spacing_m
        return self.eikonal_loss()+.1*(dxx.square().mean()+dzz.square().mean())

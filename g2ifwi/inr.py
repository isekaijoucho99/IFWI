"""Implicit velocity representation v_theta(r) = v_init + dv_scale * MLP(hash(r))  (Eq. 1, 7; Alg. 1 line 5)."""
import numpy as np
import torch
from torch import nn

from .hashgrid import HashEncoder2D


class SirenMLP(nn.Module):
    """SIREN with `layers` hidden sine layers and a linear output layer."""

    def __init__(self, in_dim, hidden=128, layers=2, out_dim=1, omega=30.0, zero_last=True):
        super().__init__()
        self.omega = omega
        dims = [in_dim] + [hidden] * layers
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(hidden, out_dim)
        with torch.no_grad():
            self.hidden[0].weight.uniform_(-1 / in_dim, 1 / in_dim)
            for lin in list(self.hidden)[1:]:
                bound = np.sqrt(6 / lin.in_features) / omega
                lin.weight.uniform_(-bound, bound)
            if zero_last:   # start exactly from v_init
                self.out.weight.zero_()
                self.out.bias.zero_()

    def forward(self, x):
        for lin in self.hidden:
            x = torch.sin(self.omega * lin(x))
        return self.out(x)


class HashSirenVelocity(nn.Module):
    def __init__(self, nz, nx, v_init, enc_kwargs, mlp_kwargs, dv_scale=1000.0):
        super().__init__()
        self.nz, self.nx, self.dv_scale = nz, nx, dv_scale
        self.encoder = HashEncoder2D(**enc_kwargs)
        self.mlp = SirenMLP(self.encoder.out_dim, **mlp_kwargs)
        zz, xx = torch.meshgrid(torch.linspace(0, 1, nz), torch.linspace(0, 1, nx), indexing="ij")
        self.register_buffer("coords", torch.stack([xx, zz], -1).reshape(-1, 2))
        self.register_buffer("v_init", torch.as_tensor(v_init, dtype=torch.float32))

    def forward(self):
        dv = self.mlp(self.encoder(self.coords)).reshape(self.nz, self.nx)
        return self.v_init + self.dv_scale * dv

"""Multiresolution hash encoding (Mueller et al., 2022) for 2D coordinates, pure PyTorch.

Implements Eqs. (2)-(6) of the paper: level resolutions N_l = N_min * b**l, spatial hash
h(i) = (i_x * pi_1  XOR  i_z * pi_2) mod T, bilinear interpolation of the four hashed
corner features, concatenation over levels.
"""
import math

import torch
from torch import nn

PI1, PI2 = 1, 2654435761  # primes used by instant-ngp for the first two dimensions


class HashEncoder2D(nn.Module):
    def __init__(self, n_levels=4, n_features=1, base_res=64, finest_res=256,
                 log2_hashmap_size=18, init_scale=1e-4):
        super().__init__()
        self.n_levels, self.n_features = n_levels, n_features
        self.table_size = 2 ** log2_hashmap_size
        growth = math.exp((math.log(finest_res) - math.log(base_res)) / (n_levels - 1)) if n_levels > 1 else 1.0
        self.resolutions = [int(math.floor(base_res * growth ** l + 1e-6)) for l in range(n_levels)]
        self.tables = nn.Parameter(torch.empty(n_levels, self.table_size, n_features).uniform_(-init_scale, init_scale))
        self.register_buffer("res", torch.tensor(self.resolutions, dtype=torch.float32), persistent=False)
        self.register_buffer("corner", torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]), persistent=False)

    @property
    def out_dim(self):
        return self.n_levels * self.n_features

    def hash(self, ix, iz):
        return ((ix * PI1) ^ (iz * PI2)) % self.table_size

    def forward(self, coords):
        """coords: [N, 2] with (x, z) in [0, 1]. Returns [N, n_levels * n_features]."""
        coords = coords.clamp(0.0, 1.0)
        feats = []
        for l in range(self.n_levels):
            u = coords * self.res[l]                       # Eq. (4)
            i0 = torch.floor(u)
            w = u - i0                                     # [N, 2]
            i0 = i0.long()
            f = 0.0
            for c in self.corner:                          # four neighbouring vertices, Eq. (5)
                idx = self.hash(i0[:, 0] + c[0], i0[:, 1] + c[1])
                wx = w[:, 0] if c[0] else 1 - w[:, 0]
                wz = w[:, 1] if c[1] else 1 - w[:, 1]
                f = f + (wx * wz)[:, None] * self.tables[l][idx]
            feats.append(f)
        return torch.cat(feats, dim=-1)                    # Eq. (6)

"""Baseline velocity models for the paper's supplementary experiments (docs/paper_supplement.md).

Both share the interface of experiments.residual_ifwi_experiment.VelocityParameterization that
the runner and train_update use: forward() -> [1, nz, nx] velocity in m/s, a fixed background
buffer ``fixed_init_mps``, ``parameterization == "residual"`` and empty omega/Fourier lists.

* GridVelocity: conventional grid FWI, v = v0 + dv with dv in R^{nz x nx} (m/s) starting at 0.
  Adam acts directly on dv, so the learning rate is a velocity step in m/s.
* FourierFeatureVelocity: input Fourier features (Tancik et al., 2020),
  gamma(q) = [sin(2 pi G q), cos(2 pi G q)] with G ~ N(0, s^2) in cycles/km and q = (x, z) in km
  (the coordinates the SIREN uses), followed by a ReLU MLP and a zero-initialized linear head:
  v = v0 + 1000 f(q), the same residual scale and zero-output start as E2.

linear_depth_background builds a background that depends on depth only, v0(z) = top + g z, from
explicit constants (no truth involved).
"""
from __future__ import annotations

import numpy as np
import torch


class GridVelocity(torch.nn.Module):
    parameterization = "residual"

    def __init__(self, initial_mps):
        super().__init__()
        initial = torch.as_tensor(initial_mps, dtype=torch.float32).detach().clone()
        if initial.ndim != 2 or not torch.isfinite(initial).all() or initial.min() <= 0:
            raise ValueError("initial_mps must be a finite positive [nz,nx] model")
        self.register_buffer("fixed_init_mps", initial[None])
        self.delta_mps = torch.nn.Parameter(torch.zeros_like(initial)[None])
        self.net, self.low_rank, self.std_kmps = None, None, 1.
        self.fourier_layers, self.omega_layers = [], []

    def forward(self):
        return self.fixed_init_mps + self.delta_mps


class _Linears(torch.nn.Module):
    """Holds ``linear`` like the original IRN, so layer diagnostics can read the weights."""

    def __init__(self, layers):
        super().__init__()
        self.linear = torch.nn.ModuleList(layers)


class FourierFeatureVelocity(torch.nn.Module):
    parameterization = "residual"

    def __init__(self, initial_mps, dx_m, scale_cpkm, features=128, hidden=(128, 128, 128, 128), seed=0):
        super().__init__()
        initial = torch.as_tensor(initial_mps, dtype=torch.float32).detach().clone()
        if initial.ndim != 2 or not torch.isfinite(initial).all() or initial.min() <= 0:
            raise ValueError("initial_mps must be a finite positive [nz,nx] model")
        if not scale_cpkm > 0 or features < 1 or not hidden:
            raise ValueError("Fourier features need a positive scale, features and hidden layers")
        self.register_buffer("fixed_init_mps", initial[None])
        x, z = np.meshgrid(np.arange(initial.shape[1]) * dx_m / 1000, np.arange(initial.shape[0]) * dx_m / 1000)
        self.register_buffer("coords", torch.from_numpy(np.stack((x, z), axis=-1).astype(np.float32))[None])
        # Own generator: the frequency matrix does not depend on (or consume) the global torch RNG.
        generator = torch.Generator().manual_seed(int(seed))
        self.register_buffer("frequencies_cpkm", torch.randn(features, 2, generator=generator) * float(scale_cpkm))
        widths = [2 * features, *hidden, 1]
        layers = [torch.nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:])]
        with torch.no_grad():
            layers[-1].weight.zero_()
            layers[-1].bias.zero_()
        self.net = _Linears(layers)
        self.low_rank, self.std_kmps, self.scale_cpkm = None, 1., float(scale_cpkm)
        self.fourier_layers, self.omega_layers = [], []

    def raw_output(self):
        phase = 2 * np.pi * self.coords @ self.frequencies_cpkm.T
        h = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        for layer in self.net.linear[:-1]:
            h = torch.relu(layer(h))
        return self.net.linear[-1](h).squeeze(-1)

    def forward(self):
        return self.fixed_init_mps + 1000. * self.raw_output()


def linear_depth_background(shape, dz_m, top_mps, gradient_per_s):
    """v0(z) = top + gradient * z (m/s, z in m), constant along x; float32 like the Gaussian background."""
    if not top_mps > 0 or gradient_per_s < 0:
        raise ValueError("Linear background needs a positive top velocity and a nonnegative gradient")
    z = np.arange(shape[0], dtype=np.float64) * dz_m
    return np.repeat((top_mps + gradient_per_s * z)[:, None], shape[1], axis=1).astype(np.float32)

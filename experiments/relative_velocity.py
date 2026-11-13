"""Fixed background-relative perturbations for the existing IFWI network.

This wrapper changes the velocity parameterization only. Fourier layers, when
requested, are constructed by the unchanged ``VelocityParameterization`` API.
The scale is fixed from the initial background, not from any later iterate.
"""
from __future__ import annotations

import torch

from experiments.residual_ifwi_experiment import VelocityParameterization


class RelativeVelocityParameterization(VelocityParameterization):
    """v = v0 + (v0 / rms(v0)) * (1000 * std_kmps) * f_theta.

    ``relative_scale`` is a persistent, nontrainable float32 buffer. Its RMS
    normalization is accumulated in float64 once at construction to avoid
    unnecessary roundoff in the global scalar. The inherited zero residual
    head makes the initial physical velocity exactly equal to v0.

    This is a background-relative perturbation, not an exact log-velocity
    parameterization or a substitute for the physical illumination operator.
    """

    def __init__(self, initial_mps, dx_m, neurons, omega_0=30., mean_kmps=3.,
                 std_kmps=1., parameterization='residual', fourier=None,
                 trainable_omega=False, low_rank=None, zero_residual_head=True):
        if parameterization != 'residual':
            raise ValueError('Background-relative velocity requires residual parameterization')
        super().__init__(initial_mps, dx_m, neurons, omega_0, mean_kmps,
                         std_kmps, parameterization, fourier, trainable_omega,
                         low_rank, zero_residual_head)
        rms = self.fixed_init_mps.double().square().mean().sqrt()
        rms = rms.to(dtype=self.fixed_init_mps.dtype)
        self.register_buffer('relative_scale', self.fixed_init_mps / rms)

    def forward(self):
        return (self.fixed_init_mps
                + self.relative_scale * (1000 * self.std_kmps) * self.raw_output())

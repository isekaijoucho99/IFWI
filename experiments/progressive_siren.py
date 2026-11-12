"""User-specified two-SIREN frequency-progression adaptation.

Reuses this project's existing VelocityParameterization/author IRN module
(SIREN: https://arxiv.org/abs/2006.09661). The half-cosine window follows the
coarse-to-fine window pattern in BARF's author code:
https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/model/barf.py
BARF applies its window to input encodings; here it scales a second SIREN's
output as explicitly requested. This is neither an original-author SIREN-FWI
pipeline nor a guarantee of physical spatial band limitation or denoising.

Both residual heads start at zero. Both branches are always evaluated, even
when alpha is zero, so backward creates zero (not missing) high-branch grads.
There are two complete SIRENs: capacity/optimizer cost doubles, so any result
is a whole-adaptation comparison, not an isolated frequency-only effect.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch

from experiments.residual_ifwi_experiment import VelocityParameterization


def _positive_real(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _nonnegative_integer(value, label, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < (1 if positive else 0) or value > torch.iinfo(torch.int64).max):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} int64 integer")
    return int(value)


class ProgressiveSirenVelocity(torch.nn.Module):
    """Return one background plus two gated, velocity-scaled residuals.

    alpha(k) = (1-cos(pi*clip(k/warmup_updates,0,1)))/2, where k is the
    number of completed optimizer updates, before the next forward. Progress
    and warmup are saved buffers. No branch is conditionally skipped.
    """

    def __init__(self, initial_mps, dx_m, neurons, mean_kmps=3., std_kmps=1.,
                 warmup_updates=200, low_omega=6., high_omega=30.):
        super().__init__()
        dx_m = _positive_real(dx_m, "dx_m")
        self.mean_kmps = _positive_real(mean_kmps, "mean_kmps")
        self.std_kmps = _positive_real(std_kmps, "std_kmps")
        low_omega = _positive_real(low_omega, "low_omega")
        high_omega = _positive_real(high_omega, "high_omega")
        warmup_updates = _nonnegative_integer(warmup_updates, "warmup_updates", positive=True)
        if (not isinstance(neurons, (tuple, list)) or len(neurons) < 3
                or neurons[0] != 2 or neurons[-1] != 1):
            raise ValueError("neurons must describe a SIREN with 2 inputs and 1 output")
        for width in neurons:
            _nonnegative_integer(width, "neuron width", positive=True)
        initial = torch.as_tensor(initial_mps, dtype=torch.float32).detach()
        if (initial.ndim != 2 or 0 in initial.shape or not torch.isfinite(initial).all()
                or initial.min() <= 0):
            raise ValueError("initial_mps must be a finite positive nonempty [nz,nx] model")
        branch_options = dict(dx_m=dx_m, neurons=list(neurons), mean_kmps=self.mean_kmps,
            std_kmps=self.std_kmps, parameterization="residual", zero_residual_head=True)
        self.low = VelocityParameterization(initial, omega_0=low_omega, **branch_options)
        self.high = VelocityParameterization(initial, omega_0=high_omega, **branch_options)
        self.register_buffer("completed_updates", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_warmup_updates", torch.tensor(warmup_updates, dtype=torch.int64))

    @property
    def fixed_init_mps(self):
        return self.low.fixed_init_mps

    @property
    def warmup_updates(self):
        return int(self._warmup_updates.item())

    @property
    def alpha(self):
        dtype = self.fixed_init_mps.dtype
        fraction = (self.completed_updates.to(dtype=dtype)
                    / self._warmup_updates.to(dtype=dtype)).clamp(0, 1)
        return .5 * (1 - torch.cos(math.pi * fraction))

    def set_progress(self, completed_updates):
        completed_updates = _nonnegative_integer(completed_updates, "completed_updates")
        self.completed_updates.fill_(completed_updates)

    def forward(self):
        low_raw = self.low.raw_output()
        high_raw = self.high.raw_output()
        return self.fixed_init_mps + (1000 * self.std_kmps) * (low_raw + self.alpha * high_raw)

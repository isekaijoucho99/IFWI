"""Physical velocities and a full-shot waveform objective for Direction A.

Arrays are ordered [z,x]. Each component of the gradient differentiates the
loss with respect to one cell's velocity in m/s. The FD solver uses float32.
"""

import math
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from experiments.residual_ifwi_data import _read_velocity, _sample
from experiments.residual_ifwi_experiment import _simulate, require_finite, waveform_mse
from generator import wGenerator
from rnn_fd import rnn2D


DATA = Path(__file__).resolve().parents[2] / "data"


def _positive_int(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def load_patch(dataset, shape):
    """Return centered truth/background float32 arrays in m/s and spacing in m.

    A null shape selects the complete inversion grid. Marmousi uses the legacy
    numeric-header convention followed by a stride of four on both axes.
    Overthrust uses the supplied vp and vi arrays at their native spacing.
    """
    if dataset.lower() == "marmousi":
        fields = []
        for name in ("vel_marmousi_376x1151.csv", "vel_marmousi_smooth400_376x1151.csv"):
            values, _ = _read_velocity(DATA / name, "m/s", "legacy")
            field, _ = _sample(values, stride=4, crop=None)
            fields.append(field)
        spacing = 15.
    elif dataset.lower() == "overthrust":
        with np.load(DATA / "overthrust2d.npz") as archive:
            fields = [archive[name].astype(np.float32) * 1000 for name in ("vp", "vi")]
            spacing = float(archive["dz"])
    else:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose marmousi or overthrust.")

    grid = fields[0].shape
    if len(grid) != 2 or fields[1].shape != grid or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Truth and background must share a 2-D grid with positive finite spacing")
    if any(not np.isfinite(field).all() or field.min() <= 0 for field in fields):
        raise ValueError("Truth and background must contain positive finite velocities in m/s")
    if shape is None:
        return fields[0], fields[1], spacing
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2
            or not all(_positive_int(n) and n <= available for n, available in zip(shape, grid))):
        raise ValueError(f"Patch shape must fit inside the inversion grid {grid}")
    z, x = ((available - requested) // 2 for requested, available in zip(shape, grid))
    truth, background = (np.ascontiguousarray(v[z:z + shape[0], x:x + shape[1]]) for v in fields)
    return truth, background, spacing


def _records(solver, v_mps, wavelet):
    try:
        return _simulate(solver, v_mps, wavelet)
    finally:
        # The reference cell caches its padded input internally. It rebuilds
        # that input each time step, so we can release the model and graph.
        solver.fd.velocity = None


class Survey:
    """Shared synthetic observations and differentiable time-domain shot solves.

    With spacing h, dt=min(h/10000, 0.0018) seconds. The Ricker peak is at
    1/frequency seconds. Sources occupy the centers of equal bins across the
    interior columns at z=1. Every column has a receiver at z=2. The original
    order-2 solver uses a free surface and 15-cell PML on its other sides.
    """

    def __init__(self, truth_mps, spacing_m, *, shots, nt, frequency, device="cpu", shot_batch=None):
        self.shape = tuple(truth_mps.shape)
        if len(self.shape) != 2 or min(self.shape) < 3:
            raise ValueError("The survey needs a [nz,nx] grid with at least three cells per axis")
        nz, nx = self.shape
        if not _positive_int(shots) or shots > nx - 2:
            raise ValueError("shots must fit at distinct interior source columns")
        if not _positive_int(nt) or (shot_batch is not None and not _positive_int(shot_batch)):
            raise ValueError("nt and shot_batch must be positive integers")
        self.spacing_m = float(spacing_m)
        if not math.isfinite(self.spacing_m) or self.spacing_m <= 0:
            raise ValueError("spacing_m must be finite and positive")
        self.dt = min(self.spacing_m / 10000., .0018)
        self.frequency = float(frequency)
        if not 0 < self.frequency < .5 / self.dt:
            raise ValueError("frequency must be positive and below the temporal Nyquist frequency")
        self.device = torch.device(device)
        self.nt, self.shots = int(nt), int(shots)
        self.shot_batch = self.shots if shot_batch is None else int(shot_batch)
        if not self.is_valid(truth_mps):
            raise ValueError("Invalid observation velocity grid or order-2 CFL violation")
        truth = torch.as_tensor(truth_mps, device=self.device, dtype=torch.float32).detach()[None]
        t = self.dt * torch.arange(self.nt, dtype=torch.float32)
        self.wavelet = wGenerator(t, self.frequency).ricker().to(self.device)
        self.source_x_indices = [1 + (2 * i + 1) * (nx - 2) // (2 * self.shots)
                                 for i in range(self.shots)]
        # A design speed fixes PML damping independently of the geological truth.
        pml_speed = .6 * self.spacing_m / self.dt
        self._batches = []
        observations = []
        for first in range(0, self.shots, self.shot_batch):
            last = min(first + self.shot_batch, self.shots)
            xs = torch.tensor(self.source_x_indices[first:last])[None]
            xr = torch.arange(nx)[None, None, :].repeat(1, last - first, 1)
            solver = rnn2D(nz, nx, torch.ones_like(xs), xs, torch.full_like(xr, 2), xr,
                           self.spacing_m, self.dt, npad=15, order=2, vmax=pml_speed,
                           log_para=1e-6, freeSurface=True, device=self.device)
            with torch.no_grad():
                observations.append(_records(solver, truth, self.wavelet).detach())
            self._batches.append((first, last, solver))
        self.observed = torch.cat(observations, dim=1)
        self.setup_forwards = self.shots

    def is_valid(self, v):
        """Screen the actual float32 solver input for shape, positivity and CFL."""
        candidate = torch.as_tensor(v).detach()
        if tuple(candidate.shape) != self.shape or not candidate.is_floating_point():
            return False
        low, high = torch.aminmax(candidate.to(device=self.device, dtype=torch.float32))
        return float(low) > 0. and float(high) * self.dt / self.spacing_m < 1 / math.sqrt(2)

    def evaluate(self, v, *, gradient=False):
        """Return full waveform MSE and optionally its [nz,nx] derivative in v.

        Every batch uses the same physical velocity and the global denominator
        shots*nt*nx. The optional derivative has the input's dtype and is
        detached from both the FD graph and the caller's parameterization.
        """
        candidate = torch.as_tensor(v).detach()
        if not self.is_valid(candidate):
            raise ValueError("Velocity must be a positive finite m/s field on the survey grid satisfying CFL")
        velocity = candidate.to(device=self.device, dtype=torch.float32)[None]
        with torch.set_grad_enabled(gradient):
            velocity.requires_grad_(gradient)
            total = 0.
            for first, last, solver in self._batches:
                records = _records(solver, velocity, self.wavelet)
                loss = waveform_mse(records, self.observed[:, first:last], self.shots)
                require_finite(loss, "waveform MSE")
                if gradient:
                    loss.backward()
                total += float(loss.detach())
                del records, loss
        if gradient:
            require_finite(velocity.grad, "physical velocity gradient")
            return total, velocity.grad[0].detach().to(dtype=candidate.dtype)
        return total, None

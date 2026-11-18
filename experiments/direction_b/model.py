"""Dataset patches and parameter derivatives for the frequency-domain experiments.

All grids are [depth, horizontal position]. Velocities are in m/s, squared
slowness m = 1 / v**2 is in s²/m², and isotropic grid spacing is in meters.
"""

from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from experiments.residual_ifwi_experiment import VelocityParameterization


DATA = Path(__file__).resolve().parents[2] / "data"


def load_patch(dataset, shape):
    """Return centered (truth_mps, initial_mps, spacing_m) on coincident nodes.

    Marmousi uses the published CSV pair with the repository's legacy header
    convention (skip its first numeric row), stride 4 on both axes and a 15 m
    inversion grid. Its background is the distributed smooth400 grid.
    Overthrust uses the native NPZ vp and vi grids in km/s, converted to m/s,
    and the dz value stored in that file. Crop origins round down to an index.
    """
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2 or
            any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in shape)):
        raise ValueError("shape must be a pair of positive integers (nz, nx)")
    if dataset == "marmousi":
        truth, initial = (np.loadtxt(DATA / name, delimiter=",", skiprows=1)[::4, ::4]
                          for name in ("vel_marmousi_376x1151.csv",
                                       "vel_marmousi_smooth400_376x1151.csv"))
        spacing = 15.
    elif dataset == "overthrust":
        with np.load(DATA / "overthrust2d.npz") as arrays:
            truth = arrays["vp"].astype(np.float64) * 1000.
            initial = arrays["vi"].astype(np.float64) * 1000.
            spacing = float(arrays["dz"].item())
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    if truth.shape != initial.shape or any(n > length for n, length in zip(shape, truth.shape)):
        raise ValueError(f"requested shape {shape} must fit both grids, got {truth.shape}, {initial.shape}")
    starts = [(length - n) // 2 for n, length in zip(shape, truth.shape)]
    window = tuple(slice(start, start + n) for start, n in zip(starts, shape))
    return truth[window].copy(), initial[window].copy(), spacing


def make_model(initial, spacing, seed):
    """Build a reproducibly initialized CPU float64 residual stage-1 SIREN."""
    initial = _real_tensor(initial, "initial velocity")
    if initial.ndim != 2 or initial.numel() == 0 or (initial <= 0).any():
        raise ValueError("initial velocity must be a nonempty positive [nz,nx] grid")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = VelocityParameterization(
            initial, spacing, [2, 16, 16, 1], parameterization="residual").double()
    # The stage-1 constructor stores float32 buffers. Preserve the supplied
    # float64 background and construct its local (x,z) coordinates in km.
    nz, nx = initial.shape
    z, x = torch.meshgrid(torch.arange(nz, dtype=torch.float64),
                          torch.arange(nx, dtype=torch.float64), indexing="ij")
    model.fixed_init_mps = initial.detach().clone()[None]
    model.coords = torch.stack((x, z), dim=-1)[None] * (spacing / 1000.)
    return model


def _real_tensor(value, label):
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    result = torch.as_tensor(value, device="cpu", dtype=torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def _squared_slowness(velocity):
    if (velocity <= 0).any() or not torch.isfinite(velocity).all():
        raise ValueError("velocity must be finite and positive on every grid point")
    result = velocity[0].reciprocal().square()
    if (result <= 0).any() or not torch.isfinite(result).all():
        raise ValueError("squared slowness must be finite and positive")
    return result


def slowness(model):
    """Evaluate current squared slowness [nz,nx] in s²/m²."""
    with torch.no_grad():
        return _squared_slowness(model()).numpy()


def parameter_gradient(model, grad_m):
    """Return (dm/dtheta)^T grad_m in model.parameters() order."""
    m = _squared_slowness(model())
    field = _real_tensor(grad_m, "model-space gradient")
    if field.shape != m.shape:
        raise ValueError(f"model-space gradient must have shape {tuple(m.shape)}")
    gradient = torch.autograd.grad(m, tuple(model.parameters()), grad_outputs=field)
    return torch.cat([g.reshape(-1) for g in gradient]).detach().numpy()


def _parameter_direction(model, p):
    direction = _real_tensor(p, "parameter direction")
    parameters = tuple(model.parameters())
    sizes = [param.numel() for param in parameters]
    if direction.shape != (sum(sizes),):
        raise ValueError(f"parameter direction must be a vector of length {sum(sizes)}")
    return tuple(part.reshape_as(param) for part, param in zip(direction.split(sizes), parameters))


def tangent(model, p):
    """Return (dm/dtheta) p as a [nz,nx] grid in s²/m²."""
    directions = _parameter_direction(model, p)
    names, parameters = zip(*model.named_parameters())

    def evaluate(*values):
        return _squared_slowness(functional_call(model, dict(zip(names, values)), ()))

    # The original IRN detaches its coordinates and sets requires_grad during
    # forward. Reverse-over-reverse JVP supports that untouched implementation.
    _, dm = torch.autograd.functional.jvp(evaluate, parameters, directions)
    return dm.detach().numpy()


def step(model, p, alpha):
    """Set theta <- theta + alpha*p once the entire candidate is finite and physical."""
    directions = _parameter_direction(model, p)
    scale = _real_tensor(alpha, "step size")
    if scale.ndim != 0:
        raise ValueError("step size must be scalar")
    with torch.no_grad():
        candidate = {name: param + scale * delta
                     for (name, param), delta in zip(model.named_parameters(), directions)}
        if any(not value.isfinite().all() for value in candidate.values()):
            raise ValueError("updated parameters must be finite")
        _squared_slowness(functional_call(model, candidate, ()))
        for name, param in model.named_parameters():
            param.copy_(candidate[name])

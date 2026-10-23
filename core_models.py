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

traveltime_coordinates / warp_coordinates: the SIREN input (x, z) is replaced by coordinates
measured in traveltime through the background, z' = c_z * int_0^z dz/v0 (and optionally
x' = c_x * int_0^x dx/v0 along each row). A coordinate network has the same spectral bias
everywhere in its input space, so after the warp the bias is uniform in local wavelengths of
v0 instead of in metres: finer where v0 is slow, smoother where it is fast, like the data
resolution. c_z and c_x keep the mean coordinate range of each axis unchanged, so the global
bandwidth (omega_0 times the coordinate range) stays that of the unwarped network and only the
spatial distribution changes. Only v0 is used, never the truth; the network, its seed-matched
initialization and the zero output layer are unchanged, so step 0 is still exactly v0.
"""
from __future__ import annotations

from pathlib import Path

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


WARP_AXES = ("z", "xz")


def _slowness_integral(slowness, step_km, axis):
    """Trapezoidal cumulative integral of slowness along an axis, 0 at the first sample (s)."""
    pairs = .5 * (np.take(slowness, range(1, slowness.shape[axis]), axis=axis)
                  + np.take(slowness, range(0, slowness.shape[axis] - 1), axis=axis)) * step_km
    zero = np.zeros_like(np.take(slowness, [0], axis=axis))
    return np.concatenate((zero, np.cumsum(pairs, axis=axis)), axis=axis)


def traveltime_coordinates(background_mps, dx_m, axes="z"):
    """[nz, nx, 2] (x', z') in km: traveltime through v0, scaled to the original mean axis ranges."""
    if axes not in WARP_AXES:
        raise ValueError(f"coordinate_warp.axes must be one of {WARP_AXES}")
    v0 = np.asarray(background_mps, dtype=np.float64)
    if v0.ndim != 2 or not np.isfinite(v0).all() or v0.min() <= 0:
        raise ValueError("background must be a finite positive [nz,nx] model")
    nz, nx = v0.shape
    step = dx_m / 1000
    x, z = np.meshgrid(np.arange(nx) * step, np.arange(nz) * step)
    slowness = 1000. / v0                                       # s/km
    tz = _slowness_integral(slowness, step, 0)
    z_warp = tz * (z[-1, 0] / tz[-1].mean())
    x_warp = x
    if axes == "xz":
        tx = _slowness_integral(slowness, step, 1)
        x_warp = tx * (x[0, -1] / tx[:, -1].mean())
    coords = np.stack((x_warp, z_warp), axis=-1)
    if not (np.diff(coords[..., 1], axis=0) > 0).all() or not (np.diff(coords[..., 0], axis=1) > 0).all():
        raise ValueError("Warped coordinates must increase monotonically")
    return coords.astype(np.float32)


def warp_report(background_mps, dx_m, coords):
    """Local stretch d z'/dz (and d x'/dx): > 1 means finer network detail than unwarped."""
    step = dx_m / 1000
    stretch_z = np.diff(coords[..., 1].astype(np.float64), axis=0) / step
    stretch_x = np.diff(coords[..., 0].astype(np.float64), axis=1) / step
    rows = np.linspace(0, stretch_z.shape[0] - 1, 5).round().astype(int)
    return {"stretch_z": {"min": float(stretch_z.min()), "max": float(stretch_z.max()),
                          "row_means": {int(r): float(stretch_z[r].mean()) for r in rows}},
            "stretch_x": {"min": float(stretch_x.min()), "max": float(stretch_x.max())},
            "coordinate_range_km": {"x": [float(coords[..., 0].min()), float(coords[..., 0].max())],
                                    "z": [float(coords[..., 1].min()), float(coords[..., 1].max())]},
            "background_mps": [float(np.min(background_mps)), float(np.max(background_mps))]}


def warp_coordinates(model, coords):
    """Replace the SIREN input grid of a VelocityParameterization (2-D network) in place."""
    if getattr(model, "coords", None) is None or getattr(model, "low_rank", None) is not None:
        raise ValueError("coordinate_warp needs a 2-D coordinate network (not low-rank or grid)")
    new = torch.as_tensor(coords, dtype=model.coords.dtype, device=model.coords.device)[None]
    if new.shape != model.coords.shape:
        raise ValueError(f"warped coordinates {tuple(new.shape)} do not match {tuple(model.coords.shape)}")
    model.coords.copy_(new)
    return model


SMOOTH400_CSV = "vel_marmousi_smooth400_376x1151.csv"


def smooth400_background(data_dir, shape):
    """Background of the earliest IFWI runs (pretrain_marmousi.py), reproduced step by step: the author's
    smoothed Marmousi CSV read with pandas' default header (first row consumed), Gaussian sigma = 50 on the
    full-resolution grid (reflect, truncate 4), then every 4th point; float32 m/s. Its generating recipe is
    unknown, so it may contain truth information beyond the sigma-15 background."""
    import pandas as pd
    from scipy.ndimage import gaussian_filter
    smooth = np.array(pd.read_csv(Path(data_dir) / SMOOTH400_CSV))
    background = gaussian_filter(smooth, sigma=50)[::4, ::4].astype(np.float32)
    if background.shape != tuple(shape):
        raise ValueError(f"smooth400 background {background.shape} does not match the grid {tuple(shape)}")
    return background


def slowness_gaussian_background(truth_mps, dz_m, slowness_sigma_m, velocity_sigma_m):
    """Kinematically smooth background: Gaussian smoothing of the slowness (nearest edges), then of the
    velocity (reflect edges); sigmas in m. With 135 m and 187.5 m it matches smooth400 on Marmousi to
    4.1 m/s RMS and keeps traveltimes much closer to the truth than velocity smoothing."""
    from scipy.ndimage import gaussian_filter
    if not slowness_sigma_m > 0 or not velocity_sigma_m > 0:
        raise ValueError("slowness_gaussian sigmas must be positive")
    truth = np.asarray(truth_mps, dtype=np.float64)
    slow = gaussian_filter(1.0 / truth, slowness_sigma_m / dz_m, mode="nearest")
    return gaussian_filter(1.0 / slow, velocity_sigma_m / dz_m, mode="reflect").astype(np.float32)


def linear_depth_background(shape, dz_m, top_mps, gradient_per_s):
    """v0(z) = top + gradient * z (m/s, z in m), constant along x; float32 like the Gaussian background."""
    if not top_mps > 0 or gradient_per_s < 0:
        raise ValueError("Linear background needs a positive top velocity and a nonnegative gradient")
    z = np.arange(shape[0], dtype=np.float64) * dz_m
    return np.repeat((top_mps + gradient_per_s * z)[:, None], shape[1], axis=1).astype(np.float32)

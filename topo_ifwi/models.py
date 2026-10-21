"""Synthetic models for the minimal experiment (proposal 6.2) and the residual SIREN (proposal 4.1).

Every model is a layered background v_bg(x, z) plus high-velocity bodies. The initial model and
the reference background v_ref are the Gaussian-smoothed background *without* bodies, i.e. they
come from the generating rule, not from the truth containing the anomaly.

* ``dumbbell``: two lobes joined by a thin neck. One connected body without holes, so the
  single-body prior is correct; the neck is the structure a waveform-only inversion may break.
* ``twobody``: the same two lobes without the neck. The single-body prior is wrong here
  (proposal 6.4, error-prior stress test): a topology update that connects them is a false bridge.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

MODELS = ("dumbbell", "twobody")


def _grid(nz, nx, dx):
    z, x = np.mgrid[:nz, :nx].astype(np.float64)
    return z * dx, x * dx


def background(nz, nx, dx):
    z, x = _grid(nz, nx, dx)
    v = 1800. + 0.6 * z
    v += 60. * np.sin(2 * np.pi * (z + 40. * np.sin(2 * np.pi * x / 1800.)) / 170.)   # weak layering
    return v


def _ellipse(z, x, cz, cx, rz, rx):
    return ((z - cz) / rz) ** 2 + ((x - cx) / rx) ** 2 <= 1


def bodies(name, nz, nx, dx):
    z, x = _grid(nz, nx, dx)
    W = nx * dx
    left = _ellipse(z, x, 700., 0.33 * W, 190., 0.12 * W)
    right = _ellipse(z, x, 760., 0.68 * W, 170., 0.11 * W)
    if name == "twobody":
        return left | right
    if name == "dumbbell":
        # Neck: a slightly curved band about 3 cells thick between the lobes.
        centre = 720. + 30. * (x - 0.33 * W) / (0.35 * W) + 25. * np.sin(np.pi * (x - 0.33 * W) / (0.35 * W))
        neck = (np.abs(z - centre) <= 1.6 * dx) & (x > 0.33 * W) & (x < 0.68 * W)
        return left | right | neck
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def build(name, nz, nx, dx, v_anomaly=4300., smooth_cells=6.):
    v_bg = background(nz, nx, dx)
    mask = bodies(name, nz, nx, dx)
    v_true = np.where(mask, v_anomaly, v_bg)
    v0 = gaussian_filter(v_bg, smooth_cells, mode="nearest")
    return {"v_true": v_true.astype(np.float32), "v0": v0.astype(np.float32),
            "v_ref": v0.astype(np.float32), "mask": mask}


def score_parameters(v_ref, mask, v_anomaly, rho, temp_frac):
    """Relative threshold (v0.2, change 3): dv = rho * contrast, T = temp_frac * contrast, where the
    contrast is the prior anomaly velocity minus the mean reference velocity over the prior depth
    range of the body. Only the prior velocity and the depth range are used, not the body shape."""
    rows = np.where(mask.any(axis=1))[0]
    contrast = float(v_anomaly - v_ref[rows.min():rows.max() + 1].mean())
    return rho * contrast, temp_frac * contrast, contrast


class Sine(torch.nn.Module):
    def __init__(self, fan_in, fan_out, omega, first):
        super().__init__()
        self.omega, self.linear = omega, torch.nn.Linear(fan_in, fan_out)
        bound = 1 / fan_in if first else math.sqrt(6 / fan_in) / omega
        torch.nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, h):
        return torch.sin(self.omega * self.linear(h))


class ResidualSiren(torch.nn.Module):
    """v = v0 + scale * taper(z) * f_theta(x, z); zero last layer, so step 0 is exactly v0."""

    def __init__(self, v0, hidden=128, depth=4, omega=30., scale=1000., taper_rows=(0, 0)):
        super().__init__()
        v0 = torch.as_tensor(v0, dtype=torch.float32)
        nz, nx = v0.shape
        self.register_buffer("v0", v0)
        # Near-surface taper (fixed by the acquisition depth): no update above row r0, full below r1.
        r0, r1 = taper_rows
        rows = torch.arange(nz, dtype=torch.float32)
        ramp = torch.clamp((rows - r0) / max(r1 - r0, 1), 0, 1) if r1 > 0 else torch.ones(nz)
        self.register_buffer("taper", (0.5 - 0.5 * torch.cos(np.pi * ramp))[:, None] if r1 > 0 else ramp[:, None])
        z, x = torch.meshgrid(torch.linspace(-1, 1, nz), torch.linspace(-1, 1, nx), indexing="ij")
        self.register_buffer("coords", torch.stack([x, z], -1).view(-1, 2))
        layers = [Sine(2, hidden, omega, True)] + [Sine(hidden, hidden, omega, False) for _ in range(depth - 1)]
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.scale = scale

    def forward(self):
        return self.v0 + self.scale * self.taper * self.head(self.body(self.coords)).view(self.v0.shape)

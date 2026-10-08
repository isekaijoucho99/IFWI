"""Procedurally generated 'realistic' velocity models for training the diffusion prior (Sec. 4.1, Fig. 2).

1) 1D stochastic background profile (optional water layer of random thickness),
2) lateral extension to 2D with a mild lateral trend,
3) random elastic deformation (Kazei et al., 2021): smooth random displacement fields + folding,
4) occasional intrusive high-velocity bodies.
No well/field information is used. Velocities lie in [1500, 4500] m/s.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

VMIN, VMAX = 1500.0, 4500.0


def _profile(nz, rng):
    water = int(rng.integers(4, 30)) if rng.random() < 0.5 else 0
    v = np.empty(nz)
    v[:water] = 1500.0
    z, n_layers = water, 0
    v_top = rng.uniform(1600, 2300)
    v_bot = rng.uniform(3200, 4500)
    depth_len = max(nz - water, 1)
    while z < nz:
        th = int(np.clip(rng.lognormal(np.log(depth_len / rng.uniform(6, 30)), 0.6), 2, depth_len / 2))
        frac = (z - water) / depth_len
        trend = v_top + (v_bot - v_top) * frac ** rng.uniform(0.6, 1.4)
        val = trend * (1 + rng.normal(0, rng.uniform(0.02, 0.10)))
        if rng.random() < 0.12:                      # low-velocity or high-velocity streak
            val *= rng.uniform(0.8, 1.25)
        v[z:z + th] = val
        z += th
        n_layers += 1
    v = gaussian_filter(v, rng.uniform(0.3, 1.5))
    return np.clip(v, VMIN, VMAX)


def _smooth_field(shape, sigma, amp, rng):
    f = gaussian_filter(rng.standard_normal(shape), sigma, mode="reflect")
    return f / (f.std() + 1e-12) * amp


def generate_model(nz=152, nx=708, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    v = np.tile(_profile(nz, rng)[:, None], (1, nx))
    # lateral trend
    v = v * (1 + rng.uniform(-0.08, 0.08) * np.linspace(-1, 1, nx)[None, :] * np.linspace(0.2, 1, nz)[:, None])
    zz, xx = np.meshgrid(np.arange(nz, dtype=float), np.arange(nx, dtype=float), indexing="ij")
    # elastic deformation: smooth random displacement
    dz = _smooth_field((nz, nx), (rng.uniform(15, 40), rng.uniform(60, 160)), rng.uniform(2, 14), rng)
    dx = _smooth_field((nz, nx), (rng.uniform(15, 40), rng.uniform(40, 120)), rng.uniform(0, 8), rng)
    # folding: one or two sinusoidal undulations tapered with depth
    for _ in range(int(rng.integers(0, 3))):
        lam, amp, ph = rng.uniform(80, 400), rng.uniform(3, 25), rng.uniform(0, 2 * np.pi)
        taper = np.exp(-((zz - rng.uniform(0, nz)) / rng.uniform(20, 80)) ** 2) if rng.random() < 0.5 else 1.0
        dz = dz + amp * np.sin(2 * np.pi * xx / lam + ph) * taper
    v = map_coordinates(v, [zz + dz, xx + dx], order=1, mode="nearest")
    # intrusive bodies
    for _ in range(int(rng.choice([0, 0, 1, 2]))):
        cz, cx = rng.uniform(0.3, 1.0) * nz, rng.uniform(0, nx)
        rz, rx = rng.uniform(8, 35), rng.uniform(25, 120)
        warp = _smooth_field((nz, nx), 6, rng.uniform(0.05, 0.3), rng)
        mask = (((zz - cz) / rz) ** 2 + ((xx - cx) / rx) ** 2 + warp) < 1
        mask = gaussian_filter(mask.astype(float), 1.0)
        v = v * (1 - mask) + rng.uniform(3800, VMAX) * mask
    return np.clip(v, VMIN, VMAX).astype(np.float32)


def random_crop(v, size=128, rng=None, flip=True):
    rng = np.random.default_rng() if rng is None else rng
    z0 = int(rng.integers(0, v.shape[0] - size + 1))
    x0 = int(rng.integers(0, v.shape[1] - size + 1))
    c = v[z0:z0 + size, x0:x0 + size]
    return c[:, ::-1].copy() if flip and rng.random() < 0.5 else c.copy()


def normalize(x, eps=1e-6):
    """Per-sample min-max scaling to [-1, 1]; returns (scaled, lo, hi)."""
    lo, hi = x.min(), x.max()
    return 2 * (x - lo) / (hi - lo + eps) - 1, lo, hi

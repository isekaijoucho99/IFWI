"""2-D constant-density acoustic finite differences in torch (proposal 4.1, 6.1).

u_tt = v^2 lap(u) + s, second order in time, fourth order in space, absorbing layers of
``npml`` cells on all four sides implemented as a damping term d(x) u_t with a quadratic
profile. Shots are the batch dimension; autograd provides the adjoint.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def ricker(f0, dt, nt, delay=None):
    t = np.arange(nt) * dt - (1.2 / f0 if delay is None else delay)
    a = (np.pi * f0 * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


class Acoustic2D:
    def __init__(self, nz, nx, dx, dt, nt, src, rec, f0, npml=20, device="cpu"):
        """src: [S, 2] (iz, ix) source cells; rec: [R, 2] (iz, ix) receiver cells (shared)."""
        self.nz, self.nx, self.dx, self.dt, self.nt, self.npml = nz, nx, dx, dt, nt, npml
        self.device = torch.device(device)
        self.src = np.asarray(src, dtype=np.int64)
        self.rec = np.asarray(rec, dtype=np.int64)
        self.wavelet = torch.tensor(ricker(f0, dt, nt), device=self.device)
        n = npml
        hz, hx = nz + 2 * n, nx + 2 * n
        # Quadratic damping profile, d0 from the usual R = 1e-3 rule for vmax = 4500 m/s.
        d0 = 3 * 4500 * math.log(1e3) / (2 * n * dx)
        def profile(m):
            i = np.arange(m, dtype=np.float64)
            dist = np.maximum(np.maximum(n - i, i - (m - 1 - n)), 0) / n
            return d0 * dist ** 2
        damp = np.maximum(profile(hz)[:, None], profile(hx)[None, :])
        self.damp = torch.tensor(damp * dt, dtype=torch.float32, device=self.device)[None, None]
        k = np.zeros((5, 5), np.float32)
        c = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12], np.float32) / dx ** 2
        k[2, :] += c
        k[:, 2] += c
        self.kernel = torch.tensor(k, device=self.device)[None, None]
        self.shape_pad = (hz, hx)

    def cfl(self, vmax):
        return vmax * self.dt / self.dx

    def _pad_model(self, v):
        n = self.npml
        return F.pad(v[None, None], (n, n, n, n), mode="replicate")

    def forward(self, v, shots=None):
        """v: [nz, nx] m/s tensor; returns [S, nt, R] pressure at the receivers."""
        shots = np.arange(len(self.src)) if shots is None else np.asarray(shots)
        n, (hz, hx) = self.npml, self.shape_pad
        S = len(shots)
        vp = self._pad_model(v)
        c2 = (vp * self.dt) ** 2                                  # [1,1,hz,hx]
        a = (1 - self.damp) / (1 + self.damp)
        b = 1 / (1 + self.damp)
        src_idx = torch.as_tensor((self.src[shots, 0] + n) * hx + self.src[shots, 1] + n, device=self.device)
        rec_idx = torch.as_tensor((self.rec[:, 0] + n) * hx + self.rec[:, 1] + n, device=self.device)
        inject = torch.zeros(S, hz * hx, device=self.device)
        inject[torch.arange(S, device=self.device), src_idx] = self.dt ** 2 / self.dx ** 2
        inject = inject.view(S, 1, hz, hx)
        u_prev = torch.zeros(S, 1, hz, hx, device=self.device)
        u = torch.zeros_like(u_prev)
        out = []
        for it in range(self.nt):
            lap = F.conv2d(u, self.kernel, padding=2)
            u_next = b * (2 * u + c2 * lap + inject * self.wavelet[it]) - a * u_prev
            u_prev, u = u, u_next
            out.append(u.view(S, -1)[:, rec_idx])
        return torch.stack(out, dim=1)


def bandpass(data, dt, low_hz, high_hz, taper_hz=1.0):
    """Zero-phase band-pass along time (dim 1) with cosine tapers; differentiable, same on all data."""
    nt = data.shape[1]
    f = torch.fft.rfftfreq(2 * nt, d=dt).to(data.device)
    w = torch.ones_like(f)
    if high_hz is not None:
        w = w * torch.clamp((high_hz + taper_hz - f) / (2 * taper_hz), 0, 1)
    if low_hz:
        w = w * torch.clamp((f - low_hz + taper_hz) / (2 * taper_hz), 0, 1)
    w = 0.5 - 0.5 * torch.cos(np.pi * w)
    spec = torch.fft.rfft(data, n=2 * nt, dim=1)
    return torch.fft.irfft(spec * w[None, :, None], n=2 * nt, dim=1)[:, :nt]

"""2D constant-density acoustic modelling with an exact gradient and bounded memory.

Second-order FD in space/time, free surface on top, Cerjan sponge on the other sides.
Shots are batched; time stepping is split into chunks that are re-computed during back-propagation
(gradient checkpointing), so the gradient equals the exact discrete adjoint-state gradient
d(loss)/d(v) used in Eq. (10) while memory stays at O(chunk) instead of O(nt).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def ricker(freq, dt, nt, delay=None, dtype=torch.float32):
    delay = 1.0 / freq if delay is None else delay
    t = torch.arange(nt, dtype=dtype) * dt - delay
    a = (np.pi * freq * t) ** 2
    return (1 - 2 * a) * torch.exp(-a)


class AcousticModeler:
    def __init__(self, nz, nx, dz, dt, wavelet, src_x, src_z, rec_x, rec_z,
                 npad=30, sponge=0.015, chunk=50, shot_batch=8, device="cpu", dtype=torch.float32):
        self.nz, self.nx, self.dz, self.dt, self.npad = nz, nx, dz, dt, npad
        self.nt = len(wavelet)
        self.chunk, self.shot_batch, self.device, self.dtype = chunk, shot_batch, device, dtype
        self.wavelet = wavelet.to(device, dtype)
        self.src_x = torch.as_tensor(src_x, dtype=torch.long, device=device)
        self.src_z = torch.as_tensor(src_z, dtype=torch.long, device=device) if np.ndim(src_z) else \
            torch.full_like(self.src_x, int(src_z))
        self.rec_x = torch.as_tensor(rec_x, dtype=torch.long, device=device) + npad
        self.rec_z = int(rec_z)
        self.n_shots, self.n_rec = len(self.src_x), len(self.rec_x)
        zp, xp = nz + npad, nx + 2 * npad
        ramp = torch.exp(-(sponge * torch.arange(npad, 0, -1, dtype=dtype)) ** 2)
        damp = torch.ones(zp, xp, dtype=dtype)
        damp[:, :npad] *= ramp[None, :]
        damp[:, -npad:] *= ramp.flip(0)[None, :]
        damp[-npad:, :] *= ramp.flip(0)[:, None]
        self.damp = damp.to(device)[None, None]
        self.kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=dtype, device=device)[None, None]

    def check_cfl(self, vmax):
        cfl = vmax * self.dt / self.dz
        if cfl >= 1 / np.sqrt(2):
            raise ValueError(f"CFL violated: vmax*dt/dz={cfl:.3f} >= 0.707")

    def _c2(self, v):
        vp = F.pad(v[None, None], (self.npad, self.npad, 0, self.npad), mode="replicate")
        return (vp * self.dt / self.dz) ** 2

    def _steps(self, u_prev, u_cur, c2, onehot, w, rec_idx):
        recs = []
        for k in range(w.shape[0]):
            lap = F.conv2d(u_cur, self.kernel, padding=1)
            u_next = (2 * u_cur - u_prev + c2 * lap + onehot * c2 * w[k]) * self.damp
            u_prev, u_cur = u_cur * self.damp, u_next
            recs.append(u_cur[:, 0, self.rec_z, rec_idx])
        return u_prev, u_cur, torch.stack(recs, 1)

    def simulate(self, v, shot_ids):
        """v: [nz, nx] m/s (may require grad). Returns [len(shot_ids), nt, n_rec]."""
        v = v.to(self.device, self.dtype)
        b, zp, xp = len(shot_ids), self.nz + self.npad, self.nx + 2 * self.npad
        c2 = self._c2(v)
        onehot = torch.zeros(b, 1, zp, xp, dtype=self.dtype, device=self.device)
        onehot[torch.arange(b), 0, self.src_z[shot_ids], self.src_x[shot_ids] + self.npad] = 1.0
        u_prev = torch.zeros(b, 1, zp, xp, dtype=self.dtype, device=self.device)
        u_cur = torch.zeros_like(u_prev)
        out = []
        for t0 in range(0, self.nt, self.chunk):
            w = self.wavelet[t0:t0 + self.chunk]
            if torch.is_grad_enabled() and c2.requires_grad:
                u_prev, u_cur, r = checkpoint(self._steps, u_prev, u_cur, c2, onehot, w, self.rec_x,
                                              use_reentrant=False)
            else:
                u_prev, u_cur, r = self._steps(u_prev, u_cur, c2, onehot, w, self.rec_x)
            out.append(r)
        return torch.cat(out, 1)

    @torch.no_grad()
    def simulate_all(self, v):
        outs = []
        for s in range(0, self.n_shots, self.shot_batch):
            ids = torch.arange(s, min(s + self.shot_batch, self.n_shots), device=self.device)
            outs.append(self.simulate(v, ids))
        return torch.cat(outs, 0)

    def loss_and_grad(self, v, d_obs, norm=None, transform=None):
        """Data misfit ||P(v)-d_obs||^2 / norm and its gradient wrt v (Eqs. 9/25/38).

        `transform` optionally filters predicted and observed data identically (band limiting)."""
        norm = float((d_obs ** 2).sum()) if norm is None else norm
        v_leaf = v.detach().clone().requires_grad_(True)
        total = 0.0
        for s in range(0, self.n_shots, self.shot_batch):
            ids = torch.arange(s, min(s + self.shot_batch, self.n_shots), device=self.device)
            pred = self.simulate(v_leaf, ids)
            obs = d_obs[ids]
            if transform is not None:
                pred, obs = transform(pred), transform(obs)
            loss = ((pred - obs) ** 2).sum() / norm
            loss.backward()
            total += float(loss.detach())
        return total, v_leaf.grad.detach()

"""DDPM prior: U-Net eps-predictor, noise schedule, RED-style prior gradient with optional well guidance.

Implements Eqs. (11)-(20) (DDPM), (22)-(24)/(32)-(35) (RED) and Eqs. (26)-(31) / Alg. 1 lines 7-17.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

PRESETS = {
    # exact configuration of Table 2
    "paper": dict(channels=128, mult=(1, 2, 4, 8, 16), n_res=3, heads=4, lr=1e-6),
    # laptop-scale configuration
    "small": dict(channels=64, mult=(1, 2, 4, 8), n_res=2, heads=4, lr=2e-4),
}


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


def gn(c):
    return nn.GroupNorm(32, c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Sequential(gn(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1))
        self.t = nn.Sequential(nn.SiLU(), nn.Linear(tdim, cout))
        self.c2 = nn.Sequential(gn(cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(x) + self.t(e)[:, :, None, None]
        return self.c2(h) + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        self.h, self.norm = heads, gn(c)
        self.qkv, self.proj = nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.h, c // self.h, H * W).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(b, c, H, W))


class UNet(nn.Module):
    """Fully convolutional (+ attention): works for any H, W divisible by 2**(len(mult)-1)."""

    def __init__(self, channels=128, mult=(1, 2, 4, 8, 16), n_res=3, heads=4, attn_from=2):
        super().__init__()
        tdim = channels * 4
        self.channels = channels
        self.temb = nn.Sequential(nn.Linear(channels, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(1, channels, 3, padding=1)
        self.down, chans = nn.ModuleList(), [channels]
        c = channels
        for lvl, m in enumerate(mult):
            for _ in range(n_res):
                blk = nn.ModuleList([ResBlock(c, channels * m, tdim)])
                c = channels * m
                if lvl >= attn_from:
                    blk.append(Attn(c, heads))
                self.down.append(blk); chans.append(c)
            if lvl < len(mult) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)])); chans.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, tdim), Attn(c, heads), ResBlock(c, c, tdim)])
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mult))):
            for i in range(n_res + 1):
                blk = nn.ModuleList([ResBlock(c + chans.pop(), channels * m, tdim)])
                c = channels * m
                if lvl >= attn_from:
                    blk.append(Attn(c, heads))
                if lvl > 0 and i == n_res:
                    blk.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blk.append(nn.Conv2d(c, c, 3, padding=1))
                self.up.append(blk)
        self.out = nn.Sequential(gn(c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        e = self.temb(timestep_embedding(t, self.channels))
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            if isinstance(blk[0], ResBlock):
                h = blk[0](h, e)
                for extra in blk[1:]:
                    h = extra(h)
            else:
                h = blk[0](h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, e)), e)
        for blk in self.up:
            h = blk[0](torch.cat([h, hs.pop()], 1), e)
            for extra in blk[1:]:
                h = extra(h)
        return self.out(h)


class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)

    def q_sample(self, x0, t, eps):                        # Eq. (14)
        a = self.abar[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def loss(self, net, x0):                                # Eq. (16)
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        return F.mse_loss(net(self.q_sample(x0, t, eps), t), eps)

    def mean(self, net, x, i):                              # Eq. (18) == Eq. (26) with s = -eps/sqrt(1-abar)
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        eps = net(x, t)
        return (x - self.betas[i] / (1 - self.abar[i]).sqrt() * eps) / self.alphas[i].sqrt()


def _resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


@torch.no_grad()
def red_denoise(net, ddpm, v, t_start=1, n_steps=1, size=(128, 128), well=None, gamma=0.0, generator=None):
    """Sampling_well(v): noise `v` to level t_start, run n_steps reverse steps (Alg. 1 lines 7-16).

    v: [nz, nx] velocity (m/s). Returns the denoised model on the original grid, same units.
    well: optional (W, M) tensors on the original grid ([nz, nx]) -- interpolated well model and Gaussian mask.
    The model is min-max normalised to [-1, 1] (as in training), resampled to `size`, denoised, mapped back.
    """
    nz, nx = v.shape
    lo, hi = v.min(), v.max()
    scale = (hi - lo).clamp_min(1.0)
    x0 = _resize(((2 * (v - lo) / scale) - 1)[None, None], size)
    x = ddpm.q_sample(x0, torch.tensor([t_start], device=v.device),
                      torch.randn(x0.shape, device=v.device, generator=generator))
    if well is not None:
        Wn = _resize(((2 * (well[0] - lo) / scale) - 1)[None, None], size)
        Mn = _resize(well[1][None, None], size)
    for i in range(t_start, t_start - n_steps, -1):
        if well is not None and gamma > 0:
            with torch.enable_grad():
                xi = x.clone().requires_grad_(True)
                mu = ddpm.mean(net, xi, i)
                lw = ((Mn * mu - Mn * Wn) ** 2).sum()       # Eq. (28)
                g = torch.autograd.grad(lw, xi)[0]          # Eq. (29)
            mu = mu.detach() - gamma * g                    # Eq. (30); descent direction (see README)
        else:
            mu = ddpm.mean(net, x, i)
        x = mu if i == t_start - n_steps + 1 else mu + ddpm.betas[i].sqrt() * torch.randn(
            mu.shape, device=v.device, generator=generator)  # Eq. (31), z=0 on the last step
    out = _resize(x, (nz, nx))[0, 0]
    return (out + 1) / 2 * scale + lo


def make_well_guidance(wells, nz, nx, sigma_x, device="cpu"):
    """wells: list of (x_index, velocity_profile[nz]) -> (W, M).
    W: laterally interpolated well model; M: Gaussian mask around the wells (width sigma_x in grid cells)."""
    wells = sorted(wells, key=lambda w: w[0])
    xs = torch.tensor([w[0] for w in wells], dtype=torch.float32)
    prof = torch.stack([torch.as_tensor(w[1], dtype=torch.float32) for w in wells], 1)   # [nz, n]
    x = torch.arange(nx, dtype=torch.float32)
    if len(xs) == 1:
        W = prof[:, :1].expand(nz, nx).clone()
    else:
        idx = torch.bucketize(x, xs).clamp(1, len(xs) - 1)
        x0, x1 = xs[idx - 1], xs[idx]
        w1 = ((x - x0) / (x1 - x0)).clamp(0, 1)
        W = prof[:, idx - 1] * (1 - w1)[None] + prof[:, idx] * w1[None]
    M = torch.stack([torch.exp(-((x - xw) / sigma_x) ** 2 / 2) for xw in xs]).amax(0)[None].expand(nz, nx)
    return W.to(device), M.contiguous().to(device)

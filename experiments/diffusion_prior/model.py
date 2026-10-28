"""Small epsilon-prediction DDPM and attention-only LoRA.

U-Net structure adapted from this repository's g2ifwi-repro@802150e,
not from a natural-image/text-to-image pipeline. All inputs are numerical
single-channel velocity patches with checkpoint-defined normalization.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


def gn(channels):
    return nn.GroupNorm(32, channels)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Sequential(gn(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1))
        self.t = nn.Sequential(nn.SiLU(), nn.Linear(tdim, cout))
        self.c2 = nn.Sequential(gn(cout), nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        return self.c2(self.c1(x) + self.t(e)[:, :, None, None]) + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        self.h, self.norm = heads, gn(c)
        self.qkv, self.proj = nn.Conv2d(c, 3*c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.h, c//self.h, h*w).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, channels=32, mult=(1, 2, 4, 8), n_res=2, heads=4, attn_from=2):
        super().__init__()
        if (type(channels) is not int or channels < 32 or channels % 32
                or type(n_res) is not int or n_res < 1 or type(heads) is not int or heads < 1
                or not mult or mult[0] != 1 or any(type(m) is not int or m < 1 for m in mult)
                or any(channels*m % heads for m in mult)):
            raise ValueError('Invalid U-Net channels/multipliers/heads/residual-block count')
        self.channels, self.divisor = channels, 2**(len(mult)-1)
        tdim = channels*4
        self.temb = nn.Sequential(nn.Linear(channels, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(1, channels, 3, padding=1)
        self.down, chans = nn.ModuleList(), [channels]
        c = channels
        for lvl, m in enumerate(mult):
            for _ in range(n_res):
                blk = nn.ModuleList([ResBlock(c, channels*m, tdim)])
                c = channels*m
                if lvl >= attn_from:
                    blk.append(Attn(c, heads))
                self.down.append(blk)
                chans.append(c)
            if lvl < len(mult)-1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                chans.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, tdim), Attn(c, heads), ResBlock(c, c, tdim)])
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mult))):
            for i in range(n_res+1):
                blk = nn.ModuleList([ResBlock(c+chans.pop(), channels*m, tdim)])
                c = channels*m
                if lvl >= attn_from:
                    blk.append(Attn(c, heads))
                if lvl > 0 and i == n_res:
                    blk.extend([nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c, c, 3, padding=1)])
                self.up.append(blk)
        self.out = nn.Sequential(gn(c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, t):
        if x.ndim != 4 or x.shape[1] != 1 or any(n % self.divisor for n in x.shape[-2:]):
            raise ValueError('U-Net expects [B,1,H,W] divisible by its downsampling factor')
        if t.shape != (x.shape[0],):
            raise ValueError('One timestep per input patch is required')
        e = self.temb(timestep_embedding(t, self.channels))
        h, hs = self.inp(x), []
        hs.append(h)
        for blk in self.down:
            h = blk[0](h, e) if isinstance(blk[0], ResBlock) else blk[0](h)
            for extra in blk[1:]:
                h = extra(h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, e)), e)
        for blk in self.up:
            h = blk[0](torch.cat([h, hs.pop()], 1), e)
            for extra in blk[1:]:
                h = extra(h)
        return self.out(h)


class DDPM(nn.Module):
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=.02):
        super().__init__()
        if (type(steps) is not int or steps < 2 or not 0 < beta_start <= beta_end < 1
                or not math.isfinite(beta_start+beta_end)):
            raise ValueError('Invalid diffusion schedule')
        self.steps = steps
        self.config = dict(steps=steps, beta_start=float(beta_start), beta_end=float(beta_end))
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, steps))
        self.register_buffer('abar', (1-self.betas).cumprod(0))
        if not torch.all((self.abar > 0) & (self.abar < 1)):
            raise ValueError('Diffusion schedule is not representable safely in float32')

    def coefficients(self, t):
        a = self.abar[t][:, None, None, None]
        return a.sqrt(), (1-a).sqrt()

    def q_sample(self, x, t, noise):
        a, s = self.coefficients(t)
        return a*x+s*noise

    def predict_clean(self, x_t, t, predicted_noise):
        a, s = self.coefficients(t)
        return (x_t-s*predicted_noise)/a

    def loss(self, net, x, generator=None):
        t = torch.randint(self.steps, (x.shape[0],), device=x.device, generator=generator)
        eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return F.mse_loss(net(self.q_sample(x, t, eps), t), eps)


class LoRA1x1(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if (not isinstance(base, nn.Conv2d) or base.kernel_size != (1, 1)
                or base.stride != (1, 1) or base.padding != (0, 0) or base.groups != 1):
            raise ValueError('LoRA only supports the standard 1x1 attention projections')
        if (type(rank) is not int or not 1 <= rank <= min(base.in_channels, base.out_channels)
                or not math.isfinite(alpha) or alpha <= 0):
            raise ValueError('Invalid LoRA rank/alpha')
        self.base = base.requires_grad_(False)
        self.scale = float(alpha)/rank
        kw = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_down = nn.Conv2d(base.in_channels, rank, 1, bias=False, **kw)
        self.lora_up = nn.Conv2d(rank, base.out_channels, 1, bias=False, **kw)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x)+self.scale*self.lora_up(self.lora_down(x))


def build_model(config):
    """Use the prior's recorded backbone; preserve older small checkpoints."""
    config = dict(config)
    backbone = config.pop('backbone','small')
    if backbone == 'small':
        return UNet(**config)
    if backbone == 'g2_author':
        from .author_unet import G2AuthorUNet
        return G2AuthorUNet(**config)
    raise ValueError(f'Unknown diffusion backbone: {backbone}')


def inject_lora(net, rank=4, alpha=4.):
    if any(isinstance(m, LoRA1x1) for m in net.modules()):
        raise ValueError('LoRA is already attached')
    from .author_unet import AttentionBlock
    blocks = [m for m in net.modules() if isinstance(m, (Attn, AttentionBlock))]
    if not blocks:
        raise ValueError('No compatible attention blocks')
    # Validate every target before modifying the module tree.
    if type(rank) is not int or rank < 1 or any(rank > b.qkv.in_channels for b in blocks):
        raise ValueError('Invalid LoRA rank')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Invalid LoRA alpha')
    net.requires_grad_(False)
    for m in blocks:
        m.qkv = LoRA1x1(m.qkv, rank, alpha)
        m.proj = LoRA1x1(m.proj, rank, alpha)
    return [p for p in net.parameters() if p.requires_grad]


def adapter_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if '.lora_' in k}

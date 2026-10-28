"""G2IFWI-author-compatible 2-D diffusion UNet with optional activation checkpointing.

Author's release: G2IFWI_release/synthetic/core/diffusion/diffusion.py
Architecture derived from https://github.com/LinXueyuanStdio/PyTorch-DDPM
G2IFWI code archive describes MIT licensing. No pretrained model is included.
"""
from __future__ import annotations
import math
from abc import abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int, optional): controls the minimum frequency of the embeddings. Defaults to 10000.

    Returns:
        Tensor: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def norm_layer(channels):
    return nn.GroupNorm(32, channels)



class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        """
        Attention block with shortcut

        Args:
            channels (int): channels
            num_heads (int, optional): attention heads. Defaults to 1.
        """
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, t):
        """
        Apply the module to `x` given `t` timestep embeddings.
        """
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """

    def forward(self, x, t):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        return x


class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)



class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding
    """

    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.FloatTensor, timesteps: torch.LongTensor):
        """Apply the model to an input batch.

        Args:
            x (Tensor): [N x C x H x W]
            timesteps (Tensor): [N,] a 1-D batch of timesteps.

        Returns:
            Tensor: [N x C x ...]
        """
        hs = []
        # down stage
        h: torch.FloatTensor = x
        t: torch.FloatTensor = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        for module in self.down_blocks:
            h = module(h, t)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, t)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, t)
        return self.out(h)





class G2AuthorUNet(UNetModel):
    """Author-compatible state_dict keys; optional gradient checkpointing."""
    gradient_checkpointing = False

    def forward(self, x, timesteps):
        if not self.gradient_checkpointing or not self.training or not torch.is_grad_enabled():
            return super().forward(x, timesteps)
        if x.ndim != 4 or x.shape[1] != self.in_channels or x.shape[0] != timesteps.shape[0]:
            raise ValueError('G2AuthorUNet input must be [B,C,H,W] with B timesteps')
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        hidden, skips = x, []
        for block in self.down_blocks:
            hidden = checkpoint(block, hidden, emb, use_reentrant=False)
            skips.append(hidden)
        hidden = checkpoint(self.middle_block, hidden, emb, use_reentrant=False)
        for block in self.up_blocks:
            hidden = checkpoint(block, torch.cat([hidden, skips.pop()], dim=1), emb,
                                use_reentrant=False)
        return self.out(hidden)


def paper_configuration(model_channels=128):
    if type(model_channels) is not int or model_channels < 32 or model_channels % 32:
        raise ValueError('model_channels must be a multiple of 32 >=32')
    return dict(backbone='g2_author', in_channels=1, out_channels=1,
                model_channels=model_channels, channel_mult=[1,2,4,8,16],
                num_res_blocks=3, attention_resolutions=[8,16], num_heads=4)


def memory_requirements(config, mode='base_train', microbatch=1, patch_size=128):
    """Preallocation memory audit, conservative estimate rather than measured peak."""
    if mode not in ('base_train','adapt','inference') or type(microbatch) is not int or microbatch < 1:
        raise ValueError('Invalid memory audit arguments')
    from .model import build_model
    with torch.device('meta'):
        network = build_model(config)
    count = sum(p.numel() for p in network.parameters())
    # Base train: weight, gradient, 2x Adam state, EMA = 20 FP32 bytes/parameter.
    static = count * (20 if mode == 'base_train' else 4)
    # A 6 GiB activation reserve *per sample* made microbatch 2/4 impossible
    # even on lightly loaded 16 GiB GPUs. Preserve fixed and per-sample
    # headroom, and recommend a real CUDA batch probe before increasing it.
    if mode == 'base_train':
        reserve = (2 + 1.5*microbatch*(patch_size/128.)**2)*2**30
    else:
        # Keep full-author LoRA adaptation / inference budgets conservative.
        reserve = {'inference':3,'adapt':10}[mode]*2**30*microbatch*(patch_size/128.)**2
    return dict(parameters=count,static_bytes=int(static),
                activation_reserve_bytes=int(reserve),estimated_bytes=int(static+reserve))


def check_memory_budget(config, available_bytes, mode='base_train', microbatch=1,
                        patch_size=128, safety_fraction=.80):
    """Reject model before full allocation when 16 GiB is evidently insufficient."""
    if not 0 < safety_fraction <= 1 or available_bytes <= 0:
        raise ValueError('Invalid memory budget')
    report = memory_requirements(config,mode,microbatch,patch_size)
    if report['estimated_bytes'] > available_bytes*safety_fraction:
        raise MemoryError(
            f'G2IFWI estimated {report["estimated_bytes"]/2**30:.1f} GiB required, '
            f'{available_bytes/2**30:.1f} GiB available (e.g. 16 GiB). '
            'Run experiments.diffusion_prior.batch_probe to calibrate batch size first; '
            'the 128-channel paper model needs much larger GPU/CPU offload.')
    return report

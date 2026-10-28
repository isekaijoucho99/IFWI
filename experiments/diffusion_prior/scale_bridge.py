"""Physical-coordinate sampling of an implicit velocity field for a fixed-scale diffusion prior.

The physical finite-difference grid is unchanged. The prior is evaluated on a
second lattice at its training spacing, with gradients through the SAME IRN.
The fixed initial background and (optionally) warped IRN coordinates are
bilinearly interpolated in physical space; the learned IRN output itself is
queried directly and is never resized as an image.

This first implementation only permits the prior lattice to be at least as
fine as the physical grid: downsampling a high-frequency field requires an
explicit antialiasing policy and is deliberately rejected.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

MAX_PRIOR_PIXELS = 4_000_000


def canonical_shape(source_shape, source_spacing_m, prior_spacing_m):
    """Number of grid points at exact prior spacing, never extrapolating outside the PDE domain."""
    if (len(source_shape) != 2 or any(type(n) is not int or n < 2 for n in source_shape)
            or isinstance(source_spacing_m, bool) or isinstance(prior_spacing_m, bool)
            or not all(math.isfinite(x) and x > 0 for x in (source_spacing_m, prior_spacing_m))):
        raise ValueError('Invalid source grid shape or physical spacing')
    if prior_spacing_m > source_spacing_m * (1 + 1e-8):
        raise ValueError('A coarser prior lattice requires antialiasing and is unsupported')
    # Grid point zero is the common origin. Last prior pixel is at or inside the PDE boundary.
    nz, nx = source_shape
    result = tuple(int(math.floor((n-1)*source_spacing_m / prior_spacing_m + 1e-8))+1
                   for n in (nz, nx))
    if result[0]*result[1] > MAX_PRIOR_PIXELS:
        raise ValueError('Canonical prior grid exceeds safe pixel budget')
    return result


def query_velocity(model, source_spacing_m, prior_spacing_m):
    """Query residual/absolute SIREN at exact physical prior-grid coordinates (m/s).

    No changes to model.coords or the optimizer; accepts warped SIREN coordinates
    by sampling the existing warp, and also accepts Fourier-reparameterized IRNs.
    """
    from ifwi_modules import IRN

    background = getattr(model, 'fixed_init_mps', None)
    coords = getattr(model, 'coords', None)
    net = getattr(model, 'net', None)
    if (background is None or coords is None or net is None
            or getattr(model, 'low_rank', None) is not None
            or not isinstance(net, IRN)
            or getattr(model, 'parameterization', None) not in ('residual', 'absolute')):
        raise ValueError('Physical-scale bridge requires a full 2-D SIREN velocity representation')
    if background.ndim != 3 or background.shape[0] != 1 or coords.shape != (*background.shape, 2):
        raise ValueError('SIREN background/coordinate-grid shapes are incompatible')
    source_shape = tuple(background.shape[-2:])
    target_shape = canonical_shape(source_shape, source_spacing_m, prior_spacing_m)
    if target_shape == source_shape and math.isclose(source_spacing_m, prior_spacing_m, rel_tol=1e-8):
        return model()

    nz, nx = source_shape
    tz, tx = target_shape
    xx = torch.arange(tx, device=coords.device, dtype=torch.float64) * (prior_spacing_m / ((nx-1)*source_spacing_m))
    zz = torch.arange(tz, device=coords.device, dtype=torch.float64) * (prior_spacing_m / ((nz-1)*source_spacing_m))
    gy, gx = torch.meshgrid(zz, xx, indexing='ij')
    grid = (torch.stack((gx * 2 - 1, gy * 2 - 1), dim=-1)
            .clamp(-1, 1).to(dtype=coords.dtype).unsqueeze(0))

    sampled_input_coords = F.grid_sample(
        coords.permute(0, 3, 1, 2).contiguous(), grid, mode='bilinear',
        padding_mode='border', align_corners=True).permute(0, 2, 3, 1)
    sampled_background = F.grid_sample(
        background[:, None], grid, mode='bilinear', padding_mode='border',
        align_corners=True)[:, 0]

    raw, _ = net(sampled_input_coords)
    raw = raw.squeeze(-1)
    if model.parameterization == 'residual':
        return sampled_background + raw * (1000. * model.std_kmps)
    return (raw * model.std_kmps + model.mean_kmps) * 1000.

"""Frozen native-grid patch score guidance, with explicit numerical controls.

Coverage averaging is a spatial preconditioner on the sum of patch scores.
The resulting field need not be the gradient of a single scalar energy.
The dot-product proxy below only injects that field through the IFWI Jacobian.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F

from .io import load_prior, validate_metadata

DEFAULTS = dict(enabled=True, mode='diffusion', checkpoint=None, adapter=None,
                warmup=50, every=5, seed=0, ratio=.025, fixed_weight=None,
                eta_start=.2, eta_end=.02, anneal_updates=400,
                patch_stride=None, patch_batch_size=4, weighting='kl',
                allow_untrained=False, resize_shape=[128, 128], scale_mps=1500.)


def validate_config(setting):
    if setting is None:
        return None
    if not isinstance(setting, dict) or set(setting)-set(DEFAULTS):
        raise ValueError('Unknown diffusion_prior configuration keys')
    cfg = {**DEFAULTS, **setting}
    for key in ('enabled', 'allow_untrained'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'diffusion_prior.{key} must be bool')
    if not cfg['enabled']:
        return cfg
    if cfg['mode'] not in ('diffusion', 'resize_only') or cfg['weighting'] not in ('kl', 'epsilon'):
        raise ValueError('Unsupported prior mode/weighting')
    for key in ('warmup', 'seed', 'every', 'anneal_updates', 'patch_batch_size'):
        lower = 0 if key in ('warmup', 'seed') else 1
        if type(cfg[key]) is not int or cfg[key] < lower:
            raise ValueError(f'Invalid diffusion_prior.{key}')
    for key in ('ratio', 'eta_start', 'eta_end', 'scale_mps'):
        if (isinstance(cfg[key], bool) or not isinstance(cfg[key], (float, int))
                or not math.isfinite(cfg[key])):
            raise ValueError(f'Invalid diffusion_prior.{key}')
    if not 0 <= cfg['ratio'] < 1 or not 0 < cfg['eta_end'] <= cfg['eta_start'] or cfg['scale_mps'] <= 0:
        raise ValueError('Invalid gradient ratio, noise annealing interval, or velocity scale')
    w = cfg['fixed_weight']
    if w is not None and (isinstance(w, bool) or not isinstance(w, (float, int)) or not math.isfinite(w) or w < 0):
        raise ValueError('fixed_weight must be a finite nonnegative number')
    if cfg['patch_stride'] is not None and (type(cfg['patch_stride']) is not int or cfg['patch_stride'] < 1):
        raise ValueError('Invalid patch_stride')
    if (not isinstance(cfg['resize_shape'], (tuple, list)) or len(cfg['resize_shape']) != 2
            or any(type(n) is not int or n < 2 for n in cfg['resize_shape'])):
        raise ValueError('resize_shape must contain two positive dimensions >= 2')
    if cfg['mode'] == 'diffusion':
        if not isinstance(cfg['checkpoint'], str) or not cfg['checkpoint']:
            raise ValueError('A checkpoint is required for diffusion guidance')
        if cfg['adapter'] is not None and (not isinstance(cfg['adapter'], str) or not cfg['adapter']):
            raise ValueError('adapter must be a checkpoint path or null')
    elif cfg['checkpoint'] is not None or cfg['adapter'] is not None:
        raise ValueError('resize_only control must not load a neural prior')
    return cfg


def patch_origins(height, width, patch, stride):
    if (type(patch) is not int or type(stride) is not int or patch < 2
            or stride < 1 or stride > patch or min(height, width) < patch):
        raise ValueError('Invalid patch/stride or grid smaller than patch; no implicit resize/padding')
    def starts(n):
        result = list(range(0, n-patch+1, stride))
        if result[-1] != n-patch:
            result.append(n-patch)
        return result
    return [(z, x) for z in starts(height) for x in starts(width)]


@torch.no_grad()
def patch_direction(velocity, net, ddpm, meta, timestep, noise, stride, batch_size, weighting):
    validate_metadata(meta)
    if velocity.ndim != 3 or velocity.shape[0] != 1 or not torch.isfinite(velocity).all():
        raise ValueError('Velocity must be finite [1,nz,nx] in m/s')
    if noise.shape != velocity.shape or not torch.isfinite(noise).all():
        raise ValueError('Noise shape/finiteness differs from velocity')
    if type(timestep) is not int or not 0 <= timestep < ddpm.steps:
        raise ValueError('Invalid diffusion timestep')
    if type(batch_size) is not int or batch_size < 1 or weighting not in ('kl', 'epsilon'):
        raise ValueError('Invalid patch batch size or score weighting')
    p = meta['patch_size']
    origins = patch_origins(*velocity.shape[-2:], p, stride)
    x = (velocity-meta['mean_mps'])/meta['scale_mps']
    score, coverage = torch.zeros_like(x), torch.zeros_like(x)
    a, sigma = ddpm.abar[timestep].sqrt(), (1-ddpm.abar[timestep]).sqrt()
    calls = 0
    for first in range(0, len(origins), batch_size):
        group = origins[first:first+batch_size]
        clean = torch.stack([x[:, z:z+p, q:q+p] for z, q in group])
        eps = torch.stack([noise[:, z:z+p, q:q+p] for z, q in group])
        t = torch.full((len(group),), timestep, device=x.device, dtype=torch.long)
        prediction = net(a*clean+sigma*eps, t)
        if prediction.shape != clean.shape or not torch.isfinite(prediction).all():
            raise FloatingPointError('Invalid diffusion noise prediction')
        field = prediction-eps
        if weighting == 'kl':
            field = field*(a/sigma)
        for i, (z, q) in enumerate(group):
            score[:, z:z+p, q:q+p] += field[i]
            coverage[:, z:z+p, q:q+p] += 1
        calls += 1
    if coverage.min() < 1:
        raise RuntimeError('Patch implementation left uncovered grid cells')
    score /= coverage
    direction = score/(meta['scale_mps']*velocity.numel())
    return direction, dict(patches=len(origins), coverage_min=float(coverage.min()),
                           unet_calls=calls, timestep=timestep, eta=float(sigma/a),
                           score_rms=float(score.square().mean().sqrt()))


def _norm(grads):
    return torch.stack([g.detach().double().square().sum() for g in grads]).sum().sqrt()


def add_guidance(model, direction, ratio=.025, fixed_weight=None):
    """Accumulate a frozen velocity field into already accumulated parameter grads."""
    if not 0 <= ratio < 1:
        raise ValueError('ratio must be in [0,1)')
    if fixed_weight is not None and (not math.isfinite(fixed_weight) or fixed_weight < 0):
        raise ValueError('Invalid fixed_weight')
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.grad is None for p in params):
        raise RuntimeError('All trainable IFWI parameters must have physical gradients first')
    data = [p.grad for p in params]
    nd = _norm(data)
    if not torch.isfinite(nd):
        raise FloatingPointError('Non-finite physical parameter gradient')
    report = dict(data_gradient_norm=float(nd), prior_gradient_norm=0.,
                  prior_lambda=0., effective_ratio=0., gradient_cosine=None)
    if (fixed_weight == 0) or (fixed_weight is None and (nd == 0 or ratio == 0)):
        return report
    velocity = model()
    if direction.shape != velocity.shape or not torch.isfinite(direction).all():
        raise ValueError('Prior field shape/finiteness mismatch')
    proxy = (velocity*direction.detach()).sum()
    grads = torch.autograd.grad(proxy, params, allow_unused=False)
    ng = _norm(grads)
    if not torch.isfinite(ng):
        raise FloatingPointError('Non-finite prior parameter gradient')
    report['prior_gradient_norm'] = float(ng)
    if ng == 0:
        return report
    dot = torch.stack([(a.detach().double()*b.detach().double()).sum() for a, b in zip(data, grads)]).sum()
    weight = float(fixed_weight) if fixed_weight is not None else float(ratio*nd/ng)
    report.update(prior_lambda=weight, effective_ratio=float(weight*ng/nd) if nd > 0 else None,
                  gradient_cosine=float(dot/(ng*nd)) if nd > 0 else None)
    # Validate every result before modifying any accumulated gradient.
    new_grads = [a + weight*b for a, b in zip(data, grads)]
    if any(not torch.isfinite(g).all() for g in new_grads):
        raise FloatingPointError('Non-finite combined gradient')
    with torch.no_grad():
        for p, g in zip(params, new_grads):
            p.grad.copy_(g)
    return report


class FrozenPrior:
    def __init__(self, cfg, device, grid_spacing_m, root=None):
        self.cfg, self.device = cfg, torch.device(device)
        self.net = self.ddpm = self.meta = None
        self.info = dict(mode=cfg['mode'])
        if cfg['mode'] == 'diffusion':
            def resolve(p):
                return str(Path(root)/p) if root is not None and not Path(p).is_absolute() else p
            self.net, self.ddpm, self.meta, info = load_prior(
                resolve(cfg['checkpoint']), device, grid_spacing_m,
                adapter=resolve(cfg['adapter']) if cfg['adapter'] else None,
                allow_untrained=cfg['allow_untrained'])
            self.info.update(info, normalization=self.meta)
            self.stride = cfg['patch_stride'] or max(1, self.meta['patch_size']//2)
            if self.stride > self.meta['patch_size']:
                raise ValueError('patch_stride exceeds checkpoint patch size')
            self.etas = ((1-self.ddpm.abar)/self.ddpm.abar).sqrt()
            if cfg['eta_end'] < float(self.etas.min()) or cfg['eta_start'] > float(self.etas.max()):
                raise ValueError('Requested eta interval lies outside the checkpoint diffusion schedule')

    def active(self, update):
        return (update > self.cfg['warmup'] and
                (update-self.cfg['warmup']-1) % self.cfg['every'] == 0)

    def timestep(self, update):
        fraction = min(1., max(0., (update-self.cfg['warmup']-1)/max(1, self.cfg['anneal_updates']-1)))
        eta = math.exp((1-fraction)*math.log(self.cfg['eta_start'])+fraction*math.log(self.cfg['eta_end']))
        return int((self.etas.log()-math.log(eta)).abs().argmin())

    @torch.no_grad()
    def direction(self, velocity, update):
        if self.cfg['mode'] == 'resize_only':
            x = F.interpolate(velocity[:, None], self.cfg['resize_shape'], mode='bilinear', align_corners=True)
            x = F.interpolate(x, velocity.shape[-2:], mode='bilinear', align_corners=True)[:, 0]
            return ((velocity-x)/(self.cfg['scale_mps']**2*velocity.numel()),
                    dict(patches=0, unet_calls=0, timestep=None, eta=None))
        # CPU generator makes noise independent of global RNG and CUDA device indices.
        seed = (self.cfg['seed']+1000003*int(update)) % (2**63-1)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn(velocity.shape, generator=generator, dtype=velocity.dtype).to(velocity.device)
        return patch_direction(velocity, self.net, self.ddpm, self.meta, self.timestep(update), noise,
                               self.stride, self.cfg['patch_batch_size'], self.cfg['weighting'])

    def apply(self, model, update):
        if not self.active(update):
            return dict(active=False, prior_seconds=0., unet_calls=0)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            direction, info = self.direction(model().detach(), update)
        report = add_guidance(model, direction, self.cfg['ratio'], self.cfg['fixed_weight'])
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return dict(active=True, prior_seconds=time.perf_counter()-started, **info, **report)


def build_prior(setting, device, grid_spacing_m, root=None):
    cfg = validate_config(setting)
    if cfg is None or not cfg['enabled']:
        return None
    return FrozenPrior(cfg, device, grid_spacing_m, root)

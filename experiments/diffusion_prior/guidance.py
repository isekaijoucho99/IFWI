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
from .scale_bridge import canonical_shape, query_velocity

DEFAULTS = dict(enabled=True, mode='diffusion', checkpoint=None, adapter=None,
                warmup=50, every=5, seed=0, ratio=.025, fixed_weight=None,
                eta_start=.2, eta_end=.02, anneal_updates=400,
                patch_stride=None, patch_batch_size=4, weighting='kl',
                allow_untrained=False, resize_shape=[128, 128], scale_mps=1500.,
                physical_scale_bridge=False, red_steps=1, red_image_size=128,
                red_geometry='resize', red_normalization='fixed')


def validate_config(setting):
    if setting is None:
        return None
    if not isinstance(setting, dict) or set(setting)-set(DEFAULTS):
        raise ValueError('Unknown diffusion_prior configuration keys')
    cfg = {**DEFAULTS, **setting}
    for key in ('enabled', 'allow_untrained', 'physical_scale_bridge'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'diffusion_prior.{key} must be bool')
    if not cfg['enabled']:
        return cfg
    if cfg['mode'] not in ('diffusion', 'g2_red', 'resize_only') or cfg['weighting'] not in ('kl', 'epsilon'):
        raise ValueError('Unsupported prior mode/weighting')
    for key in ('warmup', 'seed', 'every', 'anneal_updates', 'patch_batch_size', 'red_steps', 'red_image_size'):
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
    if cfg['red_geometry'] not in ('resize','patch') or cfg['red_normalization'] not in ('fixed','per_model'):
        raise ValueError('Invalid G2 RED geometry or normalization')
    if cfg['mode'] in ('diffusion','g2_red'):
        if not isinstance(cfg['checkpoint'], str) or not cfg['checkpoint']:
            raise ValueError('A checkpoint is required for diffusion guidance')
        if cfg['adapter'] is not None and (not isinstance(cfg['adapter'], str) or not cfg['adapter']):
            raise ValueError('adapter must be a checkpoint path or null')
    elif cfg['checkpoint'] is not None or cfg['adapter'] is not None or cfg['physical_scale_bridge']:
        raise ValueError('resize_only control cannot load a prior or use physical_scale_bridge')
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


@torch.no_grad()
def g2_red_direction(velocity, net, ddpm, meta, *, image_size=128, steps=1,
                     geometry='resize', normalization='fixed', patch_batch_size=1, stride=None):
    """G2IFWI short deterministic reverse-DDPM RED direction v-D(v) (m/s).

    The author's fwi_sample_wlr *mean* path starts from the current normalized
    velocity; it does not add fresh q_sample noise, and iterates t=steps-1,...,0.
    Resizing to square reproduces the author Overthrust geometry. This method
    is not the prior score-gradient formula used by the other branch.
    """
    validate_metadata(meta)
    if velocity.ndim != 3 or velocity.shape[0] != 1 or not torch.isfinite(velocity).all():
        raise ValueError('Velocity must be finite [1,nz,nx]')
    if type(steps) is not int or not 1 <= steps <= ddpm.steps:
        raise ValueError('Invalid reverse diffusion steps')
    if type(image_size) is not int or image_size < 2 or geometry not in ('resize','patch'):
        raise ValueError('Invalid RED geometry')
    if normalization not in ('fixed','per_model') or type(patch_batch_size) is not int or patch_batch_size < 1:
        raise ValueError('Invalid RED normalization or patch batch size')
    if normalization == 'fixed':
        center, scale = meta['mean_mps'], meta['scale_mps']
        x = (velocity-center)/scale
    else:
        lo, hi = velocity.amin(), velocity.amax()
        scale = (hi-lo).clamp_min(1.)/2
        center = (lo+hi)/2
        x = (velocity-center)/scale

    def reverse(current):
        for i in range(steps-1,-1,-1):
            t = torch.full((current.shape[0],),i,device=current.device,dtype=torch.long)
            a_t = ddpm.abar[i]
            a_prev = ddpm.abar[i-1] if i else torch.ones_like(a_t)
            alpha_t = 1 - ddpm.betas[i]
            eps = net(current,t)
            if eps.shape != current.shape or not torch.isfinite(eps).all():
                raise FloatingPointError('Non-finite G2IFWI denoiser prediction')
            clean = ((current - (1-a_t).sqrt()*eps)/a_t.sqrt()).clamp(-1,1)
            # For t=0 posterior mean is exactly the clean estimate; this
            # prevents FP32 coefficient rounding from leaving a fake residual.
            current = (clean if i == 0 else
                       (ddpm.betas[i]*a_prev.sqrt()/(1-a_t))*clean
                       +((1-a_prev)*alpha_t.sqrt()/(1-a_t))*current)
        return current

    if geometry == 'resize':
        resized = F.interpolate(x[:,None],size=(image_size,image_size),
                                mode='bilinear',align_corners=False)
        restored = F.interpolate(reverse(resized),size=x.shape[-2:],
                                 mode='bilinear',align_corners=False)[:,0]
        calls, patches = steps, 1
    else:
        patch = meta['patch_size']
        if image_size != patch:
            raise ValueError('red_image_size must equal patch_size for native RED patches')
        origins = patch_origins(*x.shape[-2:],patch,stride or max(1,patch//2))
        acc, counts = torch.zeros_like(x), torch.zeros_like(x)
        calls = 0
        for first in range(0,len(origins),patch_batch_size):
            group = origins[first:first+patch_batch_size]
            batch = torch.stack([x[:,z:z+patch,q:q+patch] for z,q in group])
            prediction = reverse(batch)
            for k,(z,q) in enumerate(group):
                acc[:,z:z+patch,q:q+patch] += prediction[k]
                counts[:,z:z+patch,q:q+patch] += 1
            calls += steps
        restored = acc/counts.clamp_min(1.)
        patches = len(origins)
    denoised = restored*scale+center
    direction = velocity-denoised
    if not torch.isfinite(direction).all():
        raise FloatingPointError('Non-finite RED prior direction')
    return direction, dict(red_steps=steps,red_geometry=geometry,
        red_normalization=normalization,unet_calls=calls,patches=patches,
        timestep=steps-1,eta=None,red_residual_rms_mps=float(direction.square().mean().sqrt()))


def _norm(grads):
    return torch.stack([g.detach().double().square().sum() for g in grads]).sum().sqrt()


def add_guidance(model, direction, ratio=.025, fixed_weight=None, *, velocity_fn=None):
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
    velocity = model() if velocity_fn is None else velocity_fn()
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
        self.source_spacing_m = float(grid_spacing_m)
        self.net = self.ddpm = self.meta = None
        self.info = dict(mode=cfg['mode'])
        if cfg['mode'] in ('diffusion','g2_red'):
            def resolve(p):
                return str(Path(root)/p) if root is not None and not Path(p).is_absolute() else p
            self.net, self.ddpm, self.meta, info = load_prior(
                resolve(cfg['checkpoint']), device, grid_spacing_m,
                adapter=resolve(cfg['adapter']) if cfg['adapter'] else None,
                allow_untrained=cfg['allow_untrained'],
                allow_grid_mismatch=cfg['physical_scale_bridge'])
            self.info.update(info, normalization=self.meta,
                             physical_scale_bridge=cfg['physical_scale_bridge'],
                             source_spacing_m=self.source_spacing_m)
            self.stride = cfg['patch_stride'] or max(1, self.meta['patch_size']//2)
            if self.stride > self.meta['patch_size']:
                raise ValueError('patch_stride exceeds checkpoint patch size')
            self.etas = ((1-self.ddpm.abar)/self.ddpm.abar).sqrt()
            if cfg['mode'] == 'diffusion' and (cfg['eta_end'] < float(self.etas.min()) or cfg['eta_start'] > float(self.etas.max())):
                raise ValueError('Requested eta interval lies outside the checkpoint diffusion schedule')

    def grid_shape(self, shape):
        if self.meta is None or not self.cfg['physical_scale_bridge']:
            return tuple(shape)
        return canonical_shape(tuple(shape), self.source_spacing_m, self.meta['grid_spacing_m'])

    def query_velocity(self, model):
        if self.meta is None or not self.cfg['physical_scale_bridge']:
            return model()
        return query_velocity(model, self.source_spacing_m, self.meta['grid_spacing_m'])

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
        if self.cfg['mode'] == 'g2_red':
            return g2_red_direction(velocity, self.net, self.ddpm, self.meta,
                                    image_size=self.cfg['red_image_size'],
                                    steps=self.cfg['red_steps'],
                                    geometry=self.cfg['red_geometry'],
                                    normalization=self.cfg['red_normalization'],
                                    patch_batch_size=self.cfg['patch_batch_size'],
                                    stride=self.stride)
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
            prior_velocity = self.query_velocity(model).detach()
            direction, info = self.direction(prior_velocity, update)
        report = add_guidance(model, direction, self.cfg['ratio'], self.cfg['fixed_weight'],
                              velocity_fn=lambda: self.query_velocity(model))
        if self.meta is not None:
            info['prior_grid_shape'] = list(prior_velocity.shape[-2:])
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        return dict(active=True, prior_seconds=time.perf_counter()-started, **info, **report)


def build_prior(setting, device, grid_spacing_m, root=None):
    cfg = validate_config(setting)
    if cfg is None or not cfg['enabled']:
        return None
    return FrozenPrior(cfg, device, grid_spacing_m, root)

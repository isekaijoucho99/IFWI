"""Versioned inference checkpoints. Legacy min-max checkpoints are not accepted."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import torch

from .model import build_model, DDPM, inject_lora, adapter_state

BASE_SCHEMA = 'ifwi_fixed_scale_ddpm_v1'
ADAPTER_SCHEMA = 'ifwi_attention_lora_v1'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def validate_metadata(meta, model_config=None):
    keys = {'units', 'mean_mps', 'scale_mps', 'grid_spacing_m', 'patch_size'}
    if not isinstance(meta, dict) or set(meta) != keys or meta['units'] != 'm/s':
        raise ValueError('Checkpoint metadata must explicitly define fixed m/s normalization and grid spacing')
    for k in ('mean_mps', 'scale_mps', 'grid_spacing_m'):
        if isinstance(meta[k], bool) or not isinstance(meta[k], (float, int)) or not math.isfinite(meta[k]):
            raise ValueError(f'Invalid metadata {k}')
    if meta['scale_mps'] <= 0 or meta['grid_spacing_m'] <= 0:
        raise ValueError('Normalization scale and grid spacing must be positive')
    p = meta['patch_size']
    if type(p) is not int or p < 2:
        raise ValueError('Invalid patch size')
    if model_config is not None:
        divisor = 2**(len(model_config.get('mult', model_config.get('channel_mult', [1, 2, 4, 8])))-1)
        if p % divisor or p // divisor < 2:
            raise ValueError('Patch size must fit U-Net downsampling (at least 2x2 at bottleneck)')
    return meta


def save_base(path, net, model_config, schedule_config, meta, updates, training=None):
    validate_metadata(meta, model_config)
    if type(updates) is not int or updates < 0:
        raise ValueError('Invalid update count')
    if adapter_state(net):
        raise ValueError('Save LoRA separately with save_adapter, not as base weights')
    atomic_save(dict(schema=BASE_SCHEMA, model_config=model_config, schedule_config=schedule_config,
                     normalization=meta, updates=updates,
                     model={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                     training=training or {}), path)


def save_adapter(path, net, base_path, rank, alpha, updates, training=None):
    state = adapter_state(net)
    if not state:
        raise ValueError('No LoRA parameters found')
    atomic_save(dict(schema=ADAPTER_SCHEMA, base_sha256=file_hash(base_path), rank=rank,
                     alpha=float(alpha), updates=updates, adapter=state, training=training or {}), path)


def load_prior(path, device, grid_spacing_m, adapter=None, allow_untrained=False, *, allow_grid_mismatch=False):
    path = Path(path)
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('schema') != BASE_SCHEMA:
        raise ValueError('Unsupported prior checkpoint schema; legacy min-max weights require retraining')
    meta = validate_metadata(ck['normalization'], ck['model_config'])
    if not allow_grid_mismatch and not math.isclose(float(grid_spacing_m), meta['grid_spacing_m'], rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError('Prior grid spacing differs from the inversion grid; no implicit resizing is allowed')
    if type(ck.get('updates')) is not int or ck['updates'] < 0:
        raise ValueError('Invalid prior training count')
    if ck['updates'] == 0 and not allow_untrained:
        raise ValueError('Refusing untrained prior; allow_untrained is only for an explicit control experiment')
    if str(device).startswith('cuda') and torch.cuda.is_available() and ck['model_config'].get('backbone') == 'g2_author':
        from .author_unet import check_memory_budget
        free, _ = torch.cuda.mem_get_info(device)
        check_memory_budget(ck['model_config'],free,mode='inference',microbatch=1,
                            patch_size=meta['patch_size'])
    # Initialization consumes CPU RNG; fork it even though state_dict replaces all weights.
    with torch.random.fork_rng(devices=[]):
        net = build_model(ck['model_config'])
        net.load_state_dict(ck['model'], strict=True)
        ddpm = DDPM(**ck['schedule_config'])
        info = dict(schema=BASE_SCHEMA, base_sha256=file_hash(path), updates=ck['updates'],
                    adapter_sha256=None, model_config=ck['model_config'], schedule_config=ck['schedule_config'],
                    training=ck.get('training', {}))
        if adapter is not None:
            ac = torch.load(adapter, map_location='cpu', weights_only=True)
            if ac.get('schema') != ADAPTER_SCHEMA or ac.get('base_sha256') != info['base_sha256']:
                raise ValueError('Adapter schema/base hash mismatch')
            if type(ac.get('updates')) is not int or ac['updates'] < 1:
                raise ValueError('Adapter must contain completed adaptation updates')
            inject_lora(net, ac['rank'], ac['alpha'])
            expected = adapter_state(net)
            if set(expected) != set(ac['adapter']):
                raise ValueError('Adapter key mismatch')
            for k, v in ac['adapter'].items():
                if v.shape != expected[k].shape or not torch.isfinite(v).all():
                    raise ValueError(f'Invalid adapter tensor {k}')
            net.load_state_dict({**net.state_dict(), **ac['adapter']}, strict=True)
            info.update(adapter_sha256=file_hash(adapter), adapter_updates=ac['updates'],
                        adapter_rank=ac['rank'], adapter_alpha=ac['alpha'],
                        adapter_training=ac.get('training', {}))
    if any(not torch.isfinite(p).all() for p in net.parameters()):
        raise ValueError('Prior checkpoint contains non-finite parameters')
    return net.to(device).eval().requires_grad_(False), ddpm.to(device), meta, info

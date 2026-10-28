"""Offline DDPM pretraining / LoRA adaptation on model-level-separated .npy files.

CLI: python -m experiments.diffusion_prior.training --help
All arrays are numerical [nz,nx] velocities in m/s, not plotted RGB images.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .io import (atomic_save, file_hash, load_prior, save_adapter, save_base,
                 validate_metadata)
from .model import DDPM, UNet, inject_lora

TRAIN_SCHEMA = 'ifwi_prior_training_v1'
DEFAULT_MODEL = dict(channels=32, mult=[1, 2, 4, 8], n_res=2, heads=4, attn_from=2)
DEFAULT_SCHEDULE = dict(steps=1000, beta_start=1e-4, beta_end=.02)
DEFAULT_META = dict(units='m/s', mean_mps=3500., scale_mps=2500., grid_spacing_m=15., patch_size=64)


def _json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def array_digest(array):
    values = np.ascontiguousarray(array, dtype='<f4')
    return hashlib.sha256(str(values.shape).encode()+values.tobytes()).hexdigest()


class VelocityDataset:
    def __init__(self, directory, meta):
        self.meta = validate_metadata(meta)
        directory = Path(directory).resolve()
        manifest = directory.parent/'dataset.json'
        if manifest.exists():
            info = json.loads(manifest.read_text(encoding='utf-8'))
            if info.get('units') != 'm/s' or not math.isclose(info['grid_spacing_m'], meta['grid_spacing_m']):
                raise ValueError('Dataset units/grid spacing disagree with training metadata')
        self.paths = sorted(directory.glob('*.npy'))
        if not self.paths:
            raise ValueError(f'No individual 2-D .npy velocity models in {directory}')
        self.records = []
        for path in self.paths:
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            if a.ndim != 2 or min(a.shape) < meta['patch_size']:
                raise ValueError(f'{path}: expected a 2-D model at least as large as patch_size')
            if not np.isfinite(a).all() or a.min() < 100 or a.max() > 20000:
                raise ValueError(f'{path}: invalid/nonphysical m/s array (check units)')
            self.records.append(dict(path=str(path), shape=list(a.shape), array_sha256=array_digest(a)))

    def sample(self, count, rng):
        patches, size = [], self.meta['patch_size']
        for _ in range(count):
            a = np.load(self.paths[int(rng.integers(len(self.paths)))], mmap_mode='r', allow_pickle=False)
            z = int(rng.integers(a.shape[0]-size+1))
            x = int(rng.integers(a.shape[1]-size+1))
            patch = np.asarray(a[z:z+size, x:x+size], dtype=np.float32)
            if rng.random() < .5:
                patch = patch[:, ::-1]
            patches.append((patch-self.meta['mean_mps'])/self.meta['scale_mps'])
        return torch.from_numpy(np.stack(patches)[:, None].copy())


def generate_dataset(out, train_models=64, val_models=16, shape=(96, 192), grid_spacing_m=15., seed=3, profile='mixed'):
    """Procedural smoke/research data, NOT a reproduction of the G2IFWI training set."""
    if (profile not in ('layered', 'folded', 'mixed') or len(shape) != 2 or min(shape) < 16
            or train_models < 1 or val_models < 1 or not math.isfinite(grid_spacing_m) or grid_spacing_m <= 0):
        raise ValueError('Invalid synthetic dataset configuration')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    nz, nx = shape
    z = np.arange(nz)[:, None]*grid_spacing_m
    x = np.arange(nx)[None, :]*grid_spacing_m
    depth, width = nz*grid_spacing_m, nx*grid_spacing_m
    for split, number, split_seed in [('train', train_models, seed), ('val', val_models, seed+1000003)]:
        folder = out/split
        folder.mkdir()
        rng = np.random.default_rng(split_seed)
        for i in range(number):
            count = int(rng.integers(5, 13))
            interfaces = np.sort(rng.uniform(.07*depth, .93*depth, count-1))
            values = np.linspace(rng.uniform(1600, 2200), rng.uniform(4200, 5800), count)
            values += rng.normal(0, 180, count)
            displacement = rng.uniform(-.08, .08)*(x-width/2)
            folded = profile == 'folded' or (profile == 'mixed' and rng.random() < .65)
            if folded:
                displacement = displacement + rng.uniform(.02, .12)*depth*np.sin(
                    2*np.pi*x/rng.uniform(.3*width, width)+rng.uniform(0, 2*np.pi))
            if profile == 'mixed' and rng.random() < .3:
                displacement = displacement + (x > rng.uniform(.3, .7)*width)*rng.uniform(-.12, .12)*depth
            layer = ((z-displacement)[..., None] > interfaces).sum(-1)
            v = values[layer]
            v = np.clip(gaussian_filter(v, .45), 1400, 6500).astype(np.float32)
            np.save(folder/f'model_{i:05d}.npy', v)
    _json(out/'dataset.json', dict(units='m/s', grid_spacing_m=grid_spacing_m, shape=list(shape),
          seed=seed, profile=profile, train_models=train_models, val_models=val_models,
          description='Procedural independent models; split before cropping; not field-validated geology'))
    return out


def _swap(net, values):
    parameters = dict(net.named_parameters())
    previous = {n: parameters[n].detach().clone() for n in values}
    with torch.no_grad():
        for n, value in values.items():
            parameters[n].copy_(value.to(parameters[n]))
    return previous


@torch.no_grad()
def _validate(net, ddpm, dataset, batch_size, batches, seed, device):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    eps_mse, clean_mse, noisy_mse = [], [], []
    net.eval()
    for i in range(batches):
        x = dataset.sample(batch_size, rng).to(device)
        # Fixed noise and timesteps at every validation event make selections comparable.
        levels = [max(0, min(ddpm.steps-1, int(ddpm.steps*f))) for f in (.02, .1, .3)]
        t = torch.tensor([levels[(i*batch_size+j) % len(levels)] for j in range(batch_size)], device=device)
        eps = torch.randn(x.shape, generator=gen).to(device)
        xt = ddpm.q_sample(x, t, eps)
        pred = net(xt, t)
        a, _ = ddpm.coefficients(t)
        eps_mse.append(float((pred-eps).square().mean()))
        clean_mse.append(float((ddpm.predict_clean(xt, t, pred)-x).square().mean()))
        noisy_mse.append(float((xt/a-x).square().mean()))
    scale = dataset.meta['scale_mps']
    metrics = dict(epsilon_mse=float(np.mean(eps_mse)),
                   clean_rmse_mps=scale*math.sqrt(float(np.mean(clean_mse))),
                   noisy_rmse_mps=scale*math.sqrt(float(np.mean(noisy_mse))))
    if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('Non-finite validation metrics')
    net.train()
    return metrics


def train(train_dir, val_dir, out, *, steps=20000, batch_size=8, lr=2e-4, eval_every=200,
          val_batches=4, seed=3, device='cpu', meta=None, model_config=None, schedule_config=None,
          base=None, rank=4, alpha=4., resume=None, ema_decay=.999):
    for k, v in dict(steps=steps, batch_size=batch_size, eval_every=eval_every, val_batches=val_batches).items():
        if type(v) is not int or v < 1:
            raise ValueError(f'{k} must be a positive integer')
    if not math.isfinite(lr) or lr <= 0 or not 0 <= ema_decay < 1:
        raise ValueError('Invalid learning rate or EMA decay')
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base_hash = None
    if base is not None:
        ck = torch.load(base, map_location='cpu', weights_only=True)
        base_meta = ck.get('normalization')
        if meta is not None and meta != base_meta:
            raise ValueError('LoRA adaptation must preserve base normalization/grid/patch metadata')
        meta = base_meta
        net, ddpm, meta, info = load_prior(base, device, meta['grid_spacing_m'])
        model_config, schedule_config = ck['model_config'], ck['schedule_config']
        params = inject_lora(net, rank, alpha)
        base_hash = info['base_sha256']
    else:
        meta = copy.deepcopy(DEFAULT_META if meta is None else meta)
        model_config = copy.deepcopy(DEFAULT_MODEL if model_config is None else model_config)
        schedule_config = copy.deepcopy(DEFAULT_SCHEDULE if schedule_config is None else schedule_config)
        validate_metadata(meta, model_config)
        net, ddpm = UNet(**model_config).to(device), DDPM(**schedule_config).to(device)
        params = list(net.parameters())
    data, val = VelocityDataset(train_dir, meta), VelocityDataset(val_dir, meta)
    if {r['array_sha256'] for r in data.records} & {r['array_sha256'] for r in val.records}:
        raise ValueError('Training/validation full-model overlap: split models before cropping')
    identity = dict(meta=meta, model_config=model_config, schedule_config=schedule_config,
                    train_records=data.records, val_records=val.records, base_sha256=base_hash,
                    rank=rank if base else None, alpha=alpha if base else None, batch_size=batch_size,
                    lr=lr, seed=seed, eval_every=eval_every, val_batches=val_batches, ema_decay=ema_decay,
                    implementation_sha256={name:file_hash(Path(__file__).with_name(name))
                                           for name in ('training.py','model.py','io.py')})
    out = Path(out)
    if out.exists() and any(out.iterdir()) and resume is None:
        raise FileExistsError(f'Refusing to overwrite nonempty training directory: {out}')
    if out.exists() and any(out.iterdir()) and resume is not None and out.resolve()!=Path(resume).resolve().parent:
        raise FileExistsError('Resume output must be new/empty or the checkpoint training directory')
    out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(params, lr=lr)
    names = {n for n, p in net.named_parameters() if p.requires_grad}
    ema = {n: p.detach().clone() for n, p in net.named_parameters() if n in names}
    rng = np.random.default_rng(seed+17)
    step, best_value, best_step, best_state, history = 0, math.inf, 0, None, []
    if resume:
        ck = torch.load(resume, map_location='cpu', weights_only=True)
        if ck.get('schema') != TRAIN_SCHEMA or ck['identity'] != identity:
            raise ValueError('Training resume configuration/data/base/source identity mismatch')
        if ck['step'] >= steps:
            raise ValueError('Resume needs a larger update budget')
        net.load_state_dict(ck['model'], strict=True)
        opt.load_state_dict(ck['optimizer'])
        ema = {n: value.to(device) for n, value in ck['ema'].items()}
        rng.bit_generator.state = ck['data_rng']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if str(device).startswith('cuda') and ck['cuda_rng']:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        step, history = ck['step'], ck['history']
        best_value, best_step, best_state = ck['best_value'], ck['best_step'], ck['best_state']
    _json(out/'training_config.json', {**identity, 'total_updates':steps})
    target = out/('best_adapter.pt' if base else 'best.pt')
    training_meta = dict(train_models=data.records, validation_models=val.records,
                         selection='lowest fixed-noise validation epsilon MSE, EMA weights', seed=seed)

    def export_best():
        previous = _swap(net, best_state)
        try:
            if base:
                save_adapter(target, net, base, rank, alpha, best_step, training_meta)
            else:
                save_base(target, net, model_config, schedule_config, meta, best_step, training_meta)
        finally:
            _swap(net, previous)

    if best_state is not None:
        export_best()
    started = time.perf_counter()
    net.train()
    while step < steps:
        x = data.sample(batch_size, rng).to(device)
        opt.zero_grad(set_to_none=True)
        loss = ddpm.loss(net, x)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite diffusion training loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError('Missing/non-finite diffusion training gradient')
        opt.step()
        step += 1
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in ema:
                    ema[n].lerp_(p.detach(), 1-ema_decay)
        if step % eval_every == 0 or step == steps:
            previous = _swap(net, ema)
            try:
                metrics = _validate(net, ddpm, val, batch_size, val_batches, seed+100003, device)
            finally:
                _swap(net, previous)
            record = dict(update=step, training_epsilon_mse=float(loss.detach()), **metrics)
            history.append(record)
            if metrics['epsilon_mse'] < best_value:
                best_value, best_step = metrics['epsilon_mse'], step
                best_state = {n: v.detach().cpu().clone() for n, v in ema.items()}
                export_best()
            atomic_save(dict(schema=TRAIN_SCHEMA, identity=identity, step=step,
                model={k: v.detach().cpu() for k, v in net.state_dict().items()}, optimizer=opt.state_dict(),
                ema={k: v.detach().cpu() for k, v in ema.items()}, data_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                best_value=best_value, best_step=best_step, best_state=best_state, history=history), out/'last_training.pt')
            _json(out/'history.json', history)
            print(f'prior update={step} val_eps={metrics["epsilon_mse"]:.6g} '
                  f'denoise_rmse={metrics["clean_rmse_mps"]:.3f} m/s', flush=True)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    _json(out/'summary.json', dict(completed_updates=step, selected_update=best_step,
          parameters=sum(p.numel() for p in net.parameters()), trainable_parameters=sum(p.numel() for p in params),
          seconds_this_attempt=time.perf_counter()-started, export=str(target),
          note='Validation denoising is not evidence of improved FWI recovery.'))
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate', help='generate independent procedural train/validation models')
    g.add_argument('--out', required=True)
    g.add_argument('--train-models', type=int, default=64)
    g.add_argument('--val-models', type=int, default=16)
    g.add_argument('--shape', nargs=2, type=int, default=[96,192])
    g.add_argument('--grid-spacing-m', type=float, default=15.)
    g.add_argument('--seed', type=int, default=3)
    g.add_argument('--profile', choices=['layered','folded','mixed'], default='mixed')
    r = sub.add_parser('init-control', help='save UNTRAINED weights for an explicit control only')
    r.add_argument('--out', required=True)
    r.add_argument('--grid-spacing-m', type=float, default=15.)
    r.add_argument('--seed', type=int, default=3)
    for command in ('train','adapt'):
        p = sub.add_parser(command)
        for key in ('train-dir','val-dir','out'):
            p.add_argument('--'+key, required=True)
        p.add_argument('--steps', type=int, default=20000 if command=='train' else 3000)
        p.add_argument('--batch-size', type=int, default=8)
        p.add_argument('--lr', type=float, default=2e-4 if command=='train' else 1e-4)
        p.add_argument('--eval-every', type=int, default=200)
        p.add_argument('--val-batches', type=int, default=4)
        p.add_argument('--seed', type=int, default=3)
        p.add_argument('--device', default='cpu')
        p.add_argument('--resume')
        if command=='adapt':
            p.add_argument('--base', required=True)
            p.add_argument('--rank', type=int, default=4)
            p.add_argument('--alpha', type=float, default=4.)
        else:
            p.add_argument('--patch-size', type=int, default=64)
            p.add_argument('--grid-spacing-m', type=float, default=15.)
            p.add_argument('--mean-mps', type=float, default=3500.)
            p.add_argument('--scale-mps', type=float, default=2500.)
            p.add_argument('--channels', type=int, default=32)
    args = vars(ap.parse_args())
    command = args.pop('command')
    if command == 'generate':
        print(generate_dataset(**args))
    elif command == 'init-control':
        path = Path(args['out'])
        if path.exists():
            raise FileExistsError('Refusing to overwrite existing checkpoint')
        torch.manual_seed(args['seed'])
        meta = {**DEFAULT_META, 'grid_spacing_m':args['grid_spacing_m']}
        save_base(path, UNet(**DEFAULT_MODEL), DEFAULT_MODEL, DEFAULT_SCHEDULE, meta, updates=0)
        print(path)
    else:
        if command == 'train':
            args['model_config'] = {**DEFAULT_MODEL, 'channels':args.pop('channels')}
            args['meta'] = {**DEFAULT_META, **{k:args.pop(k) for k in ('patch_size','grid_spacing_m','mean_mps','scale_mps')}}
        print(train(**args))


if __name__ == '__main__':
    main()

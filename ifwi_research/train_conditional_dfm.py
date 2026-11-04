"""Train the NEW masked conditional DFM, never auto-qualify it for inversion.

Conditions are supplied data-only estimates, not truth outside a declared synthetic
smoke fixture. One geological endpoint, one immutable condition and one scalar time
pair define a group. Log-Sinkhorn drift NEVER mixes groups. Empirical single-target
conditioning is not a calibrated multimodal conditional geological distribution.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .conditional_dfm import ARCHITECTURE, SCHEMA, make_transport
from .train_dfm import array_hash


@dataclass(frozen=True)
class TrainingConfig:
    architecture: str = ARCHITECTURE
    selection_statistic: str = 'single_draw'
    selection_nfe: int = 1
    patch_size: int = 64
    width: int = 32
    group_size: int = 16
    groups_per_step: int = 2
    steps: int = 20000
    validate_every: int = 100
    learning_rate: float = 1e-4
    temperature: float = .1
    sinkhorn_iters: int = 50
    ema_decay: float = .999
    grad_clip: float = 1.
    seed: int = 3

    def validate(self):
        if self.selection_statistic not in ('single_draw', 'ensemble_mean'):
            raise ValueError('selection_statistic must be single_draw or ensemble_mean')
        if type(self.selection_nfe) is not int or self.selection_nfe not in (1, 4):
            raise ValueError('selection_nfe must be 1 or 4')
        if self.architecture not in ('masked_two_time_cnn_v1', 'masked_two_time_unet_v1'):
            raise ValueError('Unsupported conditional architecture')
        for key in ('patch_size', 'width', 'group_size', 'groups_per_step', 'steps',
                    'validate_every', 'sinkhorn_iters', 'seed'):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == 'seed' else 1):
                raise ValueError(f'{key} must be an integer in its positive range')
        if self.patch_size < 4 or not 4 <= self.width <= 128 or self.group_size < 2:
            raise ValueError('patch_size>=4, width in [4,128], group_size>=2 required')
        if self.seed >= 2**63:
            raise ValueError('seed must be <2**63')
        for key in ('learning_rate', 'temperature', 'grad_clip'):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite positive')
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError('ema_decay must be in [0,1)')


def _array(path):
    return np.load(path, mmap_mode='r', allow_pickle=False)


def load_manifest(path, *, patch_size, spacing_m):
    """Read full-model split provenance BEFORE any random crop extraction.

    Family IDs must represent original parent geology, including all crops and
    augmentations. IDs are declarations, not proof; content checks complement them.
    Explicit test records are checked for overlap and never sampled or normalized.
    """
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'ifwi_conditional_dataset_v1' or not isinstance(data.get('records'), list):
        raise ValueError('ifwi_conditional_dataset_v1 records manifest required')
    records, seen_ids, seen_content, families = [], set(), {}, {}
    for source in data['records']:
        rec = dict(source)
        for name in ('model_id', 'family_id'):
            if not isinstance(rec.get(name), str) or not rec[name].strip():
                raise ValueError(f'Nonempty {name} provenance required')
        if rec['model_id'] in seen_ids:
            raise ValueError('Duplicate model_id; each paired condition must be unambiguous')
        seen_ids.add(rec['model_id'])
        split = rec.get('split')
        if split not in ('train', 'validation', 'test'):
            raise ValueError('split must be train, validation or test')
        if rec['family_id'] in families and families[rec['family_id']] != split:
            raise ValueError('Geological parent family overlaps train/validation/test splits')
        families[rec['family_id']] = split
        if rec.get('condition_source') not in ('data_only_reconstruction', 'synthetic_smoke'):
            raise ValueError('condition_source must declare data_only_reconstruction or synthetic_smoke')
        if rec['condition_source'] == 'data_only_reconstruction':
            if not isinstance(rec.get('condition_provenance'), str) or not rec['condition_provenance'].strip():
                raise ValueError('Data-only conditions need condition_provenance identifying reconstruction recipe')
        declared_spacing = rec.get('spacing_m')
        if (isinstance(declared_spacing, bool) or not isinstance(declared_spacing, (int, float)) or
            not math.isfinite(declared_spacing) or not math.isclose(declared_spacing, spacing_m, rel_tol=1e-7)):
            raise ValueError('All physical spacing_m declarations must match; do not resize images')
        shape, hashes = None, {}
        for name in ('clean', 'background', 'anchor', 'mask'):
            raw_path = rec.get(name + '_path')
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f'{name}_path required')
            file_path = (path.parent / raw_path).resolve()
            a = _array(file_path)
            if a.ndim != 2 or min(a.shape) < patch_size or not np.isfinite(a).all():
                raise ValueError(f'{name}: finite HxW array at least patch_size required')
            if shape is not None and a.shape != shape:
                raise ValueError('Clean/background/anchor/mask must share physical grid and shape')
            shape = a.shape
            if name == 'mask':
                if not np.logical_or(a == 0, a == 1).all():
                    raise ValueError('Anchor mask must be binary, not claimed illumination truth')
            elif not np.logical_and(a >= 100, a <= 15000).all():
                raise ValueError(f'{name}: physical m/s values in [100,15000] required')
            hashes[name] = array_hash(a)
            rec[name + '_path'] = str(file_path)
        if hashes['clean'] in seen_content:
            raise ValueError('Duplicate clean content, including float32-equivalent models, is forbidden')
        seen_content[hashes['clean']] = split
        rec.update(shape=list(shape), hashes=hashes)
        records.append(rec)
    if not all(any(r['split'] == s for r in records) for s in ('train', 'validation')):
        raise ValueError('At least one independent full train and validation model required')
    return records


def fit_normalization(records):
    """Pixel-weighted population statistics fit ONLY to full training models."""
    count, mean, m2 = 0, 0., 0.
    for rec in records:
        if rec['split'] != 'train':
            continue
        a = np.asarray(_array(rec['clean_path']), dtype=np.float64)
        n, bmean, bm2 = a.size, float(a.mean()), float(a.var()) * a.size
        delta = bmean - mean
        m2 += bm2 + delta * delta * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    if not count:
        raise ValueError('Training models needed to fit normalization')
    scale = max(math.sqrt(m2 / count), 1.)
    return dict(mean_mps=mean, scale_mps=scale, fit_split='train',
                method='pixel_weighted_population_std_floor_1_mps', pixel_count=count)


@torch.no_grad()
def sinkhorn_plan(logits, *, iterations=50):
    """Balanced uniform-mass OT in log space, separate for each leading group.

    Finite iterations approximate row constraints; column constraints are imposed
    last. Row normalization is applied separately to obtain barycentric weights.
    """
    if type(iterations) is not int or iterations < 1:
        raise ValueError('Sinkhorn iterations must be a positive integer')
    if logits.ndim < 2 or min(logits.shape[-2:]) < 1 or not torch.isfinite(logits).all():
        raise ValueError('Finite nonempty pairwise logits required')
    logits = logits.detach()
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    n, m = logits.shape[-2:]
    u, v = torch.zeros_like(logits[..., :, 0]), torch.zeros_like(logits[..., 0, :])
    for _ in range(iterations):
        u = -math.log(n) - torch.logsumexp(logits + v.unsqueeze(-2), dim=-1)
        v = -math.log(m) - torch.logsumexp(logits + u.unsqueeze(-1), dim=-2)
    return (logits + u.unsqueeze(-1) + v.unsqueeze(-2)).exp()


@torch.no_grad()
def _barycenter(q, p, temperature, iterations):
    # RMS distance makes temperature independent of patch pixel count. Official
    # Euclidean implementation matches at temperature * sqrt(number of pixels).
    logits = -torch.cdist(q, p) / (math.sqrt(q.shape[-1]) * temperature)
    weights = sinkhorn_plan(logits, iterations=iterations)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return weights @ p


def group_drift_loss(predicted, positive, *, temperature=.1, iterations=50):
    """Return loss per group [G]; only predicted carries gradients.

    Input [G,B,1,H,W]. Attraction minus self-inclusive repulsion follows the
    reference group Sinkhorn construction. Plain pixel-mean stop-gradient
    matching is an explicit adaptation, NOT the reference adaptive norm weighting.
    """
    if (predicted.ndim != 5 or positive.shape != predicted.shape or predicted.shape[1] < 2
            or predicted.shape[2] != 1):
        raise ValueError('Matching [G,B,1,H,W], group size>=2 required')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite positive')
    if not torch.isfinite(predicted).all() or not torch.isfinite(positive).all():
        raise ValueError('Finite group samples required')
    # Use fp32 for cdist/Sinkhorn even if caller chooses mixed precision later.
    q, p = predicted.detach().float().flatten(2), positive.detach().float().flatten(2)
    drift = _barycenter(q, p, temperature, iterations) - _barycenter(q, q, temperature, iterations)
    target = (q + drift).reshape_as(predicted).detach()
    return .5 * (predicted.float() - target).square().flatten(1).mean(1)


def sample_time_pair(rng):
    if float(torch.rand((), generator=rng)) < .2:
        return 0., 1.
    times = torch.randn(2, generator=rng).sigmoid().sort().values.tolist()
    t, r = map(float, times)
    # Avoid near-identity pairs with vanishing transport gradients.
    if r - t < .001:
        t, r = max(0., t - .001), min(1., r + .001)
    return t, r


def _patch(rec, side, rng, device, norm, *, center=False):
    h, w = rec['shape']
    if center:
        z, x = (h - side) // 2, (w - side) // 2
    else:
        z = int(torch.randint(h - side + 1, (1,), generator=rng))
        x = int(torch.randint(w - side + 1, (1,), generator=rng))
    values = {}
    for name in ('clean', 'background', 'anchor', 'mask'):
        a = np.array(_array(rec[name + '_path'])[z:z+side, x:x+side], dtype=np.float32, copy=True)
        values[name] = torch.from_numpy(a)[None, None].to(device)
    mean, scale = norm['mean_mps'], norm['scale_mps']
    clean = (values['clean'] - mean) / scale
    condition = torch.cat(((values['background'] - mean) / scale,
                           ((values['anchor'] - mean) / scale) * values['mask'], values['mask']), dim=1)
    return clean, condition.detach(), values


@torch.no_grad()
def validation(model, records, config, device, norm):
    """Gaussian0→1, fixed seeds, EMA; no truth enters the supplied condition.

    Center crops and two draws are inexpensive diagnostics, not inversion tests,
    full-model evaluation, uncertainty calibration or an acceptance certificate.
    """
    rng = torch.Generator().manual_seed(941)
    output = {}
    cases = [r for r in records if r['split'] == 'validation']
    for nfe in (1, 4):
        sums = dict(weak_sq=0., weak_n=0, total_sq=0., total_n=0,
                    edge_sq=0., edge_n=0, bg_weak_sq=0., diversity_sq=0., bad_n=0,
                    sample_weak_sq=0., sample_total_sq=0., rejected_draws=0)
        # Identical inputs/noise compare NFE without RNG drift.
        rng.manual_seed(941)
        for rec in cases:
            clean, condition, values = _patch(rec, config.patch_size, rng, device, norm, center=True)
            state = torch.randn((2, 1, config.patch_size, config.patch_size), generator=rng).to(device)
            for k in range(nfe):
                state = model(state, k/nfe, (k+1)/nfe, condition)
            if not torch.isfinite(state).all():
                raise FloatingPointError('Nonfinite validation generation')
            generated = state * norm['scale_mps'] + norm['mean_mps']
            estimate = generated.mean(0, keepdim=True)
            error = estimate - values['clean']
            weak = 1 - values['mask']
            sample_error = generated - values['clean']
            sums['sample_weak_sq'] += float((sample_error.square()*weak).sum())
            sums['sample_total_sq'] += float(sample_error.square().sum())
            sums['rejected_draws'] += int(((generated < 100) | (generated > 15000)).flatten(1).any(1).sum())
            sums['weak_sq'] += float((error.square()*weak).sum()); sums['weak_n'] += int(weak.sum())
            sums['total_sq'] += float(error.square().sum()); sums['total_n'] += error.numel()
            sums['bg_weak_sq'] += float(((values['background']-values['clean']).square()*weak).sum())
            sums['diversity_sq'] += float(generated.var(0, unbiased=False).sum())
            sums['bad_n'] += int(((generated < 100) | (generated > 15000)).sum())
            for axis in (-1, -2):
                edge_error = torch.diff(estimate, dim=axis) - torch.diff(values['clean'], dim=axis)
                sums['edge_sq'] += float(edge_error.square().sum()); sums['edge_n'] += edge_error.numel()
        output[f'nfe_{nfe}'] = dict(
            sample_weak_rmse_mps=math.sqrt(sums['sample_weak_sq']/(2*sums['weak_n'])) if sums['weak_n'] else None,
            sample_rmse_mps=math.sqrt(sums['sample_total_sq']/(2*sums['total_n'])),
            ensemble_mean_weak_rmse_mps=math.sqrt(sums['weak_sq']/sums['weak_n']) if sums['weak_n'] else None,
            ensemble_mean_rmse_mps=math.sqrt(sums['total_sq']/sums['total_n']),
            legacy_rmse_semantics='weak_rmse_mps and rmse_mps are two_draw_ensemble_mean aliases',
            rejected_draws=sums['rejected_draws'], total_draws=2*len(cases),
            weak_rmse_mps=math.sqrt(sums['weak_sq']/sums['weak_n']) if sums['weak_n'] else None,
            background_weak_rmse_mps=math.sqrt(sums['bg_weak_sq']/sums['weak_n']) if sums['weak_n'] else None,
            rmse_mps=math.sqrt(sums['total_sq']/sums['total_n']),
            edge_rmse_mps_per_cell=math.sqrt(sums['edge_sq']/sums['edge_n']),
            diversity_rms_mps=math.sqrt(sums['diversity_sq']/sums['total_n']),
            outside_physical_range_fraction=sums['bad_n']/(2*sums['total_n']),
            weak_pixels=sums['weak_n'], validation_models=len(cases), draws=2)
    return output


def selection_score(metrics, config):
    """Expected single-draw MSE by default, matching the frozen proposal API."""
    record = metrics[f'nfe_{config.selection_nfe}']
    prefix = 'sample' if config.selection_statistic == 'single_draw' else 'ensemble_mean'
    score = record[f'{prefix}_weak_rmse_mps']
    return record[f'{prefix}_rmse_mps'] if score is None else score


def _manifest_digest(records):
    # Relocating identical datasets does not change their experimental identity.
    canonical = [{k: v for k, v in r.items() if not k.endswith('_path')} for r in records]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, allow_nan=False).encode()).hexdigest()


def train(manifest, out, *, spacing_m, config=None, device='cpu', resume=None):
    config = config or TrainingConfig()
    config.validate()
    if isinstance(spacing_m, bool) or not math.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError('spacing_m must be finite positive')
    records = load_manifest(manifest, patch_size=config.patch_size, spacing_m=spacing_m)
    norm = fit_normalization(records)
    fingerprint = _manifest_digest(records)
    device = torch.device(device)
    # Only model construction uses the global stream; preserve the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        model = make_transport(config.architecture, config.width).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.)
    rng = torch.Generator().manual_seed(config.seed)
    start, best = 0, float('inf')
    selection_changed_on_resume = False
    best_scope = ('current_invocation_including_resume_initial' if resume is not None
                  else 'current_invocation_post_update_validation')
    resume_history = None
    if resume is not None:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        if checkpoint.get('meta', {}).get('schema') != SCHEMA or checkpoint.get('manifest_sha256') != fingerprint:
            raise ValueError('Resume manifest/schema differs from checkpoint')
        old = dict(checkpoint['config']); new = asdict(config)
        old_selection = (old.pop('selection_statistic', 'ensemble_mean'), old.pop('selection_nfe', 1))
        new_selection = (new.pop('selection_statistic'), new.pop('selection_nfe'))
        selection_changed_on_resume = old_selection != new_selection
        old.pop('steps'); new.pop('steps')
        if old != new or checkpoint['meta']['spacing_m'] != spacing_m or checkpoint['meta']['normalization'] != norm:
            raise ValueError('Resume training config, spacing or normalization differs')
        model.load_state_dict(checkpoint['raw_state_dict'], strict=True)
        ema.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        rng.set_state(checkpoint['rng_state'])
        start = checkpoint['meta']['training']['steps']
        # Scores without their historical weights cannot select a portable best.
        # Keep the old score as labeled provenance only, and re-score the loaded
        # EMA below under this invocation's declared metric/NFE.
        resume_history = dict(previous_best_score_mps=checkpoint['best_score'],
            previous_selection=dict(statistic=old_selection[0], nfe=old_selection[1]),
            historical_best_weights_carried=False)
        if start >= config.steps:
            raise ValueError('Resume target steps must exceed checkpoint steps')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    (out/'dataset_manifest.json').write_text(json.dumps({'schema': 'ifwi_conditional_dataset_v1', 'records': records, 'sha256': fingerprint}, indent=2))
    train_records = [r for r in records if r['split'] == 'train']
    meta = dict(schema=SCHEMA, architecture=config.architecture, units='m/s', spacing_m=spacing_m,
                width=config.width, mean_mps=norm['mean_mps'], scale_mps=norm['scale_mps'], normalization=norm,
                training=dict(steps=0, toy=any(r['condition_source']=='synthetic_smoke' for r in records),
                    contract='shared_background_estimate_anchors_mask_v1',
                    time_semantics='gaussian_0_to_geology_1_two_time_transport', anchor_semantics='estimated_not_truth',
                    train_model_hashes=[r['hashes']['clean'] for r in train_records],
                    val_model_hashes=[r['hashes']['clean'] for r in records if r['split']=='validation'],
                    group_time='one scalar pair per condition; 20% 0->1, otherwise sorted sigmoid(N(0,1))',
                    objective='stop-gradient group log-Sinkhorn attraction minus self-inclusive repulsion; pixel mean',
                    condition_target_count=1, weights='EMA', patch_size=config.patch_size,
                    validation_selection=dict(statistic=config.selection_statistic, nfe=config.selection_nfe)),
                qualification=dict(status='unqualified', report_ref=None, report_sha256=None))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = None
    initial_metrics = validation(ema, records, config, device, norm)
    (out/'initial_validation.json').write_text(json.dumps(initial_metrics, indent=2, allow_nan=False))

    def save(name, step):
        # Fresh metadata and CPU tensors make checkpoints portable and safe-loadable.
        current_meta = copy.deepcopy(meta); current_meta['training']['steps'] = step
        payload = dict(meta=current_meta, config=asdict(config), manifest_sha256=fingerprint,
            state_dict={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},
            raw_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), rng_state=rng.get_state(), best_score=best, validation=metrics,
            best_scope=best_scope, resume_history=resume_history)
        path = out/name; temp = path.with_suffix('.tmp')
        torch.save(payload, temp); temp.replace(path)
        inference_path = out / (path.stem + '_inference.pt')
        inference_temp = inference_path.with_suffix('.tmp')
        torch.save({'meta': current_meta, 'state_dict': payload['state_dict']}, inference_temp)
        inference_temp.replace(inference_path)

    if resume is not None:
        # Persist the actual loaded raw/EMA/optimizer/RNG state before advancing.
        # This is the best of this invocation so far, NOT the historical best.
        metrics = initial_metrics
        best = selection_score(metrics, config)
        save('best.pt', start)

    with (out/'history.jsonl').open('w', encoding='utf-8') as log:
        for step in range(start + 1, config.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            for _ in range(config.groups_per_step):
                rec = train_records[int(torch.randint(len(train_records), (1,), generator=rng))]
                clean, condition, _ = _patch(rec, config.patch_size, rng, device, norm)
                t, r = sample_time_pair(rng)
                noise = torch.randn((config.group_size, 1, config.patch_size, config.patch_size), generator=rng).to(device)
                xt, xr = (1-t)*noise+t*clean, (1-r)*noise+r*clean
                predicted = model(xt, t, r, condition)
                loss = group_drift_loss(predicted[None], xr[None], temperature=config.temperature,
                                       iterations=config.sinkhorn_iters).mean()/config.groups_per_step
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite conditional DFM loss')
                loss.backward(); value += float(loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(v).all() for v in model.parameters()):
                raise FloatingPointError('Nonfinite parameters after optimizer step')
            with torch.no_grad():
                for target, current in zip(ema.parameters(), model.parameters()):
                    target.lerp_(current, 1-config.ema_decay)
            record = dict(step=step, drift_surrogate_loss=value, gradient_norm=float(grad_norm))
            if step % config.validate_every == 0 or step == config.steps:
                metrics = validation(ema, records, config, device, norm)
                score = selection_score(metrics, config)
                record['validation'] = metrics
                if score < best:
                    best = score; save('best.pt', step)
                save('last.pt', step)
            record['elapsed_seconds'] = time.perf_counter()-started
            log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
    raw_metrics = validation(model, records, config, device, norm)
    selected = metrics[f'nfe_{config.selection_nfe}']
    baseline = selected['background_weak_rmse_mps']
    score = selection_score(metrics, config)
    diagnostics = dict(improves_on_initial=score < selection_score(initial_metrics, config),
        beats_background=(score < baseline) if baseline is not None else False,
        background_comparison_available=baseline is not None,
        raw_selected_rmse_mps=selection_score(raw_metrics, config), ema_selected_rmse_mps=score,
        selection_changed_on_resume=selection_changed_on_resume,
        note='Diagnostic comparisons only; no convergence or qualification certificate')
    summary = dict(selection=dict(statistic=config.selection_statistic, nfe=config.selection_nfe),
                   best_scope=best_scope, resume_history=resume_history,
                   initial_validation=initial_metrics, raw_validation=raw_metrics, training_diagnostics=diagnostics,
                   steps=config.steps, resumed_from_step=start, best_validation_score_mps=best,
                   validation=metrics, device=str(device), torch=str(torch.__version__),
                   elapsed_seconds=time.perf_counter()-started, parameter_count=sum(p.numel() for p in model.parameters()),
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                   qualification=meta['qualification'], manifest_sha256=fingerprint,
                   warning='Unqualified research weights. No seismic data consistency, geological generalization, '
                           'calibrated uncertainty, 16GB GPU fit, or FWI improvement established by this training run.')
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--spacing-m', type=float, required=True); parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume')
    for name, value in asdict(TrainingConfig()).items():
        parser.add_argument('--'+name.replace('_', '-'), type=type(value), default=value)
    args = vars(parser.parse_args())
    config = TrainingConfig(**{k: args.pop(k) for k in asdict(TrainingConfig())})
    print(json.dumps(train(**args, config=config), indent=2))


if __name__ == '__main__':
    main()

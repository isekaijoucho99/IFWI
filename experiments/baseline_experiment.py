"""Formal parameter experiments using the author's unchanged IFWI2D trainer."""
import contextlib
import copy
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ifwi_modules import IFWI2D
from rnn_fd import rnn2D
from generator import wGenerator
from baseline_reference.ifwi_experiment import EpochLog, LiveLog, plot_result
from improved_modules.evaluate_deep import evaluate_deep_layers
from parameter_sweep import source_positions


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def _velocity(model):
    previous = model.vel_net.training
    model.vel_net.eval()
    try:
        with torch.no_grad():
            normalized, _ = model.vel_net(model.coords)
            return ((normalized.squeeze() * model.std + model.mean) * 1000).cpu().numpy()
    finally:
        model.vel_net.train(previous)


def _common(config, data, device):
    geometry = data['geometry']
    params = data['params']
    return dict(nz=geometry['nz'], nx=geometry['nx'], zs=geometry['zs'], xs=geometry['xs'],
        zr=geometry['zr'], xr=geometry['xr'], dz=params['dz'], dt=params['dt'],
        npad=15, order=2, vmax=data['vp_true'].max(), log_para=1e-6,
        freeSurface=True, dtype=torch.float32, device=device)


def prepare_original_data(config, device):
    """Retain the original read, acquisition, wavelet and forward-modeling order."""
    settings = config['data']
    truth = np.array(pd.read_csv(ROOT / 'data' / settings['model_file']))[
        ::settings.get('downsample', 4), ::settings.get('downsample', 4)].astype(np.float32)
    vp = torch.from_numpy(truth[None]).to(device)
    nv, nz, nx = vp.shape
    xs = torch.arange(20, nx - 10, 20, dtype=torch.long).repeat(nv, 1)
    if settings.get('num_shots', xs.shape[1]) != xs.shape[1]:
        xs = torch.from_numpy(source_positions(nx, settings)).repeat(nv, 1)
    ns = xs.shape[1]
    xr = torch.arange(nx, dtype=torch.long).repeat(nv, ns, 1)
    zs = torch.full((nv, ns), 1, dtype=torch.long)
    zr = torch.full((nv, ns, nx), 2, dtype=torch.long)
    t = settings['dt'] * torch.arange(settings['nt'], dtype=torch.float32)
    wavelet = wGenerator(t, 8).ricker().to(device)
    data = {'vp_true': vp, 'wavelet': wavelet,
        'geometry': {'nz': nz, 'nx': nx, 'xs': xs, 'zs': zs, 'xr': xr, 'zr': zr},
        'params': {'dz': settings['dz'], 'dt': settings['dt'], 'nt': settings['nt']}}
    print('Forward modeling: generating observed shots...', flush=True)
    forward = rnn2D(**_common(config, data, device)).to(device)
    with torch.no_grad():
        _, _, data['shots'], _ = forward(vmodel=vp, segment_wavelet=wavelet)
    return data


def build_original_model(config, data, device):
    settings = config['model']
    return IFWI2D(**_common(config, data, device), mean=3., std=1.,
        neuron=settings['neuron'], omega_0=settings.get('omega_0', 30), prob=.2,
        activation='sine', bias=True, dropout=False, outermost_linear=True,
        segment_size=data['params']['nt'], vpadding=None, pretrained=None, netOpt='IFWI')


def _legacy_config(config, seed):
    return dict(mode='random', seed=seed, mean=3., std=1., dz=config['data']['dz'],
        dt=config['data']['dt'], nt=config['data']['nt'], frequency=8,
        source_depth_index=1, receiver_depth_index=2,
        learning_rate=config['optimizer']['learning_rate'], alpha=config['training'].get('alpha', 0),
        noise=0, dropout=0, epochs=config['training']['max_iterations'],
        log_interval=config['training']['log_interval'], pretrained=None)


def _spatial_metrics(truth, predicted, config):
    settings = config.get('evaluation', {})
    depth = settings.get('depth_threshold', .5)
    metrics = evaluate_deep_layers(truth, predicted, depth_threshold=depth,
        corner_size=settings.get('corner_size', .25), dz=config['data']['dz'])
    deep = predicted[int(predicted.shape[0] * depth):]
    metrics.update(data_mse=None, data_objective=None, data_mse_evaluated=False,
        data_mse_status='unevaluated; original training performs no endpoint waveform evaluation',
        velocity_min_mps=float(predicted.min()), velocity_max_mps=float(predicted.max()),
        deep_velocity_std_mps=float(deep.std()),
        deep_horizontal_tv_mps=float(np.abs(np.diff(deep, axis=1)).mean()))
    return metrics


def train_original(model, data, config, out, resume=None, stop_after=None):
    """Call original train/save/predict functions without changing a training step."""
    training = config['training']
    if training.get('shot_batch_size') is not None:
        raise ValueError('Original baseline training requires full shots; batching is unsupported')
    if type(model) is not IFWI2D or model.netOpt != 'IFWI':
        raise ValueError('Original baseline training requires the unchanged IFWI2D class')
    if model.segment_size != len(data['wavelet']):
        raise ValueError('Original baseline training requires the complete time record')
    total = training['max_iterations']
    interval = training['log_interval']
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError('max_iterations must be a positive integer')
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError('log_interval must be a positive integer')
    if stop_after is not None and (isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 1):
        raise ValueError('stop_after must be a positive integer')
    limit = min(total, stop_after) if stop_after is not None else total
    completed = 0
    if resume is not None:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        required = {'epoch', 'state_dict', 'optimizer', 'train_loss', 'best_loss',
                    'best_loss_epoch', 'best_loss_model'}
        if not required.issubset(saved) or 'schema' in saved:
            raise ValueError('Resume requires an original author checkpoint, not a modern schema')
        completed = saved['epoch']
        if limit <= completed:
            raise ValueError('Total update budget must exceed the resumed checkpoint')
    out = Path(out)
    checkpoints = out / 'checkpoints'
    if (out / 'loss.csv').exists() or (checkpoints.exists() and any(checkpoints.iterdir())):
        raise ValueError('Refuse to overwrite training output')
    checkpoints.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        model.load_state(str(resume), best=False)
    initial = _velocity(model)
    truth = data['vp_true'].squeeze().cpu().numpy()
    np.save(out / 'initial_velocity.npy', initial)
    np.save(out / 'true_velocity.npy', truth)
    np.save(out / 'v_true.npy', truth)
    np.save(out / 'observed.npy', data['shots'].cpu().numpy())
    _write_json(out / 'acquisition.json', {
        'num_shots': int(model.rnn.xs.shape[1]),
        'source_x_indices': model.rnn.xs.tolist(), 'source_z_indices': model.rnn.zs.tolist(),
        'receiver_x_indices': model.rnn.xr.tolist(), 'receiver_z_indices': model.rnn.zr.tolist()})
    _write_json(out / 'status.json', {'state': 'running', 'completed_updates': completed})
    prefix = str(checkpoints / 'MarmousiI_random-')
    started = time.perf_counter()
    cuda = str(model.device).startswith('cuda')
    if cuda:
        torch.cuda.reset_peak_memory_stats(model.device)
    try:
        history, _ = model.train(MaxIter=limit, vmodel=None, wavelet=data['wavelet'], shots=data['shots'],
            alpha=training.get('alpha', 0), option=0,
            learning_rate=config['optimizer']['learning_rate'], log_interval=interval,
            wandb=EpochLog(completed), resume_file_name=str(resume) if resume is not None else None,
            save_file_name=prefix)
        history = np.asarray(history)
        np.savetxt(out / 'loss.csv', np.column_stack([np.arange(1, len(history) + 1), history]),
            delimiter=',', header='completed_updates,total_loss,data_loss,regularization_statistic', comments='')
        checkpoint_path = Path(prefix + f'checkpoint-{limit}.pth')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        last_state = copy.deepcopy(model.vel_net.state_dict())
        last = _velocity(model)
        np.save(out / 'last_velocity.npy', last)
        best, _ = model.predict(resume_file_name=str(checkpoint_path), best=True)
        best = best.squeeze().cpu().numpy()
        np.save(out / 'best_velocity.npy', best)
        np.save(out / 'v_pred.npy', best)
        model.vel_net.load_state_dict(last_state)
        model.vel_net.train()
        metrics = dict(relative_model_error=float(np.linalg.norm(best - truth) / np.linalg.norm(truth)),
            velocity_rmse_mps=float(np.sqrt(np.mean((best - truth) ** 2))),
            final_data_loss=float(history[-1, 1]))
        _write_json(out / 'metrics.json', metrics)
        endpoint = _spatial_metrics(truth, last, config)
        endpoint.update(completed_updates=limit, selection_metric='legacy_preupdate_data_loss',
                        last_training_data_loss_before_update=float(history[-1, 1]))
        _write_json(out / 'final_metrics.json', endpoint)
        if config.get('evaluation', {}).get('save_plots', False):
            plot_result(out, truth, initial, best, history)
        if cuda:
            torch.cuda.synchronize(model.device)
        summary = dict(completed_updates=limit, best_update=int(checkpoint['best_loss_epoch']) + 1,
            best_loss=float(checkpoint['best_loss']), parameter_count=sum(p.numel() for p in model.vel_net.parameters()),
            total_seconds=time.perf_counter() - started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(model.device) if cuda else 0,
            selection_metric='legacy_preupdate_data_loss', resumed_from_updates=completed,
            executed_updates=limit - completed, checkpoint=str(checkpoint_path.resolve()),
            snapshot_updates=sorted(int(p.stem.rsplit('-', 1)[-1]) for p in checkpoints.glob('*.pth')),
            model_state_timing='postupdate', selection_loss_timing='preupdate',
            endpoint_waveform_evaluated=False)
        _write_json(out / 'training_summary.json', summary)
        _write_json(out / 'status.json', {'state': 'completed' if limit == total else 'paused',
                                       'completed_updates': limit})
        return dict(summary, history=history.tolist(), metrics=metrics, run_dir=str(out.resolve()))
    except BaseException as error:
        saved_updates = [int(p.stem.rsplit('-', 1)[-1]) for p in checkpoints.glob('*.pth')]
        _write_json(out / 'status.json', {'state': 'failed',
            'completed_updates': max(saved_updates, default=completed),
            'error': f'{type(error).__name__}: {error}'})
        raise


def validate_resume(config, resume):
    """Check an original checkpoint before preparing data or creating outputs."""
    previous_path = Path(resume).parent.parent / 'config.json'
    if not previous_path.is_file():
        raise ValueError('Resume requires the original run config.json')
    previous = json.loads(previous_path.read_text(encoding='utf-8-sig'))
    previous_training = previous if previous.get('mode') == 'random' else previous.get('training', {})
    if previous_training.get('log_interval') != config['training']['log_interval']:
        raise ValueError('Resume configuration differs from the original run: log_interval')
    if previous.get('mode') == 'random':
        if previous.get('backend', 'reference') != 'reference' or previous.get('gradient_clip') is not None:
            raise ValueError('Resume backend and gradient clipping must match the original baseline')
        expected = _legacy_config(config, config['seed'])
        for key in expected:
            if key != 'epochs' and previous.get(key) != expected[key]:
                raise ValueError('Resume configuration differs from the original run: ' + key)
        if config['data']['num_shots'] != 13 or config['model']['neuron'] != [2,128,128,128,128,1] or config['model']['omega_0'] != 30:
            raise ValueError('Original historical random checkpoint is the baseline configuration')
    else:
        current = copy.deepcopy(config)
        for value in (previous, current):
            value.pop('experiment_name', None)
            value.pop('description', None)
            value.get('training', {}).pop('max_iterations', None)
        if previous != current:
            raise ValueError('Resume configuration differs from the original run')
    checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
    required = {'epoch', 'state_dict', 'optimizer', 'train_loss', 'best_loss',
                'best_loss_epoch', 'best_loss_model'}
    if not required.issubset(checkpoint) or 'schema' in checkpoint:
        raise ValueError('Resume requires an original author checkpoint, not a modern schema')
    completed = checkpoint['epoch']
    if isinstance(completed, bool) or not isinstance(completed, int) or not 0 < completed < config['training']['max_iterations']:
        raise ValueError('Total update budget must exceed the original resumed checkpoint')
    return completed


def run_config(config, output_dir, device='cpu', seed=3, resume=None, stop_after=None,
               preliminary=False, allow_custom=False):
    """Create one suite-compatible run directory using the original common path."""
    from baseline_protocol import validate_baseline_protocol, validate_original_sources
    config = copy.deepcopy(config)
    config['seed'] = seed
    validate_baseline_protocol(config, allow_treatment=True, preliminary=preliminary,
                               allow_custom=allow_custom)
    hashes = validate_original_sources(ROOT)
    if resume is not None:
        validate_resume(config, resume)
    out = Path(output_dir) / (config['experiment_name'] + '_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / 'config.json', config)
    _write_json(out / 'baseline_config.json', _legacy_config(config, seed))
    _write_json(out / 'original_source_hashes.json', hashes)
    _write_json(out / 'run_metadata.json', dict(protocol='original_baseline', preliminary=preliminary,
                                               custom_parameters_allowed=allow_custom))
    _write_json(out / 'status.json', {'state': 'preparing', 'completed_updates': 0})
    with (out / 'progress.log').open('w', encoding='utf8', buffering=1) as log:
        with contextlib.redirect_stdout(LiveLog(sys.stdout, log)):
            try:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                print(f"START random: {config['training']['max_iterations']} updates, device={device}, output={out}", flush=True)
                data = prepare_original_data(config, device)
                model = build_original_model(config, data, device)
                # Keep c=0 available even when the original wrapper's initial
                # diagnostic refers to the resumed model.
                np.save(out / 'random_initial_velocity.npy', _velocity(model))
                result = train_original(model, data, config, out, resume=resume, stop_after=stop_after)
                print(f"COMPLETED: {result['metrics']}\nResults: {out}", flush=True)
                return result
            except Exception as error:
                print(f'FAILED: {type(error).__name__}: {error}', flush=True)
                if json.loads((out / 'status.json').read_text(encoding='utf8'))['state'] != 'failed':
                    _write_json(out / 'status.json', {'state': 'failed', 'completed_updates': 0,
                                                    'error': f'{type(error).__name__}: {error}'})
                raise

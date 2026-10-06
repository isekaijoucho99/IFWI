"""Real CPU Adam and small-grid FD characterization of unchanged upstream code."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT))
from test_lr_control import configs
from run_experiment import build_model
from experiment_runtime import train_loop, evaluate_loss, velocity
from generator import wGenerator
from rnn_fd import rnn2D


@pytest.fixture
def small_problem():
    torch.set_num_threads(1)
    nz, nx, ns, nt = 8, 9, 2, 24
    vp = torch.linspace(2800, 3200, nz).view(1, nz, 1).expand(1, nz, nx).clone()
    xs = torch.tensor([[2, 6]]); zs = torch.ones((1, ns), dtype=torch.long)
    xr = torch.arange(nx).view(1, 1, nx).repeat(1, ns, 1)
    zr = torch.full_like(xr, 2)
    wavelet = wGenerator(torch.arange(nt) * .001, 60).ricker()
    geom = dict(nz=nz, nx=nx, xs=xs, zs=zs, xr=xr, zr=zr)
    fd = rnn2D(**geom, dz=15, dt=.001, npad=15, order=2, vmax=vp.max(),
               freeSurface=True, dtype=torch.float32, device='cpu')
    with torch.no_grad(): shots = fd(vp, wavelet)[2]
    data = dict(vp_true=vp, shots=shots, wavelet=wavelet, geometry=geom,
                params={'dz': 15, 'dt': .001, 'nt': nt})
    _, config = configs()
    config['model']['neuron'] = [2, 8, 8, 1]
    config['training'].update(max_iterations=6, log_interval=2)
    return data, config


def make_model(config, data):
    torch.manual_seed(3)
    return build_model(config, data, 'cpu')


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested_equal(x, y)
    else: assert a == b


def test_small_fd_continuous_and_resumed_training_agree(small_problem, tmp_path):
    data, config = small_problem
    full = make_model(config, data)
    train_loop(full, data, config, tmp_path / 'continuous')
    first = make_model(config, data)
    train_loop(first, data, config, tmp_path / 'first', stop_after=2)
    resumed = make_model(config, data)
    train_loop(resumed, data, config, tmp_path / 'resumed', resume=tmp_path / 'first/last.pth')
    a = torch.load(tmp_path / 'continuous/last.pth', weights_only=False)
    b = torch.load(tmp_path / 'resumed/last.pth', weights_only=False)
    for field in ('completed_updates', 'model', 'optimizer', 'scheduler', 'history',
                  'best_state', 'best_loss', 'best_update'):
        assert_nested_equal(a[field], b[field])
    assert np.array_equal(velocity(full), velocity(resumed))


def test_small_fd_last_best_losses_match_saved_state(small_problem, tmp_path):
    data, config = small_problem
    model = make_model(config, data)
    result = train_loop(model, data, config, tmp_path)
    assert all(np.isfinite(row['gradient_norm']) and row['gradient_norm'] > 0 for row in result['history'])
    assert all(row['learning_rates'] != '[0.0001]' for row in result['history'][1:])
    for name in ('last', 'best'):
        ck = torch.load(tmp_path / (name + '.pth'), weights_only=False)
        model.vel_net.load_state_dict(ck['model'])
        loss, _ = evaluate_loss(model, data, 0)
        row = result['history'][ck['completed_updates'] - 1]
        assert loss == pytest.approx(row['loss_after_update'], rel=1e-6, abs=1e-12)
        assert np.array_equal(velocity(model), np.load(tmp_path / (name + '_velocity.npy')))
    evaluated = [r for r in result['history'] if r['loss_after_update'] is not None]
    assert result['best_update'] == min(evaluated, key=lambda r: r['loss_after_update'])['completed_updates']


def test_reject_optimizer_change_and_scheduler_horizon(small_problem, tmp_path):
    data, config = small_problem
    train_loop(make_model(config, data), data, config, tmp_path / 'first', stop_after=2)
    changed = copy.deepcopy(config); changed['optimizer']['use_scheduler'] = False
    with pytest.raises(ValueError, match='contract mismatch'):
        train_loop(make_model(changed, data), data, changed, tmp_path / 'changed', resume=tmp_path / 'first/last.pth')
    too_long = copy.deepcopy(config); too_long['training']['max_iterations'] = 4002
    with pytest.raises(ValueError, match='scheduler horizon'):
        train_loop(make_model(too_long, data), data, too_long, tmp_path / 'too_long')


def test_config_cli_routes_to_modern_runtime(small_problem, tmp_path, monkeypatch):
    import run_experiment
    import baseline_experiment
    data, _ = small_problem
    monkeypatch.setattr(run_experiment, 'prepare_data', lambda config, device: data)
    def forbidden(*args, **kwargs):
        pytest.fail('LR config incorrectly routed through original baseline')
    monkeypatch.setattr(baseline_experiment, 'run_config', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run_experiment.py', '--config',
        str(ROOT / 'experiments/configs/modern13_cosine_lr.yaml'), '--iterations', '2',
        '--log-interval', '2', '--device', 'cpu', '--output-dir', str(tmp_path)])
    assert run_experiment.main() == 0
    run, = tmp_path.iterdir()
    assert json.loads((run / 'status.json').read_text())['state'] == 'completed'
    assert (run / 'last.pth').is_file() and (run / 'best.pth').is_file()
    assert not (run / 'checkpoints').exists()

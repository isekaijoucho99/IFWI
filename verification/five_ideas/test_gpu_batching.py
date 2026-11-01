"""Exact batching contracts against the real original finite-difference solver."""
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from ifwi_research.repository import RepositoryObjective, load_problem
from ifwi_research.routing import CodePlan, ContrastRouter

ROOT = Path(__file__).resolve().parents[2]


def problem(free_surface=True, shot_batch_size=1):
    from experiments import residual_ifwi_experiment as e
    cfg = e.default_config()
    cfg['model']['neurons'] = [2, 8, 8, 1]
    cfg['forward'].update(nt=48, dt_s=.001, frequency_hz=30., npad=3,
                          free_surface=free_surface)
    cfg['data']['grid_spacing_m'] = 20.
    cfg['acquisition'].update(num_shots=3, source_x_indices=[3, 8, 13])
    cfg['training'].update(shot_batch_size=shot_batch_size,
                          frequency_schedule=[{'max_hz': 20., 'updates': 1}, {'max_hz': None}])
    initial = np.full((10, 18), 2000, dtype=np.float32)
    truth = initial.copy()
    truth[4:7, 5:12] += 150
    raw = e.build_forward_problem(truth, cfg, 'cpu')
    return raw, torch.from_numpy(initial)[None]


@pytest.mark.parametrize('free_surface', [True, False])
@pytest.mark.parametrize('cutoff', [None, 20.])
@pytest.mark.parametrize('batch_size', [2, 3])
def test_code_batches_match_serial_objective_gradient_and_energies(free_surface, cutoff, batch_size):
    raw, velocity = problem(free_surface)
    serial = RepositoryObjective(raw, code_batch_size=1)
    batched = RepositoryObjective(raw, code_batch_size=batch_size)
    full = ContrastRouter(3).plan(full=True)
    plan = CodePlan(full.indices, torch.tensor([.1, .7, 1.3], dtype=torch.float64),
                    full.draw_count, full.probabilities)
    expected = serial.evaluate(velocity, plan=plan, cutoff=cutoff, gradient=True)
    actual = batched.evaluate(velocity, plan=plan, cutoff=cutoff, gradient=True)
    assert actual.value == pytest.approx(expected.value, rel=2e-4, abs=1e-12)
    relative_error = (actual.gradient - expected.gradient).norm() / expected.gradient.norm()
    assert relative_error < 2e-4
    # Grouped FP32 convolutions have a different reduction order from one-code calls.
    torch.testing.assert_close(actual.energies, expected.energies, rtol=2e-4, atol=1e-12)
    assert actual.indices.tolist() == plan.indices.tolist()
    assert batched.counters['forward_solves'] == serial.counters['forward_solves'] == 3
    assert batched.counters['backward_solves'] == serial.counters['backward_solves'] == 3
    # Batch size2 exercises a short final batch; inference must retain weighting too.
    inference = batched.evaluate(velocity, plan=plan, cutoff=cutoff, gradient=False)
    assert inference.value == pytest.approx(expected.value, rel=2e-4, abs=1e-12)
    assert inference.gradient is None
    assert batched.counters['backward_solves'] == 3


def test_code_batch_uses_one_real_solver_call_per_chunk(monkeypatch):
    raw, velocity = problem()
    objective = RepositoryObjective(raw, code_batch_size=3)
    original = objective.e._simulate
    calls = []

    def simulate(solver, field, wavelet):
        calls.append(solver.zs.shape[1])
        return original(solver, field, wavelet)

    monkeypatch.setattr(objective.e, '_simulate', simulate)
    objective.evaluate(velocity, plan=ContrastRouter(3).plan(full=True))
    assert calls == [3]


@pytest.mark.parametrize('cutoff', [None, 20.])
def test_physical_shot_batches_match_serial_objective_and_gradient(cutoff):
    serial_problem, velocity = problem(shot_batch_size=1)
    batched_problem, _ = problem(shot_batch_size=3)
    batched_problem['observed'] = serial_problem['observed']
    serial = RepositoryObjective(serial_problem)
    batched = RepositoryObjective(batched_problem)
    expected = serial.evaluate(velocity, cutoff=cutoff)
    actual = batched.evaluate(velocity, cutoff=cutoff)
    assert actual.value == pytest.approx(expected.value, rel=2e-4, abs=1e-12)
    assert (actual.gradient - expected.gradient).norm() / expected.gradient.norm() < 2e-4
    assert serial.counters == batched.counters


@pytest.mark.parametrize('batch_size', [0, -1, True, 1.5])
def test_code_batch_size_must_be_a_positive_integer(batch_size):
    raw, _ = problem()
    with pytest.raises(ValueError, match='batch'):
        RepositoryObjective(raw, code_batch_size=batch_size)


def test_loader_respects_full_shot_batch_from_existing_spec(monkeypatch):
    from experiments import residual_ifwi_experiment as e
    from experiments import residual_ifwi_data
    cfg = e.default_config()
    cfg['model']['neurons'] = [2, 8, 8, 1]
    cfg['forward'].update(nt=48, dt_s=.001, frequency_hz=30., npad=3)
    cfg['data']['grid_spacing_m'] = 20.
    cfg['acquisition'].update(num_shots=3, source_x_indices=[3, 8, 13])
    spec = {'name': 'batch_test', 'experiment': cfg, 'control': {'kind': 'zero_output'},
            'robustness': {'condition': 'batch_test', 'noise_sigma': 0., 'noise_seed': 3}}
    initial = np.full((10, 18), 2000, dtype=np.float32)
    truth = initial.copy(); truth[4:7, 5:12] += 150
    monkeypatch.setattr(residual_ifwi_data, 'load_velocity_data',
                        lambda *_: (truth, initial, {}))
    model, background, objective, _, _ = load_problem(spec, ROOT)
    assert objective.problem['config']['training']['shot_batch_size'] is None
    assert len(objective.problem['propagators']) == 1
    assert objective.problem['propagators'][0]['last'] == 3
    assert objective.code_batch_size == 3
    assert spec['experiment']['training']['shot_batch_size'] is None
    _, _, overridden, _, _ = load_problem(spec, ROOT, shot_batch_size=2)
    assert len(overridden.problem['propagators']) == 2
    assert overridden.code_batch_size == 2


@pytest.mark.parametrize('batch_size', [0, -1])
def test_run_cli_rejects_invalid_batch_without_creating_output(tmp_path, monkeypatch, batch_size):
    import sys
    from ifwi_research import run
    baseline = tmp_path / 'baseline.json'
    baseline.write_text('{"experiment":{"training":{"shot_batch_size":null}}}')
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['run', '--baseline', str(baseline), '--research',
                        str(tmp_path / 'unused.json'), '--out', str(out),
                        '--shot-batch-size', str(batch_size)])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert not out.exists()


def test_run_records_effective_batch_override_in_exact_resume_identity(tmp_path, monkeypatch):
    import sys
    from ifwi_research import run
    from experiments import residual_ifwi_data
    raw, initial_tensor = problem()
    cfg = raw['config']
    cfg['training'].update(epochs=1, checkpoint_interval=1, shot_batch_size=None)
    spec = {'name': 'batch_identity', 'experiment': cfg, 'control': {'kind': 'zero_output'},
            'robustness': {'condition': 'batch_identity', 'noise_sigma': 0., 'noise_seed': 3}}
    baseline, research, out = tmp_path/'baseline.json', tmp_path/'research.json', tmp_path/'out'
    baseline.write_text(json.dumps(spec)); research.write_text('{"method":"routing"}')
    initial = initial_tensor[0].numpy(); truth = initial.copy(); truth[4:7, 5:12] += 150
    monkeypatch.setattr(residual_ifwi_data, 'load_velocity_data', lambda *_: (truth, initial, {}))
    monkeypatch.setattr(sys, 'argv', ['run', '--baseline', str(baseline), '--research', str(research),
                        '--out', str(out), '--device', 'cpu', '--shot-batch-size', '2'])
    run.main()
    identity = json.loads((out/'identity.json').read_text())
    assert identity['execution'] == {'shot_batch_size': 2, 'code_batch_size': 2}
    assert identity['baseline']['experiment']['training']['shot_batch_size'] == 2
    assert json.loads(baseline.read_text())['experiment']['training']['shot_batch_size'] is None
    assert json.loads((out/'summary.json').read_text())['completed_updates'] == 1

"""CPU oracles for one-solver source-code batching and shot-mean scaling.

Serial loops exist only in these reference tests. The production encoded
problem must contain one solver spanning all code channels.
"""
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

import core_speed as speed
from rnn_fd import Propagator2D, rnn2D
from experiments import residual_ifwi_experiment as experiment


@pytest.fixture(scope="module")
def tiny_problem():
    cfg = experiment.default_config()
    cfg['model']['neurons'] = [2, 8, 8, 1]
    cfg['acquisition'].update(num_shots=13, source_x_indices=list(range(1, 27, 2)))
    cfg['forward'].update(dt_s=.001, nt=24, frequency_hz=60., npad=3)
    cfg['training'].update(epochs=1, checkpoint_interval=1)
    truth = np.linspace(2300, 2800, 8, dtype=np.float32)[:, None] + np.zeros((8, 29), np.float32)
    problem = experiment.build_forward_problem(truth, cfg, 'cpu')
    return problem, torch.from_numpy(truth[None] * .96)


class PressureEnergyOracle:
    """Same raw forward/adjoint pressure-square sums, before H normalization."""
    def __init__(self, solver):
        self.source = None
        self.receiver = None
        self.handle = solver.fd.register_forward_hook(self.forward)

    def forward(self, module, args, outputs):
        pressure = outputs[1]
        source = pressure.detach().double().square().sum(dim=(0, 1))
        self.source = source if self.source is None else self.source + source
        if pressure.requires_grad:
            pressure.register_hook(self.backward)

    def backward(self, gradient):
        energy = gradient.detach().double().square().sum(dim=(0, 1))
        self.receiver = energy if self.receiver is None else self.receiver + energy

    def close(self):
        self.handle.remove()


def relative_error(actual, expected):
    return float(torch.linalg.vector_norm((actual - expected).double()) /
                 torch.linalg.vector_norm(expected.double()).clamp_min(1e-30))


def legacy_encoded_solver(problem, weights):
    """Unchanged historical single-code class is the independent serial oracle."""
    geometry, cfg = problem['geometry_tensors'], problem['config']
    fd = cfg['forward']
    vmax = problem['propagators'][0]['solver'].fd.vmax
    solver = rnn2D(nz=geometry['nz'], nx=geometry['nx'], xs=geometry['xs'][:, :1],
        zs=geometry['zs'][:, :1], xr=geometry['xr'][:, :1], zr=geometry['zr'][:, :1],
        dz=cfg['data']['grid_spacing_m'], dt=fd['dt_s'], npad=fd['npad'],
        order=2, vmax=vmax, log_para=fd['pml_reflection'], freeSurface=fd['free_surface'])
    encoded = speed.EncodedPropagator2D(geometry['nz'], geometry['nx'],
        cfg['data']['grid_spacing_m'], fd['dt_s'], fd['npad'], 2,
        fd['free_surface'], fd['pml_reflection'], vmax, torch.float32, 'cpu')
    encoded.source_z, encoded.source_x = geometry['zs'][0], geometry['xs'][0]
    encoded.weights = weights
    solver.fd = encoded
    return solver


def batch_gradient_and_energy(problem, velocity, encoder):
    encoded = encoder.problem(7)
    assert len(encoded['propagators']) == 1
    solver = encoded['propagators'][0]['solver']
    leaf = velocity.clone().requires_grad_()
    observer = PressureEnergyOracle(solver)
    try:
        predicted = solver(leaf, problem['wavelet'])[2]
        loss = speed.normalized_encoding_loss(predicted, encoded['observed'], 13)
        loss.backward()
        return predicted.detach(), loss.detach(), leaf.grad.clone(), observer.source, observer.receiver, encoded
    finally:
        observer.close()


def serial_gradient_and_energy(problem, velocity, matrix):
    predictions, gradients, losses, sources, adjoints = [], [], [], [], []
    for weights in matrix:
        solver = legacy_encoded_solver(problem, weights)
        observer = PressureEnergyOracle(solver)
        leaf = velocity.clone().requires_grad_()
        try:
            prediction = solver(leaf, problem['wavelet'])[2]
            target = torch.einsum('s,mstr->mtr', weights, problem['observed'])[:, None]
            loss = speed.normalized_encoding_loss(prediction, target, 13)
            loss.backward()
            predictions.append(prediction.detach())
            losses.append(loss.detach())
            gradients.append(leaf.grad.clone())
            sources.append(observer.source)
            adjoints.append(observer.receiver)
        finally:
            observer.close()
    return (torch.cat(predictions, 1), torch.stack(losses).sum(),
        torch.stack(gradients).sum(0), torch.stack(sources).sum(0), torch.stack(adjoints).sum(0))


def test_reference_shot_mean_loss_does_not_average_code_channels():
    assert hasattr(speed, 'normalized_encoding_loss'), 'Explicit shot-mean loss is missing'
    prediction = torch.ones(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = speed.normalized_encoding_loss(prediction, target, 13)
    assert float(loss.detach()) == pytest.approx(3/13)
    loss.backward()
    assert torch.equal(prediction.grad, torch.full_like(prediction, 2/(2*13*4*5)))


def test_normalized_loss_rejects_invalid_shapes_and_reference_count():
    assert hasattr(speed, 'normalized_encoding_loss')
    prediction = torch.ones(1, 2, 3, 4)
    for invalid in (0, -1, 1.5):
        with pytest.raises(ValueError):
            speed.normalized_encoding_loss(prediction, prediction, invalid)
    with pytest.raises(ValueError):
        speed.normalized_encoding_loss(prediction, prediction[:, :1], 13)


def test_global_encoder_uses_one_solver_and_preserves_raw_seeded_weights(tiny_problem):
    problem, _ = tiny_problem
    encoder = speed.SourceEncoder(problem, 3, 3)
    result = encoder.problem(7)
    assert len(result['propagators']) == 1, 'Production still solves code channels separately'
    assert result['propagators'][0]['first'] == 0
    assert result['propagators'][0]['last'] == 3
    assert result['propagators'][0]['solver'].zs.shape == (1, 3)
    assert result['observed'].shape == (1, 3, 24, 29)
    assert result['reference_shots'] == 13
    matrix = result['encoding_matrix']
    assert matrix.shape == (3, 13)
    for index in range(3):
        raw = encoder.weights(7, index)
        rng = np.random.RandomState((3*1000003 + 7*101 + index) % 2**32)
        assert np.array_equal(raw, rng.choice([-1., 1.], size=13))
        torch.testing.assert_close(matrix[index], torch.as_tensor(raw, dtype=matrix.dtype)/math.sqrt(3))
    expected = torch.einsum('ks,mstr->mktr', matrix, problem['observed'])
    torch.testing.assert_close(result['observed'], expected)


@pytest.mark.parametrize('count,sizes', [(2, [7, 6]), (3, [5, 4, 4])])
def test_interleaved_groups_cover_unequal_sizes_without_group_averaging(tiny_problem, count, sizes):
    problem, _ = tiny_problem
    encoder = speed.SourceEncoder(problem, count, 3, grouping='interleaved')
    result = encoder.problem(7)
    matrix = result['encoding_matrix']
    assert result['group_sizes'] == sizes
    assert torch.equal(matrix.square().sum(0), torch.ones(13))
    for group in range(count):
        expected_mask = torch.arange(13) % count == group
        assert torch.equal(matrix[group] != 0, expected_mask)
        assert torch.equal(matrix[group, expected_mask].abs(), torch.ones(sizes[group]))
    gram = matrix.T @ matrix
    cross_group = (torch.arange(13)[:, None] % count) != (torch.arange(13)[None] % count)
    assert torch.count_nonzero(gram[cross_group]) == 0
    assert len(result['propagators']) == 1


def test_identity_encoding_is_the_same_thirteen_shot_problem(tiny_problem):
    problem, velocity = tiny_problem
    encoder = speed.SourceEncoder(problem, 13, 3, grouping='identity')
    encoded = encoder.problem(7)
    assert torch.equal(encoded['encoding_matrix'], torch.eye(13))
    assert torch.equal(encoded['observed'], problem['observed'])
    with torch.no_grad():
        predicted = encoded['propagators'][0]['solver'](velocity, problem['wavelet'])[2]
        original = problem['propagators'][0]['solver'](velocity, problem['wavelet'])[2]
    assert relative_error(predicted, original) < 2e-6


@pytest.mark.parametrize('grouping,count', [('global', 1), ('global', 2), ('global', 3),
                                         ('interleaved', 2), ('interleaved', 3)])
def test_cpu_batch_matches_serial_waveforms_velocity_gradient_and_raw_energies(tiny_problem, grouping, count):
    problem, velocity = tiny_problem
    encoder = speed.SourceEncoder(problem, count, 3, grouping=grouping)
    batch = batch_gradient_and_energy(problem, velocity, encoder)
    serial = serial_gradient_and_energy(problem, velocity, batch[-1]['encoding_matrix'])
    for actual, expected, label in zip(batch[:5], serial, ['prediction', 'loss', 'velocity gradient', 'S', 'A']):
        assert relative_error(actual, expected) < 1e-5, (label, relative_error(actual, expected))
    batch_h = (batch[3]*batch[4]).sqrt()
    serial_h = (serial[3]*serial[4]).sqrt()
    assert relative_error(batch_h, serial_h) < 1e-5
    assert batch[2].abs().max() > 0


def test_column_orthogonal_hadamard_codes_preserve_full_shot_loss_and_gradient():
    assert hasattr(speed, 'normalized_encoding_loss')
    h = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(4):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    matrix = h[:, :13]/4
    torch.testing.assert_close(matrix.T @ matrix, torch.eye(13, dtype=torch.float64), rtol=0, atol=0)
    generator = torch.Generator().manual_seed(12)
    prediction = torch.randn(1, 13, 5, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn(prediction.shape, generator=generator, dtype=torch.float64)
    full = speed.normalized_encoding_loss(prediction, target, 13)
    encoded = speed.normalized_encoding_loss(torch.einsum('ks,mstr->mktr', matrix, prediction),
        torch.einsum('ks,mstr->mktr', matrix, target), 13)
    full_gradient, = torch.autograd.grad(full, prediction, retain_graph=True)
    encoded_gradient, = torch.autograd.grad(encoded, prediction)
    torch.testing.assert_close(full, encoded, rtol=2e-15, atol=1e-15)
    torch.testing.assert_close(full_gradient, encoded_gradient, rtol=2e-14, atol=1e-16)


@pytest.mark.parametrize('free_surface', [True, False])
def test_vectorized_native_source_injection_adapter_matches_original_without_base_step(free_surface, monkeypatch):
    from ifwi_research.batched_sources import BatchedEncodedPropagator2D
    native = Propagator2D(8, 9, 15., .001, npad=3, freeSurface=free_surface,
                          vmax=3000., dtype=torch.float64)
    module = BatchedEncodedPropagator2D(8, 9, 15., .001, npad=3,
        freeSurface=free_surface, vmax=3000., dtype=torch.float64)
    module.source_z = torch.tensor([1, 2, 3])
    module.source_x = torch.tensor([1, 4, 7])
    module.weights = torch.eye(3, dtype=torch.float64)
    field = torch.arange(2*3*module.nz_pad*module.nx_pad, dtype=torch.float64).reshape(2, 3, module.nz_pad, module.nx_pad)/100
    amplitude = torch.tensor([1.25, -.75], dtype=torch.float64)
    z = module.source_z[None].expand(2, -1)
    x = module.source_x[None].expand(2, -1)
    sources = [amplitude, z, x, 2, torch.arange(9)]
    velocity = torch.full((2, 8, 9), 2500., dtype=torch.float64)
    states = [field*.7, field, field*.02, field*-.03]
    expected = native(velocity, sources, *(state.clone() for state in states))

    def forbidden_native_step(*args, **kwargs):
        pytest.fail('Encoded adapter still calls the serial native source-injection step')

    monkeypatch.setattr(Propagator2D, '_Propagator2D___step_rnncell', forbidden_native_step)
    actual = module(velocity, sources, *(state.clone() for state in states))
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_grouping_and_batch_validation_rejects_invalid_requests(tiny_problem):
    problem, _ = tiny_problem
    for count, grouping in [(0, 'global'), (-1, 'global'), (14, 'interleaved'), (2, 'identity'), (2, 'unknown')]:
        with pytest.raises(ValueError):
            speed.SourceEncoder(problem, count, 3, grouping=grouping)
    speed.validate_speedup({'source_encoding': {'supershots': 2, 'seed': 3, 'grouping': 'interleaved'}}, problem['config'])
    with pytest.raises(ValueError):
        speed.validate_speedup({'source_encoding': {'supershots': 2, 'seed': 3, 'grouping': 'unknown'}}, problem['config'])


def test_different_shot_receiver_layout_is_rejected(tiny_problem):
    problem, _ = tiny_problem
    geometry = dict(problem['geometry_tensors'])
    geometry['xr'] = geometry['xr'].clone()
    geometry['xr'][0, 1, 0] += 1
    with pytest.raises(ValueError, match='same receiver layout'):
        speed.SourceEncoder(dict(problem, geometry_tensors=geometry), 2, 3)


def test_existing_batched_injector_accumulates_collocated_sources():
    from ifwi_research.batched_sources import BatchedEncodedPropagator2D
    fd = BatchedEncodedPropagator2D(8, 9, 15., .001, npad=3, vmax=3000.)
    fd.source_z = torch.tensor([1, 1, 3])
    fd.source_x = torch.tensor([2, 2, 4])
    fd.weights = torch.tensor([[1., 2., -3.], [.5, -.5, 2.]])
    zero = torch.zeros(1, 2, fd.nz_pad, fd.nx_pad)
    sources = [torch.tensor([2.]), torch.ones(1, 2, dtype=torch.long),
               torch.ones(1, 2, dtype=torch.long), 2, torch.arange(9)]
    current = fd(torch.full((1, 8, 9), 2500.), sources,
                 zero.clone(), zero.clone(), zero.clone(), zero.clone())[1]
    expected = torch.zeros_like(current)
    for channel in range(2):
        for source in range(3):
            expected[0, channel, fd.source_z[source], fd.source_x[source]+fd.npad] += 2*fd.weights[channel, source]
    torch.testing.assert_close(current, expected, rtol=0, atol=0)

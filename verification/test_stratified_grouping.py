"""CPU contracts for stratified membership, private draws and shot-mean loss."""
import itertools
import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

import core_speed as speed
from experiments import residual_ifwi_experiment as experiment


@pytest.fixture(scope="module")
def problem_factory():
    problems = {}

    def build(shots):
        if shots not in problems:
            config = experiment.default_config()
            config['acquisition'].update(num_shots=shots, source_x_indices=list(range(1, shots + 1)))
            config['forward'].update(dt_s=.001, nt=4, frequency_hz=60., npad=3)
            truth = np.full((8, max(9, shots + 2)), 2500., dtype=np.float32)
            problems[shots] = experiment.build_forward_problem(truth, config, 'cpu')
        return problems[shots]

    return build


@pytest.mark.parametrize('shots,groups,sorted_sizes', [
    (48, 4, [12, 12, 12, 12]), (52, 4, [13, 13, 13, 13]),
    (52, 5, [10, 10, 10, 11, 11]), (7, 3, [2, 2, 3]),
    (3, 3, [1, 1, 1]), (1, 1, [1]),
])
def test_each_block_partitions_sources_once_without_amplitude_scaling(problem_factory, shots, groups, sorted_sizes):
    encoder = speed.SourceEncoder(problem_factory(shots), groups, 19, grouping='stratified')
    for update in (0, 1, 17):
        encoded = encoder.problem(update)
        matrix = encoded['encoding_matrix']
        membership = matrix.abs().argmax(0)
        assert matrix.shape == (groups, shots)
        assert torch.equal((matrix != 0).sum(0), torch.ones(shots, dtype=torch.long))
        assert torch.equal(matrix.square().sum(0), torch.ones(shots))
        assert sorted(encoded['group_sizes']) == sorted_sizes
        assert encoded['group_sizes'] == (matrix != 0).sum(1).tolist()
        for start in range(0, shots, groups):
            block = membership[start:start + groups]
            assert len(torch.unique(block)) == len(block)
        assert encoded['reference_shots'] == shots
        assert encoded['encoding_grouping'] == 'stratified'
        assert len(encoded['propagators']) == 1
        assert encoded['propagators'][0]['last'] == groups
        assert torch.equal(encoded['propagators'][0]['solver'].fd.weights, matrix)
        expected_observed = torch.einsum('ks,mstr->mktr', matrix, encoder.base['observed'])
        torch.testing.assert_close(encoded['observed'], expected_observed, rtol=0, atol=0)


def test_stratified_is_accepted_by_speedup_validation(problem_factory):
    speed.validate_speedup({'source_encoding': {
        'supershots': 4, 'seed': 19, 'grouping': 'stratified',
    }}, problem_factory(48)['config'])


def test_stratified_rejects_more_groups_than_shots(problem_factory):
    with pytest.raises(ValueError):
        speed.SourceEncoder(problem_factory(3), 4, 19, grouping='stratified')


def test_membership_and_signs_are_seeded_per_update_without_global_rng_effects(problem_factory):
    problem = problem_factory(52)
    encoder = speed.SourceEncoder(problem, 4, 19, grouping='stratified')
    twin = speed.SourceEncoder(problem, 4, 19, grouping='stratified')
    other = speed.SourceEncoder(problem, 4, 20, grouping='stratified')
    numpy_state, torch_state, python_state = np.random.get_state(), torch.get_rng_state(), random.getstate()
    matrices = {update: encoder.encoding_matrix(update) for update in (0, 1, 17)}
    for update in (17, 0, 1, 0):
        assert torch.equal(encoder.encoding_matrix(update), matrices[update])
        assert torch.equal(twin.encoding_matrix(update), matrices[update])
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    assert np.array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()
    assert not torch.equal(matrices[0].abs(), matrices[1].abs())
    assert not torch.equal(matrices[0].abs(), other.encoding_matrix(0).abs())
    # Selected signs retain the historical private sign stream; permutation
    # draws must not consume it or alter checkpoint seed/update semantics.
    for update, matrix in matrices.items():
        for group in range(4):
            rng = np.random.RandomState((19 * 1000003 + update * 101 + group) % 2**32)
            raw = torch.as_tensor(rng.choice([-1., 1.], size=52), dtype=matrix.dtype)
            mask = matrix[group] != 0
            assert torch.equal(matrix[group, mask], raw[mask])


def test_blocks_are_independently_permuted_and_boundary_neighbors_can_share_group(problem_factory):
    encoder = speed.SourceEncoder(problem_factory(52), 4, 19, grouping='stratified')
    memberships = [encoder.encoding_matrix(update).abs().argmax(0) for update in range(8)]
    assert any(not torch.equal(labels[:4], labels[4:8]) for labels in memberships)
    assert any(labels[boundary - 1] == labels[boundary]
               for labels in memberships for boundary in range(4, 52, 4))
    partial = speed.SourceEncoder(problem_factory(7), 3, 19, grouping='stratified')
    assert len({int(partial.encoding_matrix(update).abs().argmax(0)[-1])
                for update in range(32)}) == 3


@pytest.mark.parametrize('update', [0, 1, 17])
def test_exact_finite_sign_average_is_identity_and_preserves_original_shot_mean(problem_factory, monkeypatch, update):
    encoder = speed.SourceEncoder(problem_factory(5), 2, 19, grouping='stratified')
    residual = torch.arange(1, 41, dtype=torch.float64).reshape(2, 5, 2, 2)
    gram_sum = torch.zeros(5, 5, dtype=torch.float64)
    loss_sum = torch.zeros((), dtype=torch.float64)
    for signs in itertools.product((-1., 1.), repeat=5):
        # Enumerate every assignment of independent source signs conditional
        # on real stratified membership, rather than approximate with sampling.
        monkeypatch.setattr(encoder, 'weights', lambda _, __, signs=signs: np.asarray(signs))
        matrix = encoder.encoding_matrix(update).double()
        gram_sum += matrix.T @ matrix
        encoded_residual = torch.einsum('ks,mstr->mktr', matrix, residual)
        loss = speed.normalized_encoding_loss(encoded_residual, torch.zeros_like(encoded_residual), 5)
        assert float(loss) == pytest.approx(float(encoded_residual.square().sum()) / 40)
        loss_sum += loss
    assert torch.equal(gram_sum / 32, torch.eye(5, dtype=torch.float64))
    torch.testing.assert_close(loss_sum / 32, residual.square().mean(), rtol=0, atol=1e-12)


@pytest.mark.parametrize('grouping,groups', [('global', 3), ('interleaved', 3), ('identity', 7)])
def test_legacy_matrices_and_group_sizes_keep_seed_update_meaning(problem_factory, grouping, groups):
    encoder = speed.SourceEncoder(problem_factory(7), groups, 19, grouping=grouping)
    result = encoder.problem(17)
    if grouping == 'identity':
        expected, sizes = torch.eye(7), [1] * 7
    else:
        raw = np.stack([np.random.RandomState((19 * 1000003 + 17 * 101 + group) % 2**32)
                        .choice([-1., 1.], size=7) for group in range(groups)])
        expected = torch.tensor(raw, dtype=torch.float32)
        if grouping == 'global':
            expected /= math.sqrt(groups)
            sizes = [7, 7, 7]
        else:
            expected *= torch.arange(groups)[:, None] == torch.arange(7)[None] % groups
            sizes = [3, 2, 2]
    assert torch.equal(result['encoding_matrix'], expected)
    assert result['group_sizes'] == sizes
    default = speed.SourceEncoder(problem_factory(7), 3, 19)
    explicit = speed.SourceEncoder(problem_factory(7), 3, 19, grouping='global')
    assert torch.equal(default.encoding_matrix(17), explicit.encoding_matrix(17))

"""Cyclic row shifts relabel interleaved groups; they do not change membership."""
from types import SimpleNamespace
import itertools

import numpy as np
import pytest
import torch

import core_speed
from core_speed import SourceEncoder, normalized_encoding_loss
from experiments.shifted_interleaved import ShiftedInterleavedAcquisition


def fake_problem(monkeypatch, nshots=52):
    def solver(problem, count):
        return SimpleNamespace(fd=SimpleNamespace(dtype=torch.float32, weights=None))
    monkeypatch.setattr(core_speed, 'build_code_solver', solver)
    observed = torch.arange(nshots*15, dtype=torch.float32).reshape(1,nshots,5,3)
    xs = torch.arange(nshots)[None]
    receivers = torch.arange(3)[None,None].expand(1,nshots,3)
    return dict(observed=observed, geometry_tensors=dict(nz=3,nx=nshots+2,
        xs=xs,zs=torch.ones_like(xs),xr=receivers,zr=torch.ones_like(receivers)))


@pytest.mark.parametrize('fields', [2,4])
def test_52_shot_shift_only_relabels_groups_and_keeps_solver_code_consistent(monkeypatch, fields):
    problem = fake_problem(monkeypatch)
    acquisition = ShiftedInterleavedAcquisition(problem, fields=fields, seed=3)
    fixed = SourceEncoder(problem, fields, 3, grouping='interleaved')
    shifts = set()
    for update in range(32):
        batch, = acquisition.batches(update)
        code = batch['encoding_matrix']
        shift = batch['encoding_shift']; shifts.add(shift)
        fixed_code = fixed.encoding_matrix(update)
        assert torch.equal(code, torch.roll(fixed_code, shifts=shift, dims=0))
        membership = code.abs().argmax(dim=0)
        assert torch.equal(membership, (torch.arange(52)+shift)%fields)
        assert torch.equal(code.abs().T@code.abs(), fixed_code.abs().T@fixed_code.abs())
        assert torch.equal(code.abs().sum(dim=0), torch.ones(52))
        assert batch['encoding_membership_changed'] is False
        assert batch['reference_shots'] == 52
        assert batch['group_sizes'] == [52//fields]*fields
        assert torch.equal(batch['propagators'][0]['solver'].fd.weights, code)
        assert torch.equal(batch['observed'], torch.einsum('ks,mstr->mktr', code, problem['observed']))
    assert shifts == set(range(fields))


def test_private_shift_determinism_independent_signs_and_global_rng_unchanged(monkeypatch):
    problem = fake_problem(monkeypatch)
    acquisition = ShiftedInterleavedAcquisition(problem, fields=4, seed=41)
    np.random.seed(19); torch.manual_seed(23)
    numpy_before = np.random.get_state(); torch_before = torch.get_rng_state().clone()
    codes = [next(acquisition.batches(update))['encoding_matrix'].clone() for update in range(8)]
    for update, code in reversed(list(enumerate(codes))):
        assert torch.equal(next(acquisition.batches(update))['encoding_matrix'], code)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1],numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before,torch.get_rng_state())
    # Undo only the row relabeling before comparing freshly drawn signs.
    signs = [torch.roll(code, shifts=-acquisition.shift(update), dims=0).sum(dim=0)
             for update,code in enumerate(codes)]
    assert all(not torch.equal(left,right) for left,right in zip(signs,signs[1:]))


@pytest.mark.parametrize('nshots,fields', [(52,2),(52,4),(13,4),(5,2)])
def test_unequal_group_sizes_and_objective_are_invariant_under_row_shift(monkeypatch,nshots,fields):
    problem = fake_problem(monkeypatch,nshots)
    acquisition = ShiftedInterleavedAcquisition(problem,fields=fields,seed=3)
    fixed = SourceEncoder(problem,fields,3,grouping='interleaved')
    residual = torch.linspace(-2,3,nshots*15).reshape(1,nshots,5,3)
    for update in range(9):
        batch, = acquisition.batches(update)
        code = batch['encoding_matrix']; old = fixed.encoding_matrix(update)
        encoded = torch.einsum('ks,mstr->mktr',code,residual)
        expected = torch.einsum('ks,mstr->mktr',old,residual)
        assert batch['group_sizes'] == code.ne(0).sum(dim=1).tolist()
        torch.testing.assert_close(normalized_encoding_loss(encoded,torch.zeros_like(encoded),nshots),
                                   normalized_encoding_loss(expected,torch.zeros_like(expected),nshots))


def test_shifted_physical_view_retains_full_identity_acquisition_and_cap(monkeypatch):
    acquisition = ShiftedInterleavedAcquisition(fake_problem(monkeypatch),batch_size=13,fields=4)
    physical = acquisition.physical()
    assert physical is acquisition.physical()
    assert physical.grouping == 'identity'
    batches = list(physical.batches(7))
    assert len(batches) == 4
    assert [batch['observed'].shape[1] for batch in batches] == [13]*4
    assert all(batch['reference_shots'] == 52 for batch in batches)
    assert all(torch.equal(batch['encoding_matrix'],torch.eye(13)) for batch in batches)


@pytest.mark.parametrize('update', [-1,1.5,True])
def test_shift_rejects_invalid_update_without_changing_solver_weights(monkeypatch,update):
    acquisition = ShiftedInterleavedAcquisition(fake_problem(monkeypatch),fields=4)
    with pytest.raises(ValueError):
        next(acquisition.batches(update))
    assert acquisition.encoders[0].propagators[0]['solver'].fd.weights is None


@pytest.mark.parametrize('fields', [2,4])
def test_exact_sign_expectation_is_identity_and_original_shot_mean(monkeypatch,fields):
    acquisition = ShiftedInterleavedAcquisition(fake_problem(monkeypatch,5),fields=fields)
    encoder = acquisition.encoders[0]
    # Integer-valued per-shot traces make each normalized loss exactly
    # representable; the finite-sign algebra does not require a float tolerance.
    residual = (5*torch.arange(1,6,dtype=torch.float64)).reshape(1,5,1,1).expand(1,5,5,3)
    gram_sum = torch.zeros(5,5,dtype=torch.float64)
    loss_sum = torch.zeros((),dtype=torch.float64)
    for signs in itertools.product((-1.,1.),repeat=5):
        monkeypatch.setattr(encoder,'weights',lambda _,__,signs=signs:np.asarray(signs))
        matrix = encoder.encoding_matrix(7).double()
        gram_sum += matrix.T@matrix
        encoded = torch.einsum('ks,mstr->mktr',matrix,residual)
        loss_sum += normalized_encoding_loss(encoded,torch.zeros_like(encoded),5)
    assert torch.equal(gram_sum/32,torch.eye(5,dtype=torch.float64))
    torch.testing.assert_close(loss_sum/32,residual.square().mean(),rtol=0,atol=0)

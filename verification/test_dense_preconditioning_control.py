"""The raw full-shot control uses the native physical gradient and one Adam step."""
import copy

import pytest
import torch

from core_speed import normalized_encoding_loss
from experiments import dense_encoding_physics as dense
from experiments.encoding_batch_correctness import _clear_graph_cache
from verification.test_dense_encoding_physics import fixture


def test_raw_control_parameter_gradient_matches_direct_physical_autograd(monkeypatch):
    problem,model,_ = fixture()
    oracle = copy.deepcopy(model)
    acquisition = dense.DenseAcquisition(problem,batch_size=2)
    full = dense.DenseAcquisition(problem,batch_size=5)
    batch, = full.batches(1)
    solver = batch['propagators'][0]['solver']
    try:
        prediction = solver(vmodel=oracle(),segment_wavelet=batch['wavelet'],option=0)[2]
        loss = normalized_encoding_loss(prediction,batch['observed'],5)
        loss.backward()
    finally:
        _clear_graph_cache(solver)
    def forbidden(*args,**kwargs):
        raise AssertionError('raw control must skip illumination observers, hooks and direction')
    monkeypatch.setattr(dense,'illumination_direction',forbidden)
    monkeypatch.setattr(dense,'PressureEnergy',forbidden)
    for encoder in acquisition.encoders:
        monkeypatch.setattr(encoder.propagators[0]['solver'].fd,'register_forward_hook',forbidden)
    events=[]
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    row = dense.train_dense_step(model,optimizer,acquisition,0,
        illumination_preconditioning=False,record_cost=lambda event,**kw:events.append(kw))
    assert row['illumination_preconditioning'] is False
    assert row['illumination']['enabled'] is False
    assert row['loss'] == pytest.approx(float(loss.detach()),rel=2e-6)
    assert row['fields'] == 5 and row['batch_fields'] == [2,2,1]
    for parameter,expected in zip(model.parameters(),oracle.parameters()):
        torch.testing.assert_close(parameter.grad,expected.grad,rtol=2e-5,atol=1e-8)
        assert int(optimizer.state[parameter]['step']) == 1
    for counter in ('forward_fields_attempt','forward_fields_complete',
                    'backward_fields_attempt','backward_fields_complete'):
        assert sum(event.get(counter,0) for event in events) == 5


@pytest.mark.parametrize('failure',['partial_optimizer','post_commit_check'])
def test_raw_control_preserves_uncertain_and_committed_failure_counts(monkeypatch,failure):
    from experiments.encoding_training_step import CommittedUpdateError,UncertainOptimizerError
    problem,model,_ = fixture()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    if failure == 'partial_optimizer':
        def partial():
            with torch.no_grad():
                next(model.parameters()).add_(1.)
            raise RuntimeError('partial Adam failure')
        monkeypatch.setattr(optimizer,'step',partial)
        error_type = UncertainOptimizerError
    else:
        original_check = dense.check_velocity; calls=0
        def post_check(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError('post-commit validation failure')
            return original_check(*args)
        monkeypatch.setattr(dense,'check_velocity',post_check)
        error_type = CommittedUpdateError
    with pytest.raises(error_type) as caught:
        dense.train_dense_step(model,optimizer,dense.DenseAcquisition(problem,batch_size=2),0,
                               illumination_preconditioning=False)
    if failure == 'partial_optimizer':
        assert caught.value.attempted_update == 1
    else:
        assert caught.value.completed_updates == 1
        assert all(int(state['step']) == 1 for state in optimizer.state.values())


def test_default_preconditioner_is_identical_to_explicit_true_and_differs_from_raw():
    problem,model,_ = fixture()
    explicit = copy.deepcopy(model); raw = copy.deepcopy(model)
    acquisition = dense.DenseAcquisition(problem,batch_size=2)
    rows=[]
    for current,kw in ((model,{}),(explicit,{'illumination_preconditioning':True}),
                       (raw,{'illumination_preconditioning':False})):
        rows.append(dense.train_dense_step(current,torch.optim.Adam(current.parameters(),lr=1e-4),
                                           acquisition,0,**kw))
    assert rows[0]['illumination_preconditioning'] is True
    assert rows[1]['illumination_preconditioning'] is True
    assert rows[0]['illumination'] == rows[1]['illumination']
    assert rows[0]['loss'] == rows[1]['loss'] == rows[2]['loss']
    for parameter,other in zip(model.parameters(),explicit.parameters()):
        assert torch.equal(parameter,other)
        assert torch.equal(parameter.grad,other.grad)
    assert any(not torch.allclose(parameter.grad,other.grad,rtol=1e-3,atol=1e-8)
               for parameter,other in zip(model.parameters(),raw.parameters()))


@pytest.mark.parametrize('invalid', [None,0,1,'false'])
def test_invalid_preconditioning_flag_fails_before_work_or_optimizer(invalid):
    problem,model,_ = fixture()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4); events=[]
    before=copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError,match='illumination_preconditioning'):
        dense.train_dense_step(model,optimizer,dense.DenseAcquisition(problem,batch_size=2),0,
            illumination_preconditioning=invalid,record_cost=lambda event,**kw:events.append(kw))
    assert not events and not optimizer.state
    assert all(torch.equal(value,before[key]) for key,value in model.state_dict().items())

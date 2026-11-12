"""Real CPU FD oracles for dense geometry and a single microbatched update."""
import copy
import numpy as np
import pytest
import torch

from experiments import residual_ifwi_experiment as original
from ifwi_research.batched_sources import BatchedEncodedPropagator2D
from ifwi_research.encoding_illumination import illumination_direction

try:
    from experiments import dense_encoding_physics as dense
except ImportError:
    dense = None


def fixture(nshots=5):
    assert dense is not None, 'dense physics implementation is required'
    cfg = original.default_config()
    cfg['model']['neurons'] = [2, 8, 8, 1]
    cfg['acquisition'].update(num_shots=nshots, source_x_indices=list(range(2, 2+3*nshots, 3)),
                              source_margin_cells=1)
    cfg['forward'].update(dt_s=.001, nt=18, frequency_hz=60., npad=3)
    truth = np.linspace(2300, 2800, 8, dtype=np.float32)[:, None] + np.zeros((8, 19), np.float32)
    problem = dense.build_dense_problem(truth, cfg, 'cpu')
    torch.manual_seed(3)
    model = original.VelocityParameterization(truth*.96, 15., [2, 8, 8, 1],
                                               parameterization='residual', zero_residual_head=False)
    return problem, model, truth


@pytest.mark.parametrize('count,first,last', [(48,26,261), (52,16,271)])
def test_dense_positions_preserve_actual_five_cell_spacing_and_margin(count, first, last):
    assert dense is not None
    positions = dense.generate_sources(288, 15., count)
    assert len(positions) == count and positions[0] == first and positions[-1] == last
    assert np.array_equal(np.diff(positions), np.full(count-1, 5))
    assert min(positions) >= 10 and max(positions) < 278


@pytest.mark.parametrize('args', [(288,15.,52,75.,20), (10,15.,3,75.,1),
                                  (288,15.,48,76.,10), (288,15.,0,75.,10)])
def test_geometry_rejects_nonfitting_or_non_grid_spacing(args):
    assert dense is not None
    with pytest.raises(ValueError): dense.generate_sources(*args)


def test_identity_chunks_slice_source_injection_and_global_denominator():
    problem, _, _ = fixture()
    acquisition = dense.DenseAcquisition(problem, batch_size=2)
    batches = list(acquisition.batches(1))
    assert [b['observed'].shape[1] for b in batches] == [2,2,1]
    assert [b['reference_shots'] for b in batches] == [5,5,5]
    for batch, positions in zip(batches, ([2,5],[8,11],[14])):
        solver = batch['propagators'][0]['solver']
        assert isinstance(solver.fd, BatchedEncodedPropagator2D)
        assert solver.fd.source_x.tolist() == positions
        assert torch.equal(batch['encoding_matrix'], torch.eye(len(positions)))
        assert not batch['observed'].requires_grad
    assert batches[0]['propagators'][0]['solver'] is list(acquisition.batches(2))[0]['propagators'][0]['solver']


def test_physical_unequal_microbatches_match_raw_maps_direction_parameter_gradient_and_adam(monkeypatch):
    problem, model, _ = fixture()
    other = copy.deepcopy(model)
    captures = []
    def capture(gradient, source, receiver, *args, **kwargs):
        result = illumination_direction(gradient, source, receiver, *args, **kwargs)
        captures.append(tuple(x.detach().clone() for x in (gradient,source,receiver,result[0])))
        return result
    monkeypatch.setattr(dense, 'illumination_direction', capture)
    full_opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    chunk_opt = torch.optim.Adam(other.parameters(), lr=1e-4)
    full_acquisition = dense.DenseAcquisition(problem,batch_size=5)
    micro_acquisition = dense.DenseAcquisition(problem,batch_size=2)
    full = dense.train_dense_step(model, full_opt, full_acquisition, 0)
    events = []
    micro = dense.train_dense_step(other, chunk_opt, micro_acquisition, 0,
                                  record_cost=lambda event, **kw: events.append(kw))
    assert micro['loss'] == pytest.approx(full['loss'], rel=2e-6)
    assert len(captures) == 2, 'illumination is applied once after summing raw maps'
    for x,y in zip(*captures): torch.testing.assert_close(x,y,rtol=2e-5,atol=1e-12)
    for maps in captures:
        h = (maps[1]*maps[2]).sqrt()
        assert torch.isfinite(h).all() and h.max() > 0
    torch.testing.assert_close((captures[0][1]*captures[0][2]).sqrt(),
                               (captures[1][1]*captures[1][2]).sqrt(),rtol=2e-5,atol=1e-12)
    for p,q in zip(model.parameters(),other.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=1e-8)
        torch.testing.assert_close(p,q,rtol=2e-6,atol=1e-8)
        assert int(chunk_opt.state[q]['step']) == 1
    assert micro['fields'] == 5 and micro['batch_fields'] == [2,2,1]
    for counter in ('forward_fields_attempt','forward_fields_complete',
                    'backward_fields_attempt','backward_fields_complete'):
        assert sum(e.get(counter,0) for e in events) == 5
    for batch in micro_acquisition.batches(2):
        solver = batch['propagators'][0]['solver']
        assert not solver.fd._forward_hooks
        for owner in (solver,solver.fd):
            for name in ('velocity','vpadding'):
                value=getattr(owner,name,None)
                assert not isinstance(value,torch.Tensor) or not value.requires_grad


@pytest.mark.parametrize('grouping,fields',[('global',1),('global',2),('interleaved',2),('stratified',2)])
def test_encoded_fields_reuse_adapter_and_count_actual_solves(grouping,fields):
    problem, model, _ = fixture()
    acquisition = dense.DenseAcquisition(problem,batch_size=2,grouping=grouping,fields=fields)
    batch, = list(acquisition.batches(1))
    assert batch['reference_shots'] == 5
    assert batch['propagators'][0]['solver'].fd.source_x.numel() == 5
    assert isinstance(batch['propagators'][0]['solver'].fd,BatchedEncodedPropagator2D)
    events=[]
    row=dense.train_dense_step(model,torch.optim.Adam(model.parameters(),lr=1e-4),acquisition,0,
                               record_cost=lambda event,**kw:events.append(kw))
    assert row['fields'] == fields
    assert sum(e.get('forward_fields_complete',0)for e in events) == fields
    assert sum(e.get('backward_fields_complete',0)for e in events) == fields


def test_evaluation_forces_identity_at_cap_and_compares_both_targets():
    problem,model,truth=fixture()
    assets=dict(truth=truth,clean=problem['observed'].numpy(),noisy=problem['observed'].numpy()+.01)
    events=[]
    metrics,pred,velocity=dense.evaluate_dense(model,
        dense.DenseAcquisition(problem,batch_size=2,grouping='global',fields=1),assets,
        record_cost=lambda event,**kw:events.append(kw))
    assert pred.shape == (1,5,18,19) and velocity.shape == truth.shape
    assert [e['fields'] for e in events if e.get('forward_fields_complete')] == [2,2,1]
    assert sum(e.get('backward_fields_attempt',0) for e in events) == 0
    for name in ('clean','noisy'):
        assert metrics[name+'_mse'] == pytest.approx(np.mean((pred.astype(np.float64)-assets[name])**2))


def test_supplied_assets_bypass_truth_simulation_and_validate_shape():
    problem,_,truth=fixture()
    rebuilt=dense.build_dense_problem(truth,problem['config'],'cpu',observed=problem['observed'],wavelet=problem['wavelet'])
    assert torch.equal(rebuilt['observed'],problem['observed'])
    with pytest.raises(ValueError):
        dense.build_dense_problem(truth,problem['config'],'cpu',observed=problem['observed'][:,:2])


def test_stop_before_commit_leaves_parameters_and_optimizer_untouched():
    problem,model,_=fixture();before=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4);calls=0
    class Stop(Exception):pass
    def stop():
        nonlocal calls
        calls+=1
        if calls==5:raise Stop()
    with pytest.raises(Stop):
        dense.train_dense_step(model,optimizer,dense.DenseAcquisition(problem,batch_size=2),0,check_stop=stop)
    assert not optimizer.state
    assert all(torch.equal(v,before[k])for k,v in model.state_dict().items())


def test_partial_optimizer_failure_is_uncertain_and_post_commit_failure_is_marked(monkeypatch):
    from experiments.encoding_training_step import UncertainOptimizerError,CommittedUpdateError
    problem,model,_=fixture();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    def partial():
        with torch.no_grad():next(model.parameters()).add_(1.)
        raise RuntimeError('partial step')
    monkeypatch.setattr(optimizer,'step',partial)
    with pytest.raises(UncertainOptimizerError):
        dense.train_dense_step(model,optimizer,dense.DenseAcquisition(problem,batch_size=2),0)
    problem,model,_=fixture();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    real_check=dense.check_velocity;calls=0
    def check(*args):
        nonlocal calls
        calls+=1
        if calls==2:raise ValueError('post commit')
        return real_check(*args)
    monkeypatch.setattr(dense,'check_velocity',check)
    with pytest.raises(CommittedUpdateError)as error:
        dense.train_dense_step(model,optimizer,dense.DenseAcquisition(problem,batch_size=2),0)
    assert error.value.completed_updates == 1


@pytest.mark.parametrize('nshots,cap,sizes',[(48,12,[12,12,12,12]),(52,13,[13,13,13,13]),
                                           (52,12,[12,12,12,12,4])])
def test_actual_dense_grid_chunk_limits(nshots,cap,sizes):
    assert dense is not None
    cfg=original.default_config();cfg['acquisition'].update(num_shots=nshots)
    cfg['forward'].update(nt=2,npad=2,dt_s=.001)
    truth=np.full((4,288),2500.,dtype=np.float32)
    problem=dense.build_dense_problem(truth,cfg,'cpu',observed=np.zeros((1,nshots,2,288),np.float32))
    batches=list(dense.DenseAcquisition(problem,batch_size=cap).batches(1))
    assert [b['observed'].shape[1]for b in batches] == sizes
    assert all(b['reference_shots']==nshots for b in batches)
    assert problem['geometry']['source_spacings_m'] == [75.]*(nshots-1)


def test_hard_gated_step_uses_precommit_masks_then_advances_progress():
    from experiments.gated_siren import GatedSirenVelocity
    from ifwi_research.masked_adam import MaskedAdam
    problem,_,truth=fixture()
    model=GatedSirenVelocity(truth*.96,15.)
    model.set_progress(200)  # Must be replaced by the supplied committed count.
    first=model.net.linear[0].weight;before=first.detach().clone()
    optimizer=MaskedAdam(model.parameters(),lr=1e-4,weight_decay=.1)
    row=dense.train_dense_step(model,optimizer,dense.DenseAcquisition(problem,batch_size=2),2)
    assert row['completed_updates']==3 and int(model.completed_updates)==3 and model.active_channels==33
    assert torch.equal(first[32:],before[32:])
    assert torch.equal(optimizer.state[first]['age'][:32],torch.ones_like(optimizer.state[first]['age'][:32]))
    assert not optimizer.state[first]['age'][32:].any()
    assert not optimizer.state[first]['exp_avg'][32:].any()


@pytest.mark.parametrize('fields',[0,False,-1,1.5])
def test_invalid_encoded_field_count_is_not_silently_replaced(fields):
    problem,_,_=fixture()
    with pytest.raises(ValueError):
        dense.DenseAcquisition(problem,grouping='global',fields=fields)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_missing_real_dense_assets_never_implicitly_launches_production_solve(device):
    assert dense is not None
    cfg=original.default_config();cfg['acquisition'].update(num_shots=52)
    with pytest.raises(ValueError,match='dense observed assets'):
        dense.build_dense_problem(np.full((94,288),2500.,dtype=np.float32),cfg,device)

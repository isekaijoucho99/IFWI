"""Tiny CPU tests for the callable batch/serial physical correctness gate."""
from pathlib import Path
import importlib
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import residual_ifwi_experiment as reference


@pytest.fixture(scope='module')
def lib():
    assert (ROOT/'experiments/encoding_batch_correctness.py').is_file(), 'correctness module not implemented'
    return importlib.import_module('experiments.encoding_batch_correctness')


@pytest.fixture(scope='module')
def tiny():
    torch.set_num_threads(1)
    cfg = reference.default_config()
    cfg['model']['neurons'] = [2, 8, 8, 1]
    cfg['acquisition'].update(num_shots=13, source_x_indices=list(range(1, 27, 2)))
    cfg['forward'].update(dt_s=.001, nt=24, frequency_hz=60., npad=3)
    cfg['training'].update(epochs=1, checkpoint_interval=1)
    truth = np.linspace(2300, 2800, 8, dtype=np.float32)[:,None]+np.zeros((8,29),np.float32)
    problem = reference.build_forward_problem(truth,cfg,'cpu')
    return problem,torch.from_numpy(truth[None]*.96)


def test_tiny_all_cases_match_and_cost_counts_each_physical_field(lib,tiny):
    problem,velocity = tiny
    costs = []
    report = lib.run_correctness(problem,velocity,record_cost=lambda event,**details:costs.append((event,details)),update=7)
    assert report['passed'] is True
    assert len(report['cases']) == 6
    assert report['actual_forward_fields'] == report['actual_backward_fields'] == 48
    assert report['cost_totals'] == dict(forward_fields_attempt=48,forward_fields_complete=48,
                                       backward_fields_attempt=48,backward_fields_complete=48)
    assert report['Adam_updates'] == 0
    for key in report['cost_totals']:
        assert sum(details.get(key,0)for _,details in costs) == report['cost_totals'][key]
    assert all(row['phase']=='correctness' and row['NT']==24 and row['wall_seconds']>=0 for _,row in costs)
    assert all(case['passed'] and case['metrics']['gradient']['reference_l2']>0 for case in report['cases'])


def test_stop_before_oracle_attempt_is_not_wrapped_or_charged(lib,tiny):
    from ifwi_research.suite_control import StopRequested
    problem,velocity=tiny
    def stop(event,**details):
        if 'forward_fields_attempt'in details:raise StopRequested('test stop')
    with pytest.raises(StopRequested)as caught:
        lib.run_correctness(problem,velocity,record_cost=stop)
    assert caught.value.report['status']=='STOPPED_BY_USER'
    assert all(value==0 for value in caught.value.report['cost_totals'].values())


def test_reference_is_detached_cpu_and_releases_solver_graph_cache(lib,tiny,monkeypatch):
    problem,velocity = tiny
    solvers = []
    original = lib.build_code_solver
    def capture(*args,**kwargs):
        solver=original(*args,**kwargs);solvers.append(solver);return solver
    monkeypatch.setattr(lib,'build_code_solver',capture)
    matrix=torch.eye(13)[:3]
    result=lib.serial_reference(problem,velocity,matrix,update=7)
    assert result['prediction'].shape==(1,3,24,29)
    for key in ('prediction','loss','gradient','source','adjoint','H','direction'):
        tensor=result[key]
        assert tensor.device.type=='cpu' and not tensor.requires_grad and tensor.grad_fn is None
    assert result['cost_totals']['forward_fields_complete']==3
    assert result['cost_totals']['backward_fields_complete']==3
    assert solvers
    for solver in solvers:
        for attr in ('velocity','vpadding'):
            value=getattr(solver.fd,attr,None)
            assert not isinstance(value,torch.Tensor)or not(value.requires_grad or value.grad_fn is not None)


def test_negative_result_retains_report_and_stops_at_failed_case(lib,tiny,monkeypatch):
    problem,velocity=tiny
    original=lib.serial_reference
    def wrong_scale(*args,**kwargs):
        result=original(*args,**kwargs)
        result['loss']=result['loss']*13
        return result
    monkeypatch.setattr(lib,'serial_reference',wrong_scale)
    with pytest.raises(lib.CorrectnessError)as error:
        lib.run_correctness(problem,velocity)
    report=error.value.report
    assert report['passed']is False
    assert len(report['cases'])==1
    assert report['cases'][0]['metrics']['loss']['passed']is False
    assert report['actual_forward_fields']==report['actual_backward_fields']==2
    assert report['Adam_updates']==0


def test_input_flags_and_caller_velocity_gradient_unchanged(lib,tiny):
    problem,velocity=tiny
    caller=velocity.clone().requires_grad_();caller.grad=torch.ones_like(caller)
    before=caller.detach().clone();grad=caller.grad.clone()
    flags=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.get_num_threads())
    lib.serial_reference(problem,caller,torch.eye(13)[:1])
    assert torch.equal(caller.detach(),before)and torch.equal(caller.grad,grad)
    assert flags==(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.get_num_threads())


def test_native_split_reference_still_matches_identity(lib,tiny):
    problem,velocity=tiny
    cfg=reference.default_config()
    cfg['acquisition'].update(num_shots=13,source_x_indices=list(range(1,27,2)))
    cfg['forward'].update(dt_s=.001,nt=24,frequency_hz=60.,npad=3)
    cfg['training']['shot_batch_size']=4
    # Cached data identity is fixed; a split changes only how native oracle executes.
    truth=np.linspace(2300,2800,8,dtype=np.float32)[:,None]+np.zeros((8,29),np.float32)
    from rnn_fd import rnn2D
    split=dict(problem,config=cfg,propagators=[])
    geometry=problem['geometry_tensors'];fd=cfg['forward']
    for first in range(0,13,4):
        last=min(first+4,13)
        solver=rnn2D(nz=8,nx=29,xs=geometry['xs'][:,first:last],zs=geometry['zs'][:,first:last],
            xr=geometry['xr'][:,first:last],zr=geometry['zr'][:,first:last],
            dz=cfg['data']['grid_spacing_m'],dt=fd['dt_s'],npad=fd['npad'],order=2,
            vmax=float(truth.max()),log_para=fd['pml_reflection'],freeSurface=fd['free_surface'])
        split['propagators'].append(dict(solver=solver,first=first,last=last))
    report=lib.run_correctness(split,velocity)
    assert report['passed']is True
    assert report['actual_forward_fields']==report['actual_backward_fields']==48


def test_invalid_input_fails_before_any_physical_cost(lib,tiny):
    problem,velocity=tiny
    costs=[]
    with pytest.raises((ValueError,lib.CorrectnessError)):
        lib.run_correctness(problem,velocity.expand(2,-1,-1),record_cost=lambda *args,**kwargs:costs.append(1))
    assert not costs


def test_shared_loss_helper_factor_error_is_not_hidden_by_matching_oracles(lib,tiny,monkeypatch):
    problem,velocity=tiny
    original=lib.normalized_encoding_loss
    monkeypatch.setattr(lib,'normalized_encoding_loss',lambda prediction,target,n:13*original(prediction,target,n))
    with pytest.raises(lib.CorrectnessError)as error:lib.run_correctness(problem,velocity)
    report=error.value.report
    assert 'normalization'in report['error'].lower()
    assert report['actual_forward_fields']==1 and report['actual_backward_fields']==0


def test_bad_global_code_scale_is_rejected_before_its_physical_solve(lib,tiny,monkeypatch):
    problem,velocity=tiny
    original=lib.SourceEncoder
    class BadScale(original):
        def encoding_matrix(self,update):
            value=super().encoding_matrix(update)
            return value*2**.5 if self.grouping=='global'and self.supershots==2 else value
    monkeypatch.setattr(lib,'SourceEncoder',BadScale)
    with pytest.raises(lib.CorrectnessError)as error:lib.run_correctness(problem,velocity)
    report=error.value.report
    assert 'code column'in report['error'].lower()
    assert report['actual_forward_fields']==report['actual_backward_fields']==2

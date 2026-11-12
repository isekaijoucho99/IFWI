"""Real small FD updates: accounting, normalized objective, and safe stop boundary."""
import copy
import numpy as np
import pytest
import torch
from core_speed import SourceEncoder, normalized_encoding_loss
from experiments import residual_ifwi_experiment as original
try:
    from experiments.encoding_training_step import train_step, evaluate
except ImportError:
    train_step = evaluate = None


def fixture():
    cfg=original.default_config()
    cfg['model']['neurons']=[2,8,8,1]
    cfg['acquisition'].update(num_shots=13,source_x_indices=list(range(1,27,2)))
    cfg['forward'].update(dt_s=.001,nt=20,frequency_hz=60.,npad=3)
    cfg['training'].update(epochs=1,shot_batch_size=13)
    truth=np.linspace(2300,2800,8,dtype=np.float32)[:,None]+np.zeros((8,29),np.float32)
    p=original.build_forward_problem(truth,cfg,'cpu')
    torch.manual_seed(3)
    m=original.VelocityParameterization(truth*.96,15.,[2,8,8,1],parameterization='residual',zero_residual_head=True)
    return p,m,torch.optim.Adam(m.parameters(),lr=1e-4),truth


@pytest.mark.parametrize('grouping,fields',[('global',2),('interleaved',3),('identity',13)])
def test_one_physical_backward_and_one_adam_commit(grouping,fields):
    assert train_step is not None
    p,m,opt,_=fixture();encoder=SourceEncoder(p,fields,3,grouping=grouping)
    events=[]
    row=train_step(m,opt,encoder,0,record_cost=lambda event,**kw:events.append(kw),arm_name=grouping)
    assert row['completed_updates']==1 and np.isfinite(row['loss'])
    for counter in ('forward_fields_attempt','forward_fields_complete','backward_fields_attempt','backward_fields_complete'):
        assert sum(e.get(counter,0)for e in events)==fields
    assert all(int(s['step'])==1 for s in opt.state.values())
    assert len(opt.state)==len(tuple(m.parameters()))
    assert all(torch.isfinite(p).all()for p in m.parameters())


def test_stop_before_adam_does_not_commit_parameters():
    assert train_step is not None
    p,m,opt,_=fixture();before=copy.deepcopy(m.state_dict());calls=0
    class Stop(Exception):pass
    def check():
        nonlocal calls
        calls+=1
        if calls==4:raise Stop()
    with pytest.raises(Stop):train_step(m,opt,SourceEncoder(p,2,3),0,check_stop=check)
    assert not opt.state
    assert all(torch.equal(v,before[k])for k,v in m.state_dict().items())


def test_evaluation_uses_one_shared_prediction_for_both_data_metrics():
    assert evaluate is not None
    p,m,_,truth=fixture();events=[]
    clean=p['observed'].numpy();assets=dict(truth=truth,clean=clean,noisy=clean+.01)
    report,pred,vel=evaluate(m,p,assets,record_cost=lambda event,**kw:events.append(kw),label='test')
    assert sum(e.get('forward_fields_complete',0)for e in events)==13
    assert sum(e.get('backward_fields_complete',0)for e in events)==0
    assert report['clean_mse']==pytest.approx(float(np.mean((pred.astype(np.float64)-clean)**2)))
    assert report['noisy_mse']==pytest.approx(float(np.mean((pred.astype(np.float64)-assets['noisy'])**2)))
    assert vel.shape==(8,29)


def test_optimizer_partial_exception_is_marked_uncertain(monkeypatch):
    from experiments.encoding_training_step import UncertainOptimizerError
    p,m,opt,_=fixture()
    def partial():
        with torch.no_grad():next(m.parameters()).add_(1.)
        raise RuntimeError('injected partial Adam failure')
    monkeypatch.setattr(opt,'step',partial)
    with pytest.raises(UncertainOptimizerError)as caught:
        train_step(m,opt,SourceEncoder(p,2,3),0)
    assert caught.value.attempted_update==1


def test_post_commit_validation_failure_reports_committed_count(monkeypatch):
    from experiments import encoding_training_step as step
    p,m,opt,_=fixture();checks=0;original_check=step.check_velocity
    def check(*args):
        nonlocal checks
        checks+=1
        if checks==2:raise ValueError('injected post-update failure')
        original_check(*args)
    monkeypatch.setattr(step,'check_velocity',check)
    with pytest.raises(step.CommittedUpdateError)as caught:
        train_step(m,opt,SourceEncoder(p,2,3),0)
    assert caught.value.completed_updates==1
    assert all(int(state['step'])==1 for state in opt.state.values())

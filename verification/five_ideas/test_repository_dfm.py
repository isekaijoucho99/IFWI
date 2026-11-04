"""CPU integration contracts for a deliberately UNQUALIFIED DFM and native RNN-FD.

These tiny tests validate wiring and safeguards, never geological prior quality.
"""
import copy
from pathlib import Path

import pytest
import torch

from verification.five_ideas.test_repository_bridge import problem

ROOT=Path(__file__).resolve().parents[2]
pytestmark=pytest.mark.skipif(not (ROOT/'core_runner.py').exists(),reason='Original IFWI checkout unavailable')


@pytest.fixture
def prior_path(tmp_path):
    from ifwi_research.dfm import TwoTimeTransport,conditional_group_loss
    torch.manual_seed(123); prior=TwoTimeTransport(8)
    clean=torch.linspace(-.8,.2,64).reshape(1,1,8,8)
    optimizer=torch.optim.Adam(prior.parameters(),lr=.001)
    conditional_group_loss(prior,clean,t=.2,r=.8,group_size=4,
                           generator=torch.Generator().manual_seed(123),kernel=3).backward()
    optimizer.step()
    path=tmp_path/'unqualified_native_bridge_fixture.pt'
    meta=dict(schema='ifwi_small_dfm_v1',trained_steps=1,width=8,spacing_m=20.,
              mean_mps=2000.,scale_mps=1000.,condition_kernel=3,patch_size=8,
              toy_only=True,qualification={'status':'unqualified','evidence':None})
    torch.save({'meta':meta,'model':prior.state_dict()},path)
    return path


def load_smoke_prior(path):
    from ifwi_research.dfm import FrozenDFMPrior
    with pytest.warns(UserWarning,match='smoke/testing'):
        return FrozenDFMPrior(path,20.,allow_unqualified=True)


def make_engine(path,**options):
    from experiments import residual_ifwi_experiment as e
    from ifwi_research.engine import ResearchEngine,ResearchConfig
    cfg,initial,raw,objective=problem(); torch.manual_seed(3)
    config=dict(method='dfm',prior_start=1,prior_every=1)
    config.update(options)
    engine=ResearchEngine(e.model_from_config(initial,cfg),torch.from_numpy(initial)[None],
                          objective,20.,ResearchConfig(**config),prior=load_smoke_prior(path))
    return engine


def equal_tensors(actual,expected):
    if isinstance(expected,torch.Tensor):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for k in expected: equal_tensors(actual[k],expected[k])
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected): equal_tensors(a,b)
    else: assert actual==expected


def test_native_dfm_runs_frozen_prior_vjp_and_real_candidate_checks(prior_path):
    engine=make_engine(prior_path)
    weights=copy.deepcopy(engine.prior.model.state_dict())
    old=engine.model().detach().clone(); report=engine.step()
    assert report['prior_calls']>0 and report['prior']['ratio']>0
    assert report['step']['accepted']
    after=engine.objective.evaluate(engine.model().detach(),gradient=False).value
    assert after==pytest.approx(report['step']['physical_after'],rel=1e-7)
    assert after<report['step']['physical_before']
    assert not torch.equal(old,engine.model())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in engine.model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in engine.prior.model.parameters())
    equal_tensors(engine.prior.model.state_dict(),weights)
    assert report['counters']['backward_solves']==3
    assert report['counters']['forward_solves']==3*(1+report['step']['forward_calls'])


def test_native_dfm_before_enable_matches_unmodified_baseline(prior_path):
    from ifwi_research.engine import ResearchEngine,ResearchConfig
    delayed=make_engine(prior_path,prior_start=20)
    baseline=ResearchEngine(copy.deepcopy(delayed.model),delayed.model().detach().clone(),
                            problem()[3],20.,ResearchConfig())
    for _ in range(3):
        a=delayed.step(); b=baseline.step()
        equal_tensors(delayed.model.state_dict(),baseline.model.state_dict())
        equal_tensors(delayed.optimizer.state_dict(),baseline.optimizer.state_dict())
        assert a['physical_objective']==b['physical_objective']
    assert delayed.prior.calls==0


def test_native_dfm_resume_restores_proposal_identity_and_optimizer(prior_path):
    engine=make_engine(prior_path); engine.step()
    saved=copy.deepcopy(engine.state_dict()); engine.step()
    resumed=make_engine(prior_path); resumed.load_state_dict(saved); resumed.step()
    equal_tensors(resumed.model.state_dict(),engine.model.state_dict())
    equal_tensors(resumed.optimizer.state_dict(),engine.optimizer.state_dict())
    assert resumed.prior.calls==engine.prior.calls
    assert resumed.objective.counters==engine.objective.counters


def test_native_dfm_invalid_candidates_roll_back_model_and_optimizer(prior_path):
    engine=make_engine(prior_path,learning_rate=1e20,max_trials=1)
    saved_model=copy.deepcopy(engine.model.state_dict())
    saved_opt=copy.deepcopy(engine.optimizer.state_dict())
    report=engine.step()
    assert not report['step']['accepted']
    assert report['step']['forward_calls']==0
    assert report['step']['recovery_trials']==1
    equal_tensors(engine.model.state_dict(),saved_model)
    equal_tensors(engine.optimizer.state_dict(),saved_opt)


def test_native_dfm_recovery_uses_physical_vjp_not_prior_gradient(prior_path):
    engine=make_engine(prior_path,prior_ratio=.3)
    velocity=engine.model(); parameters=list(engine.model.parameters())
    measured=engine.objective.evaluate(velocity.detach(),gradient=True)
    physical=torch.autograd.grad(velocity,parameters,measured.gradient)
    start=[p.detach().clone() for p in parameters]
    # Force the real Adam update uphill without substituting an optimizer or
    # physical objective. Ordinary trials must reject the stale moment.
    for p,g in zip(parameters,physical):
        engine.optimizer.state[p]={'step':torch.tensor(20.),'exp_avg':-100*g.clone(),
                                   'exp_avg_sq':100*g.square()}
    report=engine.step()
    assert report['step']['accepted'] and report['step']['recovery']=='gradient_descent'
    scale=engine.config.learning_rate*report['step']['step_scale']
    for p,x,g in zip(parameters,start,physical):
        expected=x.clone().add_(g,alpha=-scale)
        torch.testing.assert_close(p,expected,rtol=0,atol=0)
    assert not engine.optimizer.state

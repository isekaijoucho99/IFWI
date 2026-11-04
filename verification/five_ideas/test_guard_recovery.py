"""Real optimizer regression tests for transactional physical step recovery."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from ifwi_research.safety import guarded_optimizer_step, rng_state


def scalar_problem(lr=.1):
    model=nn.Linear(1,1,bias=False).double()
    with torch.no_grad(): model.weight.fill_(1.)
    return model,torch.optim.Adam(model.parameters(),lr=lr)


def equal_state(actual,expected):
    if isinstance(expected,torch.Tensor):
        assert torch.equal(actual,expected)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected: equal_state(actual[key],expected[key])
    elif isinstance(expected,(tuple,list)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected): equal_state(a,b)
    else: assert actual==expected


def test_guard_recovers_after_adam_momentum_turns_uphill():
    # Adam crosses zero on update 12; scaling its stale moment cannot descend.
    model,opt=scalar_problem()
    recoveries=[]
    for _ in range(16):
        opt.zero_grad(); before=model.weight.square().sum(); before.backward()
        report=guarded_optimizer_step(model,opt,lambda:model.weight.square().sum(),
                                      float(before.detach()),max_trials=5)
        assert report['accepted'], 'A finite physical descent direction must not freeze behind stale Adam moments'
        assert report['physical_after']<report['physical_before']
        assert opt.param_groups[0]['lr']==.1
        if report.get('recovery')=='gradient_descent':
            recoveries.append(report)
            # The next call must not inherit the moment that caused rejection.
            assert model.weight not in opt.state
    assert recoveries
    assert recoveries[0]['optimizer_trials']==5
    assert recoveries[0]['recovery_trials']==1
    assert recoveries[0]['trials']==6
    assert recoveries[0]['forward_calls']==6


def test_recovery_backtracks_supplied_physical_gradient_and_resets_history():
    model,opt=scalar_problem(lr=4.)
    # Intentionally uphill proposal; recovery uses the independently supplied
    # physical derivative. Three trial steps fail to decrease; an eighth step is exact.
    model.weight.grad=torch.tensor([[-1.]],dtype=torch.float64)
    report=guarded_optimizer_step(model,opt,lambda:model.weight.square().sum(),1.,
                                  fallback_gradients=[torch.tensor([[2.]],dtype=torch.float64)],max_trials=4)
    assert report['accepted'] and report['recovery']=='gradient_descent'
    assert report['step_scale']==.125
    assert float(model.weight.detach())==0.
    assert not opt.state and opt.param_groups[0]['lr']==4.


def test_rejected_recovery_restores_parameters_buffers_gradients_moments_and_rng():
    model,opt=scalar_problem()
    model.register_buffer('evaluations',torch.zeros((),dtype=torch.float64))
    model.weight.grad=torch.full_like(model.weight,-2.); opt.step()
    model.weight.grad=torch.full_like(model.weight,-1.)
    saved_model=copy.deepcopy(model.state_dict()); saved_opt=copy.deepcopy(opt.state_dict())
    saved_grad=model.weight.grad.clone(); saved_rng=rng_state()
    def objective():
        model.evaluations.add_(1); torch.rand(2); np.random.rand(); random.random()
        return model.weight.square().sum()
    report=guarded_optimizer_step(model,opt,objective,float(model.weight.detach().square()),max_trials=3)
    assert not report['accepted'] and report['recovery_trials']==3
    equal_state(model.state_dict(),saved_model); equal_state(opt.state_dict(),saved_opt)
    equal_state(model.weight.grad,saved_grad); equal_state(rng_state(),saved_rng)


@pytest.mark.parametrize('candidate_value',[float('inf'),float('nan')])
def test_recovery_never_accepts_nonfinite_physical_value(candidate_value):
    model,opt=scalar_problem(); model.weight.grad=torch.full_like(model.weight,2.)
    saved=copy.deepcopy(model.state_dict())
    report=guarded_optimizer_step(model,opt,lambda:candidate_value,1.,max_trials=2)
    assert not report['accepted'] and report['recovery_trials']==2
    equal_state(model.state_dict(),saved); assert not opt.state


def test_invalid_recovery_candidate_is_not_evaluated():
    model,opt=scalar_problem(); model.weight.grad=torch.full_like(model.weight,2.)
    calls=[]
    report=guarded_optimizer_step(model,opt,lambda:calls.append(1) or 0.,1.,
                                  valid_candidate=lambda:False,max_trials=2)
    assert not report['accepted'] and report['recovery_trials']==2
    assert not calls and report['forward_calls']==0
    assert float(model.weight.detach())==1. and not opt.state


def test_exception_during_recovery_restores_transaction():
    model,opt=scalar_problem(); model.weight.grad=torch.full_like(model.weight,-1.)
    saved_model=copy.deepcopy(model.state_dict()); saved_opt=copy.deepcopy(opt.state_dict())
    saved_rng=rng_state(); calls=[]
    def objective():
        torch.rand(2); calls.append(1)
        if len(calls)>2: raise RuntimeError('physical evaluation failed')
        return model.weight.square().sum()
    with pytest.raises(RuntimeError,match='physical evaluation failed'):
        guarded_optimizer_step(model,opt,objective,1.,max_trials=2)
    equal_state(model.state_dict(),saved_model); equal_state(opt.state_dict(),saved_opt)
    equal_state(rng_state(),saved_rng)


@pytest.mark.parametrize('gradient',[float('inf'),float('nan')])
def test_guard_rejects_nonfinite_gradient_before_trying_an_update(gradient):
    model,opt=scalar_problem(); model.weight.grad=torch.full_like(model.weight,gradient)
    with pytest.raises(ValueError,match='finite'):
        guarded_optimizer_step(model,opt,lambda:0.,1.)
    assert float(model.weight.detach())==1. and not opt.state

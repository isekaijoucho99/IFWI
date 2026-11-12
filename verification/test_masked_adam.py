import copy

import pytest
import torch

from ifwi_research.masked_adam import MaskedAdam


@pytest.mark.parametrize('decay', [0., .2])
def test_changing_masks_match_independent_scalar_adam(decay):
    p = torch.nn.Parameter(torch.tensor([1., -2., 3.], dtype=torch.float64))
    scalars = [torch.nn.Parameter(x.clone()) for x in p.detach()]
    opts = [torch.optim.Adam([x], lr=.03, betas=(.7,.91), weight_decay=decay) for x in scalars]
    opt = MaskedAdam([p], lr=.03, betas=(.7,.91), weight_decay=decay)
    ages = torch.zeros(3, dtype=torch.int64)
    for k, flags in enumerate([[1,0,0], [1,1,0], [0,1,0], [0,0,0], [1,1,1], [1,0,1]]):
        mask = torch.tensor(flags, dtype=torch.bool)
        before = p.detach().clone()
        state = copy.deepcopy(opt.state.get(p, {}))
        grad = torch.tensor([.4+k, -.3*k, .1-k], dtype=p.dtype)
        p.grad = grad
        opt.step(masks={p:mask})
        for j in range(3):
            if mask[j]:
                scalars[j].grad = grad[j].clone()
                opts[j].step()
        ages += mask
        torch.testing.assert_close(p, torch.stack(scalars), rtol=1e-12, atol=1e-12)
        assert torch.equal(p[~mask], before[~mask])
        assert torch.equal(opt.state[p]['age'], ages)
        if state:
            for key in ['exp_avg', 'exp_avg_sq', 'age']:
                assert torch.equal(opt.state[p][key][~mask], state[key][~mask])
        for j in range(3):
            if ages[j]:
                for key in ['exp_avg', 'exp_avg_sq']:
                    torch.testing.assert_close(opt.state[p][key][j], opts[j].state[scalars[j]][key])


def test_all_active_matches_adam_and_integer_state_restores_exactly():
    torch.manual_seed(37)
    p = torch.nn.Parameter(torch.randn(2,3,dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    a = MaskedAdam([p], lr=.01, weight_decay=.15)
    b = torch.optim.Adam([q], lr=.01, weight_decay=.15)
    for _ in range(4):
        p.grad = torch.randn_like(p); q.grad = p.grad.clone()
        a.step(); b.step()
    torch.testing.assert_close(p,q,rtol=1e-12,atol=1e-12)
    saved = copy.deepcopy(a.state_dict())
    r = torch.nn.Parameter(p.detach().clone())
    c = MaskedAdam([r]); c.load_state_dict(saved)
    assert c.state[r]['age'].dtype == torch.int64
    for key in a.state[p]:
        assert torch.equal(a.state[p][key], c.state[r][key])
    p.grad = torch.randn_like(p); r.grad = p.grad.clone()
    a.step(); c.step()
    assert torch.equal(p,r)


def test_invalid_mask_rejected_before_any_mutation():
    p = torch.nn.Parameter(torch.ones(2)); q = torch.nn.Parameter(torch.ones(2))
    opt = MaskedAdam([p,q]); p.grad = torch.ones(2); q.grad = torch.ones(2)
    with pytest.raises(ValueError):
        opt.step(masks={p:torch.ones(2,dtype=torch.bool), q:torch.ones(1,dtype=torch.bool)})
    assert torch.equal(p,torch.ones(2))


def test_large_age_survives_float32_restore_without_rounding():
    p = torch.nn.Parameter(torch.ones(2))
    opt = MaskedAdam([p]); p.grad = torch.ones_like(p); opt.step()
    opt.state[p]['age'].fill_(2**24+1)
    q = torch.nn.Parameter(p.detach().clone()); restored = MaskedAdam([q])
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert torch.equal(restored.state[q]['age'],opt.state[p]['age'])


def test_float32_dense_and_no_gradient_skip_match_adam():
    torch.manual_seed(9)
    p = torch.nn.Parameter(torch.randn(8,9)); q = torch.nn.Parameter(p.detach().clone())
    opt = MaskedAdam([p],lr=.002,weight_decay=.13)
    reference = torch.optim.Adam([q],lr=.002,weight_decay=.13)
    for k in range(20):
        p.grad = None if k%4==0 else torch.randn_like(p)
        q.grad = None if p.grad is None else p.grad.clone()
        opt.step(); reference.step()
    torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-7)
    assert (opt.state[p]['age']==15).all()


@pytest.mark.parametrize('options',[{'lr':-1.},{'eps':float('nan')},{'betas':(.9,1.)},{'weight_decay':-.2}])
def test_invalid_hyperparameters(options):
    with pytest.raises(ValueError):
        MaskedAdam([torch.nn.Parameter(torch.ones(2))],**options)


def test_unsupported_adam_variant_is_rejected():
    with pytest.raises(ValueError,match='amsgrad'):
        MaskedAdam([{'params':[torch.nn.Parameter(torch.ones(2))],'amsgrad':True}])

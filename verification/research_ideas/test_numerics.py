"""Numerical contracts, including failure modes, for the five independent methods."""
import copy
import itertools
import math
import numpy as np
import pytest
import torch

torch.set_num_threads(1)


def test_conflicting_auxiliary_is_projected_and_bounded():
    from research_ideas.safety import combine_gradients
    g = [torch.tensor([1., 0.]), torch.tensor([2.])]
    a = [torch.tensor([-20., 3.]), torch.tensor([-10.])]
    out, info = combine_gradients(g, a, ratio=.25)
    delta = torch.cat([x-y for x,y in zip(out,g)])
    flat = torch.cat(g)
    assert float(flat @ delta) >= -1e-6
    assert float(delta.norm()) <= .25*float(flat.norm()) + 1e-6
    assert info['conflict']


def test_zero_physics_does_not_invent_a_direction():
    from research_ideas.safety import combine_gradients
    out, _ = combine_gradients([torch.zeros(3)], [torch.ones(3)], ratio=.3)
    assert torch.equal(out[0], torch.zeros(3))


def test_nan_gradient_rejected():
    from research_ideas.safety import combine_gradients
    with pytest.raises(ValueError):
        combine_gradients([torch.ones(2)], [torch.tensor([float('nan'), 1.])])


def test_armijo_uses_actual_adam_displacement_and_rolls_back():
    from research_ideas.safety import safeguarded_step
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad(): model.weight.fill_(1.)
    opt = torch.optim.Adam(model.parameters(), lr=1.)
    before = copy.deepcopy(model.state_dict()); state = copy.deepcopy(opt.state_dict())
    # Deliberately ascent proposal: must not mutate either model or Adam state.
    report = safeguarded_step(model, opt, [torch.tensor([[1.]],dtype=torch.double)],
                              [torch.tensor([[-1.]],dtype=torch.double)],
                              lambda: .5*model.weight.square().sum(), before_loss=.5)
    assert not report['accepted']
    assert torch.equal(model.weight, before['weight'])
    assert opt.state_dict()['state'] == state['state']


def test_armijo_backtracking_accepts_true_decrease():
    from research_ideas.safety import safeguarded_step
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad(): model.weight.fill_(1.)
    opt = torch.optim.SGD(model.parameters(), lr=10.)
    report = safeguarded_step(model,opt,[torch.ones_like(model.weight)],
        [torch.ones_like(model.weight)],lambda: .5*model.weight.square().sum(),before_loss=.5)
    assert report['accepted'] and report['alpha'] < 1
    assert .5*float(model.weight.detach().square().sum()) < .5


@pytest.mark.parametrize('shape', [(2,1,8,10),(1,2,7,9),(1,1,3,4)])
def test_haar_exact_roundtrip_and_gradient(shape):
    from research_ideas.wavelet import haar2d, inverse_haar2d
    x=torch.randn(shape,dtype=torch.double,requires_grad=True)
    coeff,size=haar2d(x); y=inverse_haar2d(coeff,size)
    torch.testing.assert_close(y,x,rtol=1e-12,atol=1e-12)
    g=torch.autograd.grad(y.square().sum(),x)[0]
    torch.testing.assert_close(g,2*x,rtol=1e-12,atol=1e-12)


def test_wavelet_identity_gate_and_bounds():
    from research_ideas.wavelet import WaveletAttention
    gate=WaveletAttention(hidden=8,strength=.5)
    x=torch.randn(1,1,9,11)
    torch.testing.assert_close(gate(x), x,rtol=2e-6,atol=2e-6)
    gate(x).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in gate.parameters())
    assert gate.last_gain.min() >= .5 and gate.last_gain.max() <= 1.5


def test_interfaces_start_at_background_with_position_gradient():
    from research_ideas.interfaces import OrderedInterfaceResidual
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.offset=torch.nn.Parameter(torch.zeros(1,12,15))
            self.register_buffer('fixed_init_mps',torch.full((1,12,15),2500.))
        def forward(self): return self.fixed_init_mps+self.offset
    model=OrderedInterfaceResidual(Base(),spacing_m=15.,interfaces=2,width_m=25.)
    torch.testing.assert_close(model(),model.fixed_init_mps,rtol=0,atol=0)
    grad=torch.autograd.grad(model().sum(),tuple(model.horizon.parameters()),allow_unused=True)
    assert any(g is not None and g.abs().sum()>0 for g in grad)
    h=model.heights(); assert torch.all(h[1:]>h[:-1])
    with torch.no_grad():
        for p in model.horizon.parameters(): p.add_(torch.randn_like(p)*3)
    h=model.heights(); assert torch.all(h[1:]>h[:-1]) and h.min()>0 and h.max()<165


def test_interface_rejects_impossible_minimum_thickness():
    from research_ideas.interfaces import OrderedInterfaceResidual
    base=torch.nn.Linear(1,1)
    base.register_buffer('fixed_init_mps',torch.full((1,4,4),2000.))
    with pytest.raises(ValueError):
        OrderedInterfaceResidual(base,spacing_m=1.,interfaces=4,min_thickness_m=1.)


def test_drift_zero_for_identical_empirical_distributions():
    from research_ideas.dfm import drift_field
    x=torch.randn(5,1,4,4)
    torch.testing.assert_close(drift_field(x,x),torch.zeros_like(x),rtol=0,atol=2e-7)


def test_drift_translation_and_stopgradient_sign():
    from research_ideas.dfm import drift_field, drift_loss
    x=torch.zeros(4,1,3,3,requires_grad=True); target=torch.ones_like(x)
    v=drift_field(x,target)
    assert torch.all(v>0)
    loss=drift_loss(x,target); grad=torch.autograd.grad(loss,x)[0]
    assert torch.all(grad<0)


def test_two_time_identity_and_training_group_contract():
    from research_ideas.dfm import TwoTimeTransport, grouped_dfm_loss
    net=TwoTimeTransport(width=8)
    x=torch.randn(8,1,8,8)
    assert torch.equal(net(x,.3,.3),x)
    loss=grouped_dfm_loss(net,x,torch.randn_like(x),torch.tensor([.1,.2]),
                          torch.tensor([.8,.9]),group_size=4)
    loss.backward(); assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.parameters())
    with pytest.raises(ValueError):
        grouped_dfm_loss(net,x,x,torch.tensor([.1]),torch.tensor([.8]),group_size=1)
    with pytest.raises(ValueError): net(x,.7,.2)


def test_patch_prior_identity_no_resize():
    from research_ideas.dfm import patch_transport
    class Identity(torch.nn.Module):
        def forward(self,x,t,r): return x
    x=torch.randn(1,1,9,11)
    out,calls=patch_transport(Identity(),x,t=.8,steps=2,patch_size=6,batch_size=2)
    torch.testing.assert_close(out,x,rtol=2e-6,atol=2e-6)
    assert calls>0


def test_router_conditional_expectation_and_full_coverage():
    from research_ideas.routing import CompressedShotRouter
    # Enumerate every possible complementary draw, not a noisy Monte Carlo assertion.
    router=CompressedShotRouter(torch.arange(4.).reshape(4,1),top_k=1,samples=1,seed=9)
    router.scores=torch.tensor([1.,2.,9.,3.])
    top,rest,p=router.distribution()
    losses=torch.tensor([2.,5.,8.,14.]); expected=0.
    for j,prob in zip(rest,p):
        weights=router.weights_for(top,rest,p,[int(j)])
        expected+=float(prob)*float(weights@losses)
    assert expected==pytest.approx(float(losses.mean()),rel=1e-6)
    full=CompressedShotRouter(torch.ones(4,2),top_k=4,samples=0)
    torch.testing.assert_close(full.sample(),torch.full((4,),.25))


def test_router_no_deterministic_biased_partial_mode():
    from research_ideas.routing import CompressedShotRouter
    with pytest.raises(ValueError):
        CompressedShotRouter(torch.ones(5,2),top_k=2,samples=0)


def test_router_rng_restores_and_all_probabilities_positive():
    from research_ideas.routing import CompressedShotRouter
    router=CompressedShotRouter(torch.randn(8,3),top_k=2,samples=3,seed=17)
    saved=router.state_dict(); a=router.sample(); router.load_state_dict(saved); b=router.sample()
    assert torch.equal(a,b)
    assert torch.all(router.distribution()[2]>0)


def test_gradient_attention_is_bounded_and_no_second_order_wavefield():
    from research_ideas.gradient_attention import GradientAttention, fit_cross_group
    net=GradientAttention(width=8,strength=.5)
    opt=torch.optim.Adam(net.parameters(),lr=.01)
    v=torch.full((1,8,10),2500.,requires_grad=True)
    ga=torch.randn_like(v,requires_grad=True); gb=ga.detach().clone()
    value=fit_cross_group(net,opt,v,ga,gb)
    gate=net(v,ga)
    assert .5<=float(gate.detach().min()) and float(gate.detach().max())<=1.5
    assert math.isfinite(value)
    assert v.grad is None and ga.grad is None

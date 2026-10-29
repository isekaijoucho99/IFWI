"""Core mathematical contracts; no expensive FWI data is needed."""
import copy
import importlib.util
import math

import pytest
import torch
from torch import nn

# A real failing test before any module is implemented.
def test_package_exists():
    assert importlib.util.find_spec('ifwi_research.wavelets') is not None

class Field(nn.Module):
    def __init__(self, h=7, w=10):
        super().__init__()
        self.register_buffer('background', torch.full((1, h, w), 2000.0))
        self.residual = nn.Parameter(torch.zeros_like(self.background))
    def forward(self):
        return self.background + self.residual

@pytest.mark.parametrize('shape', [(2, 1, 8, 10), (1, 3, 7, 9), (1, 1, 1, 3)])
def test_haar_reconstruction(shape):
    from ifwi_research.wavelets import haar_forward, haar_inverse
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    bands, size = haar_forward(x)
    y = haar_inverse(bands, size)
    torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-12)
    y.square().sum().backward()
    torch.testing.assert_close(x.grad, 2*x.detach())


def test_wavelet_identity_and_learning():
    from ifwi_research.wavelets import WaveletVelocity
    base = Field(); base.residual.data.normal_()
    model = WaveletVelocity(base, base.background, strength=0.2)
    torch.testing.assert_close(model(), base(), atol=1e-5, rtol=0)
    model().square().mean().backward()
    assert model.gates[-1].weight.grad.abs().sum() > 0
    assert base.residual.grad is not None


def test_wavelet_disabled_exact():
    from ifwi_research.wavelets import WaveletVelocity
    base = Field(); base.residual.data.normal_()
    model = WaveletVelocity(base, base.background, strength=0)
    assert torch.equal(model(), base())


def test_wavelet_gradcheck():
    from ifwi_research.wavelets import haar_forward, haar_inverse
    x = torch.randn(1, 1, 3, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: haar_inverse(*haar_forward(z)), (x,))


def test_levelset_initialization_gradients_and_units():
    from ifwi_research.interfaces import InterfaceVelocity
    base = Field(h=9, w=11)
    model = InterfaceVelocity(base, base.background.shape[-2:], spacing_m=15.,
                              depth_m=60., epsilon_m=30.)
    assert torch.equal(model(), base())
    assert float(model.eikonal_loss().detach()) < 1e-10
    weights = torch.arange(99).reshape(1,9,11).float()
    (model()*weights).sum().backward()
    assert model.shape_net[-1].weight.grad.abs().sum() > 0
    before = model().detach().clone()
    model.set_epsilon(15.)
    assert torch.equal(before, model())
    with pytest.raises(ValueError): model.set_epsilon(1.)


def test_levelset_shapes_and_external_prior():
    from ifwi_research.interfaces import InterfaceVelocity
    base = Field(h=8,w=10)
    phi = torch.arange(8)[:,None].expand(8,10)*20.-70.
    model = InterfaceVelocity(base, (8,10), 20., 70., 40., phi_reference=phi)
    torch.testing.assert_close(model.phi_reference, phi)
    assert model().shape == (1,8,10)
    assert torch.isfinite(model.regularization())


@pytest.mark.parametrize('s', [2, 3, 13, 49])
def test_orthogonal_source_codes(s):
    from ifwi_research.routing import common_contrast_basis
    q = common_contrast_basis(s, dtype=torch.float64)
    torch.testing.assert_close(q@q.T, torch.eye(s,dtype=q.dtype), atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(q[0], torch.full((s,),1/math.sqrt(s),dtype=q.dtype))


def test_routed_loss_and_gradient_expectation():
    from ifwi_research.routing import common_contrast_basis, weighted_code_loss
    s = 4; q = common_contrast_basis(s,dtype=torch.float64)
    theta = torch.randn(2,dtype=torch.float64,requires_grad=True)
    a = torch.randn(s,5,2,dtype=torch.float64)
    d = torch.randn(s,5,dtype=torch.float64)
    raw = a@theta-d; encoded=q@raw
    p=torch.tensor([.1,.3,.6],dtype=torch.float64)
    total=0.
    for i in range(s-1):
        total=total+p[i]*weighted_code_loss(encoded,torch.tensor([i+1]),p)
    exact=raw.square().mean()
    torch.testing.assert_close(total,exact,rtol=1e-12,atol=1e-12)
    g1=torch.autograd.grad(total,theta,retain_graph=True)[0]
    g2=torch.autograd.grad(exact,theta)[0]
    torch.testing.assert_close(g1,g2,rtol=1e-12,atol=1e-12)


def test_router_support_duplicates_and_resume():
    from ifwi_research.routing import ContrastRouter
    router=ContrastRouter(5, samples=20, topk=1, exploration=.2, seed=123)
    router.update_scores(torch.tensor([1,2,3,4]),torch.tensor([0.,0.,0.,100.]))
    p=router.probabilities()
    assert (p>=.05-1e-8).all()
    state=copy.deepcopy(router.state_dict())
    a=router.plan()
    restored=ContrastRouter(5,20,1,.2,123); restored.load_state_dict(state)
    b=restored.plan()
    assert torch.equal(a.indices,b.indices)
    torch.testing.assert_close(a.weights,b.weights)
    assert a.indices[0]==0 and a.weights[0]==.2
    assert len(a.indices)<=5 and a.draw_count==20


def test_router_rejects_zero_probability():
    from ifwi_research.routing import ContrastRouter
    with pytest.raises(ValueError): ContrastRouter(4,2,1,exploration=0)


def test_gradient_attention_reanchor_and_learning():
    from ifwi_research.gradient_attention import GradientAttentionVelocity
    base=Field(); model=GradientAttentionVelocity(base, base.background)
    torch.testing.assert_close(model(),base())
    g=torch.randn_like(base.background,requires_grad=True)
    model.refresh(g)
    assert not model.gradient_context.requires_grad
    with torch.no_grad():
        model.gates[-1].weight.normal_()
        base.residual.add_(torch.randn_like(base.residual))
    before=model().detach().clone()
    model.refresh(-3*g)
    torch.testing.assert_close(before,model(),rtol=0,atol=2e-4)
    with torch.no_grad(): base.residual.add_(torch.randn_like(base.residual))
    model().square().mean().backward()
    assert model.gates[-1].weight.grad.abs().sum()>0
    assert g.grad is None
    restored=GradientAttentionVelocity(Field(),base.background)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(model(),restored())


def test_dfm_transport_time_identity_and_gradient():
    from ifwi_research.dfm import TwoTimeTransport, grouped_drift_loss
    model=TwoTimeTransport(width=8)
    x=torch.randn(4,1,6,8); condition=torch.randn(1,1,6,8)
    torch.testing.assert_close(model(x,.4,.4,condition),x)
    y=model(x,0.,1.,condition)
    loss=grouped_drift_loss(y,torch.randn_like(y),temperature=1.)
    loss.backward()
    assert model.net[-1].weight.grad.abs().sum()>0
    with pytest.raises(ValueError): model(x,.5,.4,condition)
    with pytest.raises(ValueError): grouped_drift_loss(y[:1],y[:1])


def test_drift_field_direction_detach_and_equilibrium():
    from ifwi_research.dfm import drift_field
    neg=torch.tensor([[[-1.]],[[1.]]]).reshape(2,1,1,1).requires_grad_()
    pos=neg.detach()+2.
    drift=drift_field(neg,pos,temperature=1.)
    assert not drift.requires_grad and drift.mean()>0
    zero=drift_field(neg,neg.detach(),temperature=1.,exclude_self=False)
    torch.testing.assert_close(zero,torch.zeros_like(zero))


def test_projected_gradient_protects_physics():
    from ifwi_research.safety import protect_physical_gradient
    gp=torch.tensor([1.,2.,3.],dtype=torch.float64)
    for gs in (-gp,torch.zeros_like(gp),torch.tensor([-5.,4.,-10.],dtype=torch.float64)):
        mixed,info=protect_physical_gradient([gp],[gs],ratio=.25)
        assert float(gp@mixed[0])>=float(gp@gp)-1e-10
        assert torch.linalg.vector_norm(mixed[0]-gp)<=.25*gp.norm()+1e-10
    mixed,_=protect_physical_gradient([gp*0],[gp],ratio=.25)
    assert torch.equal(mixed[0],gp*0)


def test_guard_rejects_and_restores_adam_rng():
    from ifwi_research.safety import guarded_optimizer_step
    model=nn.Linear(1,1,bias=False); model.weight.data.fill_(1.)
    opt=torch.optim.Adam(model.parameters(),lr=.1)
    model.weight.grad=torch.tensor([[-1.]])
    old=copy.deepcopy(model.state_dict()); state=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone()
    def objective():
        torch.rand(2)
        return model.weight.square().sum()
    report=guarded_optimizer_step(model,opt,objective,reference_loss=1.,max_trials=3)
    assert not report['accepted']
    assert torch.equal(model.weight,old['weight'])
    assert opt.state_dict()==state
    assert torch.equal(torch.get_rng_state(),rng)


def test_guard_accepts_a_real_descent_step():
    from ifwi_research.safety import guarded_optimizer_step
    model=nn.Linear(1,1,bias=False); model.weight.data.fill_(1.)
    opt=torch.optim.Adam(model.parameters(),lr=4.)
    model.weight.grad=torch.tensor([[2.]])
    report=guarded_optimizer_step(model,opt,lambda:model.weight.square().sum(),1.,max_trials=5)
    assert report['accepted'] and report['step_scale']<1
    assert float(model.weight.detach().square())<1.
    assert opt.param_groups[0]['lr']==4.
    assert float(next(iter(opt.state.values()))['step'])==1.

import copy
from dataclasses import replace
import pytest
import torch
from torch import nn


def components(method='baseline', **kwargs):
    from ifwi_research.engine import ResearchEngine, ResearchConfig
    from ifwi_research.toy_wave import ToyWaveObjective, TinyINR
    h,w=10,14
    z=torch.arange(h)[:,None].expand(h,w).float()
    initial=(2000+10*z)[None]
    true=initial.clone(); true[:,5:8,5:10]+=200
    obj=ToyWaveObjective(true, shots=3, nt=45, spacing_m=20., dt_s=.002)
    torch.manual_seed(3)
    model=TinyINR(initial,width=8)
    cfg=ResearchConfig(method=method, learning_rate=.0001, **kwargs)
    return ResearchEngine(model,initial,obj,20.,cfg), obj


@pytest.mark.parametrize('method',['baseline','routing','wavelet','interface','gradient_attention'])
def test_numerical_wave_step_all_methods(method):
    engine,obj=components(method, audit_every=3, attention_block=2)
    start=engine.model().detach().clone()
    for _ in range(3):
        report=engine.step()
        assert torch.isfinite(engine.model()).all()
        assert report['physical_objective']>=0
    assert report['update']==3
    assert obj.counters['forward_solves']>0 and obj.counters['backward_solves']>0
    assert not torch.equal(start,engine.model())


def test_engine_resume_exact_with_routing_and_cache():
    for mode in ('routing','gradient_attention'):
        engine,obj=components(mode,attention_block=2)
        engine.step()
        checkpoint=copy.deepcopy(engine.state_dict())
        engine.step(); engine.step()
        expected=engine.model().detach().clone()
        resumed,other=components(mode,attention_block=2)
        resumed.load_state_dict(checkpoint)
        resumed.step(); resumed.step()
        torch.testing.assert_close(expected,resumed.model(),rtol=0,atol=0)
        assert obj.counters==other.counters


def test_source_basis_matches_physical_shots():
    from ifwi_research.routing import ContrastRouter
    engine,obj=components()
    v=engine.model().detach()
    a=obj.evaluate(v,gradient=True)
    b=obj.evaluate(v,plan=ContrastRouter(3).plan(full=True),gradient=True)
    assert a.value==pytest.approx(b.value,rel=1e-5,abs=1e-8)
    torch.testing.assert_close(a.gradient,b.gradient,rtol=2e-4,atol=1e-9)


def test_velocity_gradient_finite_difference():
    engine,obj=components()
    v=engine.model().detach().double()
    # Numerical solver is differentiable in v and uses its supplied dtype.
    ev=obj.evaluate(v,gradient=True)
    direction=torch.randn(v.shape,dtype=v.dtype)
    h=.5
    a=obj.evaluate(v+h*direction,gradient=False).value
    b=obj.evaluate(v-h*direction,gradient=False).value
    predicted=float((ev.gradient*direction).sum())
    assert (a-b)/(2*h)==pytest.approx(predicted,rel=5e-3,abs=1e-8)


def test_dfm_requires_checkpoint():
    with pytest.raises(ValueError,match='trained'):
        components('dfm')


def test_config_invalid_combination():
    from ifwi_research.engine import ResearchConfig
    with pytest.raises(ValueError): ResearchConfig(attention_block=1).validate()
    with pytest.raises(ValueError): ResearchConfig(method='all').validate()
    with pytest.raises(ValueError): ResearchConfig(learning_rate=float('nan')).validate()


def test_dfm_checkpoint_training_and_rejection(tmp_path):
    from ifwi_research.dfm import TwoTimeTransport, FrozenDFMPrior, conditional_group_loss
    model=TwoTimeTransport(width=8)
    clean=torch.randn(1,1,8,8)
    rng=torch.Generator().manual_seed(123)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    opt.zero_grad(); conditional_group_loss(model,clean,t=.2,r=.8,group_size=4,generator=rng).backward(); opt.step()
    meta=dict(schema='ifwi_small_dfm_v1',trained_steps=1,width=8,spacing_m=20.,mean_mps=2000.,
              scale_mps=1000.,condition_kernel=3,patch_size=8,toy_only=False)
    path=tmp_path/'synthetic_test_fixture.pt'
    torch.save({'meta':meta,'model':model.state_dict()},path)
    prior=FrozenDFMPrior(path,20.,start_t=.8,nfe=2)
    v=torch.full((1,9,13),2200.,requires_grad=True)
    target=prior.proposal(v,1)
    assert target.shape==v.shape and not target.requires_grad
    assert all(not p.requires_grad for p in prior.model.parameters())
    prior.penalty(v,target).backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    with pytest.raises(ValueError,match='spacing'): FrozenDFMPrior(path,15.)
    meta['trained_steps']=0; torch.save({'meta':meta,'model':model.state_dict()},path)
    with pytest.raises(ValueError,match='trained'): FrozenDFMPrior(path,20.)


def test_group_condition_must_be_shared():
    from ifwi_research.dfm import TwoTimeTransport
    model=TwoTimeTransport(8); x=torch.randn(4,1,5,5)
    with pytest.raises(ValueError,match='SHARED'): model(x,.2,.6,x)

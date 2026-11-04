"""Analytic candidate fixtures test coupling, not a pretrained geological DFM."""
import copy
import pytest
import torch
from torch import nn

from ifwi_research.illumination_prior import IlluminationSoftPrior, SoftPriorConfig


class AnalyticCandidateFixture:
    """Explicit deterministic fixture. It is NOT DFM and has no trained weights."""
    def __init__(self,offset=10.):
        self.offset=offset; self.model=nn.Identity().eval(); self.calls=0
        self.scale_mps=1000.; self.spacing_m=20.
        self.checkpoint_hash='analytic_fixture_not_dfm'
        self.meta={'schema':'analytic_fixture_not_dfm'}
        self.conditions=[]
    def generate(self,background_mps,anchor_mps,anchor_mask,*,sample_id):
        self.calls+=1
        self.conditions.append((background_mps.clone(),anchor_mps.clone(),anchor_mask.clone(),sample_id))
        return anchor_mps.detach()+self.offset


def make_prior(**options):
    background=torch.full((1,3,4),2000.,dtype=torch.float64)
    sensitivity=torch.tensor([[[0.,.2,.8,1.]]],dtype=torch.float64).expand_as(background).clone()
    source=AnalyticCandidateFixture()
    return IlluminationSoftPrior(source,background,sensitivity,spacing_m=20.,
        config=SoftPriorConfig(**options)),source


def test_weights_distinguish_weak_strong_and_estimated_anchors():
    prior,source=make_prior(weak_threshold=.2,strong_threshold=.8)
    torch.testing.assert_close(prior.weights,torch.tensor([[[1.,1.,0.,0.]]],dtype=torch.float64).expand(1,3,4))
    value=torch.full_like(prior.background,2100.,requires_grad=True)
    target=prior.proposal(value,1)
    assert not target.requires_grad
    _,anchor,mask,sample=source.conditions[0]
    torch.testing.assert_close(anchor,value.detach())
    torch.testing.assert_close(mask,torch.tensor([[[0.,0.,1.,1.]]],dtype=torch.float64).expand_as(mask))
    assert sample==1 and not anchor.requires_grad
    assert prior.diagnostics()['source_schema']=='analytic_fixture_not_dfm'


def test_cache_freezes_target_between_refreshes_and_restores_exactly():
    prior,source=make_prior(refresh_every=3)
    v=prior.background.clone(); first=prior.proposal(v,1)
    second=prior.proposal(v+100,2)
    assert source.calls==1 and torch.equal(first,second)
    saved=copy.deepcopy(prior.state_dict())
    expected=prior.proposal(v+200,4)
    resumed,other=make_prior(refresh_every=3); resumed.load_state_dict(saved)
    torch.testing.assert_close(resumed.proposal(v+200,4),expected,rtol=0,atol=0)
    assert other.calls==source.calls==2
    assert prior.diagnostics()['refreshes']==2


def test_zero_weight_needs_no_source_and_produces_exact_zero_gradient():
    v=torch.full((1,3,4),2000.,requires_grad=True)
    prior=IlluminationSoftPrior(None,v.detach(),torch.ones_like(v),spacing_m=20.,
                                config=SoftPriorConfig(weak_weight=0.,strong_weight=0.))
    assert not prior.enabled
    loss=prior.penalty(v,v+100); loss.backward()
    assert loss.item()==0. and torch.count_nonzero(v.grad)==0


def test_edge_penalty_does_not_directly_push_zero_weight_strong_pixels():
    prior,_=make_prior(velocity_weight=0.,edge_weight=1.,weak_threshold=.2,strong_threshold=.8)
    v=prior.background.clone().requires_grad_(); target=v.detach().clone()
    target[...,1]=target[...,1]+100
    loss=prior.penalty(v,target); loss.backward()
    assert torch.count_nonzero(v.grad[...,:2])>0
    assert torch.count_nonzero(v.grad[...,2:])==0


def test_soft_penalty_has_physical_units_independent_value_and_gradcheck():
    prior,_=make_prior(velocity_weight=1.,edge_weight=0.,weak_threshold=.2,strong_threshold=.8)
    v=prior.background.clone().requires_grad_(); target=v.detach()+100
    # Half pixels have w=1; (100/1000)^2 * 1/2 * 1/2 = .0025.
    assert prior.penalty(v,target).item()==pytest.approx(.0025)
    assert torch.autograd.gradcheck(lambda x:prior.penalty(x,target),(v,))


def test_spatial_derivative_uses_declared_meter_spacing():
    prior,_=make_prior(velocity_weight=0.,edge_weight=1.,edge_length_m=20.,weak_threshold=.2,strong_threshold=.8)
    target=prior.background.clone(); target[...,1]+=100
    # One of three x edges active: .5 * (.1*20/20)^2 /3. No vertical residual differences.
    assert prior.penalty(prior.background,target).item()==pytest.approx(1/600)


def test_missing_nonzero_source_and_zero_sensitivity_fail_closed():
    background=torch.full((1,3,4),2000.)
    with pytest.raises(ValueError,match='source'):
        IlluminationSoftPrior(None,background,torch.full_like(background,.5),spacing_m=20.)
    with pytest.raises(ValueError,match='sensitivity'):
        IlluminationSoftPrior(AnalyticCandidateFixture(),background,torch.zeros_like(background),spacing_m=20.)


@pytest.mark.parametrize('kind',['nan','negative','over_one'])
def test_invalid_sensitivity_rejected(kind):
    value={'nan':float('nan'),'negative':-.1,'over_one':1.1}[kind]
    background=torch.full((1,3,4),2000.)
    with pytest.raises(ValueError,match='sensitivity'):
        IlluminationSoftPrior(AnalyticCandidateFixture(),background,torch.full_like(background,value),spacing_m=20.)


def test_context_change_and_bad_candidate_do_not_overwrite_cache():
    prior,source=make_prior(refresh_every=1)
    v=prior.background; target=prior.proposal(v,1).clone()
    with pytest.raises(ValueError,match='frequency'):
        prior.prepare_context(3.)
    source.offset=float('nan')
    with pytest.raises(ValueError,match='candidate'):
        prior.proposal(v,2)
    assert torch.equal(prior.state_dict()['target'],target)


def test_resource_and_source_spacing_checked_before_generation():
    prior,source=make_prior(max_cells=12)
    with pytest.raises(ValueError,match='grid'):
        prior.proposal(torch.full((1,4,4),2000.),1)
    assert source.calls==0
    source.spacing_m=15.
    with pytest.raises(ValueError,match='spacing'):
        IlluminationSoftPrior(source,prior.background,prior.illumination,spacing_m=20.)


def test_checkpoint_rejects_changed_mask_or_source_identity():
    prior,source=make_prior(); prior.proposal(prior.background,1)
    saved=copy.deepcopy(prior.state_dict())
    changed,_=make_prior(strong_threshold=.9)
    with pytest.raises(ValueError,match='identity'):
        changed.load_state_dict(saved)


def test_source_must_have_frozen_eval_model_and_stable_identity():
    prior,source=make_prior()
    source.model=nn.Linear(1,1).eval()
    with pytest.raises(ValueError,match='frozen'):
        IlluminationSoftPrior(source,prior.background,prior.illumination,spacing_m=20.)
    source.model=nn.Linear(1,1).requires_grad_(False).train()
    with pytest.raises(ValueError,match='eval'):
        IlluminationSoftPrior(source,prior.background,prior.illumination,spacing_m=20.)
    source.model.eval(); source.checkpoint_hash=None
    with pytest.raises(ValueError,match='identity'):
        IlluminationSoftPrior(source,prior.background,prior.illumination,spacing_m=20.)


def test_corrupted_negative_cached_target_is_rejected():
    prior,_=make_prior(); prior.proposal(prior.background,1)
    saved=copy.deepcopy(prior.state_dict()); saved['target'].fill_(-1.)
    with pytest.raises(ValueError,match='positive'):
        prior.load_state_dict(saved)


def test_resume_binds_sampler_configuration_and_rejects_runtime_change():
    prior,source=make_prior()
    source.inference_identity='seed_3_nfe_1'
    prior=IlluminationSoftPrior(source,prior.background,prior.illumination,spacing_m=20.)
    saved=prior.state_dict()
    source.inference_identity='seed_4_nfe_2'
    changed=IlluminationSoftPrior(source,prior.background,prior.illumination,spacing_m=20.)
    with pytest.raises(ValueError,match='identity'):
        changed.load_state_dict(saved)
    with pytest.raises(ValueError,match='identity'):
        prior.proposal(prior.background,1)
    assert source.calls==0

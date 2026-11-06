"""Safety coupling uses real gradients/optimizers and explicitly analytic sources."""
import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from ifwi_research import safety
from ifwi_research.engine import Evaluation, ResearchConfig, ResearchEngine
from ifwi_research.illumination_prior import IlluminationSoftPrior, SoftPriorConfig


def test_small_prior_gradient_is_not_amplified_to_the_cap():
    physical=torch.tensor([3.,4.],dtype=torch.float64)
    proposal=torch.tensor([.03,.04],dtype=torch.float64)
    mixed,report=safety.protect_physical_gradient([physical],[proposal],ratio=.25)
    torch.testing.assert_close(mixed[0],torch.tensor([3.03,4.04],dtype=torch.float64),rtol=0,atol=1e-15)
    assert report['scale']==1.
    assert report['ratio']==pytest.approx(.01)
    assert report['raw_proposal_norm']==pytest.approx(.05)
    assert report['projected_proposal_norm']==pytest.approx(.05)
    assert report['applied_proposal_norm']==pytest.approx(.05)


@pytest.mark.parametrize('physical,proposal,ratio,raw,projected,applied,scale',[
    ([3.,4.],[0.,10.],.25,10.,10.,1.25,.125),
    ([1.,0.],[-4.,.01],.25,(16.0001)**.5,.01,.01,1.),
    ([1.,0.],[-1.,0.],.25,1.,0.,0.,0.),
    ([0.,0.],[3.,4.],.25,5.,5.,0.,0.),
    ([3.,4.],[0.,10.],0.,10.,10.,0.,0.),
])
def test_gradient_diagnostics_report_projection_and_applied_budget(physical,proposal,ratio,raw,projected,applied,scale):
    gp=torch.tensor(physical,dtype=torch.float64); gs=torch.tensor(proposal,dtype=torch.float64)
    mixed,report=safety.protect_physical_gradient([gp],[gs],ratio)
    contribution=mixed[0]-gp
    assert report['raw_proposal_norm']==pytest.approx(raw)
    assert report['projected_proposal_norm']==pytest.approx(projected)
    assert report['applied_proposal_norm']==pytest.approx(applied)
    assert report['scale']==pytest.approx(scale)
    assert contribution.norm().item()==pytest.approx(applied,abs=1e-15)
    assert report['ratio']==pytest.approx(applied/gp.norm().item() if gp.norm()>0 else 0.)
    assert 0<=report['scale']<=1
    assert torch.dot(gp,mixed[0])>=torch.dot(gp,gp)-1e-12


class GridModel(nn.Module):
    def __init__(self):
        super().__init__(); self.velocity=nn.Parameter(torch.full((1,2,2),2000.,dtype=torch.float64))
    def forward(self): return self.velocity


class QuadraticObjective:
    """Analytic objective fixture with known derivative; not seismic evidence."""
    operator_identity='analytic_quadratic_not_wave_operator'
    def __init__(self,target=1900.): self.target=target; self.counters={'forward_solves':0,'backward_solves':0}
    def valid(self,velocity):
        return bool(torch.isfinite(velocity).all() and ((velocity>=100)&(velocity<=15000)).all())
    def evaluate(self,velocity,*,plan=None,cutoff=None,gradient=False):
        assert plan is None and cutoff is None
        self.counters['forward_solves']+=1
        self.counters['backward_solves']+=int(gradient)
        residual=velocity-self.target
        return Evaluation(float(.5*residual.detach().square().mean()),
                          residual.detach()/velocity.numel() if gradient else None)


class CandidateSource:
    """Analytic output fixture preserving the frozen-source contract."""
    def __init__(self,offset=-10.):
        self.offset=offset; self.calls=0; self.model=nn.Identity().eval()
        self.checkpoint_hash='analytic_candidate_not_dfm'; self.meta={'schema':'analytic_fixture_not_dfm'}
        self.spacing_m=20.; self.scale_mps=1000.
    def generate(self,background_mps,anchor_mps,anchor_mask,*,sample_id):
        self.calls+=1
        return anchor_mps+self.offset


def prior_fixture(offset=-10.,**config):
    source=CandidateSource(offset); background=torch.full((1,2,2),2000.,dtype=torch.float64)
    prior=IlluminationSoftPrior(source,background,torch.full_like(background,.2),spacing_m=20.,
        operator_identity=QuadraticObjective.operator_identity,config=SoftPriorConfig(**config))
    return prior,source


def engine_fixture(offset=-10.,*,policy=None,**config):
    prior,source=prior_fixture(offset,refresh_every=1)
    options=dict(method='dfm',prior_start=1,prior_every=1,learning_rate=.1,max_trials=2,**config)
    if policy is not None: options['prior_invalid_policy']=policy
    engine=ResearchEngine(GridModel(),prior.background,QuadraticObjective(),20.,ResearchConfig(**options),prior=prior)
    return engine,source


@pytest.mark.parametrize('offset,reason',[(float('nan'),'nonfinite'),(float('inf'),'nonfinite'),
    (-2001.,'physical_range'),(-1950.,'physical_range'),(14000.,'physical_range'),(1100.,'rms_budget')])
def test_invalid_generated_values_raise_typed_errors_without_caching(offset,reason):
    prior,_=prior_fixture(offset)
    with pytest.raises(ValueError) as caught: prior.proposal(prior.background,1)
    assert isinstance(caught.value,safety.InvalidProposalError)
    assert caught.value.reason==reason
    assert prior.state_dict()['target'] is None


def test_cached_target_is_revalidated_against_current_rms_budget():
    prior,source=prior_fixture(max_target_rms_change_mps=20.,refresh_every=5)
    prior.proposal(prior.background,1)
    with pytest.raises(ValueError) as caught: prior.proposal(prior.background+100.,2)
    assert isinstance(caught.value,safety.InvalidProposalError)
    assert caught.value.reason=='rms_budget'
    assert source.calls==1 and prior.diagnostics()['cache_hits']==0


def test_default_invalid_policy_still_fails_closed_without_optimizer_step():
    engine,_=engine_fixture(float('nan')); before=engine.model().detach().clone()
    with pytest.raises(ValueError) as caught: engine.step()
    assert isinstance(caught.value,safety.InvalidProposalError)
    assert torch.equal(before,engine.model()) and engine.completed==0 and not engine.optimizer.state


@pytest.mark.parametrize('offset,reason',[(float('nan'),'nonfinite'),(-2001.,'physical_range'),(1100.,'rms_budget')])
def test_opt_in_invalid_proposal_uses_guarded_physical_only_update(offset,reason):
    engine,_=engine_fixture(offset,policy='physical_only')
    report=engine.step()
    assert report['step']['accepted'] and report['step']['update_kind']=='physical_only'
    assert report['step']['physical_after']<report['step']['physical_before']
    assert report['prior']['status']=='invalid' and report['prior']['applied'] is False
    assert report['prior']['invalid_reason']==reason and report['prior']['ratio']==0.
    assert report['prior']['applied_proposal_norm']==0.
    assert report['prior']['proposal']['target_refreshed_at'] is None
    assert report['prior_seconds']>0 and engine.completed==1


def test_failed_refresh_never_reuses_the_previously_cached_target():
    engine,source=engine_fixture(policy='physical_only'); engine.step()
    saved=copy.deepcopy(engine.state_dict()); before=engine.model().detach().clone()
    source.offset=float('nan'); report=engine.step()
    assert report['prior']['status']=='invalid' and not report['prior']['applied']
    assert report['step']['update_kind']=='physical_only'
    assert not torch.equal(before,engine.model())
    # With identical history, a physical-only guarded step must be identical.
    control,_=engine_fixture(policy='physical_only'); control.load_state_dict(saved)
    control.config=replace(control.config,prior_ratio=0.,guard_every=1)
    control.step()
    torch.testing.assert_close(control.model(),engine.model(),rtol=0,atol=0)


@pytest.mark.parametrize('broken_contract',['frequency','operator','source_identity','shape','source_input'])
def test_fallback_does_not_hide_context_or_source_contract_errors(broken_contract):
    engine,source=engine_fixture(policy='physical_only'); before=engine.model().detach().clone()
    cutoff=None
    if broken_contract=='frequency': cutoff=20.
    elif broken_contract=='operator': engine.objective.operator_identity='wrong_operator'
    elif broken_contract=='source_identity': source.checkpoint_hash='changed_identity'
    elif broken_contract=='shape':
        source.generate=lambda *args,**kwargs:torch.ones(1,3,3,dtype=torch.float64)
    else:
        def invalid_input(*args,**kwargs): raise ValueError('Invalid source normalization contract')
        source.generate=invalid_input
    with pytest.raises(ValueError) as caught: engine.step(cutoff)
    assert not isinstance(caught.value,safety.InvalidProposalError)
    assert torch.equal(before,engine.model()) and engine.completed==0 and not engine.optimizer.state


def test_invalid_policy_is_rejected_before_solves():
    with pytest.raises(ValueError,match='prior_invalid_policy'): engine_fixture(policy='ignore')


def test_valid_prior_reports_whether_accepted_update_applied_it():
    engine,_=engine_fixture(); report=engine.step()
    assert report['prior']['status']=='valid' and report['prior']['applied']
    assert report['step']['update_kind']=='prior_applied'
    assert 0<report['prior']['ratio']<engine.config.prior_ratio


def test_physical_recovery_does_not_claim_prior_application():
    engine,_=engine_fixture(); parameter=engine.model.velocity
    # A stale negative Adam moment is uphill for the known positive physical gradient.
    engine.optimizer.state[parameter]={'step':torch.tensor(1.),'exp_avg':torch.full_like(parameter,-1000.),
                                      'exp_avg_sq':torch.ones_like(parameter)}
    report=engine.step()
    assert report['step']['accepted'] and report['step']['recovery']=='gradient_descent'
    assert report['step']['update_kind']=='physical_recovery'
    assert report['prior']['status']=='valid' and report['prior']['applied'] is False


def test_total_rejection_does_not_claim_prior_application_or_change_model():
    engine,_=engine_fixture(); engine.objective.target=2000.
    before=engine.model().detach().clone(); report=engine.step()
    assert not report['step']['accepted'] and report['step']['update_kind']=='rejected'
    assert report['prior']['status']=='valid' and report['prior']['applied'] is False
    assert torch.equal(before,engine.model()) and not engine.optimizer.state


@pytest.mark.parametrize('bias,reason',[(float('nan'),'nonfinite'),(1e6,'physical_range')])
def test_conditional_sampler_classifies_output_failures_not_input_errors(tmp_path,bias,reason):
    from verification.five_ideas.test_conditional_dfm import checkpoint
    from ifwi_research.conditional_dfm import FrozenConditionalDFM
    path=checkpoint(tmp_path)
    # A checkpoint with finite but explosive output tests runtime output rejection.
    adapter=FrozenConditionalDFM(path,10.,nfe=1)
    with torch.no_grad(): adapter.model.net[-1].bias.fill_(bias)
    value=torch.full((1,2,2),2500.)
    with pytest.raises((ValueError,FloatingPointError)) as caught:
        adapter.generate(value,value,torch.zeros_like(value),sample_id=1)
    assert isinstance(caught.value,safety.InvalidProposalError)
    assert caught.value.reason==reason


def test_zero_ratio_is_exact_physical_only_even_for_extreme_finite_gradients():
    physical=torch.tensor([1e-40],dtype=torch.float32)
    proposal=torch.tensor([-1e38],dtype=torch.float32)
    mixed,report=safety.protect_physical_gradient([physical],[proposal],ratio=0.)
    assert torch.equal(mixed[0],physical)
    assert report['ratio']==0. and report['applied_proposal_norm']==0.

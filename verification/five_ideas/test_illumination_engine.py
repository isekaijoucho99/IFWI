import copy
import pytest
import torch


def test_zero_dfm_ratio_requires_no_checkpoint_and_matches_baseline():
    from verification.five_ideas.test_engine import components
    from ifwi_research.engine import ResearchEngine,ResearchConfig
    base,obj=components()
    off=ResearchEngine(copy.deepcopy(base.model),base.model().detach(),copy.deepcopy(obj),20.,
                       ResearchConfig(method='dfm',prior_ratio=0.))
    for _ in range(3):
        a=base.step(); b=off.step()
        assert a['physical_objective']==b['physical_objective']
        for p,q in zip(base.model.parameters(),off.model.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert off.prior is None and a['counters']==b['counters']


def native_engine(*,cutoff=None,zero_weights=False):
    from experiments import residual_ifwi_experiment as e
    from verification.five_ideas.test_repository_bridge import problem
    from verification.five_ideas.test_illumination_prior import AnalyticCandidateFixture
    from ifwi_research.illumination import NativeSensitivity
    from ifwi_research.illumination_prior import IlluminationSoftPrior,SoftPriorConfig
    from ifwi_research.engine import ResearchEngine,ResearchConfig
    cfg,initial,raw,obj=problem()
    # Do not reuse truth-derived PML coefficients: these are separately declared.
    estimator=NativeSensitivity(geometry=raw['geometry_tensors'],wavelet=raw['wavelet'],
        dt_s=.001,spacing_m=20.,npad=3,free_surface=True,pml_vmax_mps=3000.,shot_batch_size=1)
    bg=torch.from_numpy(initial)[None]
    sensitivity=estimator.estimate(bg,probes=2,cutoff=cutoff)
    # Synthetic truth is used ONLY to manufacture the fixture observations,
    # never to set sensitivity/PML parameters. Both paths use the same operator.
    fixture_truth=bg.clone(); fixture_truth[:,4:7,5:12]+=150
    raw['propagators']=[{'first':a,'last':b,'solver':solver} for a,b,solver in estimator.batches]
    with torch.no_grad():
        raw['observed']=torch.cat([e._simulate(solver,fixture_truth,raw['wavelet'])
                                   for _,_,solver in estimator.batches],dim=1)
    from ifwi_research.repository import RepositoryObjective
    obj=RepositoryObjective(raw)
    source=AnalyticCandidateFixture(offset=10.)
    config=SoftPriorConfig(refresh_every=3,weak_weight=0. if zero_weights else 1.,strong_weight=0.)
    prior=IlluminationSoftPrior(source,bg,sensitivity['illumination'],spacing_m=20.,config=config,cutoff=cutoff,
                                operator_identity=sensitivity['operator_identity'])
    torch.manual_seed(3)
    engine=ResearchEngine(e.model_from_config(initial,cfg),bg,obj,20.,
                         ResearchConfig(method='dfm',prior_start=1,prior_every=1),prior=prior)
    return engine,estimator,sensitivity


def test_native_soft_prior_cached_candidate_guard_and_costs():
    engine,estimator,sensitivity=native_engine()
    first=engine.step(); second=engine.step()
    assert first['step']['accepted'] and second['step']['accepted']
    assert first['step']['physical_after']<first['step']['physical_before']
    assert second['step']['physical_after']<second['step']['physical_before']
    assert engine.prior.calls==1
    assert second['prior']['proposal']['cache_hits']==1
    assert second['prior']['proposal']['source_schema']=='analytic_fixture_not_dfm'
    assert sensitivity['diagnostics']['source_backward_solves']==6
    # Probes are separate diagnostic solves, not hidden in physical optimization.
    expected_forwards=3*(2+first['step']['forward_calls']+second['step']['forward_calls'])
    assert second['counters']['forward_solves']==expected_forwards


def test_native_zero_spatial_weights_match_baseline_without_prior_evaluations():
    from ifwi_research.engine import ResearchEngine,ResearchConfig
    engine,_,_=native_engine(zero_weights=True)
    other,_,_=native_engine(zero_weights=True)
    baseline=ResearchEngine(copy.deepcopy(engine.model),engine.model().detach(),
                            other.objective,20.,ResearchConfig())
    for _ in range(2):
        a=engine.step(); b=baseline.step()
        for p,q in zip(engine.model.parameters(),baseline.model.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert engine.prior.calls==0 and a['counters']==b['counters']


def test_native_cached_soft_target_survives_exact_resume():
    engine,_,_=native_engine(); engine.step(); saved=copy.deepcopy(engine.state_dict())
    assert saved['prior_state']['target'] is not None
    engine.step(); engine.step()
    resumed,_,_=native_engine(); resumed.load_state_dict(saved)
    resumed.step(); resumed.step()
    for p,q in zip(engine.model.parameters(),resumed.model.parameters()):
        torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert engine.prior.calls==resumed.prior.calls==1
    assert engine.prior.diagnostics()==resumed.prior.diagnostics()
    assert engine.objective.counters==resumed.objective.counters


def test_engine_rejects_wrong_sensitivity_frequency_before_prior_generation():
    engine,_,_=native_engine(cutoff=20.)
    with pytest.raises(ValueError,match='frequency'):
        engine.step(cutoff=None)
    assert engine.prior.calls==0


def test_cached_prior_is_bound_to_actual_native_operator():
    engine,estimator,_=native_engine()
    assert engine.objective.operator_identity==estimator.operator_identity
    engine.objective.problem['wavelet']=engine.objective.problem['wavelet']*.9
    assert engine.objective.operator_identity!=estimator.operator_identity
    counters=copy.deepcopy(engine.objective.counters)
    with pytest.raises(ValueError,match='operator'):
        engine.step()
    assert engine.objective.counters==counters and engine.prior.calls==0


def test_wrong_stage_is_rejected_before_expensive_physical_solve():
    engine,_,_=native_engine(cutoff=20.)
    counters=copy.deepcopy(engine.objective.counters)
    with pytest.raises(ValueError,match='frequency'):
        engine.step(None)
    assert engine.objective.counters==counters


def test_masked_sensitivity_cannot_attach_to_unmasked_objective():
    from ifwi_research.illumination_prior import IlluminationSoftPrior
    engine,estimator,_=native_engine()
    mask=torch.ones(1,estimator.shots,estimator.nt,estimator.receivers)
    mask[...,::2]=0.
    report=estimator.estimate(engine.prior.background,data_mask=mask)
    engine.prior=IlluminationSoftPrior(engine.prior.source,engine.prior.background,
        report['illumination'],spacing_m=20.,operator_identity=report['operator_identity'])
    counters=copy.deepcopy(engine.objective.counters)
    with pytest.raises(ValueError,match='operator'):
        engine.step()
    assert engine.objective.counters==counters and engine.prior.calls==0

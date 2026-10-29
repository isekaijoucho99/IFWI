"""Adversarial and numerical checks added during the separate review pass."""
import copy
import json
import numpy as np
import pytest
import torch
from verification.research_ideas.test_integration import tiny_spec


def test_original_fd_directional_derivative(tmp_path):
    from experiments import residual_ifwi_experiment as e
    from research_ideas.physics import OriginalPhysics
    cfg=tiny_spec(tmp_path)['experiment']
    truth=np.loadtxt(cfg['data']['model_file'],delimiter=',').astype(np.float32)
    physics=OriginalPhysics(e.build_forward_problem(truth,cfg,'cpu'))
    v=torch.full((1,9,11),2500.)
    r=physics.value_and_gradient(v); d=r['gradient']/r['gradient'].norm()
    numerical=(physics.value(v+2*d)-physics.value(v-2*d))/4
    analytic=float((r['gradient']*d).sum())
    assert numerical==pytest.approx(analytic,rel=.02,abs=1e-10)


def test_router_refresh_uses_observed_energy_in_runner(tmp_path):
    from research_ideas.runner import run
    from research_ideas.routing import CompressedShotRouter,compress_traces
    from research_ideas.physics import OriginalPhysics
    from experiments import residual_ifwi_experiment as e
    from experiments.residual_ifwi_data import load_velocity_data
    from pathlib import Path
    spec=tiny_spec(tmp_path); cfg=copy.deepcopy(spec['experiment'])
    truth,back,_=load_velocity_data(cfg,Path('.'))
    p=e.build_forward_problem(truth,cfg,'cpu'); adapter=OriginalPhysics(p)
    _,pred=adapter.value(torch.tensor(back)[None],return_records=True)
    expected=CompressedShotRouter(compress_traces(p['observed']),top_k=1,samples=1)
    expected.refresh(p['observed'],pred)
    result=run(spec,dict(method='routing',steps=1,options=dict(top_k=1,samples=1)),tmp_path/'route')
    saved=torch.load(result/'checkpoint.pt',weights_only=False)
    torch.testing.assert_close(saved['router']['scores'],expected.scores,rtol=1e-6,atol=1e-9)


@pytest.mark.parametrize('method,options',[
    ('wavelet',dict(high_start=1.)),('interfaces',dict(interfaces=2,width_m=30.,warmup=0,ramp_updates=1)),
    ('gradient_attention',dict(warmup=0,fit_every=1))])
def test_model_and_gate_replay(tmp_path,method,options):
    from research_ideas.runner import run
    spec=tiny_spec(tmp_path)
    r=dict(method=method,options=options,steps=3,eval_every=1)
    a=run(spec,r,tmp_path/'a'); b=run(spec,{**r,'steps':1},tmp_path/'b')
    c=run(spec,r,tmp_path/'c',resume=b/'checkpoint.pt')
    np.testing.assert_array_equal(np.load(a/'final_velocity.npy'),np.load(c/'final_velocity.npy'))
    sa=torch.load(a/'checkpoint.pt',weights_only=False); sc=torch.load(c/'checkpoint.pt',weights_only=False)
    if method=='gradient_attention':
        for key in sa['gate']: torch.testing.assert_close(sa['gate'][key],sc['gate'][key],rtol=0,atol=0)


def test_resume_rejects_data_and_recipe_change(tmp_path):
    from research_ideas.runner import run
    spec=tiny_spec(tmp_path); r=dict(method='baseline',steps=1,eval_every=1)
    old=run(spec,r,tmp_path/'a')
    with pytest.raises(ValueError):
        run(spec,{**r,'steps':2,'safeguard':True},tmp_path/'b',resume=old/'checkpoint.pt')
    a=np.loadtxt(spec['experiment']['data']['model_file'],delimiter=','); a+=1
    np.savetxt(spec['experiment']['data']['model_file'],a,delimiter=',')
    with pytest.raises(ValueError): run(spec,{**r,'steps':2},tmp_path/'c',resume=old/'checkpoint.pt')


def test_runner_rejects_silent_original_shot_sampling(tmp_path):
    from research_ideas.runner import validate
    spec=tiny_spec(tmp_path); spec['experiment']['training']['shots_per_update']=1
    with pytest.raises(ValueError): validate(spec,dict(method='baseline'),False)


def test_zero_scale_prior_is_rejected(tmp_path):
    from research_ideas.dfm import TwoTimeTransport,FrozenDFMPrior
    file=tmp_path/'bad.pt'
    torch.save(dict(schema='ifwi_dfm_v1',model=TwoTimeTransport(8).state_dict(),
        metadata=dict(width=8,patch_size=8,spacing_m=15.,trained_steps=1,smoke_only=True,
                      center_mps=3000.,scale_mps=0.)),file)
    with pytest.raises(ValueError): FrozenDFMPrior(file,'cpu',15.,allow_smoke=True)


def test_prior_training_truth_leak_is_refused_by_runner(tmp_path):
    from research_ideas.dfm import TwoTimeTransport
    from research_ideas.runner import run
    from research_ideas.train_dfm import model_digest
    spec=tiny_spec(tmp_path); truth=np.loadtxt(spec['experiment']['data']['model_file'],delimiter=',')
    file=tmp_path/'leaked.pt'
    torch.save(dict(schema='ifwi_dfm_v1',model=TwoTimeTransport(8).state_dict(),
      metadata=dict(width=8,patch_size=8,spacing_m=15.,trained_steps=1,smoke_only=False,
        center_mps=3000.,scale_mps=1500.,training_models=[dict(sha256=model_digest(truth))],validation_models=[])),file)
    with pytest.raises(ValueError):
        run(spec,dict(method='dfm',steps=1,options=dict(checkpoint=str(file))),tmp_path/'bad')


def test_dfm_cached_prior_resume_and_rng_isolation(tmp_path):
    from research_ideas.dfm import TwoTimeTransport,FrozenDFMPrior
    from research_ideas.runner import run
    file=tmp_path/'prior.pt'; model=TwoTimeTransport(8)
    torch.save(dict(schema='ifwi_dfm_v1',model=model.state_dict(),
      metadata=dict(width=8,patch_size=8,spacing_m=15.,trained_steps=1,smoke_only=True,
        center_mps=3000.,scale_mps=1500.,training_models=[],validation_models=[])),file)
    prior=FrozenDFMPrior(file,'cpu',15.,allow_smoke=True)
    before=torch.get_rng_state().clone(); prior.gradient(torch.full((1,9,11),2500.),1)
    assert torch.equal(before,torch.get_rng_state())
    spec=tiny_spec(tmp_path)
    r=dict(method='dfm',steps=3,eval_every=1,options=dict(checkpoint=str(file),allow_smoke=True,warmup=0,every=1))
    a=run(spec,r,tmp_path/'a',smoke=True); b=run(spec,{**r,'steps':1},tmp_path/'b',smoke=True)
    c=run(spec,r,tmp_path/'c',resume=b/'checkpoint.pt',smoke=True)
    np.testing.assert_array_equal(np.load(a/'final_velocity.npy'),np.load(c/'final_velocity.npy'))


def test_interface_regularizer_moves_without_truth(tmp_path):
    from research_ideas.runner import run
    spec=tiny_spec(tmp_path)
    result=run(spec,dict(method='interfaces',steps=3,eval_every=1,options=dict(
        interfaces=2,width_m=30.,warmup=0,ramp_updates=1,curvature_weight=.01)),tmp_path/'result')
    assert (result/'interfaces_m.npy').exists()
    metrics=json.loads((result/'metrics.json').read_text())
    assert np.isfinite(metrics['velocity']['rmse_mps'])


@pytest.mark.parametrize('blend,steps,expected_calls',[(1.,4,1),(0.,4,4)])
def test_dfm_single_mode_ablations_do_not_pay_for_unused_mode(tmp_path,blend,steps,expected_calls):
    from research_ideas.dfm import TwoTimeTransport,FrozenDFMPrior
    file=tmp_path/'prior.pt'
    torch.save(dict(schema='ifwi_dfm_v1',model=TwoTimeTransport(8).state_dict(),
      metadata=dict(width=8,patch_size=8,spacing_m=15.,trained_steps=1,smoke_only=True,
        center_mps=3000.,scale_mps=1500.)),file)
    prior=FrozenDFMPrior(file,'cpu',15.,allow_smoke=True,blend=blend,steps=steps)
    prior.gradient(torch.full((1,8,8),2500.),1)
    assert prior.calls==expected_calls


def test_experiment_matrix_and_pareto(tmp_path):
    from research_ideas.experiments import generate,pareto_front
    folder=generate(tmp_path/'matrix',seeds=[0,3],datasets=['marmousi','overthrust'],include_dfm=False)
    jobs=json.loads((folder/'manifest.json').read_text())
    assert len(jobs)==7*2*2
    for job in jobs:
        spec=json.loads((folder/job['spec']).read_text()); recipe=json.loads((folder/job['recipe']).read_text())
        assert spec['control']==dict(kind='zero_output') and 'target' not in spec
        assert recipe['steps']==1000
    assert pareto_front([(1,5),(2,4),(3,6),(2,4),(4,3)])==[0,1,3,4]

"""First-stage diagnostics use explicit synthetic fixtures, never qualified weights."""
import copy
import importlib
import json
import pytest
import torch


def api():
    return importlib.import_module('ifwi_research.validate_conditional_dfm')


def test_diagnostic_loader_keeps_unqualified_metadata_and_normal_loader_closed(tmp_path):
    from ifwi_research.conditional_dfm import make_transport, FrozenConditionalDFM
    meta = dict(schema='ifwi_masked_conditional_dfm_v1', architecture='masked_two_time_cnn_v1',
        width=4, mean_mps=2000., scale_mps=10., spacing_m=20., units='m/s',
        training=dict(steps=1, toy=True, contract='shared_background_estimate_anchors_mask_v1',
            time_semantics='gaussian_0_to_geology_1_two_time_transport',
            anchor_semantics='estimated_not_truth',train_model_hashes=['a'*64],val_model_hashes=['b'*64]),
        qualification=dict(status='unqualified', report_ref=None, report_sha256=None))
    path=tmp_path/'unqualified.pt'; torch.save(dict(meta=meta,state_dict=make_transport(width=4).state_dict()),path)
    with pytest.raises(ValueError): FrozenConditionalDFM(path,20.)
    source=api().DiagnosticConditionalDFM(path,20.)
    assert source.diagnostic_only and source.meta==meta
    v=torch.full((1,3,4),2000.)
    assert source.generate(v,v,torch.zeros_like(v),sample_id=1).shape==v.shape
    meta['training']['anchor_semantics']='true_known_pixels'
    torch.save(dict(meta=meta,state_dict=make_transport(width=4).state_dict()),path)
    with pytest.raises(ValueError): api().DiagnosticConditionalDFM(path,20.)


def test_candidate_report_uses_snapshot_estimate_not_truth_and_preserves_cache():
    from verification.five_ideas.test_illumination_engine import native_engine
    engine,_,_=native_engine()
    estimate=engine.model().detach(); truth=estimate+25
    before=copy.deepcopy(engine.prior.state_dict())
    result=api().evaluate_candidates(engine.prior,estimate,engine.objective,sample_ids=[1,2],truth=truth)
    assert result['selection']=='fixed_single_draws_no_truth_selection'
    assert result['draw_count']==2 and result['valid_draw_count']==2
    assert result['draws'][0]['truth_evaluation_only']['all_rmse_mps']==pytest.approx(15.)
    assert result['estimate']['truth_evaluation_only']['all_rmse_mps']==pytest.approx(25.)
    assert result['draws'][0]['estimate_rms_change_mps']==pytest.approx(10.)
    assert engine.prior.state_dict()['target'] is before['target'] is None
    for _,anchors,_,_ in engine.prior.source.conditions:
        torch.testing.assert_close(anchors,estimate)


def test_candidate_invalid_fraction_excludes_rejected_draws_from_quality_averages():
    from verification.five_ideas.test_illumination_engine import native_engine
    engine,_,_=native_engine(); engine.prior.source.offset=float('nan')
    result=api().evaluate_candidates(engine.prior,engine.model().detach(),engine.objective,sample_ids=[1,2])
    assert result['valid_draw_count']==0 and result['invalid_fraction']==1.
    assert all(row['valid'] is False and row['reason'] for row in result['draws'])
    assert result['mean_valid_candidate_physical_objective'] is None
    json.dumps(result,allow_nan=False)


def test_matched_zero_ratio_has_identical_updates_and_does_not_mutate_snapshot():
    from verification.five_ideas.test_illumination_engine import native_engine
    engine,_,_=native_engine(); initial=copy.deepcopy(engine.model.state_dict())
    result=api().compare_coupling(engine,engine.prior,steps=2,ratio=0.)
    assert result['dfm']['velocity_rmse_vs_physics_mps']==0.
    assert result['dfm']['parameter_l2_vs_physics']==0.
    assert result['dfm']['prior_calls']==0
    assert result['dfm']['optimizer']['name']=='Adam'
    assert result['dfm']['optimizer']['parameter_groups'][0]['eps']==1e-8
    assert engine.completed==0 and engine.prior.calls==0
    for name,value in initial.items(): torch.testing.assert_close(engine.model.state_dict()[name],value,rtol=0,atol=0)


def test_matched_branches_measure_actual_difference_and_oracle_is_separate():
    from verification.five_ideas.test_illumination_engine import native_engine
    engine,_,_=native_engine(); truth=engine.model().detach()+25
    result=api().compare_coupling(engine,engine.prior,steps=2,ratio=.1,oracle_truth=truth)
    assert result['dfm']['velocity_rmse_vs_physics_mps']>=0.
    assert result['dfm']['parameter_l2_vs_physics']>0.
    assert result['dfm']['prior_calls']==1
    assert result['oracle']['oracle_truth_used_for_updates'] is True
    assert result['dfm']['oracle_truth_used_for_updates'] is False
    assert result['qualification_changed'] is False
    assert len(result['physics_only']['history'])==2
    json.dumps(result,allow_nan=False)


def test_snapshot_requires_matching_observations_and_baseline_config():
    from verification.five_ideas.test_illumination_engine import native_engine
    from ifwi_research.run import tensor_hash
    engine,_,_=native_engine()
    spec={'experiment':{'seed':3}}
    snapshot=dict(identity=dict(baseline=spec,initial_hash=tensor_hash(engine.prior.background),
        observations_hash=tensor_hash(engine.objective.problem['observed'])),engine=engine.state_dict())
    # Only data-only baseline checkpoints may seed a causal comparison.
    with pytest.raises(ValueError,match='baseline'):
        api().restore_snapshot(engine,snapshot,spec,engine.prior.background)
    snapshot['engine']['config']['method']='baseline'
    snapshot['engine']['prior_hash']=None; snapshot['engine']['prior_state']=None
    api().restore_snapshot(engine,snapshot,spec,engine.prior.background)
    snapshot['identity']['observations_hash']='different'
    with pytest.raises(ValueError,match='observations'):
        api().restore_snapshot(engine,snapshot,spec,engine.prior.background)


def test_cli_synthetic_oracle_never_invents_dfm_weights(tmp_path):
    result=api().main(['--synthetic-smoke','--oracle','--steps','2','--out',str(tmp_path/'run')])
    assert result['candidates']['status']=='not_run_missing_checkpoint'
    assert result['scientific_qualification'] is False
    assert 'dfm' not in result['coupling']
    assert result['coupling']['oracle']['prior_calls']==1
    assert result['coupling']['oracle']['oracle_truth_used_for_updates']
    assert (tmp_path/'run'/'physics_only_velocity.npy').is_file()
    assert (tmp_path/'run'/'oracle_velocity.npy').is_file()
    saved=json.loads((tmp_path/'run'/'summary.json').read_text())
    assert saved['checkpoint_meta'] is None and saved['qualification_changed'] is False


def test_snapshot_mismatch_does_not_partially_restore_model():
    from verification.five_ideas.test_illumination_engine import native_engine
    from ifwi_research.run import tensor_hash
    engine,_,_=native_engine(); initial=copy.deepcopy(engine.model.state_dict())
    state=engine.state_dict(); state['config']['method']='baseline'; state['prior_hash']=None; state['prior_state']=None
    saved=dict(identity=dict(baseline={},initial_hash=tensor_hash(engine.prior.background),observations_hash='wrong'),engine=state)
    with pytest.raises(ValueError,match='observations'): api().restore_snapshot(engine,saved,{},engine.prior.background)
    for k,v in initial.items(): torch.testing.assert_close(engine.model.state_dict()[k],v,rtol=0,atol=0)


def test_cli_restores_persisted_baseline_optimizer_and_exports_snapshot_conditions(tmp_path):
    import numpy as np
    from pathlib import Path
    from experiments import residual_ifwi_experiment as e
    from ifwi_research.engine import ResearchEngine,ResearchConfig
    from ifwi_research.repository import load_problem
    from ifwi_research.run import tensor_hash
    cfg=e.default_config(); cfg['model']['neurons']=[2,8,8,1]
    cfg['forward'].update(nt=48,dt_s=.001,frequency_hz=30.,npad=3)
    cfg['data'].update(grid_spacing_m=20.,downsample=1,csv_header='none',model_file=str(tmp_path/'fixture.csv'))
    cfg['acquisition'].update(num_shots=3,source_x_indices=[3,8,13],receiver_stride=1)
    cfg['training'].update(shot_batch_size=1,epochs=4,checkpoint_interval=1)
    truth=np.full((10,18),2000.,dtype=np.float32); truth[4:7,5:12]+=150.
    np.savetxt(tmp_path/'fixture.csv',truth,delimiter=',')
    spec=dict(name='synthetic_snapshot_test',experiment=cfg,control={'kind':'zero_output'},
        robustness={},background=dict(kind='linear_depth',top_mps=2000.,gradient_per_s=0.))
    base,bg,obj,_,_=load_problem(spec,Path(__file__).resolve().parents[2])
    engine=ResearchEngine(base,bg,obj,20.,ResearchConfig()); engine.step()
    snapshot=dict(identity=dict(baseline=spec,initial_hash=tensor_hash(bg),
        observations_hash=tensor_hash(obj.problem['observed'])),engine=engine.state_dict())
    torch.save(snapshot,tmp_path/'snapshot.pt'); (tmp_path/'baseline.json').write_text(json.dumps(spec))
    summary=api().main(['--baseline',str(tmp_path/'baseline.json'),'--snapshot',str(tmp_path/'snapshot.pt'),
        '--steps','1','--out',str(tmp_path/'result')])
    assert summary['coupling']['start_update']==1
    assert summary['coupling']['physics_only']['history'][0]['update']==2
    np.testing.assert_array_equal(np.load(tmp_path/'result'/'anchor.npy'),engine.model().detach()[0].numpy())
    state=torch.load(tmp_path/'result'/'physics_only_state.pt',weights_only=True)['engine']
    assert state['completed']==2 and state['optimizer']['state']


@pytest.mark.parametrize('background',[None,{'kind':'slowness_gaussian','slowness_sigma_m':50.,'velocity_sigma_m':50.}])
def test_real_cli_rejects_truth_derived_initialization_before_loading_snapshot(tmp_path,background):
    spec=dict(experiment={'initialization':{'sigma':0.}},background=background)
    path=tmp_path/'bad.json'; path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='independent linear_depth'):
        api().main(['--baseline',str(path),'--snapshot',str(tmp_path/'missing.pt'),'--out',str(tmp_path/'out')])


def test_cli_records_effective_prior_and_lr_policy(tmp_path):
    result=api().main(['--synthetic-smoke','--steps','1','--refresh-every','2','--out',str(tmp_path/'out')])
    assert result['effective_prior_config']['refresh_every']==2
    assert result['inference_sampler'] is None
    assert result['lr_policy']=='fixed_base_learning_rate_no_schedule_multiplier'


def test_real_cli_resource_cap_precedes_snapshot_or_pde(tmp_path):
    spec=dict(background={'kind':'linear_depth'},experiment={'forward':{'nt':1000}})
    path=tmp_path/'large.json'; path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='256'):
        api().main(['--baseline',str(path),'--snapshot',str(tmp_path/'missing.pt'),'--out',str(tmp_path/'out')])


def test_training_export_runs_diagnostic_candidate_and_coupling_pipeline_without_qualification(tmp_path):
    from verification.five_ideas.test_conditional_training import fixture_manifest,config
    from ifwi_research.train_conditional_dfm import train
    manifest=fixture_manifest(tmp_path)
    train(manifest=manifest,out=tmp_path/'training',spacing_m=20.,device='cpu',config=config())
    path=tmp_path/'training'/'best.pt'
    original=torch.load(path,weights_only=True)['meta']
    result=api().main(['--synthetic-smoke','--diagnostic-checkpoint',str(path),'--steps','1',
        '--draws','2','--nfe','4','--seed','7','--refresh-every','2','--out',str(tmp_path/'validation')])
    assert result['checkpoint_meta']==original
    assert result['checkpoint_meta']['qualification']['status']=='unqualified'
    assert result['scientific_qualification'] is False and result['qualification_changed'] is False
    assert result['inference_sampler']['seed']==7 and result['inference_sampler']['nfe']==4
    assert result['effective_prior_config']['refresh_every']==2
    assert result['candidates']['network_evaluations']==8
    assert result['coupling']['dfm']['prior_calls']==4
    assert (tmp_path/'validation'/'dfm_velocity.npy').is_file()

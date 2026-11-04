"""Synthetic numerical contracts only; these do not qualify a geological prior."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ifwi_research.train_conditional_dfm import (
    TrainingConfig, fit_normalization, group_drift_loss, load_manifest,
    sample_time_pair, sinkhorn_plan, train,
)


def fixture_manifest(tmp_path):
    records = []
    z, x = np.mgrid[:10, :12]
    for i, split in enumerate(('train', 'train', 'validation')):
        clean = (1800 + 100 * i + 30 * z + 25 * np.sin(x / 2)).astype('float32')
        background = (1800 + 100 * i + 28 * z + np.zeros_like(x)).astype('float32')
        mask = (z < 4).astype('float32')
        anchor = (clean + 20 * np.cos(x)).astype('float32')
        record = dict(model_id=f'm{i}', family_id=f'f{i}', split=split,
                      spacing_m=20., condition_source='synthetic_smoke')
        for name, value in [('clean', clean), ('background', background), ('anchor', anchor), ('mask', mask)]:
            path = tmp_path / f'{i}_{name}.npy'
            np.save(path, value)
            record[f'{name}_path'] = path.name
        records.append(record)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'schema': 'ifwi_conditional_dataset_v1', 'records': records}))
    return path


def config(**kwargs):
    return TrainingConfig(patch_size=8, width=8, group_size=4, groups_per_step=1,
                          steps=2, validate_every=1, sinkhorn_iters=20, **kwargs)


def test_sinkhorn_uniform_marginals_and_finite_extremes():
    logits = torch.tensor([[[1., -3., 4.], [5., 2., -2.]]], dtype=torch.float64)
    plan = sinkhorn_plan(logits, iterations=200)
    torch.testing.assert_close(plan.sum(-1), torch.full((1, 2), .5, dtype=torch.float64))
    torch.testing.assert_close(plan.sum(-2), torch.full((1, 3), 1 / 3, dtype=torch.float64))
    assert torch.isfinite(sinkhorn_plan(logits * 1e6, iterations=20)).all()


@pytest.mark.parametrize('value', [0, -1, True])
def test_invalid_sinkhorn_iterations_fail(value):
    with pytest.raises(ValueError):
        sinkhorn_plan(torch.ones(1, 2, 2), iterations=value)


def test_drift_groups_do_not_influence_other_groups_and_have_finite_gradients():
    torch.manual_seed(3)
    q = torch.randn(2, 4, 1, 4, 4, requires_grad=True)
    p = torch.randn_like(q)
    loss = group_drift_loss(q, p, temperature=.5, iterations=50)
    loss.sum().backward()
    gradient = q.grad.clone()
    changed = p.clone(); changed[1] += 100
    q2 = q.detach().clone().requires_grad_()
    group_drift_loss(q2, changed, temperature=.5, iterations=50).sum().backward()
    torch.testing.assert_close(gradient[0], q2.grad[0], rtol=0, atol=0)
    assert torch.isfinite(q2.grad).all() and q2.grad.abs().sum() > 0
    assert not torch.allclose(gradient[1], q2.grad[1])


def test_identical_marginals_have_zero_drift_loss():
    q = torch.randn(2, 4, 1, 3, 3, requires_grad=True)
    loss = group_drift_loss(q, q.detach().clone())
    assert torch.equal(loss, torch.zeros_like(loss))


def test_group_time_pairs_are_scalars_ordered_and_include_endpoint():
    rng = torch.Generator().manual_seed(1)
    pairs = [sample_time_pair(rng) for _ in range(100)]
    assert all(type(t) is float and type(r) is float and 0 <= t < r <= 1 for t, r in pairs)
    assert (0., 1.) in pairs


def test_family_overlap_rejected_even_when_arrays_differ(tmp_path):
    path = fixture_manifest(tmp_path)
    data = json.loads(path.read_text()); data['records'][2]['family_id'] = 'f0'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='family'):
        load_manifest(path, patch_size=8, spacing_m=20.)


def test_duplicate_content_rejected_even_if_family_ids_differ(tmp_path):
    path = fixture_manifest(tmp_path)
    np.save(tmp_path / '2_clean.npy', np.load(tmp_path / '0_clean.npy').astype('float64'))
    with pytest.raises(ValueError, match='content'):
        load_manifest(path, patch_size=8, spacing_m=20.)


@pytest.mark.parametrize('key,value,match', [('spacing_m', 10., 'spacing'),
    ('condition_source', 'ground_truth', 'condition_source')])
def test_manifest_rejects_unverified_contracts(tmp_path, key, value, match):
    path = fixture_manifest(tmp_path)
    data = json.loads(path.read_text()); data['records'][0][key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_manifest(path, patch_size=8, spacing_m=20.)


def test_nonbinary_mask_rejected(tmp_path):
    path = fixture_manifest(tmp_path)
    np.save(tmp_path / '0_mask.npy', np.full((10, 12), .5))
    with pytest.raises(ValueError, match='binary'):
        load_manifest(path, patch_size=8, spacing_m=20.)


def test_normalization_uses_train_only(tmp_path):
    path = fixture_manifest(tmp_path)
    records = load_manifest(path, patch_size=8, spacing_m=20.)
    first = fit_normalization(records)
    np.save(tmp_path / '2_clean.npy', np.full((10, 12), 9000.))
    second = fit_normalization(load_manifest(path, patch_size=8, spacing_m=20.))
    assert first == second
    a = np.concatenate([np.load(tmp_path / f'{i}_clean.npy').ravel() for i in (0, 1)])
    assert first['mean_mps'] == pytest.approx(a.mean(), rel=1e-6)
    assert first['scale_mps'] == pytest.approx(a.std(), rel=1e-6)


def test_training_ema_checkpoint_metrics_and_unqualified_gate(tmp_path):
    path = fixture_manifest(tmp_path)
    result = train(path, tmp_path / 'run', spacing_m=20., config=config())
    payload = torch.load(tmp_path / 'run' / 'last.pt', weights_only=True)
    assert payload['meta']['training']['steps'] == 2
    assert payload['meta']['qualification']['status'] == 'unqualified'
    assert payload['meta']['training']['toy'] is True
    assert payload['meta']['training']['anchor_semantics'] == 'estimated_not_truth'
    assert payload['meta']['normalization']['fit_split'] == 'train'
    assert payload['state_dict'].keys() == payload['raw_state_dict'].keys()
    assert any(not torch.equal(payload['state_dict'][k], payload['raw_state_dict'][k]) for k in payload['state_dict'])
    assert result['validation']['nfe_1']['weak_rmse_mps'] > 0
    assert 'edge_rmse_mps_per_cell' in result['validation']['nfe_4']
    from ifwi_research.conditional_dfm import FrozenConditionalDFM
    with pytest.raises(ValueError, match='[Qq]ualif|Non-toy'):
        FrozenConditionalDFM(tmp_path / 'run' / 'last.pt', 20.)


def test_resume_matches_uninterrupted_training(tmp_path):
    path = fixture_manifest(tmp_path)
    train(path, tmp_path / 'first', spacing_m=20., config=config())
    train(path, tmp_path / 'resumed', spacing_m=20., config=TrainingConfig(**{**config().__dict__, 'steps': 4}),
          resume=tmp_path / 'first' / 'last.pt')
    train(path, tmp_path / 'full', spacing_m=20., config=TrainingConfig(**{**config().__dict__, 'steps': 4}))
    a = torch.load(tmp_path / 'resumed' / 'last.pt', weights_only=True)
    b = torch.load(tmp_path / 'full' / 'last.pt', weights_only=True)
    for field in ('state_dict', 'raw_state_dict'):
        for key in a[field]:
            torch.testing.assert_close(a[field][key], b[field][key], rtol=0, atol=0)
    assert torch.equal(a['rng_state'], b['rng_state'])


def test_resume_rejects_changed_data(tmp_path):
    path = fixture_manifest(tmp_path)
    train(path, tmp_path / 'first', spacing_m=20., config=config())
    np.save(tmp_path / '0_anchor.npy', np.load(tmp_path / '0_anchor.npy') + 1.)
    with pytest.raises(ValueError, match='manifest'):
        train(path, tmp_path / 'resumed', spacing_m=20., config=TrainingConfig(**{**config().__dict__, 'steps': 4}),
              resume=tmp_path / 'first' / 'last.pt')


def test_condition_excludes_clean_and_unmasked_anchor_values(tmp_path):
    from ifwi_research.train_conditional_dfm import _patch
    path = fixture_manifest(tmp_path)
    records = load_manifest(path, patch_size=8, spacing_m=20.)
    norm = fit_normalization(records)
    rng = torch.Generator().manual_seed(11)
    _, first, _ = _patch(records[0], 8, rng, 'cpu', norm, center=True)
    mask = np.load(tmp_path / '0_mask.npy')
    anchor = np.load(tmp_path / '0_anchor.npy'); anchor[mask == 0] += 500
    np.save(tmp_path / '0_anchor.npy', anchor)
    np.save(tmp_path / '0_clean.npy', np.load(tmp_path / '0_clean.npy') + 1000)
    _, second, _ = _patch(records[0], 8, rng, 'cpu', norm, center=True)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(first[:, 1:2] * (1 - first[:, 2:3]), torch.zeros_like(first[:, 1:2]))


def test_validation_uses_gaussian_not_truth_and_preserves_rng(tmp_path):
    from ifwi_research.conditional_dfm import MaskedTwoTimeTransport
    from ifwi_research.train_conditional_dfm import validation
    path = fixture_manifest(tmp_path)
    records = load_manifest(path, patch_size=8, spacing_m=20.)
    norm = fit_normalization(records)
    model = MaskedTwoTimeTransport(8)  # Zero final layer -> identity for any number of steps.
    state = torch.get_rng_state().clone()
    first = validation(model, records, config(), 'cpu', norm)
    assert torch.equal(state, torch.get_rng_state())
    assert first == validation(model, records, config(), 'cpu', norm)
    assert first['nfe_1'] == first['nfe_4']
    # Gaussian draws under identity are not a leaked clean endpoint.
    assert first['nfe_1']['weak_rmse_mps'] > 100
    assert first['nfe_1']['diversity_rms_mps'] > 1


def test_plain_training_does_not_qualify_data_only_declaration(tmp_path):
    path = fixture_manifest(tmp_path)
    data = json.loads(path.read_text())
    for rec in data['records']:
        rec['condition_source'] = 'data_only_reconstruction'
        rec['condition_provenance'] = 'test declaration only; values are synthetic fixture'
    path.write_text(json.dumps(data))
    train(path, tmp_path / 'run', spacing_m=20., config=config())
    payload = torch.load(tmp_path / 'run' / 'last.pt', weights_only=True)
    assert payload['meta']['training']['toy'] is False
    assert payload['meta']['qualification']['status'] == 'unqualified'
    from ifwi_research.conditional_dfm import FrozenConditionalDFM
    with pytest.raises(ValueError, match='[Qq]ualif'):
        FrozenConditionalDFM(tmp_path / 'run' / 'last.pt', 20.)


def test_training_preserves_callers_cpu_rng(tmp_path):
    path = fixture_manifest(tmp_path)
    torch.manual_seed(222)
    before = torch.get_rng_state().clone()
    train(path, tmp_path / 'run', spacing_m=20., config=config())
    assert torch.equal(before, torch.get_rng_state())


def test_optional_multiscale_training_and_inference_export(tmp_path):
    path = fixture_manifest(tmp_path)
    cfg = TrainingConfig(**{**config().__dict__, 'architecture': 'masked_two_time_unet_v1'})
    train(path, tmp_path / 'run', spacing_m=20., config=cfg)
    checkpoint = torch.load(tmp_path / 'run' / 'last.pt', weights_only=True)
    inference = torch.load(tmp_path / 'run' / 'last_inference.pt', weights_only=True)
    assert inference['meta']['architecture'] == 'masked_two_time_unet_v1'
    assert set(inference) == {'meta', 'state_dict'}
    assert inference['meta']['qualification']['status'] == 'unqualified'
    for key in inference['state_dict']:
        assert torch.equal(inference['state_dict'][key], checkpoint['state_dict'][key])
    from ifwi_research.conditional_dfm import make_transport
    model = make_transport('masked_two_time_unet_v1', 8)
    model.load_state_dict(inference['state_dict'], strict=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_saved_manifest_can_reproduce_dataset_inventory(tmp_path):
    path = fixture_manifest(tmp_path)
    original = load_manifest(path, patch_size=8, spacing_m=20.)
    train(path, tmp_path / 'run', spacing_m=20., config=config())
    saved = load_manifest(tmp_path / 'run' / 'dataset_manifest.json', patch_size=8, spacing_m=20.)
    assert saved == original


def test_validation_distinguishes_samples_from_ensemble_mean(tmp_path):
    from ifwi_research.conditional_dfm import make_transport
    from ifwi_research.train_conditional_dfm import validation
    path = fixture_manifest(tmp_path)
    records = load_manifest(path, patch_size=8, spacing_m=20.)
    norm = fit_normalization(records)
    metrics = validation(make_transport('masked_two_time_cnn_v1', 8), records, config(), 'cpu', norm)['nfe_1']
    assert metrics['sample_weak_rmse_mps'] > metrics['ensemble_mean_weak_rmse_mps']
    assert metrics['weak_rmse_mps'] == metrics['ensemble_mean_weak_rmse_mps']  # Historical alias, no silent change.
    assert metrics['sample_rmse_mps'] ** 2 == pytest.approx(
        metrics['ensemble_mean_rmse_mps'] ** 2 + metrics['diversity_rms_mps'] ** 2, rel=1e-6)


def test_selection_matches_single_sample_deployment_by_default(tmp_path):
    path = fixture_manifest(tmp_path)
    summary = train(path, tmp_path / 'run', spacing_m=20., config=config())
    assert summary['selection']['statistic'] == 'single_draw'
    assert summary['selection']['nfe'] == 1
    records = [json.loads(line) for line in (tmp_path / 'run' / 'history.jsonl').read_text().splitlines()]
    expected = min(r['validation']['nfe_1']['sample_weak_rmse_mps'] for r in records)
    assert summary['best_validation_score_mps'] == expected
    assert summary['initial_validation']['nfe_1']['sample_weak_rmse_mps'] > 0
    assert summary['raw_validation']['nfe_1']['sample_weak_rmse_mps'] > 0
    assert isinstance(summary['training_diagnostics']['beats_background'], bool)


def test_invalid_selection_statistic_rejected():
    with pytest.raises(ValueError, match='selection'):
        TrainingConfig(selection_statistic='best_looking').validate()


def test_legacy_resume_resets_incompatible_best_metric(tmp_path):
    path = fixture_manifest(tmp_path)
    train(path, tmp_path / 'first', spacing_m=20., config=config())
    checkpoint = torch.load(tmp_path / 'first' / 'last.pt', weights_only=True)
    checkpoint['config'].pop('selection_statistic'); checkpoint['config'].pop('selection_nfe')
    checkpoint['meta']['training'].pop('validation_selection')
    checkpoint['best_score'] = 0.0
    legacy = tmp_path / 'legacy.pt'; torch.save(checkpoint, legacy)
    summary = train(path, tmp_path / 'resumed', spacing_m=20.,
                    config=TrainingConfig(**{**config().__dict__, 'steps': 3}), resume=legacy)
    assert summary['training_diagnostics']['selection_changed_on_resume'] is True
    assert summary['best_validation_score_mps'] > 0
    assert (tmp_path / 'resumed' / 'best_inference.pt').is_file()


def test_validation_counts_each_out_of_range_draw(tmp_path):
    from ifwi_research.train_conditional_dfm import validation
    class OutsideRange(torch.nn.Module):
        def forward(self, x, t, r, condition):
            return torch.full_like(x, 10000.)
    path = fixture_manifest(tmp_path)
    records = load_manifest(path, patch_size=8, spacing_m=20.)
    metrics = validation(OutsideRange(), records, config(), 'cpu', fit_normalization(records))['nfe_1']
    assert metrics['rejected_draws'] == metrics['total_draws'] == 2
    assert metrics['outside_physical_range_fraction'] == 1.


def test_resume_without_improvement_writes_self_contained_best(tmp_path):
    path=fixture_manifest(tmp_path)
    cfg=TrainingConfig(**{**config().__dict__,'learning_rate':1e-40,'steps':1})
    train(path,tmp_path/'first',spacing_m=20.,config=cfg)
    initial=torch.load(tmp_path/'first'/'last.pt',weights_only=True)
    summary=train(path,tmp_path/'resumed',spacing_m=20.,
        config=TrainingConfig(**{**cfg.__dict__,'steps':2}),resume=tmp_path/'first'/'last.pt')
    assert (tmp_path/'resumed'/'best.pt').is_file()
    assert (tmp_path/'resumed'/'best_inference.pt').is_file()
    best=torch.load(tmp_path/'resumed'/'best.pt',weights_only=True)
    inference=torch.load(tmp_path/'resumed'/'best_inference.pt',weights_only=True)
    assert summary['best_scope']=='current_invocation_including_resume_initial'
    assert best['best_scope']==summary['best_scope']
    assert best['meta']['training']['steps']==1  # Preserve the actual chosen state, not the new last.
    assert best['validation']==summary['initial_validation']
    assert best['best_score']==summary['best_validation_score_mps']
    for field in ('state_dict','raw_state_dict'):
        for key in initial[field]:
            torch.testing.assert_close(best[field][key],initial[field][key],rtol=0,atol=0)
    assert torch.equal(best['rng_state'],initial['rng_state'])
    assert best['optimizer']['param_groups']==initial['optimizer']['param_groups']
    for index,state in initial['optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(best['optimizer']['state'][index][key],value,rtol=0,atol=0)
    assert inference['meta']==best['meta']
    for key in best['state_dict']:
        assert torch.equal(inference['state_dict'][key],best['state_dict'][key])
    # Full best preserves raw/optimizer/RNG, so it remains a genuine resume source.
    train(path,tmp_path/'from_best',spacing_m=20.,
        config=TrainingConfig(**{**cfg.__dict__,'steps':3}),resume=tmp_path/'resumed'/'best.pt')
    train(path,tmp_path/'full',spacing_m=20.,config=TrainingConfig(**{**cfg.__dict__,'steps':3}))
    resumed=torch.load(tmp_path/'from_best'/'last.pt',weights_only=True)
    full=torch.load(tmp_path/'full'/'last.pt',weights_only=True)
    for field in ('state_dict','raw_state_dict'):
        for key in full[field]:
            torch.testing.assert_close(resumed[field][key],full[field][key],rtol=0,atol=0)
    assert torch.equal(resumed['rng_state'],full['rng_state'])
    assert resumed['optimizer']['param_groups']==full['optimizer']['param_groups']
    for index,state in full['optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(resumed['optimizer']['state'][index][key],value,rtol=0,atol=0)


def test_moved_last_does_not_claim_unavailable_historical_best(tmp_path):
    import shutil
    path=fixture_manifest(tmp_path)
    cfg=TrainingConfig(**{**config().__dict__,'learning_rate':1e-40,'steps':1})
    train(path,tmp_path/'first',spacing_m=20.,config=cfg)
    checkpoint=torch.load(tmp_path/'first'/'last.pt',weights_only=True)
    checkpoint['best_score']=0.0  # A historical score without corresponding carried weights.
    moved=tmp_path/'moved_last.pt'; torch.save(checkpoint,moved)
    shutil.rmtree(tmp_path/'first')
    summary=train(path,tmp_path/'resumed',spacing_m=20.,
        config=TrainingConfig(**{**cfg.__dict__,'steps':2}),resume=moved)
    assert summary['best_validation_score_mps']>0
    assert summary['resume_history']['previous_best_score_mps']==0.0
    assert summary['resume_history']['historical_best_weights_carried'] is False
    assert (tmp_path/'resumed'/'best.pt').is_file()


def test_resumed_initial_best_is_rescored_for_changed_nfe(tmp_path):
    path=fixture_manifest(tmp_path)
    cfg=TrainingConfig(**{**config().__dict__,'learning_rate':1e-40,'steps':1})
    train(path,tmp_path/'first',spacing_m=20.,config=cfg)
    summary=train(path,tmp_path/'resumed',spacing_m=20.,
        config=TrainingConfig(**{**cfg.__dict__,'steps':2,'selection_nfe':4}),resume=tmp_path/'first'/'last.pt')
    best=torch.load(tmp_path/'resumed'/'best.pt',weights_only=True)
    assert summary['training_diagnostics']['selection_changed_on_resume'] is True
    assert best['meta']['training']['validation_selection']=={'statistic':'single_draw','nfe':4}
    assert best['best_score']==summary['initial_validation']['nfe_4']['sample_weak_rmse_mps']

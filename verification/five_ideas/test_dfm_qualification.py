"""Synthetic weights exercise metadata gates, never establish scientific quality."""
import pytest
import torch
from ifwi_research.dfm import FrozenDFMPrior, TwoTimeTransport


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / 'synthetic.pt'
    meta = dict(schema='ifwi_small_dfm_v1', trained_steps=1, width=8,
                spacing_m=20., mean_mps=2000., scale_mps=1000.,
                condition_kernel=3, patch_size=8)
    payload = {'meta': meta, 'model': TwoTimeTransport(8).state_dict()}
    return path, payload


@pytest.mark.parametrize('flag', [None, False, True, 0, '', 'false'])
def test_legacy_or_unqualified_checkpoint_rejected(checkpoint, flag):
    path, payload = checkpoint
    if flag is not None:
        payload['meta']['toy_only'] = flag
    torch.save(payload, path)
    with pytest.raises(ValueError, match='qualification'):
        FrozenDFMPrior(path, 20.)


@pytest.mark.parametrize('qualification', [None, {}, 'reviewed_for_inversion',
    {'status': 'unqualified', 'evidence': 'fixture'},
    {'status': 'reviewed_for_inversion'},
    {'status': 'reviewed_for_inversion', 'evidence': ''},
    {'status': 'reviewed_for_inversion', 'evidence': '  '},
    {'status': 'reviewed_for_inversion', 'evidence': 1}])
def test_incomplete_review_cannot_qualify_checkpoint(checkpoint, qualification):
    path, payload = checkpoint
    payload['meta'].update(toy_only=False, qualification=qualification)
    torch.save(payload, path)
    with pytest.raises(ValueError, match='qualification'):
        FrozenDFMPrior(path, 20.)


@pytest.mark.parametrize('flag', [None, True, 0, '', 'false'])
def test_review_does_not_override_missing_or_nonboolean_toy_flag(checkpoint, flag):
    path, payload = checkpoint
    payload['meta']['qualification'] = {
        'status': 'reviewed_for_inversion', 'evidence': 'synthetic metadata contract fixture only'}
    if flag is not None:
        payload['meta']['toy_only'] = flag
    torch.save(payload, path)
    with pytest.raises(ValueError, match='qualification'):
        FrozenDFMPrior(path, 20.)


def test_explicit_review_record_is_accepted_as_declaration_only(checkpoint):
    path, payload = checkpoint
    payload['meta'].update(toy_only=False, qualification={
        'status': 'reviewed_for_inversion', 'evidence': 'synthetic metadata contract fixture only'})
    torch.save(payload, path)
    prior = FrozenDFMPrior(path, 20.)
    assert not any(p.requires_grad for p in prior.model.parameters())


def test_explicit_smoke_opt_in_loads_without_rewriting_qualification(checkpoint):
    path, payload = checkpoint
    payload['meta']['toy_only'] = True
    torch.save(payload, path)
    with pytest.warns(UserWarning, match='unqualified'):
        prior = FrozenDFMPrior(path, 20., allow_unqualified=True)
    assert prior.meta['toy_only'] is True
    assert torch.load(path, weights_only=True)['meta']['toy_only'] is True
    target = prior.proposal(torch.full((1, 8, 8), 2200.), 1)
    assert torch.isfinite(target).all()


@pytest.mark.parametrize('value', ['false', 'true', 1, None])
def test_smoke_opt_in_requires_boolean(checkpoint, value):
    path, payload = checkpoint
    torch.save(payload, path)
    with pytest.raises(ValueError, match='allow_unqualified'):
        FrozenDFMPrior(path, 20., allow_unqualified=value)


@pytest.mark.parametrize('changes,spacing,error', [
    ({'trained_steps': 0}, 20., 'trained'),
    ({}, 15., 'spacing'),
    ({'scale_mps': 0.}, 20., 'normalization')])
def test_smoke_opt_in_preserves_other_checks(checkpoint, changes, spacing, error):
    path, payload = checkpoint
    payload['meta'].update(changes)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=error):
        FrozenDFMPrior(path, spacing, allow_unqualified=True)


def test_smoke_opt_in_rejects_nonfinite_weights(checkpoint):
    path, payload = checkpoint
    next(iter(payload['model'].values())).fill_(float('nan'))
    torch.save(payload, path)
    with pytest.raises(ValueError, match='Nonfinite.*weights'):
        FrozenDFMPrior(path, 20., allow_unqualified=True)


@pytest.mark.parametrize('value', [True, 'false', 0, 1, None])
def test_runner_config_rejects_smoke_override(tmp_path, value):
    import json
    from ifwi_research.run import read_research_config
    path = tmp_path / 'research.json'
    path.write_text(json.dumps({'method': 'dfm', 'prior': {
        'checkpoint': 'synthetic.pt', 'allow_unqualified': value}}))
    with pytest.raises(ValueError, match='allow_unqualified'):
        read_research_config(path)


def test_runner_config_allows_explicit_closed_gate(tmp_path):
    import json
    from ifwi_research.run import read_research_config
    path = tmp_path / 'research.json'
    path.write_text(json.dumps({'method': 'dfm', 'prior': {
        'checkpoint': 'reviewed.pt', 'allow_unqualified': False}}))
    _, prior, _ = read_research_config(path)
    assert prior['allow_unqualified'] is False

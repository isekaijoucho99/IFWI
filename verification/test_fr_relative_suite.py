import copy
import numpy as np
import pytest

from experiments.fr_relative_suite import ARMS, budget, make_model, threshold_crossings, validate_target
from experiments.residual_ifwi_experiment import default_config


def test_fourteen_case_budget_includes_warmup_and_evaluations():
    b = budget('formal', 500)
    assert len(ARMS) == 7
    assert b['training_forward_fields'] == 119000
    assert b['warmup_forward_fields'] == 476
    assert b['evaluation_forward_fields'] == 3692
    assert b['forward_fields'] == 123168
    assert b['backward_fields'] == 119476
    assert b['optimizer_updates'] == 7028


def test_timing_and_preflight_are_additional_cost():
    assert budget('timing')['forward_fields'] == 1428
    assert budget('timing')['backward_fields'] == 1428
    assert budget('timing')['optimizer_updates'] == 84
    assert budget('preflight')['forward_fields'] == 34
    assert budget('preflight')['optimizer_updates'] == 0


def test_unreached_thresholds_are_not_interpolated_or_promised():
    rows = [dict(update=0, rmse_mps=537., training_seconds=0., elapsed_seconds=1.),
            dict(update=10, rmse_mps=470., training_seconds=30., elapsed_seconds=36.)]
    result = threshold_crossings(rows, [500., 475., 450.])
    assert result[0]['first_observed_update'] == 10
    assert result[0]['training_seconds'] == 30.
    assert result[2]['status'] == 'NOT_REACHED'
    assert result[2]['first_observed_update'] is None


def test_no_automatic_1000_expansion_and_explicit_resume_required():
    validate_target('formal', 500, None)
    validate_target('formal', 1000, 'verified500')
    with pytest.raises(ValueError): validate_target('formal', 1000, None)
    with pytest.raises(ValueError): validate_target('formal', 501, None)
    with pytest.raises(ValueError): validate_target('timing', 1000, 'old')


def test_encoding_controls_isolate_single_factors():
    by = {a['name']: a for a in ARMS}
    assert not by['full_raw']['preconditioned']
    assert by['full_preconditioned']['preconditioned']
    for name in ('global1', 'interleaved2', 'interleaved4'):
        assert by[name]['model'] == 'plain'
        assert by[name]['preconditioned']
    assert by['interleaved4_relative']['model'] == 'relative'
    assert by['interleaved4_relative']['preconditioned']
    assert by['interleaved4_relative_fr']['model'] == 'relative_fr'
    assert by['interleaved4_relative_fr']['preconditioned']


def test_relative_plain_arm_preserves_50049_parameters_and_canonical_hidden_weights():
    import torch
    cfg = default_config()
    background = np.linspace(1500., 3500., 12, dtype=np.float32).reshape(3, 4)
    assets = dict(config=cfg, background=background, canonical_checkpoint=None)
    by = {a['name']: a for a in ARMS}
    plain, _ = make_model(assets, by['interleaved4'], 'cpu')
    rel, _ = make_model(assets, by['interleaved4_relative'], 'cpu')
    rel_fr, _ = make_model(assets, by['interleaved4_relative_fr'], 'cpu')
    assert sum(p.numel() for p in plain.parameters()) == 50049
    assert sum(p.numel() for p in rel.parameters()) == 50049
    assert sum(p.numel() for p in rel_fr.parameters()) == 3146625
    for (n1, p1), (n2, p2) in zip(plain.net.named_parameters(), rel.net.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)

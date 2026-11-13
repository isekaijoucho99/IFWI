import copy
import numpy as np
import pytest

from experiments.fr_relative_suite import ARMS, budget, threshold_crossings, validate_target


def test_twelve_case_budget_includes_warmup_and_six_evaluations():
    b = budget('formal', 500)
    assert len(ARMS) == 6
    assert b['training_forward_fields'] == 115000
    assert b['warmup_forward_fields'] == 460
    assert b['evaluation_forward_fields'] == 3172
    assert b['forward_fields'] == 118632
    assert b['backward_fields'] == 115460
    assert b['optimizer_updates'] == 6024


def test_timing_and_preflight_are_additional_cost():
    assert budget('timing')['forward_fields'] == 1380
    assert budget('timing')['backward_fields'] == 1380
    assert budget('timing')['optimizer_updates'] == 72
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


def test_encoding_controls_match_full_preconditioned_except_encoding():
    by = {a['name']: a for a in ARMS}
    assert not by['full_raw']['preconditioned']
    assert by['full_preconditioned']['preconditioned']
    for name in ('global1', 'interleaved2', 'interleaved4'):
        assert by[name]['model'] == 'plain'
        assert by[name]['preconditioned']
    assert by['interleaved4_relative_fr']['model'] == 'relative_fr'

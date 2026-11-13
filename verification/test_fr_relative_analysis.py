import copy
import json

import numpy as np
import pytest

from experiments.fr_relative_analysis import (
    AnalysisError, canonical_hash, check_metric, first_crossings,
    physical_budget, validate_curve, verify_identity, verify_plan, verify_stage,
    waveform_mse,
)


def test_physical_budget_charges_one_shared_initial_and_supports_1000_resume():
    assert physical_budget('preflight') == dict(forward_fields=34, backward_fields=34, optimizer_updates=0)
    assert physical_budget('timing') == dict(forward_fields=1428, backward_fields=1428, optimizer_updates=84)
    assert physical_budget('formal', target=500, resume=False) == dict(
        forward_fields=123168, backward_fields=119476, optimizer_updates=7028)
    assert physical_budget('formal', target=1000, resume=True) == dict(
        forward_fields=122640, backward_fields=119000, optimizer_updates=7000)


def test_crossings_report_observed_samples_and_unreached_without_interpolation():
    rows = [dict(update=0, rmse_mps=537., training_seconds=0., elapsed_seconds=4.),
            dict(update=10, rmse_mps=470., training_seconds=30., elapsed_seconds=37.)]
    got = first_crossings(rows, (500., 475., 450.))
    assert got[0]['first_observed_update'] == 10
    assert got[0]['training_seconds'] == 30.
    assert got[0]['elapsed_seconds'] == 37.
    assert got[2]['status'] == 'NOT_REACHED'
    assert got[2]['first_observed_update'] is None


def test_counterfactual_better_later_sample_does_not_backdate_threshold():
    rows = [dict(update=0, rmse_mps=510., training_seconds=0., elapsed_seconds=2.),
            dict(update=10, rmse_mps=490., training_seconds=35., elapsed_seconds=42.),
            dict(update=20, rmse_mps=480., training_seconds=70., elapsed_seconds=80.)]
    assert first_crossings(rows, (500.,))[0]['first_observed_update'] == 10


def test_curve_rejects_missing_duplicate_reordered_and_time_reversal():
    rows = [dict(update=k, rmse_mps=537.-k, training_seconds=float(k),
                 elapsed_seconds=float(k)+2., evaluation_seconds=float(k)/10.)
            for k in (0, 10, 20)]
    validate_curve(rows, target=20)
    for bad in (rows[:-1], [rows[0], rows[0], rows[2]], list(reversed(rows))):
        with pytest.raises(AnalysisError): validate_curve(bad, target=20)
    bad = copy.deepcopy(rows); bad[2]['elapsed_seconds'] = 1.
    with pytest.raises(AnalysisError): validate_curve(bad, target=20)


def test_waveform_mse_uses_float64_and_rejects_broadcasting_nonfinite():
    pred = np.array([[[[1.e4, 2.]]]], np.float32)
    target = np.array([[[[1.e4+1., 4.]]]], np.float32)
    assert waveform_mse(pred, target) == 2.5
    with pytest.raises(AnalysisError): waveform_mse(pred, target[0])
    bad = pred.copy(); bad.flat[0] = np.nan
    with pytest.raises(AnalysisError): waveform_mse(bad, target)


def test_metric_validation_retains_deterioration_and_rejects_false_scores():
    check_metric('rmse_mps', 550., 550.)
    with pytest.raises(AnalysisError): check_metric('rmse_mps', 450., 550.)
    with pytest.raises(AnalysisError): check_metric('clean_mse', float('nan'), 1.)


def test_identity_requires_matching_run_pid_and_normal_exit():
    launch = dict(run_id='unique', pid=41, parent_pid=7, owner_pid=41)
    end = dict(launch, status='COMPLETED', exit_code=0)
    verify_identity(launch, end)
    for changes in (dict(run_id='other'), dict(pid=42), dict(parent_pid=8),
                    dict(status='FAILED', exit_code=1)):
        with pytest.raises(AnalysisError): verify_identity(launch, dict(end, **changes))


def test_plan_hash_checks_sources_and_rejects_metadata_tampering(tmp_path):
    source = tmp_path/'example.py'; source.write_text('known code\n')
    import hashlib
    plan = dict(schema='fr_relative_52_v1', source_sha256={'example.py': hashlib.sha256(source.read_bytes()).hexdigest()})
    plan['signature'] = canonical_hash(plan)
    plan['status'] = 'FROZEN_BEFORE_GPU'; plan['budgets'] = {}
    verify_plan(plan, source_root=tmp_path)
    bad = dict(plan, schema='changed')
    with pytest.raises(AnalysisError): verify_plan(bad, source_root=tmp_path)
    source.write_text('changed code\n')
    with pytest.raises(AnalysisError): verify_plan(plan, source_root=tmp_path)


def _preflight_fixture(tmp_path):
    control = tmp_path/'control'; control.mkdir()
    launch = dict(run_id='child-one', pid=12, owner_pid=12, parent_pid=10)
    totals = dict(forward_fields_attempt=34, forward_fields_complete=34,
                  backward_fields_attempt=34, backward_fields_complete=34)
    plan = dict(signature='known-plan')
    report = dict(status='COMPLETED', stage='preflight', signature='known-plan',
                  target_updates=500, resume_root=None, cost_totals=totals,
                  cost_by_phase={'preflight':totals}, optimizer_updates=dict(warmup=0,training=0),
                  cases=[dict(passed=True),dict(passed=True)], passed=True)
    for path, value in ((control/'LAUNCH.json',launch),
                        (control/'EXIT.json',dict(launch,status='COMPLETED',exit_code=0,cost_totals=totals)),
                        (tmp_path/'plan.json',plan), (tmp_path/'result.json',report),
                        (tmp_path/'budget.json',physical_budget('preflight'))):
        path.write_text(json.dumps(value), encoding='utf-8')
    events = [dict(launch,event='preflight.'+key,phase='preflight',**{key:34}) for key in totals]
    ledger = tmp_path/'ledger.jsonl'
    ledger.write_text(''.join(json.dumps(row)+'\n' for row in events),encoding='utf-8')
    return plan, events


def test_preflight_acceptance_reconciles_full_counter_ledger_and_receipts(tmp_path):
    plan, _ = _preflight_fixture(tmp_path)
    assert verify_stage(tmp_path,'preflight',plan)['report']['cost_totals']['forward_fields_complete'] == 34


def test_preflight_rejects_duplicate_attempt_even_when_terminal_says_success(tmp_path):
    plan, events = _preflight_fixture(tmp_path)
    with (tmp_path/'ledger.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(events[0])+'\n')
    with pytest.raises(AnalysisError, match='cost'):
        verify_stage(tmp_path,'preflight',plan)


def test_preflight_rejects_foreign_ledger_identity(tmp_path):
    plan, events = _preflight_fixture(tmp_path)
    events[0]['run_id'] = 'a-different-attempt'
    (tmp_path/'ledger.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events),encoding='utf-8')
    with pytest.raises(AnalysisError, match='identity'):
        verify_stage(tmp_path,'preflight',plan)

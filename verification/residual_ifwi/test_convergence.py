"""CPU-only stopping tests using full-data MSE and checkpoint velocities."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def convergence_api():
    name = "experiments.residual_ifwi_convergence"
    assert importlib.util.find_spec(name) is not None, "Convergence implementation is missing"
    return importlib.import_module(name)


def records(losses, *, first=800, interval=50):
    return [{"evaluated_updates": first + index * interval, "data_mse": loss}
            for index, loss in enumerate(losses)]


def velocities_for(history, *, step_mps=0.):
    return {row["evaluated_updates"]: np.full((2, 3), 2500. + index * step_mps, dtype=np.float32)
            for index, row in enumerate(history)}


def analyze(history, velocities=None, **policy_options):
    api = convergence_api()
    return api.analyze_convergence(
        history, velocities_for(history) if velocities is None else velocities,
        resumed_from_updates=800, policy=api.ConvergencePolicy(**policy_options))


def test_actual_700_750_800_descent_and_continued_improvement_cannot_stop():
    history = records([2.462115e-4, 2.10483e-4, 1.66370375e-4,
                       1.40e-4, 1.20e-4, 1.00e-4, 8.5e-5, 7.2e-5,
                       6.0e-5, 5.0e-5, 4.0e-5], first=700)
    velocity = velocities_for(history)
    del velocity[700], velocity[750]
    result = analyze(history, velocity)
    assert result["converged"] is False
    assert result["state"] == "improving_or_unstable"
    assert result["confirmation_count"] == 0
    assert result["evaluated_updates"] == 1200
    assert all(check["start_updates"] >= 800 for check in result["checks"])


def test_stable_loss_and_velocity_stop_only_after_400_additional_updates():
    history = records([1e-4] * 9)
    early = analyze(history[:-1])
    assert early["converged"] is False
    assert early["state"] == "insufficient_history"
    final = analyze(history)
    assert final["converged"] is True
    assert final["state"] == "plateau"
    assert final["confirmation_count"] >= 3
    assert final["evaluated_updates"] == 1200
    assert all(check["candidate"] for check in final["checks"])
    json.dumps(final, allow_nan=False)


def test_three_consecutive_candidate_windows_are_required_when_minimum_is_met():
    history = records([1e-4] * 7)
    assert analyze(history[:5], min_additional_updates=0)["converged"] is False
    assert analyze(history[:6], min_additional_updates=0)["confirmation_count"] == 2
    assert analyze(history, min_additional_updates=0)["converged"] is True


def test_fixed_loss_with_continuously_changing_velocity_does_not_stop():
    history = records([1e-4] * 9)
    result = analyze(history, velocities_for(history, step_mps=10.))
    assert result["converged"] is False
    assert result["state"] == "improving_or_unstable"
    assert result["latest_window"]["velocity_step_rtol"] > .001


def test_small_individual_velocity_steps_still_require_window_stability():
    history = records([1e-4] * 9)
    result = analyze(history, velocities_for(history, step_mps=2.))
    assert result["converged"] is False
    assert result["latest_window"]["velocity_step_rtol"] < .001
    assert result["latest_window"]["velocity_window_rtol"] > .002


def test_oscillating_loss_with_zero_fitted_trend_does_not_stop():
    history = records([1e-4, 1.1e-4] * 4 + [1e-4])
    result = analyze(history)
    assert result["converged"] is False
    assert result["latest_window"]["loss_trend_rtol"] < .005
    assert result["latest_window"]["loss_range_rtol"] > .02


def test_new_improvement_resets_confirmation_after_multiple_plateaus():
    history = records([1e-4] * 7 + [8e-5] * 7 + [6e-5])
    result = analyze(history)
    assert result["converged"] is False
    assert result["confirmation_count"] == 0
    assert sum(check["candidate"] for check in result["checks"]) >= 6
    assert result["checks"][-1]["candidate"] is False


def test_ground_truth_metrics_and_other_loss_scopes_are_ignored():
    history = records([1e-4] * 9)
    changed = copy.deepcopy(history)
    for index, row in enumerate(changed):
        row.update(rmse_mps=float("nan"), ssim=-index,
                   common_1p9s_data_mse=float("inf"), sampled_data_mse=10. ** index)
    assert analyze(changed) == analyze(history)


def test_window_metrics_match_hand_calculated_loss_and_velocity_changes():
    history = records([1., 1.1, 1.2])
    velocity = {800: np.full((2, 3), 100.), 850: np.full((2, 3), 101.),
                900: np.full((2, 3), 102.)}
    result = analyze(history, velocity, window_evaluations=3, confirmations=1,
                     min_additional_updates=100, loss_trend_rtol=.5,
                     loss_range_rtol=.5, velocity_step_rtol=.5, velocity_window_rtol=.5)
    window = result["latest_window"]
    assert window["loss_trend_rtol"] == pytest.approx(2. / 11.)
    assert window["loss_range_rtol"] == pytest.approx(2. / 11.)
    assert window["velocity_step_rtol"] == pytest.approx(.01)
    assert window["velocity_window_rtol"] == pytest.approx(.02)
    assert result["converged"] is True


def test_zero_loss_is_a_defined_finite_plateau():
    result = analyze(records([0.] * 9))
    assert result["converged"] is True
    assert result["latest_window"]["loss_trend_rtol"] == 0.
    assert result["latest_window"]["loss_range_rtol"] == 0.
    json.dumps(result, allow_nan=False)


def test_empty_history_reports_insufficient_history():
    result = analyze([], {})
    assert result["converged"] is False
    assert result["state"] == "insufficient_history"
    assert result["evaluated_updates"] is None
    assert result["latest_window"] is None
    assert result["checks"] == []


def test_history_before_resume_boundary_cannot_supply_confirmation_windows():
    history = records([1e-4] * 17, first=0)
    result = analyze(history, {800: np.full((2, 3), 2500.)})
    assert result["converged"] is False
    assert result["state"] == "insufficient_history"
    assert result["confirmation_count"] == 0
    assert result["checks"] == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_invalid_full_data_loss_raises(value):
    history = records([1e-4] * 9)
    history[4]["data_mse"] = value
    with pytest.raises(ValueError):
        analyze(history)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0., -1.])
def test_invalid_velocity_raises(value):
    history = records([1e-4] * 9)
    velocity = velocities_for(history)
    velocity[1000][0, 0] = value
    with pytest.raises(ValueError):
        analyze(history, velocity)


def test_checkpoint_velocity_shape_mismatch_raises():
    history = records([1e-4] * 9)
    velocity = velocities_for(history)
    velocity[1000] = np.full((3, 2), 2500.)
    with pytest.raises(ValueError):
        analyze(history, velocity)


@pytest.mark.parametrize("updates", [[800, 850, 850], [800, 900, 850]])
def test_duplicate_or_nonincreasing_updates_raise(updates):
    history = [{"evaluated_updates": update, "data_mse": 1e-4} for update in updates]
    with pytest.raises(ValueError):
        analyze(history)


def test_missing_expected_interval_raises_instead_of_skipping_evaluation():
    history = records([1e-4] * 9)
    del history[3]
    with pytest.raises(ValueError):
        analyze(history)


def test_off_interval_evaluation_raises():
    history = records([1e-4] * 9)
    history[2]["evaluated_updates"] = 901
    with pytest.raises(ValueError):
        analyze(history)


def test_missing_resume_boundary_evaluation_raises():
    with pytest.raises(ValueError):
        analyze(records([1e-4] * 9, first=850))


def test_missing_velocity_used_by_a_window_raises():
    history = records([1e-4] * 9)
    velocity = velocities_for(history)
    del velocity[900]
    with pytest.raises(ValueError):
        analyze(history, velocity)


@pytest.mark.parametrize("options", [
    {"eval_interval": 0}, {"eval_interval": True}, {"window_evaluations": 1},
    {"confirmations": 0}, {"min_additional_updates": -1},
    {"loss_trend_rtol": -1.}, {"loss_range_rtol": float("nan")},
    {"loss_trend_rtol": "0.005"},
    {"velocity_step_rtol": float("inf")}, {"velocity_window_rtol": -1.},
])
def test_invalid_policy_raises(options):
    with pytest.raises(ValueError):
        analyze(records([1e-4] * 9), **options)


@pytest.mark.parametrize("update", [True, 800.5, -1])
def test_invalid_evaluation_update_raises(update):
    history = records([1e-4] * 9)
    history[0]["evaluated_updates"] = update
    with pytest.raises(ValueError):
        analyze(history)

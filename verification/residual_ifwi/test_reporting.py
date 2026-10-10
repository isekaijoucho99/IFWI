"""Scientific metric contracts on known CPU-only velocity fields."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def metrics(prediction, truth):
    path = ROOT / "experiments" / "residual_ifwi_report.py"
    assert path.is_file(), "Missing independent residual IFWI reporting module"
    spec = importlib.util.spec_from_file_location("residual_ifwi_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.velocity_metrics(prediction, truth)


def test_constant_offset_reports_physical_rmse_and_mae_without_clipping():
    truth = np.arange(35, dtype=np.float32).reshape(5, 7) + 5000
    result = metrics(truth + 800, truth)

    assert result["rmse_mps"] == pytest.approx(800)
    assert result["mae_mps"] == pytest.approx(800)
    assert set(result) == {"rmse_mps", "mae_mps", "ssim"}
    assert all(type(value) is float for value in result.values())
    json.dumps(result, allow_nan=False)


def test_signed_nonuniform_errors_distinguish_rmse_from_mae():
    truth = np.full((3, 3), 3000.0)
    prediction = truth + np.array([[-3.0, 0.0, 4.0]] * 3)
    result = metrics(prediction, truth)

    assert result["rmse_mps"] == pytest.approx(2.886751345948129)
    assert result["mae_mps"] == pytest.approx(2.3333333333333335)


@pytest.mark.parametrize("shape", [(3, 5), (4, 6), (7, 9)])
def test_identical_velocity_fields_have_unit_ssim_and_zero_errors(shape):
    truth = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1500
    result = metrics(truth.copy(), truth)

    assert result == {"rmse_mps": 0.0, "mae_mps": 0.0, "ssim": 1.0}


def test_constant_truth_uses_one_mps_ssim_data_range():
    # Constant images have zero variance: C1/(1+C1), C1=(0.01*1 m/s)^2.
    result = metrics(np.ones((3, 3)), np.zeros((3, 3)))

    assert result["rmse_mps"] == 1.0
    assert result["mae_mps"] == 1.0
    assert result["ssim"] == pytest.approx(0.0000999900009999, rel=1e-10)


@pytest.mark.parametrize(
    "prediction,truth",
    [
        (np.zeros(9), np.zeros(9)),
        (np.zeros((1, 3, 3)), np.zeros((1, 3, 3))),
        (np.zeros((3, 4)), np.zeros((4, 3))),
        (np.zeros((2, 5)), np.zeros((2, 5))),
    ],
)
def test_invalid_velocity_grid_shapes_are_rejected(prediction, truth):
    with pytest.raises(ValueError):
        metrics(prediction, truth)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("invalid_target", ["prediction", "truth"])
def test_nonfinite_values_in_either_velocity_field_are_rejected(invalid, invalid_target):
    prediction = np.ones((3, 3))
    truth = np.ones((3, 3))
    target = prediction if invalid_target == "prediction" else truth
    target[1, 1] = invalid

    with pytest.raises(ValueError, match="finite"):
        metrics(prediction, truth)

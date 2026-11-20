"""Physical update diagnostics on analytic fields, independent of the wave solver."""

import json
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class LinearVelocity(torch.nn.Module):
    """Two coefficients map to two velocity cells with a known Jacobian."""

    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def forward(self):
        a, b = self.weights
        return torch.stack([2 * a - b, 3 * a + 4 * b]).reshape(1, 1, 2)


def test_predicts_jvps_of_the_weighted_step_and_unclipped_kernel():
    from experiments.direction_a.diagnostics import predict_fields
    from experiments.direction_a.optimizer import UpdateRule

    rule = UpdateRule("adam", 0.25, beta1=0.5, beta2=0.8, clip_norm=0.5)
    rule.accept(rule.propose(torch.tensor([-1., 0.], dtype=torch.float64)))
    raw = torch.tensor([3., 4.], dtype=torch.float64, requires_grad=True)
    proposal = rule.propose(raw)
    fields = predict_fields(LinearVelocity(), proposal, 0.5, raw, rule.lr)
    assert set(fields) == {"linear", "current", "history", "instant", "sgd_kernel"}
    jacobian = torch.tensor([[2., -1.], [3., 4.]], dtype=torch.float64)
    for name, delta in dict(linear=proposal.delta, current=proposal.current,
                            history=proposal.history, instant=proposal.instant).items():
        expected = (0.5 * (jacobian @ delta)).reshape(1, 2)
        torch.testing.assert_close(fields[name], expected)
    torch.testing.assert_close(fields["sgd_kernel"], torch.tensor([[-0.25, -3.125]], dtype=torch.float64))
    assert not any(field.requires_grad for field in fields.values())


def first_cell(value):
    return torch.tensor([[value, 0.], [0., 0.]], dtype=torch.float64)


def test_physical_gradient_dots_rms_angles_and_posthoc_rescaling():
    from experiments.direction_a.diagnostics import measure

    actual = torch.tensor([[3., 4.], [0., 0.]])
    fields = {name: first_cell(value) for name, value in
              dict(linear=2, current=4, history=-2, instant=6, sgd_kernel=3).items()}
    row = measure(actual, first_cell(-1), fields, spacing_m=15.)
    assert row["actual_rms_mps"] == 2.5
    assert row["linear_relative_error"] == pytest.approx(math.sqrt(17) / 5)
    assert row["instant_relative_error"] == 1
    assert row["sgd_kernel_relative_error"] == pytest.approx(0.8)
    assert row["physical_gradient_dot_actual"] == -3
    assert row["current_gradient_dot"] == -4
    assert row["history_gradient_dot"] == 2
    assert row["history_first_order_gain_vs_instant"] == -4
    assert row["decomposition_error_rms_mps"] == 0
    assert row["cosine_to_descent"] == pytest.approx(0.6)
    assert row["angle_to_descent_deg"] == pytest.approx(53.13010235)
    for name, scale in dict(linear=1.5, instant=0.5, sgd_kernel=1., steepest_descent=3.).items():
        fit = row["posthoc_scalar_fit"][name]
        assert fit["cosine_to_actual"] == pytest.approx(0.6)
        assert fit["best_scalar"] == pytest.approx(scale)
        assert fit["rms_matched_relative_error"] == pytest.approx(math.sqrt(0.8))
        assert fit["best_scalar_relative_error"] == pytest.approx(0.8)
    for name, rms in dict(actual=2.5, linear=1, history=1, remainder=math.sqrt(17) / 2).items():
        bands = row["band_rms_mps"][name]
        assert sum(b ** 2 for b in bands.values()) == pytest.approx(rms ** 2)
    json.dumps(row, allow_nan=False)


def test_history_can_assist_descent_and_signed_rescaling_can_hide_reversal():
    from experiments.direction_a.diagnostics import measure

    fields = {name: first_cell(value) for name, value in
              dict(linear=4, current=1, history=3, instant=2, sgd_kernel=-1).items()}
    row = measure(first_cell(2), first_cell(-1), fields, spacing_m=15)
    assert row["history_gradient_dot"] == -3
    assert row["history_first_order_gain_vs_instant"] == 2
    fit = row["posthoc_scalar_fit"]["sgd_kernel"]
    assert fit["best_scalar"] == -2
    assert fit["best_scalar_relative_error"] == 0
    assert fit["nonnegative_scalar_relative_error"] == 1
    assert fit["rms_matched_relative_error"] == 2
    assert fit["cosine_to_actual"] == -1


def test_zero_norms_and_inconsistent_decomposition():
    from experiments.direction_a.diagnostics import measure

    fields = {name: first_cell(0) for name in ("linear", "current", "history", "instant", "sgd_kernel")}
    fields["history"] = first_cell(3)
    row = measure(first_cell(0), first_cell(0), fields, spacing_m=15)
    assert row["linear_relative_error"] is None
    assert row["cosine_to_descent"] is None
    assert row["angle_to_descent_deg"] is None
    assert row["decomposition_error_rms_mps"] == 1.5
    assert row["posthoc_scalar_fit"]["linear"]["best_scalar"] is None
    json.dumps(row, allow_nan=False)


def test_single_dc_band_has_correct_physical_units_and_nonzero_orthogonal_error():
    from experiments.direction_a.diagnostics import measure

    actual = torch.full((2, 2), 7.)
    fields = {name: torch.zeros_like(actual) for name in ("linear", "current", "history", "instant", "sgd_kernel")}
    fields["linear"] = torch.tensor([[1., -1.], [1., -1.]])
    row = measure(actual, torch.ones_like(actual), fields, spacing_m=15)
    bands = row["band_rms_mps"]["actual"]
    assert bands["0-1cpkm"] == pytest.approx(7.)
    assert sum(b for label, b in bands.items() if label != "0-1cpkm") == pytest.approx(0.)
    assert row["posthoc_scalar_fit"]["linear"]["best_scalar_relative_error"] == 1
    assert row["posthoc_scalar_fit"]["linear"]["best_scalar"] == 0

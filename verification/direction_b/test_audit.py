import json

import numpy as np
import pytest

from experiments.direction_b import audit
from experiments.direction_b.model import load_patch, make_model, parameter_gradient, slowness, step, tangent


def test_multishot_multifrequency_reference_matches_network_finite_difference():
    truth, initial, spacing = load_patch("overthrust", (8, 12))
    model = make_model(initial, spacing, 5)
    frequencies = [4., 7.]
    geometry = audit.geometry(initial.shape, 2)
    survey = audit.observations(truth, spacing, frequencies, 0.04, geometry)
    def oracle():
        cases = audit.build_cases(slowness(model), spacing, 0.04, survey)
        return audit.reference(cases)

    loss, gm, _ = oracle()
    gradient = parameter_gradient(model, gm.reshape(initial.shape))
    direction = -gradient / np.linalg.norm(gradient)
    directional = np.dot(gradient, direction)
    np.testing.assert_allclose(directional, np.vdot(gm, tangent(model, direction).ravel()), rtol=1e-10)
    eps = 1e-5
    step(model, direction, eps)
    plus = oracle()[0]
    step(model, -direction, 2 * eps)
    minus = oracle()[0]
    assert loss > 0
    np.testing.assert_allclose((plus - minus) / (2 * eps), directional, rtol=1e-5)


def test_cli_reports_matched_checkpoints_and_oracle_scores(tmp_path):
    out = tmp_path / "trial.json"
    audit.main(["--datasets", "marmousi", "overthrust", "--shape", "8", "12",
                "--shots", "2", "--frequencies", "5", "8", "--checkpoints", "0", "1",
                "--sweep-tols", ".03", ".003", "--output", str(out)])
    record = json.loads(out.read_text())
    for run in record["datasets"]:
        assert [c["update"] for c in run["checkpoints"]] == [0, 1]
        for checkpoint in run["checkpoints"]:
            rows = checkpoint["comparisons"]
            assert sorted(r["method"] for r in rows) == ["dual", "dual_corrected"] + ["fixed"]*4 + ["sloppiness"]
            for row in rows:
                assert row["absolute_error"] <= row["bound"] + 1e-12
                assert -1 <= row["gradient_cosine"] <= 1
                assert row["online_seconds"] >= row["candidate_seconds"] >= 0
                if row["certified"]:
                    assert row["reference_derivative"] < 0
            assert checkpoint["reference_seconds"] > 0
            assert checkpoint["reference_max_relative_residual"] < 1e-10
        assert run["stop_reason"] == "finished"


def test_fail_closed_driver_does_not_advance(monkeypatch):
    from experiments.direction_b import control
    real = control.evaluate

    def uncertified(*args, **kwargs):
        gradient, report = real(*args, **kwargs)
        report["certified"] = False
        return gradient, report

    monkeypatch.setattr(control, "evaluate", uncertified)
    args = audit.parser().parse_args(["--checkpoints", "0", "2", "--shape", "8", "12",
                                     "--shots", "1", "--frequencies", "6", "--sweep-tols", ".02"])
    run = audit.run_dataset("overthrust", args)
    assert [c["update"] for c in run["checkpoints"]] == [0]
    assert run["completed_updates"] == 0
    assert run["stop_reason"] == "uncertified_direction"


def test_duplicate_sources_rejected():
    with pytest.raises(ValueError, match="shot"):
        audit.geometry((8, 8), 100)


def test_failed_fixed_solve_is_charged_and_other_comparisons_run(monkeypatch):
    real = audit.control.evaluate

    def fail_fixed(cases, dm, method, *args, **kwargs):
        if method == "fixed":
            raise audit.SolveFailure("test iteration limit", iterations=8, matvecs=11, seconds=.01)
        return real(cases, dm, method, *args, **kwargs)

    monkeypatch.setattr(audit.control, "evaluate", fail_fixed)
    args = audit.parser().parse_args(["--shape", "8", "12", "--shots", "1", "--frequencies", "5",
                                     "--sweep-tols", ".03", "--checkpoints", "0"])
    run = audit.run_dataset("overthrust", args)
    checkpoint, = run["checkpoints"]
    failed, *others = checkpoint["comparisons"]
    assert failed["stop_reason"] == "solver_failed"
    assert failed["online_matvecs"] == checkpoint["candidate_work"]["matvecs"] + 11
    assert all("absolute_error" in row for row in others)

import json
from types import SimpleNamespace

import numpy as np

from experiments.direction_b import benchmark, control, survey
from experiments.direction_b.model import load_patch, make_model, parameter_gradient, slowness, step, tangent


def settings(**overrides):
    values = dict(shape=(8, 12), shots=2, frequencies=[5., 8.], damping=.04, sponge=.12,
                  seed=3, candidate_tol=.0002, start_tol=.03, min_tol=1e-10, eta=.05,
                  step_mps=5., fixed_tol=.0001, target_ratio=.999999, max_updates=2)
    return SimpleNamespace(**(values | overrides))


def test_target_scored_after_complete_independent_paths():
    args = settings()
    report = benchmark.compare_trajectories("marmousi", args)
    json.dumps(report, allow_nan=False)
    methods = {row["method"]: row for row in report["methods"]}
    assert set(methods) == {"fixed", "sloppiness", "dual", "dual_corrected"}
    assert report["target_misfit"] == report["initial_misfit"] * args.target_ratio
    for name, row in methods.items():
        initial, *accepted = row["trajectory"]
        assert initial["reference_misfit"] == report["initial_misfit"]
        assert initial["online_matvecs"] == 0
        assert row["completed_updates"] == len(accepted) == args.max_updates
        for attempt, point in zip(row["attempts"], accepted, strict=True):
            pilot = attempt["pilot"]
            assert (pilot is None) == (name in {"fixed", "sloppiness"})
            assert attempt["online_matvecs"] == attempt["policy"]["matvecs"] + (pilot["matvecs"] if pilot else 0)
            assert point["online_seconds"] >= attempt["online_seconds"] > 0
        assert row["total_online_matvecs"] == sum(a["online_matvecs"] for a in row["attempts"])
        expected = next((point for point in accepted if point["reference_misfit"] <= report["target_misfit"]), None)
        assert row["first_crossing"] == expected
        assert row["matched_target"] == (expected is not None)
    assert methods["fixed"]["first_crossing"]["update"] == 1

    # Reproduce the first fixed-baseline update with just its own gradient
    # and score the resulting physical model using the separate reference API.
    truth, initial, h = load_patch("marmousi", args.shape)
    model = make_model(initial, h, args.seed)
    data = survey.observations(truth, h, args.frequencies, args.damping,
                               survey.geometry(args.shape, args.shots), sponge=args.sponge)
    m = slowness(model)
    cases = survey.build_cases(m, h, args.damping, data, sponge=args.sponge)
    gm, _ = control.baseline_gradient(cases, "fixed", args.fixed_tol, args.fixed_tol)
    g = parameter_gradient(model, gm.reshape(m.shape))
    p = -g / np.linalg.norm(g)
    step(model, p, args.step_mps / np.max(np.abs(tangent(model, p)) / (2 * m**1.5)))
    updated = survey.build_cases(slowness(model), h, args.damping, data, sponge=args.sponge)
    np.testing.assert_allclose(methods["fixed"]["trajectory"][1]["reference_misfit"],
                               survey.reference(updated)[0], rtol=1e-12, atol=0)


def test_unmet_target_and_rejected_attempts_are_charged():
    report = benchmark.compare_trajectories("overthrust", settings(
        target_ratio=1e-12, max_updates=1, candidate_tol=.25, start_tol=.25, min_tol=.25))
    for row in report["methods"]:
        assert not row["matched_target"]
        assert row["first_crossing"] is None
        assert row["total_online_matvecs"] > 0
        if row["method"] in {"dual", "dual_corrected"}:
            assert row["stop_reason"] == "uncertified_direction"
            assert row["completed_updates"] == 0
            assert len(row["trajectory"]) == 1
            assert row["total_online_matvecs"] == row["attempts"][0]["online_matvecs"]


def test_failed_solves_charge_partial_work_and_allow_other_paths(monkeypatch):
    from experiments.direction_b.physics import SolveFailure

    args = settings(fixed_tol=.00331, candidate_tol=.0022, start_tol=.0014,
                    shots=1, frequencies=[6.], max_updates=1)
    real_solve = control.solve
    completed = {args.fixed_tol: [], args.candidate_tol: []}

    def solve(matrix, rhs, rtol, x0=None):
        # Interrupt the first fixed and pilot adjoints, and the corrected
        # controller's first warm-started refinement after its successful pilot.
        if (rtol in completed and len(completed[rtol]) == 1
                or rtol == args.start_tol and x0 is not None):
            if rtol in completed:
                completed[rtol].append(None)
            raise SolveFailure("injected GMRES failure", matvecs=7, iterations=5, seconds=0.)
        solved = real_solve(matrix, rhs, rtol, x0)
        if rtol in completed:
            completed[rtol].append(solved)
        return solved

    monkeypatch.setattr(control, "solve", solve)
    report = benchmark.compare_trajectories("marmousi", args)
    json.dumps(report, allow_nan=False)
    methods = {row["method"]: row for row in report["methods"]}
    assert methods["sloppiness"]["completed_updates"] == 1
    for name in ("fixed", "dual", "dual_corrected"):
        row = methods[name]
        assert row["stop_reason"] == "solver_failed" and row["completed_updates"] == 0
        assert row["first_crossing"] is None and len(row["trajectory"]) == 1
        attempt, = row["attempts"]
        failure = attempt["failure"]
        assert "injected GMRES failure" in failure["diagnostic"]
        assert failure["stage"] == ("candidate" if name == "dual" else "policy")
        partial_matvecs, partial_iterations = 7, 5
        if name == "dual_corrected":
            assert attempt["pilot"]["matvecs"] > 0
        else:
            prev = completed[args.fixed_tol if name == "fixed" else args.candidate_tol][0]
            partial_matvecs += prev.matvecs
            partial_iterations += prev.iterations
        assert failure["work"]["matvecs"] == partial_matvecs
        assert failure["work"]["iterations"] == partial_iterations
        pilot = attempt["pilot"] or {}
        assert row["total_online_matvecs"] == partial_matvecs + pilot.get("matvecs", 0)
        assert row["total_online_iterations"] == partial_iterations + pilot.get("iterations", 0)
        assert row["total_online_seconds"] == attempt["online_seconds"] > 0


def test_nonphysical_trial_leaves_grid_at_previous_update():
    report = benchmark.compare_trajectories("marmousi", settings(
        step_mps=1e200, frequencies=[6.], shots=1, max_updates=1))
    for row in report["methods"]:
        assert row["stop_reason"] == "invalid_trial_velocity"
        assert row["attempts"][0]["failure"]["diagnostic"]
        assert row["completed_updates"] == 0
        only_point, = row["trajectory"]
        assert only_point["reference_misfit"] == report["initial_misfit"]
        assert row["total_online_matvecs"] > 0


def test_zero_approximate_gradient_keeps_positive_reference_misfit(monkeypatch):
    prepare = control.prepare_candidate

    def zero_candidate(*args, **kwargs):
        grad, stats, pilot = prepare(*args, **kwargs)
        return np.zeros_like(grad), stats, pilot

    monkeypatch.setattr(control, "prepare_candidate", zero_candidate)
    report = benchmark.compare_trajectories("marmousi", settings(
        frequencies=[6.], shots=1, max_updates=1))
    assert report["initial_misfit"] > 0
    for row in report["methods"]:
        if row["method"] in {"dual", "dual_corrected"}:
            assert row["stop_reason"] == "zero_candidate_direction"
            assert row["completed_updates"] == 0
            assert row["total_online_matvecs"] == row["attempts"][0]["pilot"]["matvecs"] > 0

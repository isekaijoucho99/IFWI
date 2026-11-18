from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import norm
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from experiments.direction_b import control
from experiments.direction_b.physics import Helmholtz


def test_weak_source_residual_does_not_certify_a_wrong_sign():
    op = Helmholtz(np.ones((6, 6)), 0.5, 0.3, 0.1, 0.3)
    rng = np.random.default_rng(1)
    source = np.zeros(36, dtype=complex)
    source[14] = 1e-180
    data = rng.normal(size=36) + 1j * rng.normal(size=36)
    case = control.Case(op, source, np.arange(36), data)
    u = spsolve(op.A, source)
    adjoint = spsolve(op.AH, u - data)
    exact = op.coefficient * (adjoint.conj() * u).real
    approx, _ = control.baseline_gradient([case], "fixed", .5, .5)
    dm = exact / norm(exact) - approx / norm(approx)
    _, row = control.evaluate([case], dm, "dual", .5, .5, min_tol=.5)
    assert row["derivative"] < 0 < exact @ dm
    assert not row["certified"]
    assert row["derivative"] + row["bound"] >= exact @ dm


def case(data=0.0):
    matrix = diags([1.0 + 0j], format="csr")
    op = SimpleNamespace(A=matrix, AH=matrix.getH().tocsr(), K=1.0,
                         shape=(1, 1), coefficient=1.0)
    return control.Case(op, np.array([1.0]), np.array([0]), np.array([data]))


def mock_solve(monkeypatch, approximation):
    calls = []

    def solve(matrix, rhs, rtol, x0=None):
        calls.append((matrix, rhs.copy(), rtol, x0))
        return SimpleNamespace(x=approximation(matrix, rhs, rtol), iterations=2,
                               seconds=0.125, matvecs=3)

    monkeypatch.setattr(control, "solve", solve)
    return calls


def test_forward_residual_contaminates_even_an_exact_adjoint_solve(monkeypatch):
    c = case()
    mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs * (0.75 if matrix is c.op.A else 1))
    g, report = control.evaluate([c], [-1], "fixed", 0.3, 0.01)
    assert g[0] == 0.75**2
    assert report["derivative"] == -g[0]
    assert report["bound"] == abs(-1.0 - report["derivative"])
    assert report["case_checks"][0]["forward_residual_norm"] == 0.25
    assert report["case_checks"][0]["adjoint_residual_norm"] == 0
    assert report["global_bound"] == report["bound"]
    assert report["certified"]
    assert report["matvecs"] == 8


def test_corrected_gradient_misfit_and_bound_use_same_fields(monkeypatch):
    c = case()
    mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs * 0.75)
    g, report = control.evaluate([c], [-1], "dual_corrected", 0.3, 0.3)
    np.testing.assert_array_equal(g, [1.0])
    assert report["derivative"] == -1.0
    assert report["uncorrected_derivative"] == -(0.75 ** 3)
    assert report["bound"] == report["global_bound"] == 0
    assert report["misfit"] == 0.5
    assert report["certified"]
    assert report["matvecs"] == 10
    assert report["correction_seconds"] > 0


def test_strict_gate_refuses_cancelling_case_derivatives(monkeypatch):
    mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs.copy())
    g, report = control.evaluate([case(), case(2.0)], [-1], "dual", 0.01, 0.01,
                                 min_tol=0.01)
    np.testing.assert_array_equal(g, [0.0])
    assert report["derivative"] == report["bound"] == 0.0
    assert not report["certified"]


def test_certified_ascent_is_refused_immediately(monkeypatch):
    calls = mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs.copy())
    _, report = control.evaluate([case()], [1], "dual", 0.5, 0.5)
    assert report["derivative"] > report["bound"]
    assert not report["certified"]
    assert report["stop_reason"] == "certified_ascent"
    assert len(calls) == 2


def test_refining_forward_rebuilds_and_warm_starts_adjoint(monkeypatch):
    c = case()
    calls = mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs * (1 - tol if matrix is c.op.A else 1))
    _, report = control.evaluate([c], [-1], "dual", 0.5, 0.45)
    assert report["certified"] and report["rounds"] == 2
    assert calls[2][0] is c.op.A and calls[2][2] < 0.5
    assert calls[3][0] is c.op.AH and calls[3][2] == 0.45
    assert calls[3][1][0] != calls[1][1][0]
    assert calls[2][3] is not None and calls[3][3] is not None


def test_sloppiness_stops_per_case_and_matches_adjoint_tolerance(monkeypatch):
    first, second = case(), case(100)
    calls = mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs * (1 - tol))
    _, report = control.evaluate([first, second], [-1], "sloppiness", 0.4, 0.01, eta=0.14)
    for c, expected in ((first, [0.4, 0.2, 0.1, 0.05]), (second, [0.4, 0.2])):
        fwd = [call for call in calls if call[0] is c.op.A]
        assert [call[2] for call in fwd] == expected
        assert all(call[3] is not None for call in fwd[1:])
        assert [call[2] for call in calls if call[0] is c.op.AH] == expected[-1:]
    assert report["rounds"] == 4


def test_aggregate_interval_rejects_false_descent_under_cancellation(monkeypatch):
    first, second = case(), case(2.01)
    # Exact contributions are -1 and +1.01. The second inexact adjoint
    # reduces the positive term enough to reverse the computed sign.
    mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs * (0.97 if matrix is second.op.AH else 1))
    _, report = control.evaluate([first, second], [-1], "dual", 0.1, 0.1, min_tol=0.1)
    assert report["derivative"] < 0
    assert report["derivative"] + report["bound"] >= 0.01 - 1e-14
    assert not report["certified"] and report["stop_reason"] == "tolerance_floor"


def test_directional_bound_uses_frozen_support(monkeypatch):
    matrix = diags([1.0 + 0j, 1.0 + 0j], format="csr")
    op = SimpleNamespace(A=matrix, AH=matrix.getH().tocsr(), K=1.0,
                         shape=(1, 2), coefficient=1.0)
    c = control.Case(op, np.array([1.0, 100.0]), np.array([0, 1]), np.zeros(2))
    mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs * np.array([0.9, 1.0]))
    g, report = control.evaluate([c], [-1, 0], "dual_corrected", 0.2, 0.2)
    np.testing.assert_allclose(g, [1.0, 10000.0])
    _, raw = control.evaluate([c], [-1, 0], "fixed", 0.2, 0.2)
    assert abs(raw["derivative"] + 1.0) <= raw["bound"] + 1e-14
    assert raw["bound"] < raw["global_bound"] / 50
    assert raw["certified"] and report["certified"]


def test_baseline_matches_fixed_gradient_and_only_pays_for_system_solves(monkeypatch):
    c = case()
    calls = mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs * (1 - tol))
    expected, fixed = control.evaluate([c], [-1], "fixed", 0.25, 0.05)
    calls.clear()
    g, costs = control.baseline_gradient([c], "fixed", 0.25, 0.05)
    assert [call[2] for call in calls] == [0.25, 0.05]
    np.testing.assert_array_equal(expected, g)
    assert costs["misfit"] == fixed["misfit"]
    assert costs["matvecs"] == fixed["solve_matvecs"] == 6
    assert costs["iterations"] == 4
    assert costs["check_matvecs"] == costs["correction_matvecs"] == 0
    assert costs["forward_seconds"] == costs["adjoint_seconds"] == 0.125
    assert costs["forward_tols"] == [0.25] and costs["adjoint_tols"] == [0.05]


def test_sloppy_baseline_counts_all_stabilization_solves(monkeypatch):
    c = case()
    calls = mock_solve(monkeypatch, lambda matrix, rhs, tol: rhs * (1 - tol))
    _, costs = control.baseline_gradient([c], "sloppiness", 0.4, 0.01, eta=0.14)
    assert [call[2] for call in calls] == [0.4, 0.2, 0.1, 0.05, 0.05]
    assert costs["matvecs"] == 15 and costs["iterations"] == 10
    assert costs["check_seconds"] > 0
    assert costs["forward_seconds"] == 0.5
    assert costs["rounds"] == 4 and costs["stop_reason"] == "misfit_stable"


def test_common_pilot_serves_both_certificates_without_repeated_solves(monkeypatch):
    c = case()
    calls = mock_solve(monkeypatch, lambda m, rhs, tol: rhs * (1 - tol))
    g, cost, pilot = control.prepare_candidate([c], 0.04)
    assert cost["matvecs"] == 6 and len(calls) == 2
    for method in ("dual", "dual_corrected", "dual"):
        grad, r = control.evaluate([c], [-1], method, 0.08, 0.07, pilot=pilot)
        assert r["certified"] and r["rounds"] == 1
        assert r["forward_tols"] == r["adjoint_tols"] == [0.04]
        assert r["solve_matvecs"] == r["iterations"] == 0
        assert r["matvecs"] == (4 if method == "dual_corrected" else 2)
        if method == "dual":
            np.testing.assert_array_equal(grad, g)
    assert len(calls) == 2


def test_refinement_failure_charges_prior_fields_checks_corrections_and_failed_work(monkeypatch):
    op = Helmholtz(np.ones((4, 4)), 0.3, 0.5, 0.1)
    survey = control.Case(op, np.ones(16), np.arange(16), np.zeros(16))
    dm = np.zeros(16)
    dm[[0, 3]] = [1, -1]  # Symmetric fields cancel, forcing refinement.
    cold = []
    solve = control.solve
    failed_side = []

    def singular_on_refinement(matrix, rhs, rtol, x0=None):
        side = "forward" if matrix is op.A else "adjoint"
        if x0 is not None:
            failed_side.append(side)
            return solve(csr_matrix(matrix.shape), rhs, rtol, x0=x0)
        result = solve(matrix, rhs, rtol)
        cold.append((side, result))
        return result

    monkeypatch.setattr(control, "solve", singular_on_refinement)
    with pytest.raises(RuntimeError) as captured:
        control.evaluate([survey], dm, "dual_corrected", 0.8, 0.8)
    error = captured.value
    assert isinstance(error, control.SolveFailure)
    assert len(cold) == 2 and len(failed_side) == 1
    work = error.work
    assert error.matvecs > 0 and error.iterations > 0 and error.seconds > 0
    assert work["solve_matvecs"] == sum(s.matvecs for _, s in cold) + error.matvecs
    assert work["iterations"] == sum(s.iterations for _, s in cold) + error.iterations
    assert work["check_matvecs"] == work["correction_matvecs"] == 2
    assert work["matvecs"] == work["solve_matvecs"] + 4
    assert work["check_seconds"] > 0 and work["correction_seconds"] > 0
    for side, result in cold:
        assert work[side + "_seconds"] == result.seconds + (error.seconds if side == failed_side[0] else 0)


def test_tighter_initial_request_warm_starts_pilot_and_updates_adjoint_rhs(monkeypatch):
    c = case()
    calls = mock_solve(monkeypatch, lambda m, rhs, tol: rhs * (1 - tol))
    _, _, pilot = control.prepare_candidate([c], 0.04)
    _, r = control.evaluate([c], [-1], "dual", 0.01, 0.5, pilot=pilot)
    assert r["certified"] and r["solve_matvecs"] == 6 and len(calls) == 4
    assert calls[2][0] is c.op.A and calls[2][2] == 0.01
    assert calls[3][0] is c.op.AH and calls[3][2] == 0.04
    np.testing.assert_array_equal(calls[2][3], [0.96])
    np.testing.assert_array_equal(calls[3][1], [0.99])
    np.testing.assert_array_equal(calls[3][3], [0.96**2])
    _, a = control.evaluate([c], [-1], "dual", 0.1, 0.02, pilot=pilot)
    assert a["certified"] and a["solve_matvecs"] == 3 and len(calls) == 5
    assert calls[4][0] is c.op.AH and calls[4][2] == 0.02
    np.testing.assert_array_equal(calls[4][1], [0.96])


def test_pilot_is_scoped_to_original_ordered_cases(monkeypatch):
    import pytest
    cs = [case(), case(0.2)]
    calls = mock_solve(monkeypatch, lambda m, rhs, tol: rhs.copy())
    _, _, pilot = control.prepare_candidate(cs, 0.05)
    for incompatible in ([cs[1], cs[0]], [cs[0]], [case(), cs[1]]):
        with pytest.raises(ValueError, match="pilot"):
            control.evaluate(incompatible, [-1], "dual", 0.1, 0.1, pilot=pilot)
    with pytest.raises(ValueError, match="pilot"):
        control.evaluate(cs, [-1], "fixed", 0.1, 0.1, pilot=pilot)
    cs[0].data[0] += 0.1
    with pytest.raises(ValueError, match="pilot"):
        control.evaluate(cs, [-1], "dual_corrected", 0.1, 0.1, pilot=pilot)
    assert len(calls) == 4

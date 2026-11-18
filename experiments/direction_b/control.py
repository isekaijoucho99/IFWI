"""Receiver least-squares derivatives for one frozen squared-slowness direction."""

from dataclasses import dataclass
from math import fsum
from time import perf_counter

import numpy as np
from scipy.linalg import norm

from .physics import Helmholtz, SolveFailure, solve


@dataclass
class Case:
    """One source and one frequency, with unweighted, distinct flat receiver indices."""

    op: Helmholtz
    source: np.ndarray
    receivers: np.ndarray
    data: np.ndarray


@dataclass
class _State:
    case: Case
    ftol: float
    atol: float
    u: np.ndarray | None = None
    lam: np.ndarray | None = None


@dataclass(frozen=True)
class _Pilot:
    owners: tuple[Case, ...]
    fields: tuple[_State, ...]


def _prepare(case, n, ftol, atol):
    if np.prod(case.op.shape) != n or case.op.A.shape != (n, n):
        raise ValueError("direction and operator dimensions must agree")
    src = np.asarray(case.source, dtype=np.complex128).reshape(-1)
    rx = np.asarray(case.receivers)
    data = np.asarray(case.data, dtype=np.complex128).reshape(-1)
    if (rx.ndim != 1 or not np.issubdtype(rx.dtype, np.integer)
            or np.any(rx < 0) or np.any(rx >= n) or np.unique(rx).size != rx.size):
        raise ValueError("receivers must be distinct, in-range flat integer indices")
    if src.size != n or rx.size != data.size:
        raise ValueError("source and receiver data must match the survey dimensions")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(data))):
        raise ValueError("source and data must be finite")
    return _State(Case(case.op, src, rx, data), ftol, atol)


def _rhs(case, u):
    residual = u[case.receivers] - case.data
    rhs = np.zeros_like(u)
    rhs[case.receivers] = residual
    return rhs


def _misfit(case, u):
    error = u[case.receivers] - case.data
    return 0.5 * np.vdot(error, error).real


def _charge_solve(stats, side, result):
    stats[side + "_seconds"] += result.seconds
    stats["iterations"] += result.iterations
    stats["solve_matvecs"] += result.matvecs


def _work(stats):
    total = stats["solve_matvecs"] + stats["check_matvecs"] + stats["correction_matvecs"]
    return dict(stats, matvecs=total)


def _solve(state, side, stats):
    c = state.case
    try:
        if side == "forward":
            solution = solve(c.op.A, c.source, state.ftol, x0=state.u)
            state.u = solution.x
        else:
            solution = solve(c.op.AH, _rhs(c, state.u), state.atol, x0=state.lam)
            state.lam = solution.x
    except SolveFailure as failure:
        _charge_solve(stats, side, failure)
        failure.work = _work(stats)
        raise
    _charge_solve(stats, side, solution)


def _gradient(c, u, lam):
    return c.op.coefficient * (np.conj(lam) * u).real


def _inspect(state, dm, corrected, stats):
    c, u, lam = state.case, state.u, state.lam
    s = -c.op.coefficient * dm
    if corrected:
        start = perf_counter()
        raw_slope = float(-np.vdot(lam, s * u).real)
        # One Jacobi sweep for each field. Changing u also changes the adjoint RHS.
        u = u + (c.source - c.op.A @ u) / c.op.A.diagonal()
        lam = lam + (_rhs(c, u) - c.op.AH @ lam) / c.op.AH.diagonal()
        stats["correction_seconds"] += perf_counter() - start
        stats["correction_matvecs"] += 2

    start = perf_counter()
    rf = c.source - c.op.A @ u
    ra = _rhs(c, u) - c.op.AH @ lam
    # E_lam <= K ||ra|| + K E_u because ||R^H R|| <= 1.
    K = c.op.K
    # Scaled BLAS norms avoid squaring tiny wavefields into a false zero bound.
    rf_norm = norm(rf, check_finite=False)
    ra_norm = norm(ra, check_finite=False)
    eu = K * rf_norm
    ea = K * ra_norm
    ec = K * eu
    snorm = np.max(np.abs(s))
    su = norm(s * u, check_finite=False)
    slam = norm(s.conj() * lam, check_finite=False)
    forward = eu * slam + ec * (su + eu * snorm)
    adjoint = ea * (su + eu * snorm)
    global_bound = snorm * ((ea + ec) * norm(u, check_finite=False)
                            + eu * norm(lam, check_finite=False) + (ea + ec) * eu)
    gradient = _gradient(c, u, lam)
    slope = float(-np.vdot(lam, s * u).real)
    check = dict(derivative=slope, uncorrected_derivative=raw_slope if corrected else slope,
                 forward_bound=forward, adjoint_bound=adjoint, global_bound=global_bound,
                 misfit=float(_misfit(c, u)), forward_residual_norm=rf_norm,
                 adjoint_residual_norm=ra_norm)
    stats["check_seconds"] += perf_counter() - start
    stats["check_matvecs"] += 2
    return gradient, check


def _summary(states, dm, corrected, stats):
    rows = [_inspect(st, dm, corrected, stats) for st in states]
    start = perf_counter()
    gradients, checks = zip(*rows)
    gradient = np.sum(gradients, axis=0)
    report = {key: fsum(row[key] for row in checks)
              for key in ("derivative", "uncorrected_derivative", "forward_bound", "adjoint_bound",
                          "global_bound", "misfit")}
    slope = report["derivative"]
    bound = report["forward_bound"] + report["adjoint_bound"]
    report.update(bound=bound, case_checks=list(checks),
                  certified=bool(np.isfinite(slope) and np.isfinite(bound) and slope < -bound))
    stats["check_seconds"] += perf_counter() - start
    return gradient, report


def _sloppy_solve(state, min_tol, eta, stats, max_rounds):
    _solve(state, "forward", stats)
    start = perf_counter()
    previous = _misfit(state.case, state.u)
    stats["check_seconds"] += perf_counter() - start
    rounds = 1
    converged = False
    while state.ftol > min_tol and rounds < max_rounds:
        state.ftol = max(min_tol, 0.5 * state.ftol)
        _solve(state, "forward", stats)
        start = perf_counter()
        current = _misfit(state.case, state.u)
        converged = abs(current - previous) <= eta * current
        stats["check_seconds"] += perf_counter() - start
        rounds += 1
        if converged:
            break
        previous = current
    state.atol = state.ftol
    _solve(state, "adjoint", stats)
    return rounds, bool(converged)


def _initialize(cases, forward_tol, adjoint_tol, min_tol, eta, n=None):
    if any(not 0 < t < 1 for t in (forward_tol, adjoint_tol, min_tol)):
        raise ValueError("relative tolerances must be positive and less than one")
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    cases = list(cases)
    if not cases:
        raise ValueError("at least one shot-frequency case is required")
    if n is None:
        n = int(np.prod(cases[0].op.shape))
    states = [_prepare(case, n, forward_tol, adjoint_tol) for case in cases]
    stats = dict(forward_seconds=0.0, adjoint_seconds=0.0, check_seconds=0.0,
                 correction_seconds=0.0, solve_matvecs=0, correction_matvecs=0,
                 check_matvecs=0, iterations=0)
    return states, stats


_MAX_ROUNDS = 64


def _initial_fields(states, method, min_tol, eta, stats):
    rounds = 1
    if method == "sloppiness":
        results = [_sloppy_solve(st, min_tol, eta, stats, _MAX_ROUNDS) for st in states]
        rounds = max(count for count, _ in results)
        if all(done for _, done in results):
            stop = "misfit_stable"
        else:
            stop = "round_limit" if rounds >= _MAX_ROUNDS else "tolerance_floor"
    else:
        for st in states:
            _solve(st, "forward", stats)
            _solve(st, "adjoint", stats)
        stop = "fixed"

    return rounds, stop


def _costs(states, stats, rounds, stop):
    return dict(_work(stats), stop_reason=stop, rounds=rounds,
                forward_tol=max(st.ftol for st in states), adjoint_tol=max(st.atol for st in states),
                forward_tols=[st.ftol for st in states], adjoint_tols=[st.atol for st in states])


def _baseline(cases, method, forward_tol, adjoint_tol, min_tol, eta):
    start = perf_counter()
    if method not in {"fixed", "sloppiness"}:
        raise ValueError("baseline_gradient supports fixed and sloppiness")
    states, stats = _initialize(cases, forward_tol, adjoint_tol, min_tol, eta)
    rounds, stop = _initial_fields(states, method, min_tol, eta, stats)
    gradient_start = perf_counter()
    gradient = np.zeros(states[0].u.size)
    misfits = []
    for st in states:
        gradient += _gradient(st.case, st.u, st.lam)
        misfits.append(_misfit(st.case, st.u))
    result = _costs(states, stats, rounds, stop)
    result.update(method=method, misfit=fsum(misfits),
                  gradient_seconds=perf_counter() - gradient_start, wall_seconds=perf_counter() - start)
    return gradient, result, states


def baseline_gradient(cases, method, forward_tol, adjoint_tol, *, min_tol=1e-10, eta=0.05):
    """Compute an ordinary baseline gradient, with all stabilization solves charged."""
    gradient, result, _ = _baseline(cases, method, forward_tol, adjoint_tol, min_tol, eta)
    return gradient, result


def prepare_candidate(cases, rtol):
    """Solve once to propose an update and preserve its wavefields for certification.

    A pilot belongs to the ordered Case objects and their assembled operators at
    one model checkpoint. A changed model requires newly assembled operators.
    """
    start = perf_counter()
    owners = tuple(cases)
    gradient, result, states = _baseline(owners, "fixed", rtol, rtol, rtol, 0.0)
    for st in states:
        c = st.case
        st.case = Case(c.op, c.source.copy(), c.receivers.copy(), c.data.copy())
    pilot = _Pilot(owners, tuple(states))
    result["wall_seconds"] = perf_counter() - start
    return gradient, result, pilot


def _start_from_pilot(pilot, cases, states, stats):
    if (not isinstance(pilot, _Pilot) or len(cases) != len(pilot.fields)
            or len(cases) != len(pilot.owners)):
        raise ValueError("pilot must match the complete ordered set of cases")
    for original, owner, st, cached in zip(cases, pilot.owners, states, pilot.fields):
        c, old = st.case, cached.case
        if (original is not owner or c.op is not old.op or not np.array_equal(c.source, old.source)
                or not np.array_equal(c.data, old.data) or not np.array_equal(c.receivers, old.receivers)):
            raise ValueError("pilot belongs to different or modified shot-frequency cases")

    for st, cached in zip(states, pilot.fields):
        # Each evaluation owns its warm starts, including when two controller
        # variants are compared with the same candidate direction.
        st.u, st.lam = cached.u.copy(), cached.lam.copy()
        forward_changed = st.ftol < cached.ftol
        tighten_adjoint = st.atol < cached.atol
        st.ftol = min(st.ftol, cached.ftol)
        st.atol = min(st.atol, cached.atol)
        if forward_changed:
            _solve(st, "forward", stats)
        if forward_changed or tighten_adjoint:
            _solve(st, "adjoint", stats)


def evaluate(cases, delta_m, method, forward_tol, adjoint_tol, *, min_tol=1e-10, eta=0.05, pilot=None):
    """Sum physical gradients, report the descent gate for the same real, frozen direction.

    Sloppiness compares consecutive per-case forward misfits. The dual policies
    tighten the dominant error contribution, re-solving every affected system.
    All checks use explicit residuals at the fields that produced the gradient.
    The gate is diagnostic for fixed/sloppiness; a dual update requires certification.
    """
    if method not in {"fixed", "sloppiness", "dual", "dual_corrected"}:
        raise ValueError(f"unknown method {method!r}")
    dm = np.asarray(delta_m).reshape(-1)
    if dm.size == 0 or np.iscomplexobj(dm) or not np.all(np.isfinite(dm)):
        raise ValueError("the frozen squared-slowness direction must be finite and real")
    cases = tuple(cases)
    states, stats = _initialize(cases, forward_tol, adjoint_tol, min_tol, eta, n=dm.size)
    if pilot is None:
        rounds, stop = _initial_fields(states, method, min_tol, eta, stats)
    else:
        if method not in {"dual", "dual_corrected"}:
            raise ValueError("only dual controllers can reuse a pilot")
        _start_from_pilot(pilot, cases, states, stats)
        rounds, stop = 1, None

    while True:
        gradient, result = _summary(states, dm, method == "dual_corrected", stats)
        if method in {"fixed", "sloppiness"}:
            break
        slope, bound = result["derivative"], result["bound"]
        if result["certified"]:
            stop = "certified_descent"
            break
        if not (np.isfinite(slope) and np.isfinite(bound)):
            stop = "nonfinite_check"
            break
        if slope > bound:
            stop = "certified_ascent"
            break
        if rounds >= _MAX_ROUNDS:
            stop = "round_limit"
            break
        can_fwd, can_adj = states[0].ftol > min_tol, states[0].atol > min_tol
        if not (can_fwd or can_adj):
            stop = "tolerance_floor"
            break
        if bound == 0:
            stop = "zero_directional_derivative"
            break
        refine_fwd = can_fwd and (not can_adj or result["forward_bound"] >= result["adjoint_bound"])
        for st in states:
            if refine_fwd:
                st.ftol = max(min_tol, 0.1 * st.ftol)
                _solve(st, "forward", stats)
            else:
                st.atol = max(min_tol, 0.1 * st.atol)
            _solve(st, "adjoint", stats)
        rounds += 1

    result.update(_costs(states, stats, rounds, stop), method=method)
    return gradient, result

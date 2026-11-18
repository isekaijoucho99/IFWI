"""Compare independent IFWI trajectories at a common offline receiver misfit."""

from copy import deepcopy
from math import fsum
from time import perf_counter

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import splu

from . import control
from .model import load_patch, make_model, parameter_gradient, slowness, step, tangent
from .physics import SolveFailure
from .survey import build_cases, geometry, observations


def normalized_direction(model, grad_m, m):
    """Pull a physical gradient back to theta and push its unit descent direction forward."""
    gtheta = parameter_gradient(model, grad_m.reshape(m.shape))
    length = norm(gtheta)
    if length == 0:
        return np.zeros_like(gtheta), np.zeros_like(m)
    p = -gtheta / length
    return p, tangent(model, p)


def advance(model, p, dm, m, step_mps):
    """Apply the theta step whose maximum linearized velocity change is step_mps."""
    max_dv = float(np.max(np.abs(dm) / (2 * m**1.5)))
    if not np.isfinite(max_dv) or max_dv <= 0:
        raise FloatingPointError("direction has no finite, positive maximum velocity change")
    alpha = step_mps / max_dv
    step(model, p, alpha)
    return alpha


def _reference_misfit(m, spacing, survey, args):
    """Forward-only sparse LU, sharing each frequency factor among all its shots."""
    cases = build_cases(m, spacing, args.damping, survey, sponge=args.sponge)
    factors = {}
    losses = []
    for case in cases:
        if case.op not in factors:
            factors[case.op] = splu(case.op.A.tocsc())
        field = factors[case.op].solve(case.source)
        residual = field[case.receivers] - case.data
        losses.append(0.5 * np.vdot(residual, residual).real)
    return fsum(losses)


def _attempt(model, m, spacing, survey, method, args):
    start = perf_counter()
    cases = build_cases(m, spacing, args.damping, survey, sponge=args.sponge)
    pilot_stats = policy = None
    dual = method in {"dual", "dual_corrected"}
    alpha = stop = failure = failed_work = None
    try:
        if dual:
            grad_m, pilot_stats, pilot = control.prepare_candidate(cases, args.candidate_tol)
        else:
            tol = args.fixed_tol if method == "fixed" else args.start_tol
            grad_m, policy = control.baseline_gradient(
                cases, method, tol, tol, min_tol=args.min_tol, eta=args.eta)
        p, dm = normalized_direction(model, grad_m, m)
        if not np.any(dm):
            stop = "zero_candidate_direction"
        elif dual:
            _, policy = control.evaluate(cases, dm.ravel(), method, args.start_tol, args.start_tol,
                                         min_tol=args.min_tol, eta=args.eta, pilot=pilot)
            if not policy["certified"]:
                stop = "uncertified_direction"
    except SolveFailure as exc:
        stop = "solver_failed"
        stage = "candidate" if dual and pilot_stats is None else "policy"
        failed_work = exc.work
        failure = dict(stage=stage, diagnostic=str(exc), work=failed_work)

    if stop is None:
        try:
            alpha = advance(model, p, dm, m, args.step_mps)
        except ValueError as exc:
            # step() validates the complete trial before touching the parameters.
            stop = "invalid_trial_velocity"
            failure = dict(stage="step", diagnostic=str(exc))
        else:
            m = slowness(model)

    seconds = perf_counter() - start
    work = [part for part in (pilot_stats, policy, failed_work) if part is not None]
    row = dict(alpha=alpha, stop_reason=stop, pilot=pilot_stats, policy=policy,
               online_seconds=seconds, online_matvecs=sum(part["matvecs"] for part in work),
               online_iterations=sum(part["iterations"] for part in work))
    if failure is not None:
        row["failure"] = failure
    return m, row


def _trajectory(model, initial_m, spacing, survey, method, args):
    m = initial_m
    totals = dict(online_seconds=0., online_matvecs=0, online_iterations=0)
    trajectory = [dict(update=0, **totals)]
    # Only physical grids are retained for post-hoc scoring. Each slowness()
    # evaluation returns fresh storage and subsequent network steps cannot alter it.
    grids = [m]
    attempts = []
    stop = "budget_exhausted"
    for k in range(1, args.max_updates + 1):
        m, row = _attempt(model, m, spacing, survey, method, args)
        row["from_update"] = k - 1
        attempts.append(row)
        for key in totals:
            totals[key] += row[key]
        if row["stop_reason"]:
            stop = row["stop_reason"]
            break
        grids.append(m)
        trajectory.append(dict(update=k, **totals))
    return dict(method=method, completed_updates=len(grids) - 1, stop_reason=stop,
                trajectory=trajectory, attempts=attempts,
                **{"total_" + k: v for k, v in totals.items()}), grids


def compare_trajectories(dataset, args):
    """Run the four policies to their update budget; score the first crossing afterward.

    Online times cover operator assembly, iterative PDE work, autograd, and model
    updates. Dataset/survey creation, shared framework warm-up, initial model
    replication and offline sparse-LU scoring have separate accounting.
    """
    if not 0 < args.target_ratio <= 1 or args.max_updates < 0 or not 0 < args.fixed_tol < 1:
        raise ValueError("require 0 < target ratio <= 1, nonnegative update budget, and 0 < fixed tol < 1")
    started = perf_counter()
    truth, initial, spacing = load_patch(dataset, args.shape)
    acquisition = geometry(initial.shape, args.shots)
    survey_start = perf_counter()
    survey = observations(truth, spacing, args.frequencies, args.damping, acquisition,
                          sponge=args.sponge)
    observation_seconds = perf_counter() - survey_start
    model = make_model(initial, spacing, args.seed)
    initial_m = slowness(model)
    warmup_start = perf_counter()
    normalized_direction(model, np.ones(initial_m.size), initial_m)
    warmup_seconds = perf_counter() - warmup_start
    methods = ("fixed", "sloppiness", "dual", "dual_corrected")
    replicas = [deepcopy(model) for _ in methods]
    setup_seconds = perf_counter() - started
    paths = [_trajectory(replica, initial_m, spacing, survey, method, args)
             for method, replica in zip(methods, replicas)]

    score_start = perf_counter()
    initial_misfit = _reference_misfit(initial_m, spacing, survey, args)
    initial_scoring_seconds = perf_counter() - score_start
    target = args.target_ratio * initial_misfit
    reports = []
    for row, grids in paths:
        path_score_start = perf_counter()
        for point, grid in zip(row["trajectory"], grids):
            point["reference_misfit"] = (initial_misfit if point["update"] == 0
                                         else _reference_misfit(grid, spacing, survey, args))
        crossing = next((p for p in row["trajectory"] if p["reference_misfit"] <= target), None)
        row.update(first_crossing=crossing, matched_target=crossing is not None,
                   scoring_seconds=perf_counter() - path_score_start)
        reports.append(row)
    return dict(dataset=dataset, shape=list(initial.shape), spacing_m=spacing,
                source_indices=acquisition[0].tolist(), receiver_indices=acquisition[1].tolist(),
                setup_seconds=setup_seconds, observation_seconds=observation_seconds,
                warmup_seconds=warmup_seconds, scoring_seconds=perf_counter() - score_start,
                initial_scoring_seconds=initial_scoring_seconds, initial_misfit=initial_misfit,
                target_misfit=target, target_ratio=args.target_ratio, max_updates=args.max_updates,
                methods=reports, wall_seconds=perf_counter() - started)

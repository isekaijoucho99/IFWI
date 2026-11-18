"""Compare inexact forward/adjoint solves on shared, actual IFWI iterates."""

import argparse
from itertools import product
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from scipy.linalg import norm

from . import control
from .benchmark import advance, compare_trajectories, normalized_direction
from .model import load_patch, make_model, parameter_gradient, slowness
from .physics import SolveFailure
from .survey import build_cases, geometry, observations, reference


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("audit", "benchmark"), default="audit")
    p.add_argument("--datasets", nargs="+", choices=("marmousi", "overthrust"),
                   default=["marmousi", "overthrust"])
    p.add_argument("--shape", type=int, nargs=2, metavar=("NZ", "NX"), default=[16, 24])
    p.add_argument("--shots", type=int, default=2)
    p.add_argument("--frequencies", type=float, nargs="+", default=[5., 8.], metavar="HZ")
    p.add_argument("--damping", type=float, default=.04, help="positive dimensionless imaginary shift")
    p.add_argument("--sponge", type=float, default=.12, help="additional dimensionless absorption at the edges")
    p.add_argument("--checkpoints", nargs="+", type=int, default=[0, 2, 4], metavar="UPDATE")
    p.add_argument("--sweep-tols", nargs="+", type=float, default=[.03, .003, .0003],
                   help="all forward x adjoint combinations for the fixed baseline")
    p.add_argument("--candidate-tol", type=float, default=.01, help="relative tolerance to construct p")
    p.add_argument("--start-tol", type=float, default=.03, help="initial adaptive solver tolerance")
    p.add_argument("--min-tol", type=float, default=1e-10)
    p.add_argument("--eta", type=float, default=.05, help="relative misfit change for sloppiness")
    p.add_argument("--step-mps", type=float, default=25.,
                   help="maximum linearized velocity change per network update (m/s)")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--fixed-tol", type=float, default=.003, help="fixed baseline tolerance in benchmark mode")
    p.add_argument("--max-updates", type=int, default=8, help="update budget per benchmark method")
    p.add_argument("--target-ratio", type=float, default=.8,
                   help="benchmark quality threshold as a fraction of the initial data misfit")
    p.add_argument("--output", type=Path, help="JSON report (default: outputs/direction_b/<mode>.json)")
    return p


def _validate(args):
    geometry(args.shape, args.shots)
    if min(args.checkpoints) < 0:
        raise ValueError("checkpoint indices must be nonnegative")
    if not all(np.isfinite(x) and x > 0 for x in [*args.frequencies, args.damping, args.step_mps]):
        raise ValueError("frequencies, damping and step-mps must be positive and finite")
    if not all(0 < x < 1 for x in [*args.sweep_tols, args.fixed_tol, args.start_tol, args.candidate_tol, args.min_tol]):
        raise ValueError("relative solver tolerances must lie between zero and one")
    if args.min_tol > args.start_tol or not (np.isfinite(args.eta) and args.eta >= 0):
        raise ValueError("min-tol must be <= start-tol and eta finite and nonnegative")
    if not np.isfinite(args.sponge) or args.sponge < 0:
        raise ValueError("sponge must be nonnegative and finite")
    if not 0 < args.target_ratio <= 1 or args.max_updates < 0:
        raise ValueError("benchmark target-ratio must lie in (0, 1] and max-updates must be nonnegative")
    if any(len(set(values)) != len(values) for values in
           (args.frequencies, args.sweep_tols, args.datasets, args.checkpoints)):
        raise ValueError("frequencies, sweep tolerances, datasets and checkpoints must each be distinct")


def _score(row, gtheta, ref_grad, ref_slope):
    n_approx, n_ref = norm(gtheta), norm(ref_grad)
    cosine = float(np.clip((gtheta / n_approx) @ (ref_grad / n_ref), -1., 1.)) if n_approx and n_ref else None
    error = abs(row["derivative"] - ref_slope)
    row.update(reference_derivative=ref_slope, absolute_error=error,
               uncorrected_absolute_error=abs(row["uncorrected_derivative"] - ref_slope),
               sign_error=bool(np.sign(row["derivative"]) != np.sign(ref_slope)),
               gradient_cosine=cosine,
               gradient_angle_deg=None if cosine is None else float(np.degrees(np.arccos(cosine))),
               bound_to_error=None if error == 0 else row["bound"] / error)


def _policies(args, audit_checkpoint):
    if audit_checkpoint:
        for ftol, atol in product(args.sweep_tols, repeat=2):
            yield "fixed", ftol, atol
        yield "sloppiness", args.start_tol, args.start_tol
        yield "dual", args.start_tol, args.start_tol
    yield "dual_corrected", args.start_tol, args.start_tol


def _compare(cases, dm, pilot, work, candidate_seconds, policies, args):
    rows = []
    for method, ftol, atol in policies:
        start = perf_counter()
        try:
            grad, row = control.evaluate(cases, dm.ravel(), method, ftol, atol,
                                         min_tol=args.min_tol, eta=args.eta,
                                         pilot=pilot if method in ("dual", "dual_corrected") else None)
        except SolveFailure as error:
            grad = None
            row = dict(error.work, method=method, stop_reason="solver_failed",
                       certified=False, diagnostic=str(error))
        measured = perf_counter() - start
        row.update(requested_forward_tol=ftol, requested_adjoint_tol=atol,
                   candidate_seconds=candidate_seconds, online_seconds=candidate_seconds + measured,
                   online_matvecs=work["matvecs"] + row["matvecs"])
        rows.append((row, grad))
    return rows


def _checkpoint(model, m, cases, direction, rows, update, work):
    misfit, exact_grad, ref = reference(cases)
    start = perf_counter()
    exact_theta = parameter_gradient(model, exact_grad.reshape(m.shape))
    ref_slope = float(exact_theta @ direction)
    for row, grad in rows:
        if grad is not None:
            _score(row, parameter_gradient(model, grad.reshape(m.shape)), exact_theta, ref_slope)
    return dict(update=update, reference_misfit=misfit, reference_derivative=ref_slope,
                reference_seconds=ref["seconds"], reference_max_relative_residual=ref["max_relative_residual"],
                scoring_seconds=perf_counter() - start, candidate_work=work,
                comparisons=[row for row, _ in rows])


def run_dataset(dataset, args):
    _validate(args)
    started = perf_counter()
    truth, initial, spacing = load_patch(dataset, args.shape)
    model = make_model(initial, spacing, args.seed)
    acquisition = geometry(initial.shape, args.shots)
    survey = observations(truth, spacing, args.frequencies, args.damping, acquisition, sponge=args.sponge)
    # All policies see only the survey and the current parameter state.
    run = dict(dataset=dataset, spacing_m=spacing, shape=list(initial.shape),
               source_indices=acquisition[0].tolist(), receiver_indices=acquisition[1].tolist(),
               setup_seconds=perf_counter() - started, checkpoints=[], trajectory=[])
    requested = set(args.checkpoints)
    last = max(requested)
    run["stop_reason"] = "finished"
    for update in range(last + 1):
        candidate_start = perf_counter()
        m = slowness(model)
        cases = build_cases(m, spacing, args.damping, survey, sponge=args.sponge)
        try:
            pilot_grad, work, pilot = control.prepare_candidate(cases, args.candidate_tol)
        except SolveFailure as error:
            run.update(stop_reason="solver_failed", failure=dict(
                update=update, stage="candidate", diagnostic=str(error), work=error.work,
                online_seconds=perf_counter() - candidate_start))
            break
        direction, dm = normalized_direction(model, pilot_grad, m)
        candidate_seconds = perf_counter() - candidate_start

        audit_checkpoint = update in requested
        rows = _compare(cases, dm, pilot, work, candidate_seconds, _policies(args, audit_checkpoint), args)
        driver = rows[-1][0]
        stop = None
        if driver["stop_reason"] == "solver_failed":
            stop = "solver_failed"
        elif update < last and not driver["certified"]:
            stop = "uncertified_direction"
        if audit_checkpoint or stop:
            run["checkpoints"].append(_checkpoint(model, m, cases, direction, rows, update, work))

        if stop:
            run["stop_reason"] = stop
            break
        if update < last:
            step_start = perf_counter()
            try:
                alpha = advance(model, direction, dm, m, args.step_mps)
            except (ValueError, FloatingPointError) as error:
                run.update(stop_reason="invalid_trial_velocity", failure=dict(
                    update=update, stage="step", diagnostic=str(error), online_matvecs=driver["online_matvecs"],
                    online_seconds=driver["online_seconds"] + (perf_counter() - step_start)))
                break
            run["trajectory"].append(dict(update=update + 1, alpha=alpha,
                                          directional_upper=driver["derivative"] + driver["bound"],
                                          online_matvecs=driver["online_matvecs"],
                                          online_seconds=driver["online_seconds"] + (perf_counter() - step_start)))

    run["completed_updates"] = len(run["trajectory"])
    run["wall_seconds"] = perf_counter() - started
    return run


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        _validate(args)
    except ValueError as error:
        parser().error(str(error))
    runs = []
    for dataset in args.datasets:
        if args.mode == "benchmark":
            run = compare_trajectories(dataset, args)
            print(f"{dataset}: benchmark complete", flush=True)
        else:
            run = run_dataset(dataset, args)
            print(f"{dataset}: {run['completed_updates']} updates, {run['stop_reason']}, "
                  f"{run['wall_seconds']:.2f} s", flush=True)
        runs.append(run)
    report = dict(schema="ifwi_direction_b_v1", settings={k: v for k, v in vars(args).items() if k != "output"},
                  datasets=runs)
    out = args.output or Path("outputs/direction_b") / f"{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile("w", dir=out.parent, encoding="utf-8", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temp.replace(out)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    print(out)

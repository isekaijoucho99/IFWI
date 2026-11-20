"""Run matched-velocity, finite-step Adam diagnostics on the time-domain IFWI objective."""

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from .control import choose_step, rms
from .diagnostics import measure, predict_fields
from .model import apply_delta, jvp, make_model, matched_copy, pullback, trial_velocity, velocity
from .optimizer import UpdateRule
from .physics import Survey, load_patch
from .report import write_summary


POLICIES = {"sgd": "plain", "adam_no_momentum": "plain", "adam": "plain", "adam_clipped": "plain",
            "adam_armijo": "armijo", "adam_guarded_no_remainder": "guarded_no_remainder",
            "adam_guarded": "guarded"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=("marmousi", "overthrust"), default=["marmousi", "overthrust"])
    grid = p.add_mutually_exclusive_group()
    grid.add_argument("--shape", nargs=2, type=int, default=[16, 24], metavar=("NZ", "NX"))
    grid.add_argument("--full-grid", action="store_true", help="use the complete project inversion grid")
    p.add_argument("--shots", type=int, default=2)
    p.add_argument("--shot-batch", type=int, help="shots per forward call; all shots contribute to every update")
    p.add_argument("--frequency", type=float, default=18., help="Ricker peak frequency in Hz")
    p.add_argument("--nt", type=int, default=110)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--representations", nargs="+", choices=("siren", "fr"), default=["siren", "fr"])
    p.add_argument("--scales", type=float, nargs="+", default=[1., 4.], help="matched-function output gains")
    p.add_argument("--methods", nargs="+", choices=tuple(POLICIES), default=list(POLICIES))
    p.add_argument("--warmup-steps", type=int, default=1, help="shared FR Adam updates before the fork")
    p.add_argument("--warmup-lr", type=float, default=.0003)
    p.add_argument("--steps", type=int, default=5, help="attempt budget per independent trajectory")
    p.add_argument("--lr", type=float, default=.001, help="native Adam family learning rate")
    p.add_argument("--sgd-lr", type=float, default=.003)
    p.add_argument("--lr-modes", nargs="+", choices=("native", "matched"), default=["native", "matched"])
    p.add_argument("--first-rms", type=float, default=5., help="matched-mode initial linear step RMS in m/s")
    p.add_argument("--clip-norm", type=float, default=.1, help="parameter-gradient norm cap for adam_clipped")
    p.add_argument("--beta1", type=float, default=.9)
    p.add_argument("--beta2", type=float, default=.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--max-rms", type=float, default=15., help="guarded actual velocity RMS cap in m/s")
    p.add_argument("--linear-tol", type=float, default=.25, help="guarded remainder RMS / actual RMS limit")
    p.add_argument("--max-halvings", type=int, default=8)
    p.add_argument("--misfit-ratio", type=float, default=.8, help="target fraction of checkpoint data MSE")
    p.add_argument("--rmse-ratio", type=float, default=.99, help="target fraction of checkpoint velocity RMSE")
    p.add_argument("--save-steps", nargs="*", type=int, help="NPZ maps at these one-based steps (default: first and last)")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/direction_a"))
    p.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    p.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    return p


def _validate(args):
    for items in (args.methods, args.datasets, args.scales, args.representations, args.lr_modes):
        if len(set(items)) != len(items):
            raise ValueError("each sweep axis must contain distinct values")
    if not all(math.isfinite(x) and x > 0 for x in (
            *args.scales, args.lr, args.sgd_lr, args.warmup_lr, args.first_rms, args.max_rms,
            args.linear_tol, args.clip_norm, args.eps, args.frequency)):
        raise ValueError("scales, learning rates and control thresholds must be finite and positive")
    if min(args.warmup_steps, args.max_halvings) < 0 or min(args.steps, args.threads, args.width, args.shots, args.nt) < 1:
        raise ValueError("invalid step, thread, width or acquisition count")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        raise ValueError("Adam betas must be in [0, 1)")
    if not all(0 < x < 1 for x in (args.misfit_ratio, args.rmse_ratio)):
        raise ValueError("quality ratios must lie between 0 and 1")
    if args.save_steps is not None and any(s < 1 or s > args.steps for s in args.save_steps):
        raise ValueError("save-steps must be between 1 and steps")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("use cpu or an available CUDA device")


def _clock(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def _settings(args, method):
    base = method if method in ("sgd", "adam_no_momentum") else "adam"
    return UpdateRule(base, args.sgd_lr if method == "sgd" else args.lr,
                      beta1=args.beta1, beta2=args.beta2, eps=args.eps,
                      clip_norm=args.clip_norm if method == "adam_clipped" else None)


def _warm_checkpoint(initial, spacing, survey, args):
    model = make_model(initial, spacing, args.width, args.seed, "fr", args.device)
    rule = UpdateRule("adam", args.warmup_lr)
    for _ in range(args.warmup_steps):
        _, grad = survey.evaluate(velocity(model), gradient=True)
        proposal = rule.propose(pullback(model, grad))
        if not survey.is_valid(trial_velocity(model, proposal.delta)):
            raise ValueError("invalid common checkpoint velocity; reduce --warmup-lr")
        apply_delta(model, proposal.delta)
        rule.accept(proposal)
    return model


def _save_fields(path, actual, before, after, grad, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    maps = dict(before=before, after=after, actual=actual, gradient=grad, **fields)
    np.savez_compressed(path, **{k: value.detach().cpu().numpy() for k, value in maps.items()})


def _run_arm(reference, truth, survey, spacing, setup, out, arm, args):
    rep, scale, method, mode = arm
    started = _clock(args.device)
    model = matched_copy(reference, scale, representation=rep)
    before = velocity(model)
    reference_v = velocity(reference)
    mismatch = float((before - reference_v).abs().max())
    if mismatch > 1e-8 or not torch.equal(before.float(), reference_v.float()):
        raise RuntimeError(f"matched checkpoint differs by {mismatch:g} m/s")
    rule = _settings(args, method)
    run = dict(representation=rep, scale=scale, method=method, lr_mode=mode,
               start_max_diff_mps=mismatch, parameters=sum(p.numel() for p in model.parameters()),
               initial_misfit=setup["initial_misfit"], initial_rmse_mps=setup["initial_rmse_mps"],
               steps=[], first_misfit_target=None, first_rmse_target=None)
    saved = set(args.save_steps if args.save_steps is not None else [1, args.steps])
    forwards = adjoints = report_forwards = 0
    online = _clock(args.device) - started
    for metric, initial in (("misfit", run["initial_misfit"]), ("rmse", run["initial_rmse_mps"])):
        if initial == 0:
            run[f"first_{metric}_target"] = dict(step=0, value=0., target=0., forward_shots=0,
                adjoint_shots=0, report_forward_shots=0, online_seconds=online,
                total_seconds=online + setup["seconds"], total_forward_shots=setup["forward_shots"],
                total_adjoint_shots=setup["adjoint_shots"])
    run["stop_reason"] = "budget_exhausted"
    # The gradient and no-grad paths may select different float32 FD kernels.
    # Compare accepted losses in exactly the same forward-only evaluation mode.
    loss = setup["initial_misfit"]
    for index in range(1, args.steps + 1):
        tic = _clock(args.device)
        gradient_loss, grad = survey.evaluate(before, gradient=True)
        forwards += survey.shots
        adjoints += survey.shots
        gtheta = pullback(model, grad)
        if index == 1 and mode == "matched":
            first = rms(jvp(model, rule.propose(gtheta).delta))
            if first > 0:
                rule.lr *= args.first_rms / first
        choice = choose_step(model, before, grad, gtheta, rule, loss, survey,
                             policy=POLICIES[method], max_rms=args.max_rms,
                             linear_tol=args.linear_tol, max_halvings=args.max_halvings)
        forwards += choice.forward_shots
        online += _clock(args.device) - tic
        row = dict(step=index, status=choice.reason, misfit_before=loss, gradient_evaluation_misfit=gradient_loss,
                   factor=choice.factor,
                   restarted=choice.restarted, trials=choice.trials, extra_forward_shots=choice.forward_shots,
                   gradient_norm=choice.proposal.raw_norm, clip_factor=choice.proposal.clip_factor)
        run["steps"].append(row)
        if choice.trial is not None:
            # These Jacobian products are diagnostic. They use the pre-update network.
            fields = predict_fields(model, choice.proposal, choice.factor, gtheta, rule.lr)
            tic = _clock(args.device)
            apply_delta(model, choice.factor * choice.proposal.delta)
            rule.accept(choice.proposal)
            after = velocity(model)
            online += _clock(args.device) - tic
            actual = after - before
            row["diagnostics"] = measure(actual, grad, fields, spacing)
            if choice.verified_loss is None:
                post_loss, _ = survey.evaluate(after, gradient=False)
                report_forwards += survey.shots
            else:
                post_loss = choice.verified_loss
            row.update(misfit_after=post_loss, rmse_mps=rms(after - truth),
                       misfit_increased=bool(post_loss > loss), misfit_change=post_loss - loss,
                       misfit_taylor_remainder=post_loss - loss - row["diagnostics"]["physical_gradient_dot_actual"])
            loss = post_loss
            if index in saved:
                path = Path(setup["dataset"]) / f"{rep}_scale{scale!r}_{method}_{mode}" / f"step_{index:04d}.npz"
                _save_fields(out / path, actual, before, after, grad, fields)
                row["fields_path"] = path.as_posix()
            before = after
        costs = dict(forward_shots=forwards, adjoint_shots=adjoints, report_forward_shots=report_forwards,
                     online_seconds=online, total_seconds=setup["seconds"] + (_clock(args.device) - started),
                     total_forward_shots=setup["forward_shots"] + forwards + report_forwards,
                     total_adjoint_shots=setup["adjoint_shots"] + adjoints)
        row.update(costs)
        if choice.trial is None:
            run["stop_reason"] = choice.reason
            break
        for metric, ratio, value in (("misfit", args.misfit_ratio, post_loss),
                                     ("rmse", args.rmse_ratio, row["rmse_mps"])):
            key = f"first_{metric}_target"
            baseline = run["initial_misfit" if metric == "misfit" else "initial_rmse_mps"]
            if run[key] is None and value <= ratio * baseline:
                run[key] = dict(step=index, value=value, target=ratio * baseline, **costs)
    run.update(effective_lr=rule.lr, completed_steps=sum(s["status"] == "accepted" for s in run["steps"]),
               costs=costs)
    return run


def run_dataset(dataset, args, out):
    tic = _clock(args.device)
    truth, initial, spacing = load_patch(dataset, None if args.full_grid else args.shape)
    survey = Survey(truth, spacing, shots=args.shots, nt=args.nt, frequency=args.frequency,
                    device=args.device, shot_batch=args.shot_batch)
    model = _warm_checkpoint(initial, spacing, survey, args)
    ref_velocity = velocity(model)
    truth = torch.as_tensor(truth, device=args.device, dtype=ref_velocity.dtype)
    initial_loss, _ = survey.evaluate(ref_velocity, gradient=False)
    setup = dict(dataset=dataset, initial_misfit=initial_loss, initial_rmse_mps=rms(ref_velocity - truth),
                 forward_shots=survey.setup_forwards + (args.warmup_steps + 1) * survey.shots,
                 adjoint_shots=args.warmup_steps * survey.shots, seconds=_clock(args.device) - tic)
    runs = [_run_arm(model, truth, survey, spacing, setup, out, arm, args)
            for arm in product(args.representations, args.scales, args.methods, args.lr_modes)]
    return dict(dataset=dataset, shape=list(truth.shape), spacing_m=spacing, dt_s=survey.dt,
                source_x=list(map(int, survey.source_x_indices)), setup=setup, runs=runs)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        _validate(args)
    except ValueError as error:
        cli.error(str(error))
    torch.set_num_threads(args.threads)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_dir / f"{stamp}_{uuid4().hex[:8]}"
    out.mkdir(parents=True)
    report = dict(schema="ifwi_direction_a_v1",
                  settings={k: v for k, v in vars(args).items() if k != "output_dir"}, datasets=[])
    for dataset in args.datasets:
        result = run_dataset(dataset, args, out)
        report["datasets"].append(result)
        tmp = out / "report.json.tmp"
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(out / "report.json")
        write_summary(report, out / "summary.csv")
        print(f"{dataset}: {len(result['runs'])} trajectories -> {out / 'report.json'}", flush=True)
    return out

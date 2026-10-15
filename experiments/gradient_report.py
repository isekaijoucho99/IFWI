"""Physical velocity gradient at saved velocity states, per misfit band (no training).

For every velocity state and objective, one all-shot FD forward and backward gives
G = dL/dv of the training objective: the full-band waveform MSE, or a frequency-continuation
low-pass stage with its energy normalization (experiments/frequency_continuation.py).
Reported for each state and objective:

* cos(-G, v_true - v) on the whole grid, in radial wavenumber bands (orthonormal DCT) and in
  depth regions [0, 30 %), [30 %, 65 %), [65 %, 100 %] of the deepest node: does steepest
  descent from this state point toward the truth, and where? Truth is used only here;
* the share of |G|^2 per depth region (where the objective is sensitive);
* once per state, the full-band data residual |r|^2/|d|^2 in record quarters and in bands of
  the source frequency f0 (0-f0/2, f0/2-f0, f0-1.5f0, 1.5f0-2.5f0, above): which part of
  the records the state still does not explain.

A cosine near zero with a tiny residual means the remaining model error is mostly invisible
to these data (no optimizer recovers it); a negative cosine means descent moves away from the
truth (as the full-band gradient does at the sigma 15 initial model of the 13-shot protocol).
Networks see G only through their Jacobian and optimizer.

States: ``initial`` (the configured Gaussian background), a velocity .npy file, or a run
directory (run_start_velocity.npy and checkpoints/velocity_XXXX.npy; final_velocity.npy when
there are no checkpoints). Truth, acquisition and forward settings come from --config, or
from each run directory's config.json when --config is omitted.

    python -m experiments.gradient_report --states initial outputs/paper13_fr_ifwi/RUN \\
        --config experiments/configs/paper13_fr_ifwi.yaml --max-hz 3 5 full \\
        --device cuda:0 --output-dir outputs/gradient_report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.frequency_continuation import energy_scale
from experiments.initial_gradient_report import band_pass, cosine
from experiments.residual_ifwi_data import load_velocity_data
from experiments.residual_ifwi_experiment import (ROOT, _simulate, build_forward_problem, require_finite,
    training_objective, validate_config, waveform_mse)
from experiments.spectral_metrics import band_label

BANDS_CPKM = ((0., 1.), (1., 3.), (3., 8.), (8., np.inf))
DEPTH_REGIONS = (("shallow", 0., .3), ("middle", .3, .65), ("deep", .65, 1.))


def state_list(spec):
    """[(label, completed updates or None, .npy path or None for the configured background)]."""
    if spec == "initial":
        return [("initial", 0, None)]
    path = Path(spec)
    if path.is_file():
        return [(path.stem, None, path)]
    if not path.is_dir():
        raise FileNotFoundError(f"No velocity state at {spec}")
    states = [("run_start", 0, path / "run_start_velocity.npy")] if (path / "run_start_velocity.npy").is_file() else []
    checkpoints = sorted((path / "checkpoints").glob("velocity_*.npy"))
    states += [(f.stem, int(f.stem.split("_")[1]), f) for f in checkpoints]
    if not checkpoints and (path / "final_velocity.npy").is_file():
        states.append(("final", None, path / "final_velocity.npy"))
    if not states:
        raise ValueError(f"No saved velocities in {path}")
    return states


def depth_masks(shape, spacing_m):
    depth = np.arange(shape[0]) * spacing_m
    fraction = depth / depth.max()
    masks = {}
    for name, low, high in DEPTH_REGIONS:
        rows = (fraction >= low) & ((fraction < high) if high < 1 else (fraction <= high))
        masks[name] = np.broadcast_to(rows[:, None], shape)
    return masks


def objective_gradient(velocity_mps, problem, max_hz=None):
    """(G as float64 [nz,nx], objective, full-band data MSE, residual records on CPU) for all shots."""
    velocity = torch.as_tensor(velocity_mps, dtype=torch.float32,
                               device=problem["observed"].device)[None].requires_grad_(True)
    total, objective_value, data_mse, residuals = problem["observed"].shape[1], 0., 0., []
    for item in problem["propagators"]:
        predicted = _simulate(item["solver"], velocity, problem["wavelet"])
        observations = problem["observed"][:, item["first"]:item["last"]]
        objective = training_objective(predicted, observations, total, max_hz, problem)
        require_finite(objective, "objective")
        objective.backward()
        objective_value += float(objective.detach())
        with torch.no_grad():
            data_mse += float(waveform_mse(predicted, observations, total))
            residuals.append((predicted - observations).cpu())
        del objective, predicted
    require_finite(velocity.grad, "velocity gradient")
    return velocity.grad[0].double().cpu().numpy(), objective_value, data_mse, torch.cat(residuals, 1)


def alignment(gradient, truth, velocity, spacing_m):
    descent = -np.asarray(gradient, np.float64)
    target = np.asarray(truth, np.float64) - np.asarray(velocity, np.float64)
    energy = np.square(descent)
    row = {"cos_all": cosine(descent, target)}
    for low, high in BANDS_CPKM:
        row[f"cos_{band_label(low, high)}"] = cosine(band_pass(descent, spacing_m, low, high),
                                                     band_pass(target, spacing_m, low, high))
    for name, mask in depth_masks(descent.shape, spacing_m).items():
        row[f"cos_{name}"] = cosine(descent[mask], target[mask])
        row[f"grad_energy_{name}"] = float(energy[mask].sum() / energy.sum()) if energy.sum() > 0 else None
    return row


@torch.no_grad()
def residual_breakdown(residual, observed, dt_s, source_hz):
    """Relative full-band residual energy |r|^2/|d|^2 per record quarter and frequency band."""
    residual, observed = residual.double(), observed.double().cpu()
    nt, row = residual.shape[2], {}
    for quarter in range(4):
        first, last = quarter * nt // 4, (quarter + 1) * nt // 4
        reference = float(observed[:, :, first:last].square().sum())
        row[f"rel_t{first * dt_s:.2f}-{last * dt_s:.2f}s"] = (
            float(residual[:, :, first:last].square().sum()) / reference if reference > 0 else None)
    frequencies = torch.fft.rfftfreq(nt, dt_s, dtype=torch.float64)
    weights = torch.full_like(frequencies, 2.)
    weights[0] = 1.
    if nt % 2 == 0:
        weights[-1] = 1.
    power_r = torch.fft.rfft(residual, dim=2).abs().square() * weights[None, None, :, None]
    power_d = torch.fft.rfft(observed, dim=2).abs().square() * weights[None, None, :, None]
    edges = (0., .5 * source_hz, source_hz, 1.5 * source_hz, 2.5 * source_hz, np.inf)
    for low, high in zip(edges[:-1], edges[1:]):
        keep = (frequencies >= low) & (frequencies < high)
        reference = float(power_d[:, :, keep].sum())
        label = f"{low:g}-{high:g}Hz" if np.isfinite(high) else f">{low:g}Hz"
        row[f"rel_{label}"] = float(power_r[:, :, keep].sum()) / reference if reference > 0 else None
    row["rel_total"] = float(residual.square().sum() / observed.square().sum())
    return row


def report(config, states, cutoffs, device="cpu"):
    """Rows (state x objective) and per-state residual breakdowns for one configuration."""
    validate_config(config)
    truth, background, _ = load_velocity_data(config, ROOT)
    problem = build_forward_problem(truth, config, device)
    dt_s, spacing = config["forward"]["dt_s"], config["data"]["grid_spacing_m"]
    problem["frequency_scales"] = {float(c): energy_scale(problem["observed"], float(c), dt_s)
                                   for c in cutoffs if c is not None}
    rows, residuals = [], []
    for label, updates, path in states:
        velocity = background if path is None else np.load(path, allow_pickle=False)
        if velocity.shape != truth.shape:
            raise ValueError(f"{label}: velocity shape {velocity.shape} differs from the configured grid {truth.shape}")
        for index, cutoff in enumerate(cutoffs):
            gradient, objective, data_mse, residual = objective_gradient(velocity, problem, cutoff)
            rows.append({"state": label, "updates": updates, "objective": "full" if cutoff is None else f"<={cutoff:g}Hz",
                         "objective_value": objective, "data_mse": data_mse,
                         "rmse_mps": float(np.sqrt(np.mean((np.asarray(velocity, np.float64) - truth) ** 2))),
                         **alignment(gradient, truth, velocity, spacing)})
            if index == 0:
                residuals.append({"state": label, "updates": updates,
                                  **residual_breakdown(residual, problem["observed"], dt_s, config["forward"]["frequency_hz"])})
    return rows, residuals


def load_config(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--states", nargs="+", required=True, help="initial, a velocity .npy, or a run directory")
    parser.add_argument("--config", type=Path, help="YAML/JSON; default: each run directory's config.json")
    parser.add_argument("--max-hz", nargs="+", default=["full"], help="low-pass cutoffs in Hz and/or 'full'")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cutoffs = [None if c == "full" else float(c) for c in args.max_hz]
    torch.set_num_threads(args.threads)
    groups = {}
    for spec in args.states:
        if args.config is not None:
            config = load_config(args.config)
        elif Path(spec).is_dir():
            config = load_config(Path(spec) / "config.json")
        else:
            parser.error(f"--config is required for state {spec}")
        groups.setdefault(json.dumps(config, sort_keys=True), (config, []))[1].extend(state_list(spec))
    rows, residuals = [], []
    for config, states in groups.values():
        found = report(config, states, cutoffs, args.device)
        rows += found[0]
        residuals += found[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "gradient_report.csv", rows)
    write_csv(args.output_dir / "residual_report.csv", residuals)
    (args.output_dir / "gradient_report.json").write_text(json.dumps(
        {"states": args.states, "objectives": args.max_hz, "rows": rows, "residuals": residuals,
         "note": "truth is used for evaluation only; objectives equal the training objectives"}, indent=2), encoding="utf-8")
    keys = [k for k in rows[0] if k.startswith("cos_")]
    print(f"{'state':>16} {'objective':>9} {'RMSE':>7} " + " ".join(f"{k[4:]:>11}" for k in keys))
    for row in rows:
        print(f"{row['state']:>16} {row['objective']:>9} {row['rmse_mps']:7.1f} "
              + " ".join(f"{row[k]:+11.3f}" if row[k] is not None else f"{'-':>11}" for k in keys))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

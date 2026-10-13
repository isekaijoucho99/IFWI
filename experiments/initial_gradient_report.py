"""Offline check of the first FWI gradient at candidate initial models (no training).

For each Gaussian sigma, builds the configured observations, takes one all-shot
velocity gradient G = dL/dv at the smoothed initial model (one FD forward and
backward, no network) and reports how well the steepest-descent direction -G
points toward the missing model update v_true - v_init:

* cosine(-G, v_true - v_init) on the whole grid, below ``--shallow-m`` and in
  radial wavenumber bands (orthonormal DCT, as in spectral_metrics);
* the share of gradient energy above ``--shallow-m`` and 500 m.

A negative cosine means the first descent step moves the model away from the
truth. It is a first-step diagnostic only: networks see G through their own
Jacobian and Adam, and later steps may recover. Truth is used for this diagnostic
only and never by training. Choose initial models by a rule that does not use
these cosines (e.g. matching a paper's initial error), and report it if you do.

    python -m experiments.initial_gradient_report --config experiments/configs/paper13_ifwi.yaml \\
        --sigmas 10 15 22 25 --device cuda:0 --output-dir outputs/initial_gradient
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn
import torch

from experiments.low_rank_calibration import velocity_gradient
from experiments.residual_ifwi_data import load_velocity_data
from experiments.residual_ifwi_experiment import ROOT, build_forward_problem, validate_config
from experiments.spectral_metrics import band_label, radial_wavenumber_cpkm

DEFAULT_BANDS_CPKM = ((0., 1.), (1., 8.), (8., np.inf))


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    norm = np.sqrt((a @ a) * (b @ b))
    return float(np.clip(a @ b / norm, -1., 1.)) if norm > 0 else None


def band_pass(field, spacing_m, low, high):
    coefficients = dctn(np.asarray(field, np.float64), norm="ortho")
    k = radial_wavenumber_cpkm(coefficients.shape, spacing_m)
    coefficients[(k < low) | (k >= high)] = 0
    return idctn(coefficients, norm="ortho")


def alignment(gradient, truth, initial, spacing_m, shallow_m=150., bands=DEFAULT_BANDS_CPKM):
    """Cosines of -G with v_true - v_init and gradient energy shares, all on [nz, nx] arrays."""
    descent = -np.asarray(gradient, np.float64)
    update = np.asarray(truth, np.float64) - np.asarray(initial, np.float64)
    depth = np.broadcast_to((np.arange(descent.shape[0]) * spacing_m)[:, None], descent.shape)
    energy = np.square(descent)
    deep = depth >= shallow_m
    return {"cos_all": cosine(descent, update),
            f"cos_z_ge_{shallow_m:g}m": cosine(descent[deep], update[deep]),
            **{f"cos_{band_label(lo, hi)}": cosine(band_pass(descent, spacing_m, lo, hi), band_pass(update, spacing_m, lo, hi))
               for lo, hi in bands},
            f"grad_energy_share_z_lt_{shallow_m:g}m": float(energy[~deep].sum() / energy.sum()),
            "grad_energy_share_z_lt_500m": float(energy[depth < 500].sum() / energy.sum())}


def report(config, sigmas, device="cpu", shallow_m=150.):
    rows = []
    for sigma in sigmas:
        cfg = copy.deepcopy(config)
        cfg["initialization"]["sigma"] = float(sigma)
        validate_config(cfg)
        truth, initial, _ = load_velocity_data(cfg, ROOT)
        problem = build_forward_problem(truth, cfg, device)
        gradient, loss = velocity_gradient(initial, problem)
        spacing = cfg["data"]["grid_spacing_m"]
        rows.append({"sigma": float(sigma),
                     "initial_rmse_mps": float(np.sqrt(np.mean((initial.astype(np.float64) - truth) ** 2))),
                     "initial_data_mse": loss,
                     **alignment(gradient[0].cpu().numpy(), truth, initial, spacing, shallow_m)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sigmas", nargs="+", type=float, required=True, help="Gaussian sigma in grid cells")
    parser.add_argument("--shallow-m", type=float, default=150.)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import yaml
    with args.config.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    torch.set_num_threads(args.threads)
    rows = report(config, args.sigmas, args.device, args.shallow_m)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "initial_gradient.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "initial_gradient.json").write_text(json.dumps(
        {"config": str(args.config), "experiment_name": config["experiment_name"], "rows": rows,
         "note": "first-step diagnostic; truth used for evaluation only"}, indent=2), encoding="utf-8")
    keys = [k for k in rows[0] if k.startswith("cos_")]
    print(f"{'sigma':>6} {'init RMSE':>10} " + " ".join(f"{k[4:]:>14}" for k in keys))
    for row in rows:
        print(f"{row['sigma']:6g} {row['initial_rmse_mps']:10.1f} "
              + " ".join(f"{row[k]:+14.3f}" if row[k] is not None else f"{'-':>14}" for k in keys))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

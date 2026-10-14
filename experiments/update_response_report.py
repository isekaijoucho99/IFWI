"""Offline check of how a model turns one FWI gradient into velocity updates (no FD per step).

Computes one all-shot velocity gradient G at the first config's initial model, then
for each config builds the model exactly as the runner does (pretraining included),
applies Adam with the config's learning rates for ``--steps`` updates driven by that
fixed G, and reports after each listed step:

* rms_dm_mps: RMS velocity change from the run start (m/s);
* shares of the change's energy in vertical stripes (kx > 8 and kz < 2 c/km),
  horizontal stripes (kz > 8 and kx < 2 c/km) and above 8 c/km in either direction;
* cos(dm, -G);
* the mean stable rank of the cumulative effective hidden-weight change (FR: lamb @ bases).

The optimizer is the config's ``training.optimizer`` (Adam if unset); ``--muon-lrs`` repeats
every Muon config once per listed Muon learning rate, e.g. to bracket the lr whose velocity
change matches Adam's before a GPU sweep.

Step 1 is exact; later steps hold G fixed, so they show the parameterization's response,
not the true trajectory. Use it to compare step growth and stripe content between models
before a GPU run, e.g. a low-rank model against the 2-D SIREN it is meant to replace.
All configs must share data, initialization sigma, acquisition and forward settings.

    python -m experiments.update_response_report --configs experiments/configs/lrpaper13_ifwi.yaml \\
        experiments/configs/lrpaper13_lr_ifwi.yaml --steps 1 5 20 --output-dir outputs/update_response
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn
import torch

from experiments.initial_gradient_report import cosine
from experiments.low_rank_calibration import velocity_gradient
from experiments.residual_ifwi_data import load_velocity_data
from experiments.residual_ifwi_experiment import (ROOT, _seed, build_forward_problem, model_from_config,
    pretrain_to_initial, validate_config)
from experiments.muon_optimizer import build_optimizer, effective_hidden_weights, optimizer_name, stable_rank
from experiments.spectral_metrics import radial_wavenumber_cpkm

SHARED = ("data", "acquisition", "forward")


def stripe_shares(field, spacing_m, threshold=8., across=2.):
    power = dctn(np.asarray(field, np.float64), norm="ortho") ** 2
    nz, nx = power.shape
    kz = np.broadcast_to((np.arange(nz) / (2 * nz * spacing_m / 1000))[:, None], power.shape)
    kx = np.broadcast_to((np.arange(nx) / (2 * nx * spacing_m / 1000))[None, :], power.shape)
    total = power.sum()
    if total == 0:
        return {"vertical_stripe_share": 0., "horizontal_stripe_share": 0., "high_k_share": 0.}
    return {"vertical_stripe_share": float(power[(kx > threshold) & (kz < across)].sum() / total),
            "horizontal_stripe_share": float(power[(kz > threshold) & (kx < across)].sum() / total),
            "high_k_share": float(power[(kx > threshold) | (kz > threshold)].sum() / total)}


def response(config, background, gradient, steps, emit=lambda _: None):
    """Velocity changes of the runner-built (and pretrained) model under a fixed gradient."""
    _seed(config["seed"])
    model = model_from_config(background, config)
    pretrain = config["initialization"].get("pretrain")
    trainable_omega = config["model"].get("trainable_omega")
    if pretrain is not None:
        pretrain_to_initial(model, background, pretrain, trainable_omega,
                            config["model"]["mean_kmps"], config["model"]["std_kmps"], emit)
    optimizer = build_optimizer(model, config)
    hidden_start = effective_hidden_weights(model)
    with torch.no_grad():
        start = model()[0].detach().double().numpy().copy()
    rows, descent = [], -gradient[0].double().numpy()
    for step in range(1, max(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        model().backward(gradient)
        optimizer.step()
        if step in steps:
            with torch.no_grad():
                change = model()[0].detach().double().numpy() - start
            ranks = [stable_rank(a - b) for a, b in zip(effective_hidden_weights(model), hidden_start)]
            ranks = [r for r in ranks if r is not None]
            rows.append({"step": step, "rms_dm_mps": float(np.sqrt(np.mean(change ** 2))),
                         **stripe_shares(change, config["data"]["grid_spacing_m"]),
                         "cos_dm_descent": cosine(change, descent),
                         "hidden_change_srank_mean": float(np.mean(ranks)) if ranks else None})
    return rows


def report(configs, steps, device="cpu"):
    first = configs[0]
    for cfg in configs:
        validate_config(cfg)
        if any(cfg[s] != first[s] for s in SHARED) or cfg["initialization"]["sigma"] != first["initialization"]["sigma"]:
            raise ValueError(f"{cfg['experiment_name']}: data, sigma, acquisition and forward must match {first['experiment_name']}")
    truth, background, _ = load_velocity_data(first, ROOT)
    _seed(first["seed"])
    gradient, _ = velocity_gradient(background, build_forward_problem(truth, first, device))
    gradient = gradient.cpu()
    return [{"experiment_name": cfg["experiment_name"], **row}
            for cfg in configs for row in response(copy.deepcopy(cfg), background, gradient, steps)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--sigma", type=float, help="Override initialization.sigma for every config")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--muon-lrs", nargs="+", type=float, help="Run each Muon config once per Muon learning rate")
    parser.add_argument("--device", default="cpu", help="Device for the one FD gradient; model steps run on CPU")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import yaml
    configs = []
    for path in args.configs:
        with path.open(encoding="utf-8-sig") as stream:
            cfg = yaml.safe_load(stream)
        if args.sigma is not None:
            cfg["initialization"]["sigma"] = args.sigma
        if args.muon_lrs and optimizer_name(cfg) == "muon":
            for lr in args.muon_lrs:
                variant = copy.deepcopy(cfg)
                variant["training"]["optimizer"]["learning_rate"] = lr
                variant["experiment_name"] = f"{cfg['experiment_name']}_muonlr{lr:g}".replace(".", "p").replace("-", "m")
                configs.append(variant)
        else:
            configs.append(cfg)
    torch.set_num_threads(args.threads)
    rows = report(configs, sorted(set(args.steps)), args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "update_response.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "update_response.json").write_text(json.dumps(
        {"configs": [str(p) for p in args.configs], "rows": rows,
         "note": "step 1 exact; later steps hold the initial gradient fixed"}, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['experiment_name']:28s} step {row['step']:3d}: |dm| {row['rms_dm_mps']:8.2f} m/s, "
              f"vertical {row['vertical_stripe_share']:.3f}, horizontal {row['horizontal_stripe_share']:.3f}, "
              f"k>8 {row['high_k_share']:.3f}, cos(dm,-G) {row['cos_dm_descent']:+.3f}, "
              f"hidden srank {row['hidden_change_srank_mean'] if row['hidden_change_srank_mean'] is None else round(row['hidden_change_srank_mean'], 1)}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

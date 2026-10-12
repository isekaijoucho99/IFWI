"""Offline per-layer weight displacement between saved residual IFWI checkpoints.

Tests whether FR layers take larger steps than plain layers at the same Adam lr.
For every pair of consecutive checkpoints/update_XXXX.pt it rebuilds the model
from the saved config (FR bases are deterministic), merges W = Lambda @ B, and
reports ||A_t2 - A_t1||_F per layer, where A = omega * W is the pre-activation
weight (omega = omega_0, or the trainable value; the linear output layer has no
omega). This is net displacement over the window, not the summed path length,
so oscillation lowers it; both runs are measured the same way. Runs no FD and no
training. A resumed run is one series: join its directories with commas.

    python -m experiments.residual_step_report RUN_BASELINE,RUN_BASELINE_RESUMED RUN_FR128 --labels baseline fr128 --output-dir outputs/step_report

The first run is the reference; ratios compare each run's displacement with the
reference over identical windows. Only checkpoints written by this repository's
runner are supported (other FR implementations use different config keys).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics

import numpy as np
import torch

from experiments.residual_band_report import split_dirs


def checkpoint_paths(run_dirs):
    paths = {}
    for run_dir in run_dirs:
        for path in sorted((Path(run_dir) / "checkpoints").glob("update_*.pt")):
            paths.setdefault(int(re.fullmatch(r"update_(\d+)\.pt", path.name).group(1)), path)
    return dict(sorted(paths.items()))


def build_model(config, state):
    from experiments.residual_ifwi_experiment import VelocityParameterization, validate_config
    validate_config(config)
    model = config["model"]
    net = VelocityParameterization(state["fixed_init_mps"][0].numpy(), config["data"]["grid_spacing_m"],
        model["neurons"], model["omega_0"], model["mean_kmps"], model["std_kmps"], config["parameterization"],
        fourier=model.get("fourier"), trainable_omega=model.get("trainable_omega") is not None)
    net.load_state_dict(state, strict=True)
    return net


@torch.no_grad()
def preactivation_weights(net):
    """[omega_i * W_i for activated layers] + [W_last], as float64 arrays."""
    from experiments.trainable_omega import OmegaScaledLayer
    layers, result = net.net.linear, []
    for index, layer in enumerate(layers):
        weight = layer.weight.detach().double()
        if index < len(layers) - 1:
            weight = weight * (float(layer.omega) if isinstance(layer, OmegaScaledLayer) else net.net.omega_0)
        result.append(weight.cpu().numpy())
    return result


def load_weights(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return preactivation_weights(build_model(payload["config"], payload["model"])), payload["config"]


def displacement_series(run_dirs):
    paths = checkpoint_paths(run_dirs)
    if len(paths) < 2:
        raise ValueError(f"Need at least two checkpoints/update_*.pt in {run_dirs}")
    rows, previous, config = [], None, None
    for update, path in paths.items():
        weights, config = load_weights(path)
        if previous is not None:
            start, before = previous
            rows.append({"start": start, "end": update,
                         "displacement": [float(np.linalg.norm(a - b)) for a, b in zip(weights, before)],
                         "norm_start": [float(np.linalg.norm(b)) for b in before]})
        previous = (update, weights)
    return rows, config


def compare(series, labels):
    """Per window and layer, displacement ratio of each run to the first run."""
    reference = {(r["start"], r["end"]): r for r in series[0]}
    ratios = {}
    for label, rows in zip(labels[1:], series[1:]):
        matched = [(r, reference[(r["start"], r["end"])]) for r in rows if (r["start"], r["end"]) in reference]
        ratios[label] = [{"start": r["start"], "end": r["end"],
                          "ratio": [a / b if b > 0 else None for a, b in zip(r["displacement"], ref["displacement"])]}
                         for r, ref in matched]
    return ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run directory; join a resumed run's directories with commas")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [split_dirs(r)[-1].name for r in args.runs]
    if len(labels) != len(args.runs):
        parser.error("--labels must match runs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [displacement_series(split_dirs(run)) for run in args.runs]
    series = [rows for rows, _ in results]
    with (args.output_dir / "step_displacement.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "start", "end", "layer", "displacement", "norm_start", "relative_displacement"])
        for label, rows in zip(labels, series):
            for r in rows:
                for layer, (d, n) in enumerate(zip(r["displacement"], r["norm_start"])):
                    writer.writerow([label, r["start"], r["end"], layer, d, n, d / n if n > 0 else None])
    ratios = compare(series, labels)
    (args.output_dir / "step_displacement.json").write_text(json.dumps(
        {"reference": labels[0], "series": dict(zip(labels, series)), "ratio_to_reference": ratios},
        indent=2, ensure_ascii=False), encoding="utf-8")
    layers = len(series[0][0]["displacement"])
    print(f"Median per-window displacement ratio to {labels[0]} (layer 0 = first, {layers - 1} = output):")
    for label, rows in ratios.items():
        medians = [statistics.median(r["ratio"][i] for r in rows if r["ratio"][i] is not None) for i in range(layers)]
        early = [r for r in rows if r["end"] <= 200]
        early_hidden = [statistics.median(r["ratio"][i] for i in range(1, layers - 1)) for r in early]
        print(f"  {label}: " + ", ".join(f"L{i} {m:.2f}" for i, m in enumerate(medians))
              + (f" | hidden, windows ending <=200: {', '.join(f'{v:.2f}' for v in early_hidden)}" if early_hidden else "")
              + f" ({len(rows)} windows)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, layers, figsize=(3.2 * layers, 3), sharey=True, squeeze=False)
    for i, ax in enumerate(axes[0]):
        for label, rows in zip(labels, series):
            ax.plot([r["end"] for r in rows], [r["displacement"][i] for r in rows], marker=".", label=label)
        ax.set_yscale("log")
        ax.set_title(f"layer {i}" + (" (output)" if i == layers - 1 else ""))
        ax.set_xlabel("update")
    axes[0][0].set_ylabel("||ΔA|| per checkpoint window")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "step_displacement.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

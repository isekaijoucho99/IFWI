"""Offline stable-rank diagnostics of saved residual IFWI checkpoints (no FD, no training).

For every checkpoints/update_XXXX.pt the model is rebuilt from its saved config and,
for each original IRN in it (the 2-D SIREN, or both low-rank factor networks):

* weight: stable rank ||W||_F^2 / ||W||_2^2 and spectral norm of every effective hidden
  weight (FR: lamb @ bases; trainable omega does not change the stable rank);
* feature: stable rank of the input of every layer after the first, evaluated on the
  run's own coordinates (the activation matrices H_l of McGinnis et al. 2026, Eq. 2,
  which bound the stable rank of that layer's weight gradient).

Stable rank is a diagnostic of how evenly a matrix spreads its energy, not of how many
directions the data constrain; checkpoints are every 50 updates, so this does not show
per-step updates (the runner logs those as hidden_update_srank). A resumed run is one
series: join its directories with commas.

    python -m experiments.rank_report RUN_ADAM RUN_MUON --labels adam muon --output-dir outputs/rank_report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from experiments.muon_optimizer import hidden_layers, irn_networks, stable_rank
from experiments.residual_band_report import split_dirs
from experiments.residual_step_report import build_model, checkpoint_paths


@torch.no_grad()
def model_ranks(model):
    rows = []
    for label, layer in hidden_layers(model):
        weight = layer.weight.detach()
        rows.append({"kind": "weight", "layer": label, "stable_rank": stable_rank(weight),
                     "spectral_norm": float(torch.linalg.matrix_norm(weight.double(), ord=2))})
    captured, hooks = {}, []
    for n, irn in enumerate(irn_networks(model)):
        for index in range(1, len(irn.linear)):
            hooks.append(irn.linear[index].register_forward_hook(
                lambda module, inputs, output, key=f"net{n}.{index}": captured.__setitem__(key, inputs[0].detach())))
    try:
        model()
    finally:
        for hook in hooks:
            hook.remove()
    for key, features in captured.items():
        matrix = features.reshape(-1, features.shape[-1])
        rows.append({"kind": "feature", "layer": key, "stable_rank": stable_rank(matrix),
                     "spectral_norm": float(torch.linalg.matrix_norm(matrix.double(), ord=2))})
    return rows


def run_series(run_dirs):
    series = []
    for update, path in checkpoint_paths(run_dirs).items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(payload["config"], payload["model"])
        series.extend({"update": update, **row} for row in model_ranks(model))
    if not series:
        raise ValueError(f"No checkpoints/update_*.pt in {run_dirs}")
    return series


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
    results = {label: run_series(split_dirs(run)) for label, run in zip(labels, args.runs)}
    with (args.output_dir / "rank_report.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", "update", "kind", "layer", "stable_rank", "spectral_norm"])
        writer.writeheader()
        for label, rows in results.items():
            writer.writerows({"run": label, **row} for row in rows)
    (args.output_dir / "rank_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), squeeze=False)
    for ax, kind in zip(axes[0], ("weight", "feature")):
        for label, rows in results.items():
            for layer in sorted({r["layer"] for r in rows if r["kind"] == kind}):
                points = [(r["update"], r["stable_rank"]) for r in rows
                          if r["kind"] == kind and r["layer"] == layer and r["stable_rank"] is not None]
                if points:
                    ax.plot(*zip(*points), marker=".", label=f"{label} {layer}")
        ax.set_title(f"{kind} stable rank")
        ax.set_xlabel("update")
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(args.output_dir / "rank_report.png", dpi=150)
    plt.close(fig)
    for label, rows in results.items():
        last = max(r["update"] for r in rows)
        final = [r for r in rows if r["update"] == last]
        print(f"{label}@{last}: " + "; ".join(f"{r['kind'][0]}:{r['layer']} {r['stable_rank']:.1f}"
                                            for r in final if r["stable_rank"] is not None))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

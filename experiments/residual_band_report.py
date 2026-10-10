"""Offline wavenumber-band errors for finished residual IFWI runs.

Reads only saved arrays (true_velocity.npy, initial_velocity.npy,
checkpoints/velocity_XXXX.npy, final_velocity.npy); runs no FD and no training,
so it also works on existing baseline1000 outputs.

    python -m experiments.residual_band_report outputs/RUN_A outputs/RUN_B --labels baseline fr --output-dir outputs/band_report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from experiments.spectral_metrics import DEFAULT_EDGES_CPKM, band_errors


def run_series(run_dir, spacing_m, edges):
    run_dir = Path(run_dir)
    truth = np.load(run_dir / "true_velocity.npy")
    background = np.load(run_dir / "initial_velocity.npy")
    rows = [{"update": 0, **band_errors(background, truth, background, spacing_m, edges)}]
    for path in sorted((run_dir / "checkpoints").glob("velocity_*.npy")):
        update = int(re.fullmatch(r"velocity_(\d+)\.npy", path.name).group(1))
        rows.append({"update": update, **band_errors(np.load(path), truth, background, spacing_m, edges)})
    metrics = run_dir / "metrics.json"
    if (run_dir / "final_velocity.npy").exists() and metrics.exists():
        completed = json.loads(metrics.read_text(encoding="utf-8"))["completed_updates"]
        if all(r["update"] != completed for r in rows):
            rows.append({"update": completed, **band_errors(np.load(run_dir / "final_velocity.npy"),
                                                             truth, background, spacing_m, edges)})
    return rows


def plot(series, labels, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bands = list(series[0][0]["rmse_mps"])
    fig, axes = plt.subplots(1, len(bands), figsize=(3.2 * len(bands), 3), sharey=True, squeeze=False)
    for ax, band in zip(axes[0], bands):
        for rows, label in zip(series, labels):
            ax.plot([r["update"] for r in rows], [r["ratio_to_background"][band] for r in rows], label=label)
        ax.axhline(1, color="0.6", lw=.8, ls="--")
        ax.set_title(band)
        ax.set_xlabel("update")
    axes[0][0].set_ylabel("band RMSE / background")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "band_ratio.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--spacing-m", type=float, default=15.)
    parser.add_argument("--edges", nargs="+", type=float, help="Band edges in cycles/km; inf allowed as last edge")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [p.name for p in args.runs]
    if len(labels) != len(args.runs):
        parser.error("--labels must match runs")
    edges = tuple(args.edges) if args.edges else DEFAULT_EDGES_CPKM
    args.output_dir.mkdir(parents=True, exist_ok=True)
    series = [run_series(run, args.spacing_m, edges) for run in args.runs]
    with (args.output_dir / "band_errors.csv").open("w", encoding="utf-8", newline="") as stream:
        bands = list(series[0][0]["rmse_mps"])
        writer = csv.writer(stream)
        writer.writerow(["run", "update", *[f"rmse_{b}" for b in bands], *[f"ratio_{b}" for b in bands]])
        for label, rows in zip(labels, series):
            for r in rows:
                writer.writerow([label, r["update"], *[r["rmse_mps"][b] for b in bands],
                                 *[r["ratio_to_background"][b] for b in bands]])
    (args.output_dir / "band_errors.json").write_text(
        json.dumps(dict(zip(labels, series)), indent=2, ensure_ascii=False), encoding="utf-8")
    plot(series, labels, args.output_dir)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

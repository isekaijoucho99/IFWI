"""Offline wavenumber-band and region errors for finished residual IFWI runs.

Reads only saved arrays (true_velocity.npy, initial_velocity.npy,
checkpoints/velocity_XXXX.npy, final_velocity.npy); runs no FD and no training,
so it also works on existing baseline1000 outputs. A resumed run is one series:
join its directories with commas (earlier directory first).

    python -m experiments.residual_band_report outputs/RUN_A outputs/RUN_B,outputs/RUN_B_RESUMED --labels baseline fr --output-dir outputs/band_report

Writes band_errors.csv/json and band_ratio.png (wavenumber bands, see
spectral_metrics) and region_errors.csv/json (depth bands and the truth
high-velocity mask, see region_metrics).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from experiments.region_metrics import DEFAULT_DEPTH_EDGES_M, DEFAULT_HIGH_VELOCITY_MPS, region_errors
from experiments.spectral_metrics import DEFAULT_EDGES_CPKM, band_errors


def split_dirs(spec):
    return [Path(part) for part in str(spec).split(",") if part]


def velocity_snapshots(run_dirs):
    """Return truth, background and [(update, velocity)] across one run's directories."""
    run_dirs = split_dirs(run_dirs) if not isinstance(run_dirs, list) else run_dirs
    truth = np.load(run_dirs[0] / "true_velocity.npy")
    background = np.load(run_dirs[0] / "initial_velocity.npy")
    snapshots = {0: background}
    for run_dir in run_dirs:
        if not (np.array_equal(np.load(run_dir / "true_velocity.npy"), truth)
                and np.array_equal(np.load(run_dir / "initial_velocity.npy"), background)):
            raise ValueError(f"{run_dir} has a different truth or background")
        for path in sorted((run_dir / "checkpoints").glob("velocity_*.npy")):
            update = int(re.fullmatch(r"velocity_(\d+)\.npy", path.name).group(1))
            snapshots.setdefault(update, np.load(path))
        metrics = run_dir / "metrics.json"
        if (run_dir / "final_velocity.npy").exists() and metrics.exists():
            completed = json.loads(metrics.read_text(encoding="utf-8"))["completed_updates"]
            snapshots.setdefault(completed, np.load(run_dir / "final_velocity.npy"))
    return truth, background, sorted(snapshots.items())


def run_series(run_dirs, spacing_m, edges, depth_edges_m=DEFAULT_DEPTH_EDGES_M,
               high_velocity_mps=DEFAULT_HIGH_VELOCITY_MPS):
    truth, background, snapshots = velocity_snapshots(run_dirs)
    bands = [{"update": u, **band_errors(v, truth, background, spacing_m, edges)} for u, v in snapshots]
    regions = [{"update": u, **region_errors(v, truth, background, spacing_m, depth_edges_m, high_velocity_mps)}
               for u, v in snapshots]
    return bands, regions


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


def write_regions(region_series, labels, output_dir):
    names = list(region_series[0][0]["rmse_mps"])
    with (output_dir / "region_errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "update", *[f"rmse_{n}" for n in names], *[f"sse_share_{n}" for n in names]])
        for label, rows in zip(labels, region_series):
            for r in rows:
                writer.writerow([label, r["update"], *[r["rmse_mps"][n] for n in names],
                                 *[r["sse_share"][n] for n in names]])
    (output_dir / "region_errors.json").write_text(
        json.dumps(dict(zip(labels, region_series)), indent=2, ensure_ascii=False), encoding="utf-8")
    def cell(rmse, share):
        return f"{'-':>22s}" if rmse is None else f"{rmse:>13.1f} / {share:.2f}"

    first = region_series[0][0]
    print("cell fraction: " + ", ".join(f"{n} {first['cell_fraction'][n]:.3f}" for n in names))
    print(f"{'run@update':24s}" + "".join(f"{n:>22s}" for n in names) + "   (RMSE m/s / SSE share)")
    print(f"{'background':24s}" + "".join(
        cell(first["background_rmse_mps"][n], first["background_sse_share"][n]) for n in names))
    for label, rows in zip(labels, region_series):
        last = rows[-1]
        print(f"{label + '@' + str(last['update']):24s}" + "".join(
            cell(last["rmse_mps"][n], last["sse_share"][n]) for n in names))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run directory; join a resumed run's directories with commas")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--spacing-m", type=float, default=15.)
    parser.add_argument("--edges", nargs="+", type=float, help="Band edges in cycles/km; inf allowed as last edge")
    parser.add_argument("--depth-edges-m", nargs="+", type=float, help="Depth band edges in m; inf allowed as last edge")
    parser.add_argument("--high-velocity-mps", type=float, default=DEFAULT_HIGH_VELOCITY_MPS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [split_dirs(r)[-1].name for r in args.runs]
    if len(labels) != len(args.runs):
        parser.error("--labels must match runs")
    edges = tuple(args.edges) if args.edges else DEFAULT_EDGES_CPKM
    depth_edges = tuple(args.depth_edges_m) if args.depth_edges_m else DEFAULT_DEPTH_EDGES_M
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_series(run, args.spacing_m, edges, depth_edges, args.high_velocity_mps) for run in args.runs]
    series = [bands for bands, _ in results]
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
    write_regions([regions for _, regions in results], labels, args.output_dir)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

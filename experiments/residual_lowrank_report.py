"""Offline singular-value spectra and checkpoint relative change of residual IFWI runs.

For every saved velocity v_t (checkpoints/velocity_XXXX.npy, final_velocity.npy)
it takes the learned residual R_t = v_t - v_init and reports its singular values,
the energy ranks r90/r99 (smallest k with sum_{i<=k} s_i^2 >= 90 / 99 % of the
total) and the entropy effective rank exp(-sum p_i log p_i), p_i = s_i / sum s
(Roy & Vetterli, 2007). The truth residual truth - v_init is the reference
(LR-IFWI, Chen et al., 2025, Fig. 3). It also reports the relative change between
consecutive saved velocities, ||v_b - v_a||_F^2 / ||v_a||_F^2 (their Fig. 23 RE).
Their RE is per epoch; here it is per checkpoint window (net change, not path), so
compare runs with the same checkpoint interval.

Reads only saved arrays and runs no FD, so it works on any finished residual run,
2-D SIREN or low-rank. A resumed run is one series: join its directories with commas.

    python -m experiments.residual_lowrank_report outputs/BASELINE,outputs/BASELINE_RESUMED outputs/LR \\
        --labels baseline lr50 --output-dir outputs/lowrank_report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.residual_band_report import split_dirs, velocity_snapshots


def spectrum_metrics(residual):
    """Singular values and rank summaries of one residual; ranks are None for a zero residual."""
    singular = np.linalg.svd(np.asarray(residual, dtype=np.float64), compute_uv=False)
    energy = np.square(singular)
    total = energy.sum()
    if total == 0:
        return {"singular_values": singular.tolist(), "frobenius_mps": 0., "r90": None, "r99": None, "entropy_rank": None}
    cumulative = np.cumsum(energy) / total
    p = singular / singular.sum()
    p = p[p > 0]
    return {"singular_values": singular.tolist(), "frobenius_mps": float(np.sqrt(total)),
            "r90": int(np.searchsorted(cumulative, .90) + 1), "r99": int(np.searchsorted(cumulative, .99) + 1),
            "entropy_rank": float(np.exp(-(p * np.log(p)).sum()))}


def run_series(run_dirs):
    truth, background, snapshots = velocity_snapshots(run_dirs)
    rows, previous = [], None
    for update, velocity in snapshots:
        row = {"update": update, **spectrum_metrics(velocity - background)}
        if previous is not None:
            start, before = previous
            row.update(window_start=start,
                       relative_change=float(np.square(velocity - before).sum() / np.square(before).sum()))
        rows.append(row)
        previous = (update, velocity)
    return {"truth_residual": spectrum_metrics(truth - background), "rows": rows}


def plot(series, labels, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    truth = series[0]["truth_residual"]["singular_values"]
    axes[0].semilogy(np.arange(1, len(truth) + 1), truth, color="k", lw=1.5, label="truth - v_init")
    for result, label in zip(series, labels):
        last = result["rows"][-1]
        values = np.asarray(last["singular_values"])
        axes[0].semilogy(np.arange(1, len(values) + 1), np.maximum(values, 1e-12), label=f"{label}@{last['update']}")
        rows = [r for r in result["rows"] if r["entropy_rank"] is not None]
        axes[1].plot([r["update"] for r in rows], [r["entropy_rank"] for r in rows], marker=".", label=label)
        rows = [r for r in result["rows"] if "relative_change" in r]
        axes[2].semilogy([r["update"] for r in rows], [r["relative_change"] for r in rows], marker=".", label=label)
    axes[1].axhline(series[0]["truth_residual"]["entropy_rank"], color="k", lw=.8, ls="--", label="truth - v_init")
    axes[0].set(xlabel="index", ylabel="singular value (m/s)", title="residual spectrum (last saved)")
    axes[1].set(xlabel="update", ylabel="entropy effective rank", title="effective rank of v_t - v_init")
    axes[2].set(xlabel="update (window end)", ylabel="||v_b - v_a||² / ||v_a||²", title="relative change per window")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "lowrank_summary.png", dpi=150)
    plt.close(fig)
    for result, label in zip(series, labels):
        rows = [r for r in result["rows"] if r["frobenius_mps"] > 0]
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        colors = plt.cm.viridis(np.linspace(0, 1, max(len(rows), 1)))
        for color, row in zip(colors, rows):
            values = np.asarray(row["singular_values"])
            ax.semilogy(np.arange(1, len(values) + 1), np.maximum(values, 1e-12), color=color, lw=.9, label=str(row["update"]))
        ax.semilogy(np.arange(1, len(truth) + 1), truth, color="k", lw=1.5, ls="--", label="truth - v_init")
        ax.set(xlabel="index", ylabel="singular value (m/s)", title=f"{label}: v_t - v_init")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"lowrank_spectra_{label}.png", dpi=150)
        plt.close(fig)


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
    series = [run_series(split_dirs(run)) for run in args.runs]
    with (args.output_dir / "lowrank_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "update", "frobenius_mps", "r90", "r99", "entropy_rank", "window_start", "relative_change"])
        for label, result in zip(labels, series):
            t = result["truth_residual"]
            writer.writerow([label, "truth", t["frobenius_mps"], t["r90"], t["r99"], t["entropy_rank"], None, None])
            for r in result["rows"]:
                writer.writerow([label, r["update"], r["frobenius_mps"], r["r90"], r["r99"], r["entropy_rank"],
                                 r.get("window_start"), r.get("relative_change")])
    (args.output_dir / "lowrank_spectra.json").write_text(
        json.dumps(dict(zip(labels, series)), indent=2, ensure_ascii=False), encoding="utf-8")
    plot(series, labels, args.output_dir)
    t = series[0]["truth_residual"]
    print(f"{'truth - v_init':24s} r90 {t['r90']:>4d}  r99 {t['r99']:>4d}  entropy rank {t['entropy_rank']:.1f}")
    for label, result in zip(labels, series):
        last = result["rows"][-1]
        if last["entropy_rank"] is None:
            print(f"{label + '@' + str(last['update']):24s} zero residual")
            continue
        print(f"{label + '@' + str(last['update']):24s} r90 {last['r90']:>4d}  r99 {last['r99']:>4d}  "
              f"entropy rank {last['entropy_rank']:.1f}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()

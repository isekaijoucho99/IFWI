"""Plot saved Residual IFWI models every 100 updates, starting at update zero.

Reads existing velocity snapshots; never loads an optimizer or starts training.
Uses the original IFWI paper's colors, physical aspect and typography.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
MM = 1 / 25.4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    controller = args.controller_root.resolve()
    state = json.loads((controller / "state.json").read_text(encoding="utf-8"))
    start = Path(state["start_run"])
    runs = [start] + [Path(s["output"]) for s in state["stages"] if "output" in s]
    if state.get("active_output_parent"):
        active = Path(state["active_output_parent"])
        if active.is_dir():
            runs.extend(p for p in active.iterdir() if p.is_dir())
    runs = list(dict.fromkeys(runs))
    config = json.loads((start / "config.json").read_text(encoding="utf-8"))
    truth = np.load(start / "true_velocity.npy", allow_pickle=False)
    spacing = config["data"]["grid_spacing_m"] / 1000
    snapshots = {0: start / "initial_velocity.npy"}
    for run in runs:
        for path in (run / "checkpoints").glob("velocity_*.npy"):
            epoch = int(path.stem.split("_")[-1])
            if epoch % 100 == 0:
                snapshots[epoch] = path
    last = max(snapshots)
    expected = list(range(0, last + 1, 100))
    missing = sorted(set(expected) - set(snapshots))
    if missing:
        raise RuntimeError(f"Missing real snapshots at updates {missing}; no interpolation is permitted")
    fields = {epoch: np.load(snapshots[epoch], allow_pickle=False) for epoch in expected}
    rows = []
    for epoch, field in fields.items():
        if field.shape != truth.shape or not np.isfinite(field).all():
            raise ValueError(f"Invalid saved model at update {epoch}")
        error = field.astype(np.float64) - truth.astype(np.float64)
        rows.append(dict(epoch=epoch, rmse_mps=float(np.sqrt(np.mean(error ** 2))),
                         mae_mps=float(np.mean(np.abs(error))),
                         minimum_mps=float(field.min()), maximum_mps=float(field.max()),
                         pixels_below_color_range=int((field < 1400).sum()),
                         pixels_above_color_range=int((field > 5500).sum()),
                         source=str(snapshots[epoch]),
                         source_sha256=hashlib.sha256(snapshots[epoch].read_bytes()).hexdigest()))
    output = (args.output_dir or controller.parent /
              ("residual_every100_" + datetime.now().strftime("%Y%m%d_%H%M%S_")
               + uuid.uuid4().hex[:6])).resolve()
    output.mkdir(parents=True, exist_ok=False)
    single = output / "individual"
    single.mkdir()
    style_source = ROOT / "experiments/plot_residual_ifwi_paper.py"
    spec = importlib.util.spec_from_file_location("_residual_paper_style", style_source)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.configure("original_ifwi")
    nz, nx = truth.shape
    extent = [-spacing / 2, (nx - .5) * spacing,
              (nz - .5) * spacing, -spacing / 2]

    def draw(ax, field, title, *, xlabel=True, ylabel=True):
        im = ax.imshow(field / 1000, extent=extent, origin="upper", aspect="equal",
                       interpolation="nearest", cmap="RdBu_r", vmin=1.4, vmax=5.5)
        ax.set(xlim=(0, (nx - 1) * spacing), ylim=((nz - 1) * spacing, 0), title=title)
        ax.set_xticks(np.arange(0, (nx - 1) * spacing + .01, .5))
        ax.set_yticks([0, .5, 1.])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(direction="out", length=2.5, width=.65, pad=2,
                       labelbottom=xlabel, labelleft=ylabel)
        if xlabel:
            ax.set_xlabel("Distance [km]")
        if ylabel:
            ax.set_ylabel("Depth [km]")
        return im

    def colorbar(fig, im, axes):
        cb = fig.colorbar(im, ax=axes, fraction=.022, pad=.025,
                          ticks=[1.4, 2.5, 3.5, 4.5, 5.5], shrink=.9)
        cb.set_label("Velocity [km/s]")
        cb.ax.tick_params(length=2, pad=2, labelsize=7.5)

    def export(fig, directory, name):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(directory / f"{name}.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(110 * MM, 46 * MM), layout="constrained")
    im = draw(ax, truth, "Ground truth (reference)")
    colorbar(fig, im, [ax])
    export(fig, output, "true_reference")
    for row in rows:
        epoch = row["epoch"]
        fig, ax = plt.subplots(figsize=(110 * MM, 46 * MM), layout="constrained")
        im = draw(ax, fields[epoch], f"Epoch {epoch} | RMSE {row['rmse_mps']:.2f} m/s")
        colorbar(fig, im, [ax])
        export(fig, single, f"epoch_{epoch:04d}")
    sheets = []
    for offset in range(0, len(rows), 6):
        group = rows[offset:offset + 6]
        nrows = (len(group) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(183 * MM, (127 / 3 * nrows) * MM),
                                 layout="constrained")
        flat = np.atleast_1d(axes).ravel()
        for index, (ax, row) in enumerate(zip(flat, group)):
            im = draw(ax, fields[row["epoch"]],
                      f"Epoch {row['epoch']} | RMSE {row['rmse_mps']:.2f} m/s",
                      xlabel=index + 2 >= len(group), ylabel=index % 2 == 0)
        for ax in flat[len(group):]:
            ax.set_visible(False)
        colorbar(fig, im, flat[:len(group)].tolist())
        fig.suptitle(f"Residual IFWI: Epoch {group[0]['epoch']}-{group[-1]['epoch']}", fontsize=10)
        name = f"epochs_{group[0]['epoch']:04d}_{group[-1]['epoch']:04d}"
        export(fig, output, name)
        sheets.append(name + ".png")
    with (output / "metrics_every100.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "source_data.npz", truth=truth,
                        **{f"epoch_{k:04d}": v for k, v in fields.items()})
    manifest = dict(controller=str(controller), epochs=expected, sheets=sheets,
                    color_range_kmps=[1.4, 5.5], colormap="RdBu_r", spacing_km=spacing,
                    style_source=str(style_source),
                    style_sha256=hashlib.sha256(style_source.read_bytes()).hexdigest(),
                    note="Actual post-update velocity snapshots, no interpolation or retraining. "
                         "Epoch 0 is the original smooth initial model. Same color range in every panel.",
                    rows=rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "epochs": expected, "sheets": sheets,
                      "rmse_mps": {r['epoch']: round(r['rmse_mps'], 3) for r in rows}}, indent=2))


if __name__ == "__main__":
    main()

"""Restyle saved residual IFWI results using the original IFWI paper conventions.

The default follows Sun et al. (2023), Figures 5/7 and 8. A G2IFWI style
option retains the earlier figure adaptation. No training, resampling,
normalization, curve smoothing or added comparison method is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, LogLocator, MaxNLocator
import numpy as np

MM = 1 / 25.4
BLUE, RED, GRAY = "#0000ff", "#ff0000", "#808080"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configure(style):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"] if style == "original_ifwi" else ["DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.5,
        "axes.linewidth": .65, "lines.linewidth": .9,
        "axes.spines.top": True, "axes.spines.right": True,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def export(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=600)
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def grid(ax, logarithmic=False):
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#c5c5c5", linestyle=":", linewidth=.45)
    ax.grid(which="minor", color="#d8d8d8", linestyle=":", linewidth=.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if logarithmic:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction="out", width=.55, length=2.5)
    ax.tick_params(which="minor", direction="out", width=.4, length=1.5)
    for spine in ax.spines.values():
        spine.set_color("#808080")


def maps(output, fields, dx, limits):
    fig = plt.figure(figsize=(183 * MM, 45 * MM))
    nz, nx = fields[0].shape
    extent = [-dx / 2, (nx - .5) * dx, (nz - .5) * dx, -dx / 2]
    names = ("(a) True model", "(b) Initial model", "(c) Residual IFWI")
    for index, (field, title) in enumerate(zip(fields, names)):
        ax = fig.add_axes([(12 + 53 * index) / 183, 13 / 45, 47 / 183, 22 / 45])
        # The reference's seismic palette is intentional; one shared physical scale.
        image = ax.imshow(field, extent=extent, origin="upper", aspect="auto",
                          interpolation="nearest", cmap="seismic", vmin=limits[0], vmax=limits[1])
        ax.set(xlim=(0, (nx - 1) * dx), ylim=((nz - 1) * dx, 0),
               title=title, xlabel="Position (m)")
        ax.set_xticks(np.arange(0, (nx - 1) * dx + 1, 1000))
        ax.set_yticks(np.arange(0, (nz - 1) * dx + 1, 500))
        ax.tick_params(direction="out", width=.55, length=2, pad=2)
        if index == 0:
            ax.set_ylabel("Depth (m)", labelpad=3)
        else:
            ax.tick_params(labelleft=False)
    cax = fig.add_axes([169 / 183, 13 / 45, 2.5 / 183, 22 / 45])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_ticks(np.arange(math.ceil(limits[0] / 500) * 500, limits[1] + 1, 500))
    colorbar.ax.tick_params(labelsize=7.5, length=2, pad=2)
    colorbar.set_label("Velocity (m/s)", fontsize=8.5, labelpad=4)
    export(fig, output, "velocity_comparison_paper")


def losses(output, sampled, full, model_history):
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 75 * MM))
    fig.subplots_adjust(left=.085, right=.98, bottom=.19, top=.87, wspace=.35)
    ax = axes[0]
    x = [row["evaluated_updates"] for row in sampled]
    y = [row["data_mse"] for row in sampled]
    full_y = [row["data_mse"] for row in full]
    if min(y + full_y) <= 0:
        raise ValueError("Raw waveform MSE must be positive for the paper's log axis")
    ax.plot(x, y, color=BLUE, linewidth=.8, label="Random 8 shots (training)")
    ax.plot([row["evaluated_updates"] for row in full], full_y,
            color=RED, linestyle="--", linewidth=.9, marker="o", markersize=1.8,
            label="All 49 shots (evaluation)")
    ax.set(title="(a) Data loss", xlabel="Epoch", ylabel="Data loss",
           xlim=(0, model_history[-1]["evaluated_updates"]))
    ax.set_xticks(np.arange(0, model_history[-1]["evaluated_updates"] + 1, 100))
    grid(ax, logarithmic=True)
    ax.legend(loc="upper right", frameon=True, edgecolor="#808080", framealpha=1)
    ax = axes[1]
    ax.plot([row["evaluated_updates"] for row in model_history],
            [row["rmse_mps"] for row in model_history], color=BLUE, label="Residual IFWI")
    ax.set(title="(b) Model RMSE", xlabel="Epoch", ylabel="Model RMSE (m/s)",
           xlim=(0, model_history[-1]["evaluated_updates"]))
    ax.set_xticks(np.arange(0, model_history[-1]["evaluated_updates"] + 1, 100))
    grid(ax)
    ax.legend(loc="upper right", frameon=True, edgecolor="#808080", framealpha=1)
    export(fig, output, "loss_paper")


def metrics(output, history):
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 75 * MM))
    fig.subplots_adjust(left=.085, right=.98, bottom=.19, top=.87, wspace=.35)
    for ax, key, title, ylabel in zip(axes, ("mae_mps", "ssim"),
                                    ("(a) Model MAE", "(b) Structural similarity"),
                                    ("Model MAE (m/s)", "SSIM")):
        ax.plot([row["evaluated_updates"] for row in history],
                [row[key] for row in history], color=BLUE, label="Residual IFWI")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel,
               xlim=(0, history[-1]["evaluated_updates"]))
        ax.set_xticks(np.arange(0, history[-1]["evaluated_updates"] + 1, 100))
        grid(ax)
        ax.legend(loc="best", frameon=True, edgecolor="#808080", framealpha=1)
    export(fig, output, "metrics_paper")


def profiles(output, fields, dx, columns, limits):
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 110 * MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.98, bottom=.16, top=.88, wspace=.12)
    depth = np.arange(fields[0].shape[0]) * dx
    for index, (ax, column) in enumerate(zip(axes, columns)):
        ax.plot(fields[0][:, column], depth, color="black", label="True model")
        ax.plot(fields[1][:, column], depth, color=GRAY, linestyle="--", label="Initial model")
        ax.plot(fields[2][:, column], depth, color=BLUE, label="Residual IFWI")
        ax.set(title=f"({chr(97 + index)}) x = {column * dx:.0f} m", xlabel="Velocity (m/s)",
               ylim=(depth[-1], 0), xlim=limits)
        ax.set_xticks(np.arange(math.ceil(limits[0] / 1000) * 1000, limits[1] + 1, 1000))
        grid(ax)
    axes[0].set_ylabel("Depth (m)")
    axes[-1].legend(loc="upper right", frameon=True, edgecolor="#808080", framealpha=1)
    export(fig, output, "velocity_profiles_paper")


def original_grid(ax):
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="major", color="#808080", linestyle="--", linewidth=.55)
    ax.grid(which="minor", color="#aaaaaa", linestyle="--", linewidth=.4)
    ax.tick_params(direction="out", length=3, width=.65)


def original_maps(output, fields, dx, limits):
    """Three compact wide panels, matching one column of the paper's Figure 7."""
    fig = plt.figure(figsize=(89 * MM, 82 * MM))
    nz, nx = fields[0].shape
    spacing = dx / 1000
    height_mm = 60 * (nz - 1) / (nx - 1)
    extent = [-spacing / 2, (nx - .5) * spacing, (nz - .5) * spacing, -spacing / 2]
    for index, field in enumerate(fields):
        bottom_mm = 12 + (2 - index) * (height_mm + 1.3)
        ax = fig.add_axes([12 / 89, bottom_mm / 82, 60 / 89, height_mm / 82])
        image = ax.imshow(field / 1000, origin="upper", extent=extent,
                          aspect="equal", interpolation="nearest", cmap="RdBu_r",
                          vmin=limits[0] / 1000, vmax=limits[1] / 1000)
        ax.set(xlim=(0, (nx - 1) * spacing), ylim=((nz - 1) * spacing, 0),
               ylabel="Depth [km]")
        ax.set_xticks(np.arange(0, (nx - 1) * spacing + .01, .5))
        ax.set_yticks([0, .5, 1.])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(direction="out", length=2.5, width=.65, pad=2)
        ax.text(.018, .94, f"{chr(97 + index)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="black", linewidth=.55, pad=.8))
        if index == 2:
            ax.set_xlabel("Distance [km]")
        else:
            ax.tick_params(labelbottom=False, bottom=False)
    cax = fig.add_axes([74 / 89, 12 / 82, 1.8 / 89, height_mm / 82])
    cb = fig.colorbar(image, cax=cax, ticks=[limits[0] / 1000, 2.5, 3.5, 4.5, limits[1] / 1000])
    cb.ax.tick_params(labelsize=7.5, pad=2, length=2)
    cb.set_label("km/s", labelpad=3, fontsize=8.5)
    export(fig, output, "velocity_comparison_paper")


def epoch_ticks(total, intervals=8):
    """Keep the paper's compact axes readable for longer saved trajectories."""
    locator = MaxNLocator(nbins=intervals, integer=True, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(0, total)
    return ticks[(ticks >= 0) & (ticks <= total)]


def original_losses(output, sampled, full, model_history):
    """Figure 8 style: one linear raw-loss axis with scientific-notation offset."""
    fig, ax = plt.subplots(figsize=(89 * MM, 70 * MM))
    fig.subplots_adjust(left=.17, right=.95, bottom=.18, top=.89)
    x = [row["evaluated_updates"] for row in sampled]
    y = [row["data_mse"] for row in sampled]
    full_y = [row["data_mse"] for row in full]
    ax.plot(x, y, color=RED, linewidth=.8, label="Residual IFWI (8 shots)")
    ax.plot([row["evaluated_updates"] for row in full], full_y,
            color=BLUE, linewidth=.9, marker="o", markersize=1.8,
            label="49-shot evaluation")
    ax.set(xlabel="Epoch", ylabel="Loss", xlim=(0, model_history[-1]["evaluated_updates"]),
           ylim=(0, max(y + full_y) * 1.08))
    ax.set_xticks(epoch_ticks(model_history[-1]["evaluated_updates"]))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
    original_grid(ax)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True,
              edgecolor="#aaaaaa", framealpha=.9)
    export(fig, output, "loss_paper")


def original_metrics(output, history):
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 70 * MM))
    fig.subplots_adjust(left=.065, right=.978, bottom=.19, top=.88, wspace=.4)
    for ax, key, title, ylabel, color in zip(axes, ("rmse_mps", "mae_mps", "ssim"),
                                          ("a) RMSE", "b) MAE", "c) SSIM"),
                                          ("RMSE (m/s)", "MAE (m/s)", "SSIM"),
                                          (BLUE, "#ff7f0e", RED)):
        ax.plot([row["evaluated_updates"] for row in history], [row[key] for row in history], color=color)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel,
               xlim=(0, history[-1]["evaluated_updates"]))
        ax.set_xticks(epoch_ticks(history[-1]["evaluated_updates"], intervals=4))
        original_grid(ax)
    export(fig, output, "metrics_paper")


def original_profiles(output, fields, dx, columns, limits):
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 110 * MM), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.15, top=.88, wspace=.1)
    depth = np.arange(fields[0].shape[0]) * dx / 1000
    for index, (ax, column) in enumerate(zip(axes, columns)):
        ax.plot(fields[0][:, column] / 1000, depth, color="black", label="True model")
        ax.plot(fields[1][:, column] / 1000, depth, color=BLUE, linestyle="--", label="Initial model")
        ax.plot(fields[2][:, column] / 1000, depth, color=RED, label="Residual IFWI")
        ax.set(title=f"{chr(97 + index)}) Distance = {column * dx / 1000:.3f} km",
               xlabel="Velocity [km/s]", ylim=(depth[-1], 0),
               xlim=(limits[0] / 1000, limits[1] / 1000))
        ax.set_xticks([2, 3, 4, 5])
        original_grid(ax)
    axes[0].set_ylabel("Depth [km]")
    axes[-1].legend(loc="upper right", fontsize=7.5, frameon=True, edgecolor="#aaaaaa", framealpha=.9)
    export(fig, output, "velocity_profiles_paper")


def write_csv(path, rows, columns):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--reference-pdf", type=Path, help="Record the style-reference PDF SHA256")
    parser.add_argument("--style", choices=("original_ifwi", "g2ifwi"), default="original_ifwi")
    args = parser.parse_args()
    run, output = args.run_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error("--output-dir must not already exist")
    before = {path.name: digest(path) for path in run.iterdir() if path.is_file()}
    config, status, results = (read_json(run / name) for name in ("config.json", "status.json", "metrics.json"))
    full = read_json(run / "full_evaluation_history.json")
    sampled = read_json(run / "sampled_training_history.json")
    completed = status["completed_updates"]
    if status["state"] != "completed" or completed != results["completed_updates"] or len(sampled) != completed:
        raise ValueError("Only complete runs with a complete sampled trajectory can be restyled")
    if [row["evaluated_updates"] for row in sampled] != list(range(completed)):
        raise ValueError("Sampled losses must correspond to the original preupdate evaluation counts")
    if full[-1]["evaluated_updates"] != completed:
        raise ValueError("Missing full postupdate endpoint")
    model_history = [*sampled, {"evaluated_updates": completed, **results["final"]}]
    for row in model_history:
        for key in ("rmse_mps", "mae_mps", "ssim", "data_mse"):
            if not np.isfinite(row[key]):
                raise ValueError(f"Nonfinite source metric: {key}")
    fields = [np.load(run / f"{name}_velocity.npy", allow_pickle=False) for name in ("true", "initial", "final")]
    if any(field.ndim != 2 or field.shape != fields[0].shape or not np.isfinite(field).all() for field in fields):
        raise ValueError("Source velocity fields must be finite and have the same two-dimensional grid")
    dx = config["data"]["grid_spacing_m"]
    limits = (100 * math.floor(min(float(field.min()) for field in fields) / 100),
              100 * math.ceil(max(float(field.max()) for field in fields) / 100))
    columns = np.rint(np.array([.25, .5, .75]) * (fields[0].shape[1] - 1)).astype(int).tolist()
    output.mkdir(parents=True, exist_ok=False)
    configure(args.style)
    if args.style == "original_ifwi":
        original_maps(output, fields, dx, limits)
        original_losses(output, sampled, full, model_history)
        original_metrics(output, model_history)
        original_profiles(output, fields, dx, columns, limits)
    else:
        maps(output, fields, dx, limits)
        losses(output, sampled, full, model_history)
        metrics(output, model_history)
        profiles(output, fields, dx, columns, limits)
    loss_rows = [{"series": "Random 8 shots (training)", "timing": "preupdate", **row} for row in sampled]
    loss_rows += [{"series": "All 49 shots (evaluation)", "timing": "postupdate", **row} for row in full]
    write_csv(output / "loss_source.csv", loss_rows, ["series", "timing", "evaluated_updates", "data_mse"])
    write_csv(output / "model_metrics_source.csv", model_history,
              ["evaluated_updates", "rmse_mps", "mae_mps", "ssim"])
    profile_rows = [{"position_m": column * dx, "depth_m": z * dx,
                     "true_mps": float(fields[0][z, column]), "initial_mps": float(fields[1][z, column]),
                     "final_mps": float(fields[2][z, column])}
                    for column in columns for z in range(fields[0].shape[0])]
    write_csv(output / "profiles_source.csv", profile_rows,
              ["position_m", "depth_m", "true_mps", "initial_mps", "final_mps"])
    np.savez_compressed(output / "velocity_source.npz", true_mps=fields[0], initial_mps=fields[1],
                        final_mps=fields[2], x_m=np.arange(fields[0].shape[1]) * dx,
                        z_m=np.arange(fields[0].shape[0]) * dx)
    after = {name: digest(run / name) for name in before}
    if before != after:
        raise RuntimeError("A source result file changed during plotting")
    manifest = {
        "style": args.style,
        "style_reference": ("Sun et al. (2023), Implicit Seismic Full Waveform Inversion With Deep Neural Representation, Figures 5/7 (velocity), 8 (loss); DOI 10.1029/2022JB025964"
                            if args.style == "original_ifwi" else "G2IFWI: Figures 3 (velocity), 4/7 (optimization), 13 (profiles)"),
        "style_reference_pdf_sha256": digest(args.reference_pdf) if args.reference_pdf else None,
        "script_sha256": digest(Path(__file__)), "input_sha256": before,
        "source_run": str(run), "completed_updates": completed,
        "source_files_unchanged": True, "plotting_backend": "Python/matplotlib",
        "source_velocity_units": "m/s", "source_position_depth_units": "m", "grid_spacing_m": dx,
        "display_velocity_units": "km/s" if args.style == "original_ifwi" else "m/s",
        "display_position_depth_units": "km" if args.style == "original_ifwi" else "m",
        "grid_shape_zx": list(fields[0].shape), "velocity_color_limits_mps": list(limits),
        "velocity_colormap": "RdBu_r" if args.style == "original_ifwi" else "seismic",
        "image_display_aspect": "equal" if args.style == "original_ifwi" else "auto",
        "image_interpolation": "nearest", "spatial_crop": None,
        "model_values_clipped": False, "loss_normalization": "none; original raw MSE",
        "curve_smoothing": False, "sampled_training_points": len(sampled),
        "loss_axis_scale": "linear" if args.style == "original_ifwi" else "log",
        "full_evaluation_points": len(full), "model_metric_points": len(model_history),
        "profile_columns": columns, "profile_positions_m": [column * dx for column in columns],
        "formats": ["PNG 600 dpi", "PDF with editable text", "SVG with editable text"],
        "replicates": "one run, seed 3; no error bars or uncertainty bands inferred",
        "adaptation": "style only; three velocity panels because only one inversion method was run",
        "epoch_definition": "one Adam update using eight randomly selected shots; not a full 49-shot pass",
        "model_metric_role": "offline evaluation against synthetic truth, never a training supervision target",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    notes = """# Paper-style residual IFWI figures

Style-only adaptation of the supplied G2IFWI paper. These figures show the
completed Marmousi residual IFWI experiment, not paper-reported results or a
diffusion-regularized run. Epoch means one Adam update, selecting eight shots.

Velocity: three full sampled-grid fields, one shared seismic color scale in m/s.
The scale includes every value from every field (1400–5500 m/s for this run),
including final velocities below the true model's 1500 m/s minimum. No crop,
spatial smoothing or numeric clipping is applied. Compact wide panels adapt
the reference layout to the three available fields; display aspect is auto.

Loss: raw random-eight-shot preupdate MSE is blue; all-49-shot postupdate
evaluation MSE is dashed red. All 800 training points and all 17 full
evaluations are shown; evaluation points are connected by straight segments.
The left axis is logarithmic. Model RMSE is a separate linear panel, computed
offline against truth. Supplementary MAE and SSIM preserve the other metrics.
All 801 model-metric states are shown. Curves are not smoothed or normalized.

Profiles: the same three sampled columns used by the original report; true
model black, initial model dashed grey, residual IFWI blue. The grey curve is
an initial velocity model, not an unrun comparison method.

One seed is available. No statistical intervals, repeated-run variation,
diffusion activation line, frequency-stage boundaries or extra methods are
invented. Original result files and figures are hash-checked before/after.
Source CSV/NPZ files and the manifest provide the data mapping and provenance.

Widths are 183 mm. Body labels use 8.5 pt, legends/colorbar ticks 7.5 pt.
PNG previews use 600 dpi; PDF and SVG retain editable vector text. Publication
acceptance requirements are outside the scope of this style adaptation.
"""
    if args.style == "original_ifwi":
        notes = f"""# Original IFWI paper-style residual results

Style-only adaptation of Sun et al. (2023), JGR Solid Earth, DOI
10.1029/2022JB025964, Figures 5/7 and 8. These are saved Marmousi residual
IFWI results, not the author's reported results. The plot labels identify
Residual IFWI; no FWI, IFWI-Pretrain or IFWI-Random comparison is fabricated.

Velocity panels: a) true model; b) fixed Gaussian initial model; c) saved
postupdate-{completed} residual IFWI. The compact three-row, one-column layout
adapts one column of Figure 7 to the three available fields. Arial typography,
RdBu_r, inside-panel boxed letters, downward depth, physical aspect ratio,
outer distance ticks and a narrow bottom-panel colorbar follow the reference.
Depth/distance use km and displayed velocity uses km/s. The common full-range
scale is 1.4–5.5 km/s; the paper's 1–4.7 km/s range is not imposed on this data.
No field is cropped, smoothed or numerically clipped. Saved source units
remain m/s and m; display conversions divide by 1000.

Loss: Figure 8 uses a linear axis, not a logarithmic or normalized loss.
Original raw MSE is displayed with scientific-notation tick labels. All {len(sampled)}
sampled-shot preupdate points are red; all {len(full)} full-shot postupdate
evaluations are blue with small markers. Curves are connected by straight
segments; no smoothing is used. Epoch denotes one Adam update using eight
random shots, not a full 49-shot pass. Model RMSE, MAE and SSIM are provided
in a separate diagnostic figure, not relabeled as training losses.

Profiles use the same three columns as the initial report, with true model
black, initial model dashed blue and final residual IFWI red. This is a
style adaptation, not an exact recreation of a profile figure in the paper.

One seed is available; no repeated-run error bars or confidence bands are
invented. Source CSV/NPZ tables and input hashes preserve traceability.
Original result files and existing figures are hash-checked before/after.
Velocity and loss widths are 89 mm; metrics/profiles widths are 183 mm.
Labels are 8.5 pt, legends/colorbar ticks 7.5 pt. PNG is 600 dpi; PDF/SVG
text is editable. Scientific-notation offsets do not normalize the loss.
"""
    (output / "README.md").write_text(notes, encoding="utf-8")
    print(json.dumps({"output": str(output), "source_files_unchanged": True,
                      "sampled_points": len(sampled), "full_evaluations": len(full),
                      "metric_points": len(model_history), "color_limits_mps": limits}, ensure_ascii=False))


if __name__ == "__main__":
    main()

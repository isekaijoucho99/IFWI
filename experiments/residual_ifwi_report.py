"""CPU-only velocity metrics and local figures for residual IFWI stage 1."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


def _velocity_pair(prediction_mps, truth_mps):
    prediction = np.asarray(prediction_mps, dtype=np.float64)
    truth = np.asarray(truth_mps, dtype=np.float64)
    if prediction.ndim != 2 or truth.ndim != 2:
        raise ValueError("Velocity fields must be two-dimensional")
    if prediction.shape != truth.shape:
        raise ValueError("Velocity fields must have equal shapes")
    if min(truth.shape) < 3:
        raise ValueError("Velocity fields require at least 3 samples on each axis for SSIM")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Velocity fields must contain only finite values")
    return prediction, truth


def velocity_metrics(prediction_mps: np.ndarray, truth_mps: np.ndarray) -> dict:
    """Return RMSE/MAE in m/s and SSIM, without clipping either velocity field.

    SSIM uses the true model's velocity range. Constant truth uses an explicit
    1 m/s range; the window is the largest odd value up to 7 that fits the grid.
    """
    prediction, truth = _velocity_pair(prediction_mps, truth_mps)
    difference = prediction - truth
    data_range = float(truth.max() - truth.min())
    if data_range == 0:
        data_range = 1.0
    window = min(7, min(truth.shape))
    if window % 2 == 0:
        window -= 1
    result = {
        "rmse_mps": float(np.sqrt(np.mean(np.square(difference)))),
        "mae_mps": float(np.mean(np.abs(difference))),
        "ssim": float(structural_similarity(truth, prediction, data_range=data_range, win_size=window)),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Velocity metrics are not finite")
    return result


def save_plots(
    out: Path,
    truth_mps,
    initial_mps,
    final_mps,
    history: list[dict],
    dx_m: float,
) -> None:
    """Save velocity maps, three vertical profiles, and evaluated loss curves.

    Maps share the truth's color scale. Values outside that scale retain their
    original values in metrics and profiles. Pixel centers use the solver grid
    coordinates; distance and downward depth are displayed in km.
    """
    initial, truth = _velocity_pair(initial_mps, truth_mps)
    final, _ = _velocity_pair(final_mps, truth_mps)
    if not np.isfinite(dx_m) or dx_m <= 0:
        raise ValueError("Grid spacing must be finite and positive")
    if not history:
        raise ValueError("Evaluated loss history must not be empty")
    for row in history:
        if not np.isfinite(row["evaluated_updates"]) or not np.isfinite(row["data_mse"]) or row["data_mse"] < 0:
            raise ValueError("Evaluated update counts and waveform MSE must be finite, with nonnegative MSE")
        for key in ("rmse_mps", "mae_mps", "ssim"):
            if row.get(key) is not None and not np.isfinite(row[key]):
                raise ValueError("Evaluated velocity metrics must be finite")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    nz, nx = truth.shape
    spacing_km = dx_m / 1000
    extent = [-spacing_km / 2, (nx - .5) * spacing_km, (nz - .5) * spacing_km, -spacing_km / 2]
    vmin, vmax = float(truth.min() / 1000), float(truth.max() / 1000)
    if vmin == vmax:
        vmin -= .0005
        vmax += .0005

    fields = [(truth, "True velocity"), (initial, "Initial velocity"), (final, "Final velocity")]

    def draw_map(ax, field, title):
        raster = ax.imshow(field / 1000, origin="upper", extent=extent, aspect="auto",
                           cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set(title=title, xlabel="Distance (km)", ylabel="Depth (km)",
               xlim=(0, (nx - 1) * spacing_km), ylim=((nz - 1) * spacing_km, 0))
        return raster

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    try:
        for ax, (field, title) in zip(axes, fields):
            raster = draw_map(ax, field, title)
        fig.colorbar(raster, ax=axes, label="Velocity (km/s)", extend="both")
        fig.savefig(out / "velocity_comparison.png", dpi=160)
    finally:
        plt.close(fig)
    for filename, (field, title) in zip(
        ("true_velocity.png", "initial_velocity.png", "final_velocity.png"), fields
    ):
        fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
        try:
            raster = draw_map(ax, field, title)
            fig.colorbar(raster, ax=ax, label="Velocity (km/s)", extend="both")
            fig.savefig(out / filename, dpi=160)
        finally:
            plt.close(fig)

    columns = np.rint(np.array([.25, .5, .75]) * (nx - 1)).astype(int)
    depth_km = np.arange(nz) * spacing_km
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True, layout="constrained")
    try:
        for ax, column in zip(axes, columns):
            for (field, title), color in zip(fields, ("black", "tab:blue", "tab:red")):
                ax.plot(field[:, column] / 1000, depth_km, label=title, color=color)
            ax.set(title=f"Distance = {column * spacing_km:.3f} km", xlabel="Velocity (km/s)",
                   ylim=(depth_km[-1], 0))
            ax.grid(alpha=.25)
        axes[0].set_ylabel("Depth (km)")
        axes[-1].legend()
        fig.savefig(out / "velocity_profiles.png", dpi=160)
    finally:
        plt.close(fig)

    optional_errors = [key for key in ("rmse_mps", "mae_mps") if any(row.get(key) is not None for row in history)]
    has_ssim = any(row.get("ssim") is not None for row in history)
    panel_count = 1 + bool(optional_errors) + has_ssim
    fig, axes = plt.subplots(1, panel_count, figsize=(5 * panel_count, 3.5), squeeze=False, layout="constrained")
    axes = axes[0]
    try:
        updates = [row["evaluated_updates"] for row in history]
        losses = [row["data_mse"] for row in history]
        axes[0].plot(updates, losses, marker="o", markersize=3)
        if all(loss > 0 for loss in losses):
            axes[0].set_yscale("log")
        axes[0].set(title="Waveform fit", ylabel="Waveform MSE")
        next_panel = 1
        if optional_errors:
            for key in optional_errors:
                rows = [row for row in history if row.get(key) is not None]
                axes[next_panel].plot([row["evaluated_updates"] for row in rows], [row[key] for row in rows],
                                      marker="o", markersize=3, label="RMSE" if key == "rmse_mps" else "MAE")
            axes[next_panel].set(title="Velocity error", ylabel="Error (m/s)")
            axes[next_panel].legend()
            next_panel += 1
        if has_ssim:
            rows = [row for row in history if row.get("ssim") is not None]
            axes[next_panel].plot([row["evaluated_updates"] for row in rows], [row["ssim"] for row in rows],
                                  marker="o", markersize=3)
            axes[next_panel].set(title="Structural similarity", ylabel="SSIM")
        for ax in axes:
            ax.set_xlabel("Completed updates (evaluated)")
            ax.grid(alpha=.25)
        fig.savefig(out / "loss_curve.png", dpi=160)
    finally:
        plt.close(fig)


def save_sampling_plot(out, sampled_history, full_history, selected_shots, available_shots):
    """Keep stochastic preupdate training loss distinct from full evaluations."""
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    try:
        ax.plot([row["evaluated_updates"] for row in sampled_history],
                [row["data_mse"] for row in sampled_history], alpha=.45,
                label=f"Random {selected_shots} shots, before update")
        ax.plot([row["evaluated_updates"] for row in full_history],
                [row["data_mse"] for row in full_history], marker="o",
                label=f"All {available_shots} shots, evaluated model")
        ax.set(xlabel="Completed updates at evaluation", ylabel="Waveform MSE",
               title="Sampled training and full acquisition evaluations", yscale="log")
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(Path(out) / "sampled_loss_curve.png", dpi=160)
    finally:
        plt.close(fig)

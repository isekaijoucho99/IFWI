"""Depth-band and high-velocity-mask velocity errors for residual IFWI.

Each region reports its RMSE (m/s), cell fraction and share of the total squared
error, against the prediction and the fixed background. Depth bands partition the
grid, so their SSE shares sum to 1. The high-velocity mask selects truth cells at or
above a threshold; on the 4x-downsampled Marmousi the 5000 m/s default picks exactly
the two thin 5500 m/s layers (955 cells, 3.5% of the grid, ~1.1-1.2 km deep), far
thinner than the 8 Hz resolution. ``rest`` is everything outside that mask.
"""
from __future__ import annotations

import numpy as np

DEFAULT_DEPTH_EDGES_M = (0., 500., 1000., np.inf)
DEFAULT_HIGH_VELOCITY_MPS = 5000.


def _label(low, high):
    return f"z{low:g}-{high:g}m" if np.isfinite(high) else f"z>{low:g}m"


def region_masks(truth_mps, spacing_m, depth_edges_m=DEFAULT_DEPTH_EDGES_M,
                 high_velocity_mps=DEFAULT_HIGH_VELOCITY_MPS):
    truth = np.asarray(truth_mps, dtype=np.float64)
    if truth.ndim != 2:
        raise ValueError("Expected a [nz,nx] truth model")
    if depth_edges_m[0] != 0 or any(b <= a for a, b in zip(depth_edges_m[:-1], depth_edges_m[1:])):
        raise ValueError("Depth edges must start at 0 and increase")
    depth = (np.arange(truth.shape[0]) * spacing_m)[:, None] + np.zeros_like(truth)
    masks = {_label(low, high): (depth >= low) & (depth < high)
             for low, high in zip(depth_edges_m[:-1], depth_edges_m[1:])}
    high = truth >= high_velocity_mps
    masks[f"v>={high_velocity_mps:g}"] = high
    masks[f"v<{high_velocity_mps:g}"] = ~high
    return masks


def _stats(error, masks):
    total = float(np.square(error).sum())
    result = {}
    for name, mask in masks.items():
        sse = float(np.square(error[mask]).sum())
        result[name] = {"rmse_mps": float(np.sqrt(sse / mask.sum())) if mask.any() else None,
                        "sse_share": sse / total if total > 0 else None}
    return result


def region_errors(prediction_mps, truth_mps, background_mps, spacing_m,
                  depth_edges_m=DEFAULT_DEPTH_EDGES_M, high_velocity_mps=DEFAULT_HIGH_VELOCITY_MPS):
    """Per-region RMSE and SSE share for prediction and background against truth."""
    prediction, truth, background = (np.asarray(a, dtype=np.float64).squeeze() for a in
                                     (prediction_mps, truth_mps, background_mps))
    if not prediction.shape == truth.shape == background.shape:
        raise ValueError("prediction, truth and background must share one [nz,nx] shape")
    masks = region_masks(truth, spacing_m, depth_edges_m, high_velocity_mps)
    predicted, reference = _stats(prediction - truth, masks), _stats(background - truth, masks)
    return {"cell_fraction": {name: float(mask.mean()) for name, mask in masks.items()},
            "rmse_mps": {name: v["rmse_mps"] for name, v in predicted.items()},
            "sse_share": {name: v["sse_share"] for name, v in predicted.items()},
            "background_rmse_mps": {name: v["rmse_mps"] for name, v in reference.items()},
            "background_sse_share": {name: v["sse_share"] for name, v in reference.items()}}

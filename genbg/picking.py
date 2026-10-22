"""Truth-free background scores from one forward simulation (design doc 4.4).

First breaks: the first sample whose amplitude reaches ``alpha`` times the trace maximum, refined by
linear interpolation of the normalized amplitude between that sample and the previous one. The same
picker runs on observed and synthetic records, so its bias largely cancels in the difference.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.frequency_continuation import lowpass


def first_breaks(traces, dt_s, alpha=0.05):
    """[S, T, R] records -> [S, R] pick times (s); NaN for dead traces."""
    a = np.abs(np.asarray(traces, dtype=np.float64))
    peak = a.max(axis=1)                                            # [S, R]
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = a / peak[:, None, :]
    above = normalized >= alpha
    index = above.argmax(axis=1)                                    # first True
    previous = np.take_along_axis(normalized, np.maximum(index - 1, 0)[:, None, :], axis=1)[:, 0]
    current = np.take_along_axis(normalized, index[:, None, :], axis=1)[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(index > 0, (alpha - previous) / (current - previous), 1.)
    times = (index - 1 + np.clip(fraction, 0., 1.)) * dt_s
    times[index == 0] = 0.
    times[~(peak > 0)] = np.nan
    return times


def offsets_m(geometry, dx_m):
    sources = np.asarray(geometry["source_x_indices"], dtype=np.float64)[:, None]
    receivers = np.asarray(geometry["receiver_x_indices"], dtype=np.float64)[None, :]
    return np.abs(receivers - sources) * dx_m


def pick_scores(observed_picks, predicted_picks, offsets, frequency_hz, min_offset_m=150., max_offset_m=None):
    mask = np.isfinite(observed_picks) & np.isfinite(predicted_picks) & (offsets >= min_offset_m)
    if max_offset_m is not None:
        mask &= offsets <= max_offset_m
    if not mask.any():
        raise ValueError("no usable first-break pairs")
    delta = (predicted_picks - observed_picks)[mask]
    half_period = .5 / frequency_hz
    return {"fb_rms_ms": float(1000 * np.sqrt(np.mean(delta ** 2))),
            "fb_mean_ms": float(1000 * delta.mean()),
            "fb_median_abs_ms": float(1000 * np.median(np.abs(delta))),
            "cycle_skip_frac": float(np.mean(np.abs(delta) > half_period)),
            "fb_traces": int(mask.sum())}


@torch.no_grad()
def waveform_scores(predicted, observed, dt_s, cutoffs_hz=(3., 5.)):
    """Normalized misfits ||lp(p) - lp(o)||^2 / ||lp(o)||^2 of [B, S, T, R] records, full band and low-passed."""
    out = {}
    for cutoff in (None, *cutoffs_hz):
        p, o = lowpass(predicted, cutoff, dt_s).double(), lowpass(observed, cutoff, dt_s).double()
        key = "full_nmse" if cutoff is None else f"lp{cutoff:g}_nmse"
        out[key] = float((p - o).square().sum() / o.square().sum())
    return out

"""Multiscale frequency continuation of the waveform misfit (Bunks et al., 1995).

The first updates fit low-passed records and the last stage fits the full band. Predicted
and observed records are filtered by the same zero-phase low-pass along time, so the FD
solver, source wavelet and observations stay unchanged (the solver is linear in the
source, so filtering the records equals filtering the wavelet up to the record end).

Filter: amplitude response H(f) = 1 / (1 + (f / max_hz)^(2n)) with n = 6, applied after
zero-padding the time axis to 2 nt (no wrap-around). This is the zero-phase
(forward-backward) Butterworth of order n: -6 dB at max_hz, no time shift.

A low-pass stage's MSE is multiplied by E_full / E_band, the energy ratio of all observed
records before and after filtering (fixed per stage, from all shots). The objective is then
the band's relative misfit on the full-band scale, so gradients keep a comparable size
across stages (Adam's eps and the logged values see no 100x drop at 3 Hz). Optimizer
state carries over between stages.

    training:
      frequency_schedule:
        - {max_hz: 3.0, updates: 100}
        - {max_hz: 5.0, updates: 100}
        - {max_hz: null}              # full band for all remaining updates

null (the default) keeps the full-band misfit throughout, i.e. the previous behaviour.
"""
from __future__ import annotations

import math

import torch

FILTER_ORDER = 6


def validate_frequency_schedule(schedule, dt_s):
    if schedule is None:
        return
    if not isinstance(schedule, list) or len(schedule) < 2:
        raise ValueError("training.frequency_schedule must be null or a list of at least two stages")
    nyquist, previous = .5 / dt_s, 0.
    for index, stage in enumerate(schedule):
        last = index == len(schedule) - 1
        if not isinstance(stage, dict) or set(stage) != ({"max_hz"} if last else {"max_hz", "updates"}):
            raise ValueError("Each frequency stage needs max_hz and updates; the last stage has only max_hz")
        cutoff = stage["max_hz"]
        if last:
            if cutoff is not None:
                raise ValueError("The last frequency stage must be the full band (max_hz: null)")
            continue
        if (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff)
                or not previous < cutoff < nyquist):
            raise ValueError("Stage max_hz must be positive, increasing and below the temporal Nyquist frequency")
        updates = stage["updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
            raise ValueError("Stage updates must be a positive integer")
        previous = cutoff


def stage_cutoff(schedule, update):
    """max_hz of the stage containing the 1-based update index (None: full band)."""
    if schedule is None:
        return None
    end = 0
    for stage in schedule[:-1]:
        end += stage["updates"]
        if update <= end:
            return float(stage["max_hz"])
    return None


def stage_boundaries(schedule):
    """[(first_update, last_update or None, max_hz or None)] for logging."""
    if schedule is None:
        return []
    rows, first = [], 1
    for stage in schedule[:-1]:
        rows.append((first, first + stage["updates"] - 1, float(stage["max_hz"])))
        first += stage["updates"]
    return rows + [(first, None, None)]


def lowpass(traces, max_hz, dt_s, order=FILTER_ORDER):
    """Zero-phase low-pass of [B,S,T,R] records along T; identity for max_hz None."""
    if max_hz is None:
        return traces
    nt = traces.shape[2]
    frequencies = torch.fft.rfftfreq(2 * nt, dt_s, dtype=torch.float64, device=traces.device)
    response = (1 / (1 + (frequencies / max_hz) ** (2 * order))).to(traces.dtype)
    spectrum = torch.fft.rfft(traces, n=2 * nt, dim=2)
    return torch.fft.irfft(spectrum * response[None, None, :, None], n=2 * nt, dim=2)[:, :, :nt]


@torch.no_grad()
def energy_scale(observed, max_hz, dt_s):
    """E_full / E_band of the observed records (1 for the full band)."""
    if max_hz is None:
        return 1.
    band = float(lowpass(observed, max_hz, dt_s).double().square().sum())
    if not band > 0:
        raise ValueError(f"Observed records have no energy below {max_hz} Hz")
    return float(observed.double().square().sum()) / band


def energy_scales(observed, schedule, dt_s):
    """{max_hz: E_full / E_band} for every low-pass stage, from all observed shots."""
    return {float(s["max_hz"]): energy_scale(observed, float(s["max_hz"]), dt_s) for s in (schedule or [])[:-1]}

"""Optional training settings: late-arrival time gain, and EMA / lowest-misfit velocity estimates.

time_gain (training.time_gain: {power, start_update}): from start_update on, predicted and
observed records are multiplied by w(t) = (t / t_end)^power before the misfit (after any
frequency-continuation low-pass). At the end of a 13-shot FR run the residual sits mostly
after 1.0 s (deep reflections, relative misfit 20-30x that of the early arrivals) while the
plain MSE is dominated by the strong early arrivals. Like the low-pass stages, the gained
objective is multiplied by E_obs / E_gained (all observed records), so its size stays on the
full-band scale; data_mse remains the plain full-band MSE.

report_estimates (training.report_estimates: {ema_decay, ema_start}): truth-free extra
estimates written next to the last iterate, without changing training:
* the exponential moving average of the post-update velocity from update ema_start on;
* the evaluated state with the lowest full-band data MSE (pre-update evaluations and the
  final post-update evaluation).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from experiments.frequency_continuation import lowpass


def _number(value, label, low=None, high=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite {'integer' if integer else 'number'}")
    if (low is not None and value < low) or (high is not None and value >= high):
        raise ValueError(f"{label} must be in [{low}, {high})")


def validate_time_gain(setting):
    if setting is None:
        return
    if not isinstance(setting, dict) or set(setting) != {"power", "start_update"}:
        raise ValueError("training.time_gain must be null or contain exactly power and start_update")
    _number(setting["power"], "time_gain.power")
    if setting["power"] <= 0:
        raise ValueError("time_gain.power must be positive")
    _number(setting["start_update"], "time_gain.start_update", low=1, integer=True)


def time_power(setting, update):
    """Gain power active at the 1-based update index, or None."""
    return None if setting is None or update < setting["start_update"] else float(setting["power"])


def time_gain(traces, power, dt_s):
    """Records [B,S,T,R] times (t / t_end)^power along T."""
    nt = traces.shape[2]
    t = torch.arange(nt, dtype=torch.float64, device=traces.device) * dt_s
    weights = (t / t[-1]) ** power
    return traces * weights.to(traces.dtype)[None, None, :, None]


@torch.no_grad()
def gain_scale(observed, max_hz, power, dt_s):
    """E_obs / E of the low-passed and gained observed records."""
    weighted = time_gain(lowpass(observed, max_hz, dt_s), power, dt_s)
    energy = float(weighted.double().square().sum())
    if not energy > 0:
        raise ValueError("Gained observed records have no energy")
    return float(observed.double().square().sum()) / energy


def validate_report_estimates(setting):
    if setting is None:
        return
    if not isinstance(setting, dict) or set(setting) != {"ema_decay", "ema_start"}:
        raise ValueError("training.report_estimates must be null or contain exactly ema_decay and ema_start")
    _number(setting["ema_decay"], "report_estimates.ema_decay", low=0, high=1)
    if setting["ema_decay"] <= 0:
        raise ValueError("report_estimates.ema_decay must be in (0, 1)")
    _number(setting["ema_start"], "report_estimates.ema_start", low=1, integer=True)


class IterateEstimates:
    """EMA of post-update velocities and the lowest-misfit evaluated state (float64, m/s)."""

    def __init__(self, setting):
        self.decay, self.start = float(setting["ema_decay"]), int(setting["ema_start"])
        self.ema, self.ema_updates = None, 0
        self.best_mse, self.best_updates, self.best_velocity = math.inf, None, None

    def observe_evaluation(self, data_mse, completed_updates, velocity):
        if data_mse < self.best_mse:
            self.best_mse, self.best_updates = float(data_mse), int(completed_updates)
            self.best_velocity = np.array(velocity, dtype=np.float64)

    def observe_update(self, update, velocity_after):
        if update < self.start:
            return
        velocity_after = np.asarray(velocity_after, dtype=np.float64)
        self.ema = velocity_after.copy() if self.ema is None else self.decay * self.ema + (1 - self.decay) * velocity_after
        self.ema_updates += 1

    def state_dict(self):
        return {"ema": self.ema, "ema_updates": self.ema_updates, "best_mse": self.best_mse,
                "best_updates": self.best_updates, "best_velocity": self.best_velocity}

    def load_state_dict(self, state):
        self.ema, self.ema_updates = state["ema"], state["ema_updates"]
        self.best_mse, self.best_updates, self.best_velocity = state["best_mse"], state["best_updates"], state["best_velocity"]

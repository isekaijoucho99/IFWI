"""Detect numerical plateaus from fixed full-data MSE and saved velocities.

This module does not import the training runner or access truth-based metrics.
The resume boundary must remain the original continuation checkpoint across
training chunks; restarting a chunk must not reset the minimum update count.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _nonnegative_finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not np.isfinite(number) or number < 0.:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


@dataclass(frozen=True)
class ConvergencePolicy:
    eval_interval: int = 50
    window_evaluations: int = 5
    confirmations: int = 3
    min_additional_updates: int = 400
    loss_trend_rtol: float = .005
    loss_range_rtol: float = .02
    velocity_step_rtol: float = .001
    velocity_window_rtol: float = .002

    def __post_init__(self):
        for name, minimum in (("eval_interval", 1), ("window_evaluations", 2),
                              ("confirmations", 1), ("min_additional_updates", 0)):
            _integer(getattr(self, name), name, minimum)
        for name in ("loss_trend_rtol", "loss_range_rtol", "velocity_step_rtol",
                     "velocity_window_rtol"):
            _nonnegative_finite(getattr(self, name), name)


def _velocity_array(value, update):
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex velocity")
        velocity = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid velocity at update {update}") from error
    if velocity.ndim == 0 or not velocity.size or not np.isfinite(velocity).all() or (velocity <= 0.).any():
        raise ValueError(f"Velocity at update {update} must be a nonempty finite positive array")
    return velocity


def _relative_rms_change(previous, current):
    # The common scale cancels in the RMS ratio and avoids squaring large m/s
    # values. Subtraction occurs in float64 without mutating saved arrays.
    scale = max(float(previous.max()), float(current.max()))
    numerator = np.sqrt(np.mean(((current - previous) / scale) ** 2))
    denominator = np.sqrt(np.mean((previous / scale) ** 2))
    if denominator == 0.:
        raise ValueError("Velocity RMS ratio cannot be represented numerically")
    ratio = float(numerator / denominator)
    if not np.isfinite(ratio):
        raise ValueError("Velocity RMS ratio is not finite")
    return ratio


def analyze_convergence(full_history, velocities: dict[int, np.ndarray], *,
                        resumed_from_updates: int, policy: ConvergencePolicy) -> dict:
    """Return a JSON-compatible decision without selecting a truth-best model.

    History rows require ``evaluated_updates`` and ``data_mse``. Rows before
    ``resumed_from_updates`` never contribute convergence windows. From that
    boundary onward, every scheduled evaluation must be present and ordered.
    Velocities are required for every checkpoint used by a complete window.
    """
    origin = _integer(resumed_from_updates, "resumed_from_updates", 0)
    if not isinstance(policy, ConvergencePolicy):
        raise ValueError("policy must be a ConvergencePolicy")
    if not isinstance(velocities, Mapping):
        raise ValueError("velocities must map evaluated updates to arrays")
    if isinstance(full_history, (str, bytes, Mapping)):
        raise ValueError("full_history must be a sequence of evaluation rows")
    try:
        history = list(full_history)
    except TypeError as error:
        raise ValueError("full_history must be a sequence of evaluation rows") from error

    selected = []
    previous_update = None
    for row in history:
        if not isinstance(row, Mapping) or not {"evaluated_updates", "data_mse"}.issubset(row):
            raise ValueError("Evaluation rows require evaluated_updates and data_mse")
        update = _integer(row["evaluated_updates"], "evaluated_updates", 0)
        loss = _nonnegative_finite(row["data_mse"], "data_mse")
        if previous_update is not None and update <= previous_update:
            raise ValueError("Evaluation updates must be strictly increasing without duplicates")
        previous_update = update
        if update >= origin:
            expected = origin + len(selected) * policy.eval_interval
            if update != expected:
                raise ValueError(f"Incomplete expected evaluation interval: expected {expected}, got {update}")
            selected.append((update, loss))

    fields = {}
    shape = None
    for update, _ in selected:
        if update not in velocities:
            continue
        velocity = _velocity_array(velocities[update], update)
        if shape is not None and velocity.shape != shape:
            raise ValueError("Checkpoint velocity shapes must match")
        shape = velocity.shape
        fields[update] = velocity

    checks = []
    confirmation_count = 0
    size = policy.window_evaluations
    centered_steps = np.arange(size, dtype=np.float64) - (size - 1.) / 2.
    slope_denominator = float(np.dot(centered_steps, centered_steps))
    for stop in range(size, len(selected) + 1):
        window = selected[stop - size:stop]
        updates = [item[0] for item in window]
        missing = [update for update in updates if update not in fields]
        if missing:
            raise ValueError(f"Missing window velocity checkpoints: {missing}")
        losses = np.array([item[1] for item in window], dtype=np.float64)
        median = float(np.median(losses))
        loss_scale = max(median, np.finfo(float).eps)
        normalized_losses = (losses - median) / loss_scale
        slope = float(np.dot(centered_steps, normalized_losses) / slope_denominator)
        loss_trend = abs(slope * (size - 1))
        loss_range = float((losses.max() - losses.min()) / loss_scale)
        velocity_step = max(_relative_rms_change(fields[first], fields[second])
                            for first, second in zip(updates, updates[1:]))
        velocity_window = _relative_rms_change(fields[updates[0]], fields[updates[-1]])
        if not np.isfinite([loss_trend, loss_range, velocity_step, velocity_window]).all():
            raise ValueError("Window statistics are not finite")
        candidate = bool(loss_trend <= policy.loss_trend_rtol
                         and loss_range <= policy.loss_range_rtol
                         and velocity_step <= policy.velocity_step_rtol
                         and velocity_window <= policy.velocity_window_rtol)
        checks.append({"start_updates": updates[0], "end_updates": updates[-1],
                       "evaluated_updates": updates, "candidate": candidate,
                       "loss_trend_rtol": loss_trend, "loss_range_rtol": loss_range,
                       "velocity_step_rtol": velocity_step, "velocity_window_rtol": velocity_window})
        confirmation_count = confirmation_count + 1 if candidate else 0

    latest = selected[-1][0] if selected else None
    minimum_met = latest is not None and latest - origin >= policy.min_additional_updates
    converged = bool(minimum_met and confirmation_count >= policy.confirmations)
    if converged:
        state = "plateau"
    elif not checks or not minimum_met or (checks[-1]["candidate"] and confirmation_count < policy.confirmations):
        state = "insufficient_history"
    else:
        state = "improving_or_unstable"
    return {"converged": converged, "state": state, "evaluated_updates": latest,
            "confirmation_count": confirmation_count,
            "latest_window": checks[-1] if checks else None, "checks": checks}

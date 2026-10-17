"""Learning-rate factor per update: warmup at every frequency stage, then cosine decay.

Every optimizer parameter group keeps its configured learning rate as base (network Adam,
trainable-omega Adam, Muon) and is set to base * factor(update) before each step:

    factor = warmup * decay
    warmup = min(1, k / warmup_updates), k = 1, 2, ... counted from the first update of
             training and of every frequency-continuation stage (1 when warmup_updates is 0)
    decay  = 1 before decay_start; cosine from 1 to final_factor over decay_updates;
             final_factor afterwards

Reasons (docs/multiscale_ifwi.md): a stage switch changes the objective's curvature and
gradient scale under an Adam state tuned to the previous stage (both 13-shot multiscale runs
jumped right after the switch to the full band); late full-band training shows loss spikes
that grow over time (an FR multiscale run lost about 75 m/s in one). The factor depends
only on the update index and the configuration, so resuming reproduces it; the decay window
is explicit, so a longer resumed run does not reshape it.

    training:
      lr_schedule:
        warmup_updates: 20
        decay_start: 201
        decay_updates: 800
        final_factor: 0.1
"""
from __future__ import annotations

import math

from experiments.frequency_continuation import stage_boundaries

KEYS = ("warmup_updates", "decay_start", "decay_updates", "final_factor")


def validate_lr_schedule(setting):
    if setting is None:
        return
    if not isinstance(setting, dict) or set(setting) != set(KEYS):
        raise ValueError(f"training.lr_schedule must be null or contain exactly {list(KEYS)}")
    for key, minimum in (("warmup_updates", 0), ("decay_start", 1), ("decay_updates", 1)):
        value = setting[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"lr_schedule.{key} must be an integer >= {minimum}")
    final = setting["final_factor"]
    if isinstance(final, bool) or not isinstance(final, (int, float)) or not math.isfinite(final) or not 0 < final <= 1:
        raise ValueError("lr_schedule.final_factor must be in (0, 1]")


def stage_start(frequency_schedule, update):
    """First update of the frequency stage containing update (1 without a schedule)."""
    starts = [first for first, _, _ in stage_boundaries(frequency_schedule)] or [1]
    return max(s for s in starts if s <= update)


def lr_factor(setting, frequency_schedule, update):
    """Multiplier of every base learning rate for the 1-based update index."""
    if setting is None:
        return 1.
    warmup = setting["warmup_updates"]
    k = update - stage_start(frequency_schedule, update) + 1
    factor = min(1., k / warmup) if warmup else 1.
    start, length, final = setting["decay_start"], setting["decay_updates"], setting["final_factor"]
    if update >= start + length:
        factor *= final
    elif update >= start:
        factor *= final + (1 - final) * .5 * (1 + math.cos(math.pi * (update - start) / length))
    return factor

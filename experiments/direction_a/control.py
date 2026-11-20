"""Choose a trial displacement using the current physical gradient and acquired data."""

from dataclasses import dataclass, field
import math

import torch

from .model import jvp, trial_velocity
from .optimizer import UpdateProposal


@dataclass
class Choice:
    proposal: UpdateProposal
    factor: float = 1.
    trial: torch.Tensor | None = None
    verified_loss: float | None = None
    forward_shots: int = 0
    restarted: bool = False
    trials: list = field(default_factory=list)
    reason: str = "accepted"


def rms(v):
    return float(torch.linalg.vector_norm(v) / math.sqrt(v.numel()))


def dot(a, b):
    return float(torch.sum(a * b))


def choose_step(model, v, grad_v, grad_theta, rule, loss, survey, *,
                policy, max_rms, linear_tol, max_halvings):
    result = Choice(rule.propose(grad_theta))
    if policy == "plain":
        trial = trial_velocity(model, result.proposal.delta)
        if not survey.is_valid(trial):
            result.reason = "invalid_velocity"
        elif torch.equal(trial, v):
            result.reason = "zero_update"
        else:
            result.trial = trial
        return result

    velocity_guard = policy in ("guarded", "guarded_no_remainder")
    if not velocity_guard and policy != "armijo":
        raise ValueError(f"unknown trial policy {policy!r}")
    linear = jvp(model, result.proposal.delta) if velocity_guard else None
    slope = dot(grad_v, linear) if velocity_guard else dot(grad_theta, result.proposal.delta)
    if velocity_guard and slope >= 0 and torch.any(result.proposal.history != 0):
        result.trials.append(dict(factor=1., reason="restart_first_moment", linear_slope=slope))
        result.proposal = rule.propose(grad_theta, restart=True)
        result.restarted = True
        linear = jvp(model, result.proposal.delta)
        slope = dot(grad_v, linear)
    if not math.isfinite(slope) or slope >= 0:
        result.reason = "no_linear_descent"
        return result

    factor = min(1., max_rms / rms(linear)) if velocity_guard else 1.
    for _ in range(max_halvings + 1):
        result.factor = factor
        trial = trial_velocity(model, result.proposal.delta * factor)
        record = dict(factor=factor, linear_slope=factor * slope)
        result.trials.append(record)
        if not survey.is_valid(trial):
            record["reason"] = "invalid_velocity"
        elif velocity_guard:
            actual = trial - v
            actual_rms = rms(actual)
            remainder = rms(actual - factor * linear)
            record.update(actual_rms_mps=actual_rms, remainder_rms_mps=remainder,
                          physical_gradient_dot_actual=dot(grad_v, actual))
            if actual_rms == 0:
                result.reason = record["reason"] = "zero_update"
                return result
            if actual_rms > max_rms * (1 + 1e-12):
                record["reason"] = "velocity_cap"
            elif policy == "guarded" and remainder > linear_tol * actual_rms:
                record["reason"] = "linearization_error"
            elif record["physical_gradient_dot_actual"] >= 0:
                record["reason"] = "no_velocity_descent"
        if "reason" not in record:
            value, _ = survey.evaluate(trial, gradient=False)
            result.forward_shots += survey.shots
            record["trial_loss"] = value
            if value < loss and value <= loss + 1e-4 * factor * slope:
                result.trial, result.verified_loss = trial, value
                record["reason"] = "accepted"
                return result
            record["reason"] = "armijo"
        factor *= .5
    result.reason = "backtrack_exhausted"
    return result

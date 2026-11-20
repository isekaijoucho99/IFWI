"""Local, truth-free comparisons of physical velocity updates on a [z, x] grid."""

import math

import numpy as np
import torch

from experiments.spectral_metrics import band_rmse
from .model import jvp


def predict_fields(model, proposal, factor, grad_theta, lr):
    """Project the accepted parameter displacements through the pre-step Jacobian.

    The instantaneous Adam ablation keeps the proposal's denominator and clipping.
    The SGD kernel reference uses the raw parameter gradient at the same trial factor.
    """
    directions = {"linear": proposal.delta, "current": proposal.current, "history": proposal.history,
                  "instant": proposal.instant, "sgd_kernel": -lr * grad_theta}
    return {name: jvp(model, factor * direction) for name, direction in directions.items()}


def _field(value):
    array = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64).numpy()
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("Expected a finite, nonempty [z, x] field")
    return array


def _dot(a, b):
    return float(np.sum(a * b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _rms(a):
    return _norm(a) / math.sqrt(a.size)


def _posthoc_fit(actual, actual_norm, prediction):
    """Compare directions, then fit one unconstrained output scalar after the step."""
    pred_norm = _norm(prediction)
    dot = _dot(actual, prediction)
    scale = dot / (pred_norm ** 2) if pred_norm else None
    cosine = None
    matched_error = None
    best_error = None
    if actual_norm and pred_norm:
        cosine = float(np.clip(dot / (actual_norm * pred_norm), -1., 1.))
        matched_error = _norm(actual / actual_norm - prediction / pred_norm)
    if actual_norm:
        # With a zero prediction all scales give the same residual.
        best_error = _norm(actual - scale * prediction) / actual_norm if pred_norm else 1.
    return {"cosine_to_actual": cosine, "rms_matched_relative_error": matched_error,
            "best_scalar": scale, "best_scalar_relative_error": best_error,
            "nonnegative_scalar_relative_error": 1. if scale is not None and scale < 0 and actual_norm else best_error}


def measure(actual, grad_v, fields, spacing_m):
    """Measure a finite velocity step and its independently predicted JVPs.

    Relative errors use the norm of the observed finite step. Gradient dots are
    sums in the discrete physical coordinates used by autograd. The first-order
    gain compares the full JVP to the full instantaneous counterfactual at the
    same step factor. A positive gain predicts more loss reduction with history.
    Post-hoc fits use the observed step and are kept separate from predictions.
    """
    actual, gradient = _field(actual), _field(grad_v)
    predicted = {name: _field(fields[name]) for name in ("linear", "current", "history", "instant", "sgd_kernel")}
    if any(field.shape != actual.shape for field in (gradient, *predicted.values())):
        raise ValueError("All velocity fields and grad_v must share a [z, x] shape")
    if not math.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("Spacing in metres must be positive and finite")

    actual_norm = _norm(actual)
    linear, current, history = (predicted[name] for name in ("linear", "current", "history"))
    remainder = actual - linear
    bands = {"actual": actual, "linear": linear, "history": history, "remainder": remainder}
    comparisons = {name: predicted[name] for name in ("linear", "instant", "sgd_kernel")}
    fits = {name: _posthoc_fit(actual, actual_norm, value)
            for name, value in {**comparisons, "steepest_descent": -gradient}.items()}
    cosine_to_descent = fits["steepest_descent"]["cosine_to_actual"]
    dots = {name: _dot(gradient, predicted[name]) for name in ("linear", "current", "history", "instant")}
    return {
        **{f"{name}_relative_error": _norm(actual - value) / actual_norm if actual_norm else None
           for name, value in comparisons.items()},
        **{f"{name}_rms_mps": _rms(value) for name, value in {**bands, "current": current}.items()},
        **{f"{name}_gradient_dot": value for name, value in dots.items()},
        "physical_gradient_dot_actual": _dot(gradient, actual),
        "cosine_to_descent": cosine_to_descent,
        "angle_to_descent_deg": math.degrees(math.acos(cosine_to_descent)) if cosine_to_descent is not None else None,
        "history_first_order_gain_vs_instant": dots["instant"] - dots["linear"],
        "decomposition_error_rms_mps": _rms(linear - current - history),
        "posthoc_scalar_fit": fits,
        "band_rms_mps": {name: band_rmse(value, spacing_m) for name, value in bands.items()},
    }

"""Stratified prior importance sampling with a native, full-shot FWI likelihood."""

from numbers import Integral
from time import perf_counter

import numpy as np
from scipy.special import softmax


def importance_reference(problem, prior, *, samples_per_component, seed):
    """Return latent draws, self-normalized likelihood weights and work diagnostics.

    The frozen geological prior has equally weighted Gaussian components with
    common isotropic standard deviation. An equal number of independent draws
    per component integrates the prior with equal per-draw weights. The
    posterior weights therefore depend only on the full complex Gaussian
    likelihood at the observed frequencies. Rows are ordered by component.
    """
    if (isinstance(samples_per_component, bool) or not isinstance(samples_per_component, Integral)
            or samples_per_component < 1):
        raise ValueError("samples_per_component must be a positive integer")
    n = int(samples_per_component)
    centers = prior.centers
    components, dimension = centers.shape
    if dimension != problem.dimension:
        raise ValueError("prior and problem dimensions must agree")
    if not np.isfinite(problem.noise_std) or problem.noise_std <= 0:
        raise ValueError("noise_std must be positive and finite")
    start = perf_counter()
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((components, n, dimension))
    draws *= prior.component_std
    draws += centers[:, None, :]
    flat = draws.reshape(components * n, dimension)
    log_weights = np.empty(len(flat))
    variance = problem.noise_std**2
    for i, x in enumerate(flat):
        predicted = problem.predict(x)
        if predicted.shape != problem.data.shape:
            raise ValueError("predicted and observed gathers must have the same shape")
        residual = predicted - problem.data
        log_weights[i] = -0.5 * np.vdot(residual, residual).real / variance
    if not np.isfinite(log_weights).all():
        raise FloatingPointError("nonfinite full-shot log likelihood in importance reference")
    weights = softmax(log_weights)
    ess = float(1. / np.dot(weights, weights))
    split_ess = mean_difference = None
    if n >= 2:
        # Partition within every component to keep each half stratified.
        # Normalize each half's log weights independently to handle sharp likelihoods.
        by_component = log_weights.reshape(components, n)
        means, split_ess = [], []
        for offset in (0, 1):
            w = softmax(by_component[:, offset::2])
            means.append(np.einsum("ij,ijd->d", w, draws[:, offset::2]))
            split_ess.append(float(1. / np.sum(w*w)))
        mean_difference = float(np.mean((means[0] - means[1])**2)**0.5)
    stats = dict(method="stratified Gaussian-mixture prior importance sampling",
                 seed=int(seed), draws=len(flat), components=components, samples_per_component=n,
                 ess=ess, ess_fraction=ess / len(flat), max_weight=float(weights.max()),
                 split_half_ess=split_ess, split_half_mean_rms=mean_difference,
                 split_half_mean_units="RMS logit velocity per grid cell",
                 forward_solves=len(flat) * problem.shots * problem.frequencies_count,
                 adjoint_solves=0, factorizations=len(flat) * problem.frequencies_count,
                 wall_seconds=perf_counter() - start)
    return flat, weights, stats

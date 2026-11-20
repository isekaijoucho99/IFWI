"""Offline posterior and posterior-predictive comparisons on independent particles."""

import numpy as np
from scipy.special import ndtri
from scipy.stats import wasserstein_distance


def interval(draws, truth):
    q05, q95 = np.quantile(draws, [.05, .95], axis=0)
    return dict(coverage90=float(np.mean((q05 <= truth) & (truth <= q95))),
                interval_width90=float(np.mean(q95 - q05)),
                mean_rmse=float(np.mean((draws.mean(axis=0) - truth)**2)**.5))


def mode_score(draws, grid, weights, edges):
    bounds = np.r_[-np.inf, edges, np.inf]
    empirical = np.histogram(draws[:, 0], bounds)[0] / len(draws)
    reference = np.histogram(grid[:, 0], bounds, weights=weights)[0]
    return dict(edges=list(map(float, edges)), sample_mass=empirical.tolist(),
                reference_mass=reference.tolist(), total_variation=float(np.abs(empirical-reference).sum()/2))


def _directions(d):
    rng = np.random.default_rng(439)
    vectors = rng.standard_normal((32, d))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _relative(num, den):
    return float(num / den) if den > 0 else None


def gaussian_distance(draws, mean, covariance):
    projected = _directions(len(mean))
    quantiles = ndtri((np.arange(2048) + .5) / 2048)
    distances = [wasserstein_distance(draws @ u, u @ mean + np.sqrt(u @ covariance @ u)*quantiles)
                 for u in projected]
    delta_cov = np.atleast_2d(np.cov(draws.T)) - covariance
    return dict(mean_error_l2=float(np.linalg.norm(draws.mean(axis=0)-mean)),
                relative_covariance_error=_relative(np.linalg.norm(delta_cov), np.linalg.norm(covariance)),
                sliced_w1=float(np.mean(distances)))


def sample_distance(draws, reference, weights=None):
    """Compare to a quadrature, weighted probability measure, or iid ensemble."""
    X = (draws - draws.mean(axis=0)) / np.sqrt(len(draws)-1)
    ref_mean = np.average(reference, weights=weights, axis=0)
    if weights is None:
        Y = (reference - ref_mean) / np.sqrt(len(reference)-1)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        Y = (reference - ref_mean) * np.sqrt(weights[:, None])
    if X.shape[1] <= max(len(X), len(Y)):
        C = Y.T @ Y
        numerator, denominator = np.linalg.norm(X.T @ X - C), np.linalg.norm(C)
    else:
        yy = np.square(Y @ Y.T).sum()
        squared = np.square(X @ X.T).sum() + yy - 2*np.square(X @ Y.T).sum()
        numerator, denominator = np.sqrt(max(0., squared)), np.sqrt(yy)
    distances = [wasserstein_distance(draws @ u, reference @ u, v_weights=weights)
                 for u in _directions(draws.shape[1])]
    return dict(mean_error_l2=float(np.linalg.norm(draws.mean(axis=0) - ref_mean)),
                relative_covariance_error=_relative(numerator, denominator), sliced_w1=float(np.mean(distances)))


def _components(array):
    a = np.asarray(array).ravel()
    return np.r_[a.real, a.imag] if np.iscomplexobj(a) else a


def predictive(problem, draws, seed):
    """Score noisy held-out observations, using clean draws for the predictive mean."""
    data_rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    if hasattr(problem, "heldout_data"):
        forward, observed = problem.predict_heldout, _components(problem.heldout_data)
        count = problem.metadata["heldout_forward_solves_per_prediction"]
        setup = 0
        design = "held_out_frequency"
    else:
        forward = problem.predict
        exact = _components(forward(problem.truth))
        observed = exact + problem.noise_std * data_rng.standard_normal(exact.shape)
        count = problem.shots * problem.frequencies_count
        setup = count
        design = "independent_replicate_same_acquisition"
    clean = np.stack([_components(forward(x)) for x in draws])
    noisy = clean + problem.noise_std * noise_rng.standard_normal(clean.shape)
    scores = interval(noisy, observed)
    return dict(design=design, coverage90=scores["coverage90"],
                interval_width90_sigma=scores["interval_width90"] / problem.noise_std,
                mean_rmse_sigma=float(np.mean((clean.mean(axis=0)-observed)**2)**.5 / problem.noise_std),
                draws=len(clean), evaluation_forward_solves=len(draws)*count + setup)

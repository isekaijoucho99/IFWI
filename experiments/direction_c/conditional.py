"""Offline audit of the Gaussian clean-space conditional for a frozen patch mixture."""

from numbers import Integral

import numpy as np
from scipy.special import logsumexp


def audit_conditional(prior, sigmas, *, draws, seed):
    """Compare exact p(x|z) to N(denoise(z), tau² I), with z=x+sigma*noise.

    The mixture has equally weighted centers and one common isotropic component
    variance. For each independent (x, noise), the log density ratio at x is an
    unbiased contribution to E_z KL(p(x|z) || q(x|z)). Levels reuse the same
    pairs, so the reported Monte Carlo standard errors are marginal per level.
    RMSE averages squared errors over draws and latent coordinates. The mean
    uses latent units, the clean-space score uses inverse latent units, and
    variance uses squared latent units. Geological patch latents are logits.
    """
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws < 2:
        raise ValueError("at least two independent pairs are required for MC standard errors")
    levels = np.asarray(sigmas, dtype=float)
    if levels.ndim != 1 or not levels.size or not np.all(np.isfinite(levels) & (levels > 0)):
        raise ValueError("sigmas must be a nonempty vector of positive finite levels")

    dimension = prior.dimension
    component_var = prior.component_std**2
    total_var = component_var + levels**2
    clean_var = component_var * levels**2 / total_var
    tether_var = np.array([prior.conditional_variance(float(sigma)) for sigma in levels])
    if not np.all(np.isfinite(tether_var) & (tether_var > 0)):
        raise ValueError("conditional variance must be positive and finite")

    log_ratios = np.empty((len(levels), draws))
    mean_errors = np.zeros(len(levels))
    score_errors = np.zeros(len(levels))
    exact_variances = np.zeros(len(levels))
    rng = np.random.default_rng(seed)
    for draw in range(draws):
        clean = prior.sample(rng)
        noise = rng.standard_normal(dimension)
        for i, sigma in enumerate(levels):
            noisy = clean + sigma * noise
            offsets = prior.centers - noisy
            log_weights = -np.einsum("ij,ij->i", offsets, offsets) / (2 * total_var[i])
            log_weights -= logsumexp(log_weights)
            weights = np.exp(log_weights)
            means = noisy + (sigma**2 / total_var[i]) * offsets
            exact_mean = weights @ means
            tether_mean = prior.denoise(noisy, float(sigma))
            mean_errors[i] += np.mean((exact_mean - tether_mean)**2)

            clean_offsets = means - clean
            tether_offset = tether_mean - clean
            log_terms = log_weights - np.einsum("ij,ij->i", clean_offsets, clean_offsets) / (2 * clean_var[i])
            log_mix = logsumexp(log_terms)
            log_ratios[i, draw] = log_mix + 0.5 * (
                dimension * np.log(tether_var[i] / clean_var[i]) + tether_offset @ tether_offset / tether_var[i]
            )
            # At (x,z), these normalized component responsibilities give the
            # exact clean-space mixture score via the component Gaussian scores.
            score = np.exp(log_terms - log_mix) @ clean_offsets / clean_var[i]
            score_errors[i] += np.mean((score - tether_offset / tether_var[i])**2)
            exact_variances[i] += clean_var[i] + weights @ np.mean((means - exact_mean)**2, axis=1)

    rows = []
    for i, sigma in enumerate(levels):
        kl = float(np.mean(log_ratios[i]))
        rows.append({
            "sigma": float(sigma), "draws": int(draws), "seed": int(seed), "dimension": int(dimension),
            "expected_kl_nats": kl,
            "kl_mc_standard_error_nats": float(np.std(log_ratios[i], ddof=1) / np.sqrt(draws)),
            "expected_kl_nats_per_dimension": kl / dimension,
            "conditional_mean_rmse": float(np.sqrt(mean_errors[i] / draws)),
            "conditional_score_rmse": float(np.sqrt(score_errors[i] / draws)),
            "exact_conditional_variance_per_coordinate": float(exact_variances[i] / draws),
            "tether_variance": float(tether_var[i]),
            "units": {"mean_rmse": "latent", "score_rmse": "inverse latent",
                      "variance": "latent squared", "kl": "nats"},
            "pde_solves": 0,
        })
    return rows

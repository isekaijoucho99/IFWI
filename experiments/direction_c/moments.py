"""Exact stagewise first and second moments in the linear Gaussian experiment."""

from itertools import product

import numpy as np
from scipy.linalg import expm


def _likelihood(problem, W):
    # Remove the N(0,I) prior from the analytic (possibly sketched) posterior.
    mu, cov = problem.posterior(W)
    precision = np.linalg.inv(cov)
    return precision - np.eye(problem.dimension), precision @ np.ravel(mu)


def _kernel(H, b, tau2, eta):
    """Augmented map of [clean, frozen center, 1] for one Euler transition."""
    d = len(b)
    transform = np.eye(2*d + 1)
    transform[:d, :d] -= eta * (H + np.eye(d) / tau2)
    transform[:d, d:2*d] = (eta / tau2) * np.eye(d)
    transform[:d, -1] = eta * b
    return transform


def _record(mean, cov):
    return {"mean": mean.tolist(), "covariance": cov.tolist()}


def _distance(law, other):
    return dict(mean_l2=float(np.linalg.norm(np.subtract(law["mean"], other["mean"]))),
                covariance_frobenius=float(np.linalg.norm(np.subtract(law["covariance"], other["covariance"]))))


def gaussian_laws(problem, stages, method, fixed_weights=None):
    """Compare laws at identical step counts and eta, including re-noising.

    Exact conditional draws and continuous-time finite-horizon Langevin use
    the full likelihood. Discrete full and requested transitions then isolate
    discretization and encoding effects, respectively. Differences between
    those laws need not be additive as scalar distances.
    """
    if method not in ("full", "fixed", "refresh") or (method == "fixed" and fixed_weights is None):
        raise ValueError("a deterministic schedule and a known encoding law are required")
    d = problem.dimension
    identity = np.eye(d)
    H, b = _likelihood(problem, None)
    mu, cov = problem.posterior()
    truth = _record(np.ravel(mu), cov)
    signs = []
    if method == "refresh":
        # Global row sign cancels in W^T W; keep one representative of each pair.
        for tail in product((-1., 1.), repeat=problem.shots-1):
            signs.append(_likelihood(problem, np.array([(1., *tail)])))
    fixed = _likelihood(problem, fixed_weights) if method == "fixed" else None

    def path(kind):
        mean = np.zeros(d)
        covariance = (1 + stages[0]["sigma"]**2) * identity
        records = []
        for i, stage in enumerate(stages):
            sigma, tau2 = stage["sigma"], stage["tau2"]
            eta, n = stage["eta"], stage["steps"]
            mean = mean / (1 + sigma**2)
            covariance = covariance / (1 + sigma**2)**2
            if kind in ("exact_conditional", "finite_time"):
                precision = H + identity / tau2
                conditional_cov = np.linalg.inv(precision)
                decay = expm(-eta*n*precision) if kind == "finite_time" else np.zeros_like(H)
                gain = (identity - decay) @ conditional_cov
                center_map = decay + gain / tau2
                mean = center_map @ mean + gain @ b
                covariance = center_map @ covariance @ center_map.T + conditional_cov @ (identity - decay @ decay.T)
            else:
                initial_mean = np.r_[mean, mean, 1.]
                second = np.outer(initial_mean, initial_mean)
                second[:-1, :-1] += np.tile(covariance, (2, 2))
                law = fixed if kind == "requested" and fixed is not None else (H, b)
                T = _kernel(*law, tau2, eta)
                deviations = ([_kernel(h, g, tau2, eta) - T for h, g in signs]
                              if kind == "requested" else [])
                for _ in range(n):
                    updated = T @ second @ T.T
                    if deviations:
                        # k independent rows average k affine maps, so their
                        # conditional second-moment correction scales as 1/k.
                        updated += sum(D @ second @ D.T for D in deviations) / (len(deviations)*stage["rows"])
                    updated[:d, :d] += 2 * eta * identity
                    second = updated
                mean = second[:d, -1]
                covariance = second[:d, :d] - np.outer(mean, mean)
            covariance = (covariance + covariance.T) / 2
            records.append(_record(mean, covariance))
            if i + 1 < len(stages):
                covariance = covariance + stages[i+1]["sigma"]**2 * identity
        return records

    paths = {name: path(name) for name in ("exact_conditional", "finite_time", "full_ula", "requested")}
    final = {name: records[-1] for name, records in paths.items()}
    return dict(exact_posterior=truth, **final,
                stages=[{name: records[i] for name, records in paths.items()} for i in range(len(stages))],
                discrepancies={
                    "initialization_vs_posterior": _distance(final["exact_conditional"], truth),
                    "finite_inner_time": _distance(final["finite_time"], final["exact_conditional"]),
                    "ula_discretization": _distance(final["full_ula"], final["finite_time"]),
                    "encoding_at_matched_schedule": _distance(final["requested"], final["full_ula"]),
                })

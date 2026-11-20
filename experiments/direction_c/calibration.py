"""Repeated-dataset calibration of the Gaussian inverse problem."""

from numbers import Integral

import numpy as np

from .sampler import sample
from .toys import gaussian_problem


def _randomized_ranks(draws, truth, rng):
    """Insert truth uniformly among any draws tied with it in each coordinate."""
    smaller = np.sum(draws < truth, axis=0)
    tied = np.sum(draws == truth, axis=0)
    return smaller + rng.integers(tied + 1)


def _observe(draws, truth, tie_seed, used):
    ranks = _randomized_ranks(draws, truth, np.random.default_rng(tie_seed))
    lower, upper = np.quantile(draws, [0.05, 0.95], axis=0)
    return ranks, (lower <= truth) & (truth <= upper), upper - lower, used


def _summarize(records, particles):
    ranks, covered, widths, used = zip(*records)
    by_dimension = np.stack(ranks).T
    return {
        "ranks_by_dimension": by_dimension.tolist(),
        "rank_histogram_by_dimension": [
            np.bincount(r, minlength=particles + 1).tolist() for r in by_dimension
        ],
        "coverage90_by_dimension": np.mean(covered, axis=0).tolist(),
        "coverage90": float(np.mean(covered)),
        "interval_width90_by_dimension": np.mean(widths, axis=0).tolist(),
        "interval_width90": float(np.mean(widths)),
        "shot_gradient_equivalents": sum(used),
    }


def calibrate(*, replicates, particles, seed, methods, sampler_kwargs, mismatch_scale=1.8):
    """Return discrete SBC ranks and empirical equal-tailed interval diagnostics.

    Within each condition every replication draws a fresh Gaussian truth and
    independent observational noise. The matched and misspecified conditions
    use common standard-normal variates. Each dataset is shared by all methods.
    The exact control draws independent samples from the full assumed posterior
    in both conditions. Its matched ranks are uniform on {0,...,particles}.
    """
    for label, value in (("particles", particles), ("replicates", replicates)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if not np.isfinite(mismatch_scale) or mismatch_scale <= 0:
        raise ValueError("mismatch_scale must be positive and finite")
    methods = list(methods)
    allowed = {"full", "fixed", "refresh", "svrg", "adaptive"}
    if len(methods) != len(set(methods)) or not set(methods) <= allowed:
        raise ValueError("expected distinct sampler methods from full, fixed, refresh, svrg, adaptive")

    scales = {"matched": 1.0, "mismatched": float(mismatch_scale)}
    results = {name: {method: [] for method in ["exact", *methods]} for name in scales}
    fixed_stream, *dataset_streams = np.random.SeedSequence(seed).spawn(replicates + 1)
    fixed_seed = int(fixed_stream.generate_state(1)[0])
    fixed_weights = None
    data_seeds = []
    generator_shots = 0

    for stream in dataset_streams:
        data_seed, path_seed, sketch_seed, exact_seed, tie_seed = map(int, stream.generate_state(5))
        data_seeds.append(data_seed)
        for name, scale in scales.items():
            rows = results[name]
            problem = gaussian_problem(seed=data_seed, truth_scale=scale)
            generator_shots += problem.shots
            mu, cov = problem.posterior()
            draws = np.random.default_rng(exact_seed).multivariate_normal(mu, cov, particles)
            rows["exact"].append(_observe(draws, problem.truth, tie_seed, used=0))
            for method in methods:
                samples, stats = sample(
                    problem, problem.prior, method=method, particles=particles,
                    seed=path_seed, encoding_seed=fixed_seed if method == "fixed" else sketch_seed,
                    **sampler_kwargs,
                )
                rows[method].append(_observe(samples, problem.truth, tie_seed, stats["used"]))
                if method == "fixed":
                    fixed_weights = stats["fixed_weights"]

    conditions = {}
    sampler_shots = 0
    for name, rows in results.items():
        summaries = {method: _summarize(records, particles) for method, records in rows.items()}
        sampler_shots += sum(row["shot_gradient_equivalents"] for row in summaries.values())
        conditions[name] = {
            "assumed_prior_scale": 1.0,
            "truth_scale": scales[name],
            "independent_datasets": int(replicates),
            "methods": summaries,
        }
    return {
        "seed": int(seed), "replicates": int(replicates), "particles": int(particles),
        "dataset_seeds": data_seeds, "fixed_encoding_seed": fixed_seed, "fixed_weights": fixed_weights,
        "data_pairing": "independent datasets within condition; common standardized truth and noise across conditions",
        "encoding_policy": "one shared W for fixed; fresh encoding streams for other methods in each replication",
        "rank_support": [0, int(particles)],
        "interval_estimator": "linear empirical 5% and 95% quantiles of each ensemble",
        "conditions": conditions,
        "work": {
            "sampler_shot_gradient_equivalents": sampler_shots,
            "dataset_forward_shot_equivalents": generator_shots,
            "pde_forward_solves": 0,
            "pde_adjoint_solves": 0,
        },
    }

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.direction_c.importance import importance_reference
from experiments.direction_c.seismic import GeologicalPatchPrior, seismic_problem


def test_stratified_draws_stable_complex_weights_and_reproducibility():
    centers = np.array([[-1.], [2.]])
    prior = GeologicalPatchPrior(centers, .5)
    problem = SimpleNamespace(dimension=1, shots=2, frequencies_count=1, noise_std=.7,
                              data=np.array([1.1 + 40j, 1.8 - 50j]),
                              predict=lambda x: np.array([2*x[0] + 1j*x[0], x[0] + 0j]))
    n = 41
    draws, weights, stats = importance_reference(problem, prior, samples_per_component=n, seed=91)
    rng = np.random.default_rng(91)
    expected_draws = np.vstack([c + .5*rng.normal(size=(n, 1)) for c in centers])
    np.testing.assert_array_equal(draws, expected_draws)
    nll = np.array([np.sum(abs(problem.predict(x) - problem.data)**2) / (2*.7**2) for x in draws])
    assert np.max(np.exp(-nll)) == 0.
    scaled = np.exp(nll.min() - nll)
    np.testing.assert_allclose(weights, scaled / scaled.sum(), rtol=1e-10)
    assert stats["ess"] == pytest.approx(1 / (weights @ weights))
    assert stats["max_weight"] == pytest.approx(weights.max())
    assert stats["forward_solves"] == 2 * len(draws)
    assert stats["adjoint_solves"] == 0 and stats["factorizations"] == len(draws)
    same_draws, same_weights, _ = importance_reference(problem, prior, samples_per_component=n, seed=91)
    np.testing.assert_array_equal(draws, same_draws)
    np.testing.assert_array_equal(weights, same_weights)


def test_constant_likelihood_and_independent_balanced_halves():
    prior = GeologicalPatchPrior(np.array([[-5.], [2.], [8.]]), .4)
    problem = SimpleNamespace(dimension=1, shots=2, frequencies_count=1, noise_std=.1,
                              data=np.zeros(2), predict=lambda x: np.zeros(2))
    draws, w, stats = importance_reference(problem, prior, samples_per_component=5, seed=14)
    np.testing.assert_allclose(w, 1/15)
    assert stats["split_half_ess"] == pytest.approx([9., 6.])
    tiled = draws.reshape(3, 5)
    expected = np.abs(tiled[:, ::2].mean() - tiled[:, 1::2].mean())
    assert stats["split_half_mean_rms"] == pytest.approx(expected)


def test_independent_native_fwi_reference_matches_full_likelihood():
    problem, prior = seismic_problem(shape=(6, 8), shots=2, frequencies=(6.,))
    before = problem.work["prediction"].copy()
    draws, weights, stats = importance_reference(problem, prior, samples_per_component=1, seed=19)
    costs = np.array([problem.value_grad(x, np.eye(2))[0] for x in draws])
    raw_weights = np.exp(costs.min() - costs)
    np.testing.assert_allclose(weights, raw_weights / raw_weights.sum(), rtol=2e-11, atol=1e-20)
    assert problem.work["prediction"]["forward_rhs"] - before["forward_rhs"] == stats["forward_solves"]
    assert stats["split_half_ess"] is None and stats["split_half_mean_rms"] is None

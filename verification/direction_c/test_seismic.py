import itertools

import numpy as np
import pytest

from experiments.direction_b.model import load_patch
from experiments.direction_b.survey import geometry, observations
from experiments.direction_c.seismic import GeologicalPatchPrior, seismic_problem


@pytest.fixture(scope="module")
def survey():
    return seismic_problem()


@pytest.mark.parametrize("dataset", ["marmousi", "overthrust"])
def test_native_forward_and_heldout_acquisition(dataset):
    problem, _ = seismic_problem(dataset=dataset, shape=(7, 10), shots=3)
    truth, _, spacing = load_patch(dataset, (7, 10))
    np.testing.assert_allclose(problem.velocity(problem.truth), truth)
    native = observations(truth, spacing, (5., 8., 10.), problem.metadata["damping"],
                          geometry(truth.shape, 3), sponge=problem.metadata["sponge"])
    native = np.stack([obs.data for obs in native]).reshape(3, 3, -1)
    np.testing.assert_allclose(problem.predict(problem.truth), native[:2], rtol=1e-12)
    np.testing.assert_allclose(problem.predict_heldout(problem.truth), native[2:], rtol=1e-12)
    assert problem.data.shape == (2, 3, 4)
    assert problem.heldout_data.shape == (1, 3, 4)
    assert problem.noise_std > 0
    assert np.any((problem.data - native[:2]).real)
    assert np.any((problem.data - native[:2]).imag)
    assert problem.work["setup"] == dict(factorizations=3, forward_rhs=9, adjoint_rhs=0)
    assert problem.work["prediction"]["forward_rhs"] == 6
    assert problem.work["heldout_prediction"]["forward_rhs"] == 3


def test_encoded_real_gradient_and_actual_rhs(survey, monkeypatch):
    import experiments.direction_c.seismic as seismic
    problem, prior = survey
    rng = np.random.default_rng(271)
    x = prior.sample(rng)
    W = np.array([[1, -1, 1, -1], [-1, -1, 1, 1]]) / np.sqrt(2.)
    residual = W @ (problem.predict(x) - problem.data)
    expected = 0.5 * np.vdot(residual, residual).real / problem.noise_std**2
    real_splu = seismic.splu
    systems = []

    def record_factor(matrix):
        factor = real_splu(matrix)

        class Recorder:
            def solve(self, rhs, trans="N"):
                systems.append((trans, rhs.copy()))
                return factor.solve(rhs, trans=trans)

        return Recorder()

    monkeypatch.setattr(seismic, "splu", record_factor)
    loss, gradient = problem.value_grad(x, W)
    assert loss == pytest.approx(expected, rel=1e-11)
    assert [trans for trans, _ in systems] == ["N", "H"] * 2
    for trans, rhs in systems:
        assert rhs.shape == (problem.dimension, W.shape[0])
        if trans == "N":
            np.testing.assert_array_equal(rhs[problem.metadata["source_indices"]], W.T)
    assert np.isrealobj(gradient) and gradient.shape == x.shape
    delta = rng.normal(size=x.size)
    delta /= np.linalg.norm(delta)
    eps = 1e-5
    fd = (problem.value_grad(x + eps * delta, W)[0]
          - problem.value_grad(x - eps * delta, W)[0]) / (2 * eps)
    assert gradient @ delta == pytest.approx(fd, rel=3e-6, abs=5e-7)


def test_rademacher_expectation_recovers_native_loss_and_gradient(survey):
    problem, prior = survey
    x = prior.sample(np.random.default_rng(9))
    signs = itertools.product([-1, 1], repeat=problem.shots)
    estimates = [problem.value_grad(x, np.asarray([row])) for row in signs]
    full_loss, full_gradient = problem.value_grad(x, np.eye(problem.shots))
    np.testing.assert_allclose(np.mean([v for v, _ in estimates]), full_loss, rtol=2e-12)
    np.testing.assert_allclose(np.mean([g for _, g in estimates], axis=0),
                               full_gradient, rtol=2e-11, atol=1e-10)


def test_frozen_split_prior_and_separate_noise_streams(survey):
    problem, prior = survey
    repeat, _ = seismic_problem(seed=3)
    other_seed, other_prior = seismic_problem(seed=7)
    _, cross_medium_prior = seismic_problem(dataset="overthrust")
    np.testing.assert_array_equal(repeat.data, problem.data)
    np.testing.assert_array_equal(repeat.heldout_data, problem.heldout_data)
    assert not np.array_equal(other_seed.data, problem.data)
    assert not np.array_equal(other_seed.heldout_data, problem.heldout_data)
    for candidate in [other_prior, cross_medium_prior]:
        np.testing.assert_array_equal(prior.sample(np.random.default_rng(12)),
                                      candidate.sample(np.random.default_rng(12)))
    tile_centers = []
    nz, nx = problem.shape
    for name, split in problem.metadata["prior"]["training_splits"].items():
        grid, _, _ = load_patch(name, tuple(split["grid_shape"]))
        lo, hi = split["excluded_x"]
        tx0, tx1 = split["test_x"]
        assert lo <= tx0 and tx1 <= hi
        assert len(split["training_origins"]) > 1
        for z0, x0 in split["training_origins"]:
            assert x0 + nx <= lo or x0 >= hi
            tile_centers.append(grid[z0:z0 + nz, x0:x0 + nx])
    np.testing.assert_allclose([problem.velocity(c) for c in prior.centers], tile_centers)


def test_geological_mixture_denoiser_is_the_gaussian_conditional_mean():
    centers = np.array([[0., 1.], [2., -1.]])
    tau, sigma = 0.4, 0.7
    y = np.array([0.8, 0.1])
    prior = GeologicalPatchPrior(centers, tau)
    weights = np.exp(-np.sum((centers - y)**2, axis=1) / (2 * (tau**2 + sigma**2)))
    means = (sigma**2 * centers + tau**2 * y) / (tau**2 + sigma**2)
    np.testing.assert_allclose(prior.denoise(y, sigma),
                               np.average(means, axis=0, weights=weights))
    np.testing.assert_array_equal(prior.denoise(y, 0.), y)
    assert 0 < prior.conditional_variance(sigma) < sigma**2

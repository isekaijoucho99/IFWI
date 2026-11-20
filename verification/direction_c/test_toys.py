import itertools
import json

import numpy as np
import pytest

from experiments.direction_c.toys import NormalPrior, gaussian_problem, layered_problem, ula_audit


def test_normal_prior_and_scalar_ula_moments():
    prior = NormalPrior(2)
    noisy = np.array([2.0, -1.0])
    np.testing.assert_allclose(prior.denoise(noisy, 2.0), noisy / 5)
    assert prior.conditional_variance(2.0) == pytest.approx(0.8)
    assert prior.sample(np.random.default_rng(0)).shape == (2,)
    audit = ula_audit()
    json.dumps(audit, allow_nan=False)
    a, h, v = audit["precision"], audit["step_size"], audit["gradient_noise_variance"]
    assert audit["finite_step_excess"] == pytest.approx(h / (2 - h*a))
    assert audit["stochastic_gradient_excess"] == pytest.approx(h * v / (a * (2 - h*a)))
    for key in ("full", "stochastic"):
        row = audit[key]
        assert abs(row["empirical_variance"] - row["finite_horizon_variance"]) < 4 * row["mc_standard_error"]
    with pytest.raises(ValueError):
        ula_audit(precision=1, step_size=2)


@pytest.mark.parametrize("factory,x", [
    (gaussian_problem, np.array([-0.3, 0.7])), (layered_problem, np.array([0.4])),
])
def test_encoded_objective_gradient_and_exact_sign_expectation(factory, x):
    problem = factory(seed=19)
    identity = np.eye(problem.shots)
    full_value, full_grad = problem.value_grad(x, identity)
    residual = problem.predict(x) - problem.data
    totals = []
    for signs in itertools.product([-1, 1], repeat=problem.shots):
        w = np.array([signs])
        value, grad = problem.value_grad(x, w)
        expected = np.sum((w @ residual)**2) / (2 * problem.noise_std**2)
        assert value == pytest.approx(expected)
        direction = np.arange(1, problem.dimension + 1)
        delta = 1e-6 * direction
        difference = (problem.value_grad(x+delta, w)[0] - problem.value_grad(x-delta, w)[0]) / 2e-6
        assert grad @ direction == pytest.approx(difference, abs=1e-7, rel=1e-7)
        totals.append([value, *grad])
    np.testing.assert_allclose(np.mean(totals, axis=0), [full_value, *full_grad])
    # Concatenating two normalized rows averages the corresponding singleton objectives.
    w = np.array([[1, 1, 1, 1], [1, -1, 1, -1]])
    expected = np.mean([np.r_[v, g] for v, g in
                        [problem.value_grad(x, row[None]) for row in w]], axis=0)
    value, grad = problem.value_grad(x, w / np.sqrt(2))
    np.testing.assert_allclose(np.r_[value, grad], expected)


def test_gaussian_posterior_normal_equations_and_repeated_dataset_generator():
    problem = gaussian_problem(seed=12)
    for w in [np.eye(problem.shots), np.array([[1., -1., -1., 1.]])]:
        mean, cov = problem.posterior(w)
        _, gradient = problem.value_grad(mean, w)
        np.testing.assert_allclose(gradient + mean, 0, atol=1e-14)
        dx = np.array([0.7, -0.5])
        _, perturbed = problem.value_grad(mean + dx, w)
        np.testing.assert_allclose(cov @ (perturbed - gradient + dx), dx)
    full = problem.posterior()[1]
    fixed = problem.posterior(w)[1]
    assert not np.allclose(full, fixed)
    assert abs(full[0, 1]) > 0.01
    assert max(np.linalg.eigvalsh(fixed)) == pytest.approx(1.0)
    shifted = gaussian_problem(seed=12, truth_scale=2.0)
    np.testing.assert_allclose(shifted.truth, 2 * problem.truth)
    np.testing.assert_allclose(shifted.data - shifted.predict(shifted.truth),
                               problem.data - problem.predict(problem.truth))
    assert not np.array_equal(gaussian_problem(seed=13).truth, problem.truth)


def test_layered_reference_resolves_phase_aliases_and_mass():
    problem = layered_problem()
    grid, weights, meta = problem.reference()
    assert grid.shape == (len(weights), 1)
    assert np.all(weights >= 0)
    assert sum(weights) == pytest.approx(1)
    edges = meta["mode_edges"]
    mass = np.bincount(np.searchsorted(edges, grid[:, 0], side="right"), weights)
    np.testing.assert_allclose(mass, meta["mode_probabilities"])
    assert sum(mass > 0.10) >= 2
    assert meta["max_mode_mass_refinement_difference"] < 1e-4
    assert meta["posterior_tail_mass_upper_bound"] < 1e-8
    for v in [-1.9, -0.1, 1.8]:
        i = np.argmin(abs(grid[:, 0] - v))
        j = len(weights) // 2
        loss_i, _ = problem.value_grad(grid[i], np.eye(problem.shots))
        loss_j, _ = problem.value_grad(grid[j], np.eye(problem.shots))
        log_ratio = loss_j - loss_i + 0.5 * (grid[j, 0]**2 - grid[i, 0]**2)
        assert np.log(weights[i] / weights[j]) == pytest.approx(log_ratio)

import json

import numpy as np
import pytest


class Prior:
    def __init__(self, dimension):
        self.dimension = dimension
        self.centers = []

    def sample(self, rng):
        return np.zeros(self.dimension)

    def denoise(self, noisy, sigma):
        self.centers.append(noisy.copy())
        return noisy.copy()

    def conditional_variance(self, sigma):
        return sigma**2


class Likelihood:
    """Only shot 0 depends on x, so all normalized sketches share its Hessian."""

    shots = 3
    frequencies_count = 2
    dimension = 1

    def __init__(self, target_scale=1.0):
        self.data = np.array([1.0, -0.3, 2.0]) * target_scale
        self.calls = []

    def value_grad(self, x, weights):
        residual = -self.data.copy()
        residual[0] += x[0]
        encoded = weights @ residual
        gradient = np.array([weights[:, 0] @ encoded])
        self.calls.append((x.copy(), weights.copy(), gradient.copy()))
        return 0.5 * (encoded @ encoded), gradient


class Flat:
    shots = 3
    dimension = 8192
    frequencies_count = 1

    def value_grad(self, x, weights):
        return 0.0, np.zeros_like(x)


def draw(problem, prior, method, **overrides):
    from experiments.direction_c.sampler import sample

    options = dict(particles=2, sigmas=[1.0, 0.4], budget=11, seed=5, encoding_seed=7)
    return sample(problem, prior, method=method, **(options | overrides))


@pytest.mark.parametrize("method", ["full", "fixed", "refresh", "svrg", "adaptive"])
def test_records_every_call_in_gradient_equivalents_and_frequency_solves(method):
    model = Likelihood()
    x, report = draw(model, Prior(1), method)
    assert x.shape == (2, 1)
    json.dumps(report, allow_nan=False)
    _, replay = draw(Likelihood(), Prior(1), method)
    assert replay == report
    assert report["budget"] == 44
    work = sum(len(w) for _, w, _ in model.calls)
    assert report["used"] == work
    assert report["forward_solves"] == report["adjoint_solves"] == 2 * work
    assert report["unused"] == 44 - work
    assert report["steps"] == sum(r["steps"] for r in report["stages"])
    for r in report["stages"]:
        assert 0 < r["used"] <= 11
        assert sum(r["costs"].values()) == r["used"]
        assert r["unused"] == 11 - r["used"]
        assert r["rejected_steps"] == 0


def test_fixed_weights_shared_across_particles_stages_steps_and_replays():
    problem = Likelihood()
    first, report = draw(problem, Prior(1), "fixed", encoded_rows=2)
    second, _ = draw(Likelihood(), Prior(1), "fixed", encoded_rows=2)
    np.testing.assert_array_equal(first, second)
    fixed = np.asarray(report["fixed_weights"])
    assert fixed.shape == (2, 3)
    np.testing.assert_allclose(np.abs(fixed), 1 / np.sqrt(2))
    assert all(np.array_equal(w, fixed) for _, w, _ in problem.calls)
    problem = Likelihood()
    draw(problem, Prior(1), "refresh", encoded_rows=2)
    assert len({w.tobytes() for _, w, _ in problem.calls}) > 2
    assert all(np.allclose(np.abs(w), 1 / np.sqrt(2)) for _, w, _ in problem.calls)


def test_one_step_uses_the_raw_normalized_encoded_likelihood_gradient():
    opts = dict(sigmas=[0.5], inner_steps=1, particles=1, encoded_rows=2)
    model = Likelihood()
    actual, report = draw(model, Prior(1), "fixed", **opts)
    # Use a null likelihood with matching dimension for the coupled Langevin noise.
    flat = Flat()
    flat.dimension = 1
    baseline, _ = draw(flat, Prior(1), "fixed", **opts)
    center, weights, _ = model.calls[0]
    residual = np.array([center.item() - 1, .3, -2])
    expected_gradient = (weights.T @ weights @ residual)[0]
    np.testing.assert_allclose(baseline - actual, .03 * expected_gradient, atol=1e-14)
    assert report["stages"][0]["eta"] == pytest.approx(.03)


def test_tether_drift_starts_at_denoised_center_and_restores_toward_it():
    class ConstantCenter(Prior):
        def denoise(self, noisy, sigma):
            return np.full_like(noisy, 2.5)

    opts = dict(sigmas=[.5], inner_steps=2, particles=1)
    first = Likelihood(target_scale=0)
    zero, _ = draw(first, ConstantCenter(1), "full", **opts)
    shifted, _ = draw(Likelihood(), ConstantCenter(1), "full", **opts)
    assert first.calls[0][0].item() == 2.5
    # For a unit target shift, the coupled difference after two transitions
    # is eta * (2 - eta * (1 + 1/tau2)), with eta=.03, tau2=.25.
    np.testing.assert_allclose(shifted - zero, .0555)


def test_normalized_independent_rows_recover_the_full_gradient_in_mean():
    model = Likelihood()
    opts = dict(sigmas=[1.0], inner_steps=1, particles=1, budget=8192, encoded_rows=8192)
    draw(model, Prior(1), "fixed", **opts)
    center, w, gradient = model.calls[0]
    np.testing.assert_allclose(w.T @ w, np.eye(3), atol=.04)
    np.testing.assert_allclose(gradient, center - 1, atol=.08)


def test_svrg_reuses_first_anchor_and_uses_paired_weights_afterward():
    opts = dict(sigmas=[.6], particles=1, budget=20, inner_steps=4, encoded_rows=2)
    full, _ = draw(Likelihood(), Prior(1), "full", **opts)
    model = Likelihood()
    controlled, report = draw(model, Prior(1), "svrg", **opts)
    np.testing.assert_allclose(controlled, full, atol=1e-14)
    r = report["stages"][0]
    assert r["costs"] == dict(anchor=3, pilot=0, current=6, control=6)
    assert r["used"] == 15 and r["unused"] == 5 and r["steps"] == 4
    center, identity, _ = model.calls[0]
    np.testing.assert_array_equal(identity, np.eye(3))
    for current, control in zip(model.calls[1::2], model.calls[2::2]):
        np.testing.assert_array_equal(current[1], control[1])
        np.testing.assert_array_equal(control[0], center)


@pytest.mark.parametrize("budget, expected", [(22, "rademacher"), (26, "identity")])
def test_adaptive_counts_independent_pilots_and_limits_directional_diffusion(budget, expected):
    model = Likelihood(target_scale=100)
    _, report = draw(model, Prior(1), "adaptive", probes=20, sigmas=[1.0],
                     particles=1, inner_steps=2, budget=budget, noise_ratio=.2)
    r = report["stages"][0]
    assert r["costs"]["pilot"] == 20 and r["operator"] == expected
    pilot = np.array([g.item() for _, _, g in model.calls[:20]])
    assert all(np.array_equal(x, model.calls[0][0]) for x, _, _ in model.calls[:20])
    assert r["pilot_trace"] == pytest.approx(np.var(pilot, ddof=1))
    assert r["pilot_lambda_max"] == pytest.approx(r["pilot_trace"])
    assert r["pilot_lambda_max"] > 100
    assert r["steps"] == 2
    if expected == "identity":
        assert r["rows"] == 3 and r["eta"] == pytest.approx(.12)
        assert all(np.array_equal(w, np.eye(3)) for _, w, _ in model.calls[20:])
    else:
        assert r["rows"] == 1
        assert 0 < r["eta"] * r["pilot_lambda_max"] / 2 <= .20000001
    assert r["used"] == budget


def test_adaptive_leases_a_wider_sketch_using_leading_covariance_eigenvalue():
    class Survey:
        shots = 5
        dimension = 2
        frequencies_count = 1

        def __init__(self):
            self.grads = []

        def value_grad(self, x, w):
            residual = w @ np.r_[x, [1., 2., 3.]]
            grad = w[:, :2].T @ residual
            self.grads.append(grad)
            return .5 * residual @ residual, grad

    model = Survey()
    opts = dict(probes=24, budget=48, inner_steps=2, sigmas=[1.0], particles=1)
    draw(model, Prior(2), "adaptive", **opts)
    empirical = np.cov(np.array(model.grads[:24]).T)
    largest = np.linalg.eigvalsh(empirical)[-1]
    _, report = draw(Survey(), Prior(2), "adaptive", noise_ratio=.12 * largest / 3, **opts)
    r = report["stages"][0]
    assert r["pilot_lambda_max"] == pytest.approx(largest)
    assert r["pilot_trace"] == pytest.approx(np.trace(empirical))
    assert r["pilot_trace"] > r["pilot_lambda_max"]
    assert r["rows"] == 2 and r["eta"] == pytest.approx(.12)
    assert r["used"] == 28 and r["operator"] == "rademacher"


@pytest.mark.parametrize("method,cap", [("full", 5), ("fixed", 3), ("refresh", 3),
                                       ("svrg", 6), ("adaptive", 7)])
def test_equal_transition_request_fails_before_spending_an_insufficient_cap(method, cap):
    model = Likelihood()
    with pytest.raises(ValueError, match="budget"):
        draw(model, Prior(1), method, encoded_rows=2, probes=4, inner_steps=2, budget=cap)
    assert not model.calls


def test_independent_initial_renoise_and_ula_scales_with_coupled_method_noise():
    model = Flat()
    prior = Prior(model.dimension)
    opts = dict(inner_steps=1, particles=1, budget=5)
    initial_stage, _ = draw(model, Prior(model.dimension), "full", sigmas=[1.0], **opts)
    final, _ = draw(model, prior, "full", sigmas=[1.0, .4], **opts)
    matched, _ = draw(model, Prior(model.dimension), "refresh", sigmas=[1.0, .4], **opts)
    np.testing.assert_array_equal(final, matched)
    initial, next_center = prior.centers
    draws = np.array([initial, (initial_stage[0] - initial) / np.sqrt(.24),
                      (next_center - initial_stage[0]) / .4,
                      (final[0] - next_center) / np.sqrt(.0384)])
    np.testing.assert_allclose(np.cov(draws), np.eye(4), atol=.055)


def test_nonfinite_gradient_reports_failed_work():
    model = Likelihood(target_scale=np.nan)
    with pytest.raises(FloatingPointError, match="nonfinite") as error:
        draw(model, Prior(1), "full")
    assert error.value.stage["used"] == 3
    assert error.value.stage["rejected_steps"] == 1


def test_failed_later_stage_keeps_work_already_spent_and_marks_unspent_budget():
    class FailsOnThird(Likelihood):
        def value_grad(self, x, w):
            value, gradient = super().value_grad(x, w)
            return (np.nan if len(self.calls) == 3 else value), gradient

    with pytest.raises(FloatingPointError) as error:
        draw(FailsOnThird(), Prior(1), "full", inner_steps=2, budget=7)
    partial = error.value.stats
    assert partial["used"] == 9 and partial["unused"] == 19
    assert partial["steps"] == 2 and partial["rejected_steps"] == 1
    assert partial["forward_solves"] == partial["adjoint_solves"] == 18
    assert [r["used"] for r in partial["stages"]] == [6, 3]

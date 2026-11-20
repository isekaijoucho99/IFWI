import numpy as np

from experiments.direction_c.moments import gaussian_laws


class Toy:
    design = np.array([[1.], [2.]])
    data = np.array([[.5], [-.8]])
    shots, dimension, noise_std = 2, 1, 1.

    def posterior(self, W=None):
        W = np.eye(2) if W is None else W
        design = W @ self.design
        cov = np.linalg.inv(np.eye(1) + design.T @ design)
        return (cov @ design.T @ (W @ self.data)).ravel(), cov


def test_one_step_exact_sketch_second_moment_and_width_scaling():
    problem = Toy()
    stage = [dict(sigma=1., tau2=.5, eta=.02, steps=1, rows=1)]
    result = gaussian_laws(problem, stage, "refresh")
    H, b = 5., -1.1
    # Center has variance 1/2; tether vanishes at initialization.
    assert np.isclose(result["full_ula"]["mean"][0], .02 * b)
    var_full = .5 * (1 - .02 * H)**2 + .04
    assert np.isclose(result["full_ula"]["covariance"][0][0], var_full)
    # With two signs the cross-Hessian fluctuates by +/-4,
    # and the right-hand side by +/-(2*.5-.8).
    excess = .02**2 * (16 * .5 + .2**2)
    assert np.isclose(result["requested"]["covariance"][0][0], var_full + excess)
    stage[0]["rows"] = 2
    wide = gaussian_laws(problem, stage, "refresh")
    assert np.isclose(wide["requested"]["covariance"][0][0], var_full + excess / 2)


def test_fixed_encoding_uses_one_target_over_all_stages():
    prob = Toy()
    stages = [dict(sigma=s, tau2=s*s/(1+s*s), eta=.02, steps=5, rows=1) for s in (2., 1.)]
    # This signed encoding has a different mode and curvature from the full target.
    fixed = gaussian_laws(prob, stages, "fixed", np.array([[1., -1.]]))
    assert fixed["requested"]["mean"] != fixed["full_ula"]["mean"]
    assert len(fixed["stages"]) == 2
    np.testing.assert_allclose(fixed["exact_posterior"]["covariance"], [[1/6]])

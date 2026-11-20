import numpy as np
import pytest
from experiments.direction_c.scores import interval, mode_score, sample_distance, predictive


def test_same_ensemble_and_modes():
    draws = np.array([[-1.], [-.5], [1.], [2.]])
    stats = sample_distance(draws, draws[::-1])
    assert stats["sliced_w1"] == pytest.approx(0.)
    assert stats["relative_covariance_error"] == pytest.approx(0., abs=1e-12)
    modes = mode_score(draws, draws, np.array([.1, .1, .4, .4]), [0.])
    assert modes["sample_mass"] == [.5, .5]
    assert modes["total_variation"] == pytest.approx(.3)
    assert interval(draws, np.array([10.]))["coverage90"] == 0.


def test_high_dimensional_covariance_score_matches_explicit_covariance():
    rng = np.random.default_rng(8)
    x, y = rng.normal(size=(8, 60)), rng.normal(size=(11, 60))
    cx, cy = np.cov(x.T), np.cov(y.T)
    expected = np.linalg.norm(cx - cy) / np.linalg.norm(cy)
    assert sample_distance(x, y)["relative_covariance_error"] == pytest.approx(expected)


def test_distribution_scores_use_reference_weights():
    x = np.array([[0.], [2.]])
    y = np.array([[0.], [4.]])
    # The reference assigns 3/4 to 0 and 1/4 to 4: mean 1, variance 3.
    scores = sample_distance(x, y, weights=[.75, .25])
    assert scores["mean_error_l2"] == pytest.approx(0.)
    assert scores["relative_covariance_error"] == pytest.approx(1 / 3)
    assert scores["sliced_w1"] == pytest.approx(1.)


def test_complex_predictive_interval_includes_measurement_noise_and_cost():
    class Problem:
        shots, frequencies_count, noise_std = 3, 2, 0.25
        heldout_data = np.zeros((1, 3, 2), dtype=complex)
        metadata = {"heldout_forward_solves_per_prediction": 3}
        def predict_heldout(self, x):
            return np.zeros_like(self.heldout_data)
    row = predictive(Problem(), np.ones((24, 2)), seed=11)
    assert row["evaluation_forward_solves"] == 24*3
    assert row["interval_width90_sigma"] > 1.
    assert row["mean_rmse_sigma"] == 0.

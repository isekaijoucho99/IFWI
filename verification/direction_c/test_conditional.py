import json

import numpy as np
import pytest

from experiments.direction_c.conditional import audit_conditional
from experiments.direction_c.seismic import GeologicalPatchPrior


@pytest.mark.parametrize("count", [1, 5])
def test_identical_centers_make_clean_space_tether_exact(count):
    centers = np.tile([1., -0.7, 2., 0.1], (count, 1))
    prior = GeologicalPatchPrior(centers, .55)
    rows = audit_conditional(prior, [2., .2], draws=72, seed=77)
    json.dumps(rows, allow_nan=False)
    for row in rows:
        assert abs(row["expected_kl_nats"]) < 3e-14
        assert row["kl_mc_standard_error_nats"] < 2e-14
        assert row["conditional_mean_rmse"] < 2e-14
        assert row["conditional_score_rmse"] < 2e-13
        assert row["exact_conditional_variance_per_coordinate"] == pytest.approx(row["tether_variance"])


def test_nontrivial_mixture_and_shared_corruption_reproducibility():
    centers = np.array([[-2., 1.], [2., -1.]])
    prior = GeologicalPatchPrior(centers, .3)
    rows = audit_conditional(prior, [3., .01], draws=256, seed=77)
    assert rows == audit_conditional(prior, [3., .01], draws=256, seed=77)
    assert rows[0]["expected_kl_nats"] > 4 * rows[0]["kl_mc_standard_error_nats"]
    assert rows[0]["conditional_mean_rmse"] < 4e-15
    assert rows[0]["conditional_score_rmse"] > 1.
    assert rows[1]["expected_kl_nats"] < .01


def test_known_gaussian_kl_with_wrong_conditional_variance():
    class InflatedTether(GeologicalPatchPrior):
        def conditional_variance(self, sigma):
            return 2.5 * super().conditional_variance(sigma)

    prior = InflatedTether(np.array([[0., -1.]]), .4)
    row, = audit_conditional(prior, [1.], draws=3000, seed=487)
    exact = prior.dimension * 0.5 * (1/2.5 - 1 + np.log(2.5))
    assert abs(row["expected_kl_nats"] - exact) < 4 * row["kl_mc_standard_error_nats"]
    assert row["expected_kl_nats_per_dimension"] == row["expected_kl_nats"] / prior.dimension

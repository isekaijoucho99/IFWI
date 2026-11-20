import json

import numpy as np

from experiments.direction_c.calibration import calibrate, _randomized_ranks
from experiments.direction_c.toys import gaussian_problem


def test_tied_ranks_use_all_truth_positions_within_the_tied_block():
    rng = np.random.default_rng(32)
    draws = np.array([[0., -1.], [0., 0.], [0., 0.], [1., 1.]])
    ranks = np.array([_randomized_ranks(draws, np.zeros(2), rng) for _ in range(80)])
    assert set(ranks[:, 0]) == {0, 1, 2, 3}
    assert set(ranks[:, 1]) == {1, 2, 3}


def test_calibration_ranks_noise_predictions_and_actual_work():
    config = dict(replicates=3, particles=6, seed=3,
                  methods=["full", "fixed", "refresh", "svrg", "adaptive"],
                  sampler_kwargs=dict(sigmas=[1.], budget=6, encoded_rows=1, step_fraction=.05,
                                      noise_ratio=.2, probes=2, inner_steps=1))
    result = calibrate(**config)
    assert result == calibrate(**config)
    json.dumps(result, allow_nan=False)
    assert len(result["fixed_weights"]) == 1
    total = 0
    for condition in result["conditions"].values():
        assert condition["independent_datasets"] == config["replicates"]
        for row in condition["methods"].values():
            for ranks, counts in zip(row["ranks_by_dimension"], row["rank_histogram_by_dimension"]):
                assert len(ranks) == len(result["dataset_seeds"])
                assert len(counts) == config["particles"] + 1
                np.testing.assert_array_equal(np.bincount(ranks, minlength=len(counts)), counts)
            total += row["shot_gradient_equivalents"]
        rows = condition["methods"]
        assert rows["exact"]["shot_gradient_equivalents"] == 0
        assert rows["full"]["shot_gradient_equivalents"] == 3 * 6 * 4
        assert rows["fixed"]["shot_gradient_equivalents"] == 3 * 6
        assert rows["adaptive"]["shot_gradient_equivalents"] >= 3 * 6 * 3
    noise = []
    for seed in result["dataset_seeds"]:
        matched = gaussian_problem(seed=seed)
        mismatched = gaussian_problem(seed=seed, truth_scale=result["conditions"]["mismatched"]["truth_scale"])
        noise.append(matched.data - matched.predict(matched.truth))
        np.testing.assert_allclose(mismatched.data - mismatched.predict(mismatched.truth), noise[-1])
    assert np.all(np.linalg.norm(np.diff(np.stack(noise), axis=0), axis=(1, 2)) > 0)
    assert result["work"]["sampler_shot_gradient_equivalents"] == total
    assert result["work"]["dataset_forward_shot_equivalents"] == 3 * 2 * 4
    assert result["work"]["pde_forward_solves"] == result["work"]["pde_adjoint_solves"] == 0


def test_exact_posterior_ranks_respond_to_generative_mismatch():
    result = calibrate(replicates=400, particles=35, seed=71, methods=[], sampler_kwargs={},
                       mismatch_scale=5.0)
    matched = result["conditions"]["matched"]["methods"]["exact"]
    mismatched = result["conditions"]["mismatched"]["methods"]["exact"]
    assert .80 < matched["coverage90"] < .93
    assert matched["coverage90"] - mismatched["coverage90"] > .15
    def edge_fraction(row):
        counts = np.asarray(row["rank_histogram_by_dimension"])
        return counts[:, [0, -1]].sum() / counts.sum()
    assert edge_fraction(mismatched) > 2 * edge_fraction(matched)

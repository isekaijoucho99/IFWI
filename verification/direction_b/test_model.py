import copy

import numpy as np
import pytest
import torch

from experiments.direction_b import model as api


@pytest.mark.parametrize(
    ("dataset", "spacing", "truth_corner", "initial_corner"),
    [("marmousi", 15., 2795., 2718.2), ("overthrust", 20., 3628., 4142.148)],
)
def test_centered_patch_in_meters_per_second(dataset, spacing, truth_corner, initial_corner):
    truth, initial, grid_spacing = api.load_patch(dataset, (9, 13))
    assert truth.shape == initial.shape == (9, 13)
    assert grid_spacing == spacing
    assert truth[0, 0] == pytest.approx(truth_corner, abs=.001)
    assert initial[0, 0] == pytest.approx(initial_corner, abs=.001)
    assert np.std(truth) > np.std(initial) > 0.


def test_velocity_vjp_jvp_and_parameter_update():
    rng = np.random.default_rng(11)
    initial = rng.uniform(2000., 4000., (5, 7))
    model = api.make_model(initial, 15., 9)
    np.testing.assert_allclose(api.slowness(model), initial ** -2, rtol=1e-14)
    weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()
    assert weights.dtype == np.float64

    # Move away from the zero-head start to exercise derivatives in every layer.
    api.step(model, rng.normal(size=weights.size), .004)
    direction = rng.normal(size=weights.size)
    direction /= np.linalg.norm(direction)
    field = rng.normal(size=initial.shape)
    jvp = api.tangent(model, direction)
    vjp = api.parameter_gradient(model, field)
    assert jvp.shape == initial.shape and vjp.shape == direction.shape
    assert vjp[0] != 0.
    np.testing.assert_allclose(np.vdot(jvp, field), np.vdot(direction, vjp), rtol=1e-12)

    delta = 1e-5
    positive, negative = copy.deepcopy(model), copy.deepcopy(model)
    api.step(positive, direction, delta)
    api.step(negative, direction, -delta)
    fd = (api.slowness(positive) - api.slowness(negative)) / (2. * delta)
    np.testing.assert_allclose(jvp, fd, atol=1e-17, rtol=1e-7)


def test_invalid_steps_leave_all_weights_untouched():
    model = api.make_model(np.full((4, 6), 2100.), 16., 13)
    weights = torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy().copy()
    bad = np.zeros_like(weights)
    bad[0] = .25
    bad[-1] = -3.  # negative velocity via the additive final bias in km/s
    for direction, alpha in ((bad, 1.), (weights[:-1], .1), (weights, np.nan)):
        with pytest.raises(ValueError):
            api.step(model, direction, alpha)
        np.testing.assert_array_equal(
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy(), weights)
    with pytest.raises(ValueError):
        api.tangent(model, weights[None])
    with pytest.raises(ValueError):
        api.parameter_gradient(model, np.ones((6, 4)))


@pytest.mark.parametrize("shape", [(95, 13), (9, 0), (5.5, 9), (4,)])
def test_rejects_missing_or_truncated_cells(shape):
    with pytest.raises(ValueError):
        api.load_patch("overthrust", shape)

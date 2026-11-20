"""Checks against distributed velocities and actual finite-difference shot records."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


DATA = Path(__file__).resolve().parents[2] / "data"


def physics():
    return importlib.import_module("experiments.direction_a.physics")


@pytest.mark.parametrize("dataset,spacing,nx", [("marmousi", 15, 288), ("overthrust", 20, 401)])
def test_field_and_background_are_centered_on_native_inversion_grid(dataset, spacing, nx):
    if dataset == "marmousi":
        source = [np.loadtxt(DATA / name, delimiter=",")[1::4, ::4].astype(np.float32)
                  for name in ("vel_marmousi_376x1151.csv", "vel_marmousi_smooth400_376x1151.csv")]
    else:
        with np.load(DATA / "overthrust2d.npz") as archive:
            source = [1000 * archive[name] for name in ("vp", "vi")]
    full = physics().load_patch(dataset, None)
    patch = physics().load_patch(dataset, (11, 14))
    assert full[2] == patch[2] == spacing
    for whole, cropped, reference in zip(full[:2], patch[:2], source):
        assert whole.shape == (94, nx)
        assert cropped.dtype == np.float32 and cropped.flags.c_contiguous
        np.testing.assert_array_equal(whole, reference)
        start_x = (nx - 14) // 2
        np.testing.assert_array_equal(cropped, reference[41:52, start_x:start_x + 14])
    with pytest.raises(ValueError):
        physics().load_patch(dataset, (95, 14))


@pytest.fixture(scope="module")
def real_problem():
    truth = torch.linspace(2600, 3400, 12)[:, None] + torch.linspace(-150, 150, 15)[None, :]
    args = dict(shots=3, nt=90, frequency=15.0)
    full = physics().Survey(truth.clone(), 15, **args)
    batched = physics().Survey(truth.clone(), 15, shot_batch=2, **args)
    return truth, full, batched


def test_real_waveform_and_full_objective_match_across_unequal_shot_batches(real_problem):
    truth, full, batched = real_problem
    v = (0.94 * truth).double()
    assert full.shots == full.setup_forwards == batched.setup_forwards == 3
    assert full.observed.shape == (1, 3, 90, 15)
    assert full.observed.abs().max().item() > .1 and not full.observed.requires_grad
    assert 1 / 15 < (90 - 1) * full.dt
    assert full.evaluate(truth) == (0., None)
    loss, grad = full.evaluate(v, gradient=True)
    loss_batched, grad_batched = batched.evaluate(v, gradient=True)
    assert loss > 0 and grad.norm() > 0
    assert grad.dtype == v.dtype and grad.shape == v.shape and grad.device == v.device
    assert loss == pytest.approx(full.evaluate(v)[0], rel=1e-6)
    assert loss == pytest.approx(loss_batched, rel=3e-5)
    torch.testing.assert_close(grad, grad_batched, rtol=4e-5, atol=2e-9)


def test_physical_gradient_predicts_change_in_waveform_loss_away_from_initial_model(real_problem):
    truth, full, _ = real_problem
    state = 1.03 * truth
    _, grad = full.evaluate(state, gradient=True)
    direction = grad / grad.abs().max()
    eps_mps = 1.0
    plus, _ = full.evaluate(state + eps_mps * direction)
    minus, _ = full.evaluate(state - eps_mps * direction)
    centered_derivative = (plus - minus) / (2 * eps_mps)
    assert centered_derivative == pytest.approx(float((grad * direction).sum()), rel=6e-3)


def test_live_cfl_and_bad_fields_are_rejected_before_propagation(real_problem):
    truth, full, _ = real_problem
    assert full.is_valid(truth)
    for invalid in (truth[1:], truth * 0, truth * float("nan"), truth * float("inf"),
                    torch.full_like(truth, 7100.0)):
        assert not full.is_valid(invalid)
        with pytest.raises(ValueError):
            full.evaluate(invalid)

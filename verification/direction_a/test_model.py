"""Operator checks at a nonlinear checkpoint, in physical velocity units."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def api():
    from experiments.direction_a import model
    return model


def background():
    return torch.linspace(1900.01234567, 2400.07654321, 35, dtype=torch.float64).reshape(5, 7)


def checkpoint(representation):
    model = api().make_model(background(), 17.5, 8, 3, representation)
    with torch.no_grad():
        head = model.net.linear[-1]
        head.weight.copy_(torch.linspace(-0.04, 0.02, 8))
        head.bias.fill_(0.027)
    return model


@pytest.mark.parametrize("width", [3, 8, 17, 32])
def test_fr_seed_matches_the_plain_initial_weights_up_to_the_endpoint_tie(width):
    fr = api().make_model(background(), 17.5, width, 42, "fr")
    plain = api().make_model(background(), 17.5, width, 42, "siren")
    assert torch.equal(api().velocity(fr), background())
    assert torch.equal(api().velocity(plain), background())
    hidden = fr.net.linear[1]
    expected = plain.net.linear[1].weight.detach().clone()
    average = (expected[:, 0] + expected[:, -1]) / 2
    expected[:, 0] = expected[:, -1] = average
    assert hidden.rank() == width - 1
    torch.testing.assert_close(hidden.weight, expected, atol=2e-7, rtol=1e-5)
    assert torch.equal(fr.net.linear[0].weight, plain.net.linear[0].weight)


@pytest.mark.parametrize("representation,target", [("siren", None), ("fr", None), ("fr", "siren")])
def test_head_compensation_matches_nonzero_function_and_clones_parameters(representation, target):
    model = checkpoint(representation)
    with torch.no_grad():
        model.net.linear[1].bias.add_(0.13)
    expected = api().velocity(model)
    assert (expected - background()).abs().max() > 1.
    clone = api().matched_copy(model, 3.7, representation=target)
    torch.testing.assert_close(api().velocity(clone), expected, rtol=0, atol=3e-12)
    assert clone.std_kmps == model.std_kmps * 3.7
    assert clone.fourier_layers == ([1] if (target or representation) == "fr" else [])
    assert all(p.dtype == torch.float64 for p in clone.parameters())
    assert clone.net.linear[-1].weight.data_ptr() != model.net.linear[-1].weight.data_ptr()


@pytest.mark.parametrize("representation", ["siren", "fr"])
def test_jvp_pullback_and_trial_at_nonlinear_checkpoint(representation):
    model = checkpoint(representation)
    rng = torch.Generator().manual_seed(11)
    params = list(model.parameters())
    original = [p.detach().clone() for p in params]
    direction = torch.randn(sum(p.numel() for p in params), generator=rng, dtype=torch.float64)
    direction = direction / direction.norm()
    image_gradient = torch.randn(background().shape, generator=rng, dtype=torch.float64)

    projected = api().jvp(model, direction)
    pulled_back = api().pullback(model, image_gradient)
    torch.testing.assert_close((projected * image_gradient).sum(), pulled_back @ direction)
    eps = 1e-5
    trial = api().trial_velocity(model, eps * direction)
    opposite = api().trial_velocity(model, -eps * direction)
    finite_difference = (trial - opposite) / (2 * eps)
    torch.testing.assert_close(projected, finite_difference, atol=8e-7, rtol=3e-8)
    assert all(torch.equal(old, p) and p.grad is None for old, p in zip(original, params))
    assert all(not result.requires_grad for result in (projected, pulled_back, trial, opposite))

    api().apply_delta(model, eps * direction)
    assert torch.equal(api().velocity(model), trial)

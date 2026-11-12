"""CPU-only contracts for the user-specified two-SIREN adaptation.

Only test_branch_gradients... calls backward, once on three tiny independent
models. No PDE solver, GPU, optimizer step, or physical training is used.
"""
import copy
import io
import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from experiments.progressive_siren import ProgressiveSirenVelocity
except ModuleNotFoundError as error:
    if error.name != "experiments.progressive_siren":
        raise
    ProgressiveSirenVelocity = None

from experiments.residual_ifwi_experiment import VelocityParameterization


@pytest.fixture
def constructor():
    return ProgressiveSirenVelocity


def background():
    return torch.arange(12, dtype=torch.float32).reshape(3, 4) * 10 + 2500


def model(constructor, **options):
    assert constructor is not None, "progressive SIREN module is not implemented"
    kwargs = dict(initial_mps=background(), dx_m=15., neurons=[2, 4, 4, 1])
    kwargs.update(options)
    return constructor(**kwargs)


def test_initial_velocity_and_both_zero_heads(constructor):
    velocity = model(constructor)
    for branch in (velocity.low, velocity.high):
        assert torch.count_nonzero(branch.net.linear[-1].weight) == 0
        assert torch.count_nonzero(branch.net.linear[-1].bias) == 0
    for completed in (0, 100, 200, 1000):
        velocity.set_progress(completed)
        assert torch.equal(velocity(), background()[None])


def test_cosine_schedule_boundaries(constructor):
    velocity = model(constructor)
    for completed in (0, 50, 100, 199, 200, 1000):
        velocity.set_progress(completed)
        expected = .5 * (1 - math.cos(math.pi * min(completed / 200, 1)))
        assert float(velocity.alpha) == pytest.approx(expected, abs=1e-7)
    velocity.set_progress(0)
    assert float(velocity.alpha) == 0.
    velocity.set_progress(200)
    assert float(velocity.alpha) == 1.


def test_formula_adds_one_background_and_declared_velocity_scale(constructor):
    velocity = model(constructor, std_kmps=.2)
    with torch.no_grad():
        velocity.low.net.linear[-1].bias.fill_(2.)
        velocity.high.net.linear[-1].bias.fill_(3.)
    for completed, alpha in ((0, 0.), (100, .5), (200, 1.)):
        velocity.set_progress(completed)
        expected = background()[None] + 200 * (2 + alpha * 3)
        torch.testing.assert_close(velocity(), expected, atol=0, rtol=0)


def test_both_branches_are_evaluated_at_zero_progress(constructor):
    velocity = model(constructor)
    seen = []
    handles = [branch.net.register_forward_pre_hook(
        lambda module, args, name=name: seen.append(name))
        for name, branch in (("low", velocity.low), ("high", velocity.high))]
    try:
        velocity.set_progress(0)
        velocity()
    finally:
        for handle in handles:
            handle.remove()
    assert seen == ["low", "high"]


def test_branch_gradients_follow_alpha_with_one_toy_backward(constructor):
    torch.manual_seed(17)
    base = model(constructor)
    with torch.no_grad():
        for branch in (base.low, base.high):
            branch.net.linear[-1].weight.fill_(.125)
            branch.net.linear[-1].bias.fill_(.25)
    zero, half, full = [copy.deepcopy(base) for _ in range(3)]
    for velocity, completed in ((zero, 0), (half, 100), (full, 200)):
        velocity.set_progress(completed)
    loss = zero().sum() + half().sum() + full().sum()
    loss.backward()  # The sole toy backward in this file; no optimizer update.
    assert any(torch.count_nonzero(p.grad) for p in full.high.parameters())
    for p0, ph, p1 in zip(zero.high.parameters(), half.high.parameters(), full.high.parameters()):
        assert p0.grad is not None and ph.grad is not None and p1.grad is not None
        assert torch.count_nonzero(p0.grad) == 0
        torch.testing.assert_close(ph.grad, .5 * p1.grad, atol=1e-6, rtol=1e-5)
    for p0, ph, p1 in zip(zero.low.parameters(), half.low.parameters(), full.low.parameters()):
        assert p0.grad is not None and ph.grad is not None and p1.grad is not None
        torch.testing.assert_close(p0.grad, p1.grad, atol=0, rtol=0)
        torch.testing.assert_close(ph.grad, p1.grad, atol=0, rtol=0)


def test_state_dict_roundtrip_preserves_progress_and_output(constructor):
    velocity = model(constructor)
    with torch.no_grad():
        velocity.low.net.linear[-1].weight.fill_(.1)
        velocity.high.net.linear[-1].bias.fill_(.3)
    velocity.set_progress(75)
    expected = velocity().detach().clone()
    buffer = io.BytesIO()
    torch.save(velocity.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location="cpu", weights_only=True)
    assert "completed_updates" in saved
    restored = model(constructor, initial_mps=background() + 123, warmup_updates=500)
    restored.load_state_dict(saved, strict=True)
    assert int(restored.completed_updates) == 75
    assert float(restored.alpha) == pytest.approx(float(velocity.alpha), abs=0)
    torch.testing.assert_close(restored(), expected, atol=0, rtol=0)


def test_parameter_count_discloses_two_complete_sirens(constructor):
    neurons = [2, 128, 128, 128, 128, 1]
    baseline = VelocityParameterization(background(), 15., neurons)
    velocity = model(constructor, neurons=neurons)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    progressive_count = sum(p.numel() for p in velocity.parameters())
    assert baseline_count == 50049
    assert progressive_count == 2 * baseline_count == 100098
    assert all(p.requires_grad for p in velocity.parameters())
    assert "completed_updates" not in dict(velocity.named_parameters())


def test_invalid_configuration_fails_before_use(constructor):
    invalid = [dict(warmup_updates=0), dict(warmup_updates=-1),
        dict(warmup_updates=1.5), dict(warmup_updates=True),
        dict(low_omega=0), dict(low_omega=float("nan")),
        dict(high_omega=-1), dict(high_omega=float("inf")),
        dict(std_kmps=0), dict(std_kmps=float("nan")),
        dict(mean_kmps=float("inf")), dict(dx_m=0), dict(dx_m=True),
        dict(neurons=[]), dict(neurons=[3, 4, 1]), dict(neurons=[2, 4, 2]),
        dict(neurons=[2, 0, 1]), dict(neurons=[2, True, 1]),
        dict(initial_mps=torch.zeros(3, 4)),
        dict(initial_mps=torch.full((3, 4), float("nan"))),
        dict(initial_mps=torch.ones(3)), dict(initial_mps=torch.empty(0, 4))]
    for options in invalid:
        with pytest.raises(ValueError):
            model(constructor, **options)


def test_invalid_progress_is_rejected_without_mutating_state(constructor):
    velocity = model(constructor)
    velocity.set_progress(42)
    for completed in (-1, True, 1., "1", float("nan"), 2**65):
        with pytest.raises(ValueError):
            velocity.set_progress(completed)
        assert int(velocity.completed_updates) == 42

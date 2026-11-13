"""Background-relative velocity scale: initialization, chain rule and recovery."""
import copy
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from experiments.dense_checkpoint import load_dense_checkpoint, save_dense_checkpoint
from experiments.encoding_checkpoint import _capture_rng, _exact
from experiments.fourier_modules import FRLinear, default_fourier
from experiments.residual_ifwi_experiment import VelocityParameterization

TARGET = ROOT / 'experiments' / 'relative_velocity.py'
relative = None
if TARGET.exists():
    spec = importlib.util.spec_from_file_location('relative_velocity_target', TARGET)
    relative = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(relative)

SMALL_FOURIER = dict(high_freq_num=8, low_freq_num=8, phi_num=8,
                     alpha=.01, lambda_init='fr_inr')


def velocity_class():
    assert relative is not None, 'relative_velocity.py not implemented'
    return relative.RelativeVelocityParameterization


def background():
    return np.array([[1500., 1800., 2200.], [2600., 3100., 3500.]], dtype=np.float32)


def make_model(fourier=None, **kwargs):
    return velocity_class()(
        background(), 15., [2, 8, 8, 8, 1], fourier=fourier, **kwargs)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


@pytest.mark.parametrize('fourier', [None, SMALL_FOURIER])
def test_zero_head_starts_at_exact_background_and_preserves_network_api(fourier):
    model = make_model(fourier)
    assert isinstance(model, VelocityParameterization)
    assert torch.equal(model(), torch.from_numpy(background())[None])
    assert torch.equal(model.raw_output(), torch.zeros(1, 2, 3))
    assert model.net.omega_0 == 30.
    assert not hasattr(model, 'optimizer_masks')
    assert model.net.linear[-1].weight.count_nonzero() == 0
    assert model.net.linear[-1].bias.count_nonzero() == 0


def test_relative_scale_is_persistent_normalized_and_independent_of_input_storage():
    initial = background()
    model = velocity_class()(initial, 15., [2, 8, 8, 1])
    initial.fill(1.)
    v0 = torch.from_numpy(background())[None]
    rms = v0.double().square().mean().sqrt().float()
    assert torch.equal(model.fixed_init_mps, v0)
    assert torch.equal(model.relative_scale, v0/rms)
    assert 'relative_scale' in model.state_dict()
    assert not model.relative_scale.requires_grad
    assert 'relative_scale' not in dict(model.named_parameters())
    assert abs(float(model.relative_scale.square().mean())-1.) < 2e-7


@pytest.mark.parametrize('fourier', [None, SMALL_FOURIER])
def test_nonzero_head_uses_fixed_relative_formula_and_velocity_scale_chain_rule(fourier):
    seed(3)
    model = make_model(fourier, std_kmps=.6)
    with torch.no_grad():
        model.net.linear[-1].weight.fill_(.001)
        model.net.linear[-1].bias.fill_(.01)
    raw = model.raw_output()
    expected = model.fixed_init_mps + model.relative_scale*600.*raw
    assert torch.equal(model(), expected)
    # A nonuniform physical gradient must be scaled before the network VJP.
    direction = torch.tensor([[[.2, -.5, .8], [.9, -.7, .1]]])
    params = tuple(model.parameters())
    actual = torch.autograd.grad(model(), params, grad_outputs=direction)
    manual = torch.autograd.grad(model.raw_output(), params,
        grad_outputs=direction*model.relative_scale*600.)
    for a, b in zip(actual, manual):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-5)
    # Independent central difference of the output bias checks physical units.
    old = model.net.linear[-1].bias.detach().clone()
    eps = .001
    with torch.no_grad():
        model.net.linear[-1].bias.copy_(old+eps)
        plus = (model()*direction).sum()
        model.net.linear[-1].bias.copy_(old-eps)
        minus = (model()*direction).sum()
        model.net.linear[-1].bias.copy_(old)
    torch.testing.assert_close((plus-minus)/(2*eps), actual[-1].sum(), rtol=5e-4, atol=.1)


def test_scale_and_background_remain_fixed_during_real_adam_updates():
    model = make_model(SMALL_FOURIER)
    original = {name: value.detach().clone() for name, value in model.named_buffers()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (model()-model.fixed_init_mps-10.).square().mean().backward()
        optimizer.step()
    assert not torch.equal(model(), model.fixed_init_mps)
    for name, value in model.named_buffers():
        assert torch.equal(value, original[name]), name


def test_rejects_absolute_parameterization_instead_of_silently_reinterpreting_it():
    with pytest.raises(ValueError, match='residual'):
        make_model(parameterization='absolute')


@pytest.mark.parametrize('fourier', [None, SMALL_FOURIER])
def test_dense_checkpoint_restores_exact_model_adam_rng_and_next_update(tmp_path, fourier):
    def update(model, optimizer):
        target = torch.from_numpy(np.random.normal(size=(1, 2, 3))).float()
        target = target + random.random() + torch.randn(1, 2, 3)
        optimizer.zero_grad(set_to_none=True)
        (model()-model.fixed_init_mps-target).square().mean().backward()
        optimizer.step()

    seed(71)
    model = make_model(fourier)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    update(model, optimizer)
    metadata = dict(protocol='relative-scale-test-v1', fourier=fourier)
    path = tmp_path/'update_0001.pt'
    save_dense_checkpoint(model, optimizer, path, 1, metadata)
    saved_model = copy.deepcopy(model.state_dict())
    saved_adam = copy.deepcopy(optimizer.state_dict())
    saved_rng = _capture_rng()
    update(model, optimizer)
    expected = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), _capture_rng()

    seed(99)
    restored = make_model(fourier)
    restored_adam = torch.optim.Adam(restored.parameters(), lr=1e-4)
    assert load_dense_checkpoint(restored, restored_adam, path, metadata) == 1
    assert _exact(saved_model, restored.state_dict())
    assert _exact(saved_adam, restored_adam.state_dict())
    assert _exact(saved_rng, _capture_rng())
    update(restored, restored_adam)
    assert _exact(expected[0], restored.state_dict())
    assert _exact(expected[1], restored_adam.state_dict())
    assert _exact(expected[2], _capture_rng())


def test_actual_full_width_author_default_fr_count_and_parameter_shapes():
    model = velocity_class()(background(), 15.,
        [2, 128, 128, 128, 128, 1], fourier=default_fourier())
    assert model.fourier_layers == [1, 2, 3]
    assert sum(p.numel() for p in model.parameters()) == 3_146_625
    assert sum(b.numel() for name, b in model.named_buffers() if name.endswith('.bases')) == 3_145_728
    assert sum(p.numel()*p.element_size() for p in model.parameters()) == 12_586_500
    for index in (1, 2, 3):
        layer = model.net.linear[index]
        assert isinstance(layer, FRLinear)
        assert layer.lamb.shape == (128, 8192)
        assert layer.bases.shape == (8192, 128)
        assert layer.rank() == 127
        assert not layer.bases.requires_grad
        assert not any(k.endswith('.bases') for k in model.state_dict())
    assert torch.equal(model(), model.fixed_init_mps)

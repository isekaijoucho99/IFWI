"""Native implicit-field evaluation at the diffusion prior's physical pixel spacing."""
import copy
import json

import numpy as np
import pytest
import torch

from experiments.residual_ifwi_experiment import VelocityParameterization
from experiments.diffusion_prior.guidance import build_prior
from experiments.diffusion_prior.io import save_base
from experiments.diffusion_prior.model import UNet
from verification.residual_ifwi.test_paper_supplement import tiny_spec


def tiny_prior(tmp_path, spacing=15.0):
    cfg = dict(channels=32, mult=[1, 2], n_res=1, heads=4, attn_from=1)
    meta = dict(units='m/s', mean_mps=3000., scale_mps=1500.,
                grid_spacing_m=spacing, patch_size=8)
    filename = tmp_path / 'prior.pt'
    save_base(filename, UNet(**cfg), cfg,
              dict(steps=100, beta_start=1e-4, beta_end=.02), meta, updates=0)
    return filename


def test_canonical_shape_preserves_metric_spacing_not_fixed_image_size():
    from experiments.diffusion_prior.scale_bridge import canonical_shape
    assert canonical_shape((94, 401), 20., 15.) == (125, 534)
    assert canonical_shape((94, 288), 15., 15.) == (94, 288)
    with pytest.raises(ValueError, match='coarser'):
        canonical_shape((94, 401), 15., 20.)


def test_implicit_field_query_matches_background_and_backpropagates():
    from experiments.diffusion_prior.scale_bridge import query_velocity
    z = np.arange(9, dtype=np.float32)[:, None]
    x = np.arange(11, dtype=np.float32)[None, :]
    initial = 2400. + 33. * z + 15. * x
    model = VelocityParameterization(initial, 20., [2, 16, 16, 1])
    sample = query_velocity(model, 20., 15.)
    assert sample.shape == (1, 11, 14)
    zz = torch.arange(11).float()[:, None] * .75
    xx = torch.arange(14).float()[None, :] * .75
    expected = 2400. + 33. * zz + 15. * xx
    assert torch.allclose(sample[0], expected, atol=1e-3)
    sample.square().mean().backward()
    assert model.net.linear[-1].weight.grad.abs().sum() > 0
    assert torch.equal(query_velocity(model, 20., 20.), model())


def test_query_works_with_fourier_reparameterized_siren_and_warp():
    from experiments.diffusion_prior.scale_bridge import query_velocity
    from core_models import traveltime_coordinates, warp_coordinates
    initial = np.linspace(2200, 4100, 12, dtype=np.float32)[:, None] * np.ones((1, 16), np.float32)
    mod = VelocityParameterization(initial, 20., [2, 32, 32, 32, 1],
        fourier={'high_freq_num': 32, 'low_freq_num': 32, 'phi_num': 8, 'alpha': .01, 'lambda_init': 'fr_inr'})
    warp_coordinates(mod, traveltime_coordinates(initial, 20., 'z'))
    v = query_velocity(mod, 20., 15.)
    assert v.shape == (1, 15, 21)
    v.mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.parameters())


def test_checkpoint_spacing_rejected_by_default_accepted_only_with_opt_in(tmp_path):
    p = tiny_prior(tmp_path)
    options = dict(checkpoint=str(p), allow_untrained=True, warmup=0, every=1,
                   eta_start=.2, eta_end=.02)
    with pytest.raises(ValueError, match='grid spacing'):
        build_prior(options, 'cpu', 20.)
    prior = build_prior({**options, 'physical_scale_bridge': True}, 'cpu', 20.)
    assert prior.grid_shape((9, 11)) == (11, 14)
    model = VelocityParameterization(np.full((9, 11), 3000., np.float32), 20., [2, 16, 16, 1])
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter) * .01
    info = prior.apply(model, 1)
    assert info['active'] and info['prior_grid_shape'] == [11, 14]
    assert info['unet_calls'] >= 1
    assert info['effective_ratio'] == pytest.approx(.025)


def test_full_runner_uses_source_grid_for_fd_and_prior_grid_for_score(tmp_path):
    import core_runner
    p = tiny_prior(tmp_path)
    spec = tiny_spec(tmp_path, 'scale_bridge_smoke', method='E3')
    spec['experiment']['data']['grid_spacing_m'] = 20.
    spec['diffusion_prior'] = dict(checkpoint=str(p), allow_untrained=True,
        warmup=0, every=1, physical_scale_bridge=True,
        eta_start=.2, eta_end=.02, patch_batch_size=2)
    output = core_runner.run(spec, tmp_path/'run', 'cpu', save_figures=False, shared_dir=tmp_path/'shared')
    hist = json.loads((output/'history.json').read_text())
    assert len(hist) == 3
    assert hist[0]['diffusion_prior']['prior_grid_shape'] == [10, 11]
    assert hist[0]['diffusion_prior']['effective_ratio'] == pytest.approx(.025)
    assert np.load(output/'final_velocity.npy').shape == (8, 9)
    assert json.loads((output/'acquisition.json').read_text())['nx'] == 9


def test_bridge_rejects_incompatible_parameterizations(tmp_path):
    import core_runner
    p = tiny_prior(tmp_path)
    spec = tiny_spec(tmp_path, 'invalid_scale_bridge', method='E3')
    spec['diffusion_prior'] = dict(checkpoint=str(p), allow_untrained=True, physical_scale_bridge=True)
    spec['model'] = dict(kind='grid')
    with pytest.raises(ValueError, match='physical_scale_bridge'):
        core_runner.check_spec(spec)


def test_scale_bridge_standalone_rejects_fourier_feature_network():
    from experiments.diffusion_prior.scale_bridge import query_velocity
    from core_models import FourierFeatureVelocity
    m = FourierFeatureVelocity(np.full((9, 11), 3000., np.float32), 20., 3., 4, (8,), seed=7)
    with pytest.raises(ValueError, match='2-D SIREN'):
        query_velocity(m, 20., 15.)

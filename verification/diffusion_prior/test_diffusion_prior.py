"""CPU numerical tests: analytic scores, no test geology or pretrained weights."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

# Keep CPU smoke tests small and deterministic.
torch.set_num_threads(1)


def tiny_config():
    return dict(channels=32, mult=[1, 2], n_res=1, heads=4, attn_from=1)


def metadata():
    return dict(units='m/s', mean_mps=3000., scale_mps=1500., grid_spacing_m=15., patch_size=8)


def test_model_diffusion_and_clean_prediction():
    from experiments.diffusion_prior.model import UNet, DDPM
    net = UNet(**tiny_config())
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([0, 7])
    assert net(x, t).shape == x.shape
    d = DDPM(steps=10)
    eps = torch.randn_like(x)
    xt = d.q_sample(x, t, eps)
    assert torch.allclose(d.predict_clean(xt, t, eps), x, atol=2e-6)
    assert torch.isfinite(d.loss(net, x))


def test_lora_initial_identity_and_only_adapters_trainable():
    from experiments.diffusion_prior.model import UNet, inject_lora, adapter_state
    net = UNet(**tiny_config())
    x, t = torch.randn(2, 1, 8, 8), torch.tensor([2, 3])
    before = net(x, t).detach()
    params = inject_lora(net, rank=2, alpha=2.)
    assert torch.equal(net(x, t), before)
    assert 0 < sum(p.numel() for p in params) < sum(p.numel() for p in net.parameters())
    frozen = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}
    opt = torch.optim.Adam(params, lr=1e-3)
    net(x, t).square().mean().backward()
    opt.step()
    assert all(torch.equal(p, frozen[n]) for n, p in net.named_parameters() if n in frozen)
    assert adapter_state(net)
    assert all('lora_' in n for n, p in net.named_parameters() if p.requires_grad)
    with pytest.raises(ValueError, match='already'):
        inject_lora(net, rank=2, alpha=2.)


@pytest.mark.parametrize('kw', [dict(steps=0), dict(beta_start=-1), dict(beta_end=1), dict(beta_start=.1, beta_end=.01)])
def test_invalid_schedule(kw):
    from experiments.diffusion_prior.model import DDPM
    with pytest.raises(ValueError):
        DDPM(**kw)


def test_checkpoint_roundtrip_and_rng_isolation(tmp_path):
    from experiments.diffusion_prior.model import UNet
    from experiments.diffusion_prior.io import save_base, load_prior
    net = UNet(**tiny_config())
    p = tmp_path / 'prior.pt'
    save_base(p, net, tiny_config(), dict(steps=10, beta_start=1e-4, beta_end=.02), metadata(), updates=3)
    rng = torch.get_rng_state().clone()
    loaded, _, meta, info = load_prior(p, device='cpu', grid_spacing_m=15.)
    assert torch.equal(rng, torch.get_rng_state())
    assert meta == metadata() and info['updates'] == 3
    assert not loaded.training and not any(x.requires_grad for x in loaded.parameters())
    assert all(torch.equal(net.state_dict()[k], v) for k, v in loaded.state_dict().items())
    with pytest.raises(ValueError, match='spacing'):
        load_prior(p, device='cpu', grid_spacing_m=20.)
    torch.save({'model': net.state_dict(), 'cfg': tiny_config()}, tmp_path / 'legacy.pt')
    with pytest.raises(ValueError, match='schema'):
        load_prior(tmp_path / 'legacy.pt', device='cpu', grid_spacing_m=15.)


def test_adapter_roundtrip_and_base_hash(tmp_path):
    from experiments.diffusion_prior.model import UNet, inject_lora
    from experiments.diffusion_prior.io import save_base, save_adapter, load_prior
    net = UNet(**tiny_config())
    base = tmp_path / 'base.pt'
    save_base(base, net, tiny_config(), dict(steps=10, beta_start=1e-4, beta_end=.02), metadata(), updates=2)
    inject_lora(net, rank=2, alpha=2.)
    with torch.no_grad():
        for n, p in net.named_parameters():
            if 'lora_up' in n:
                p.add_(.001)
    adapter = tmp_path / 'adapter.pt'
    save_adapter(adapter, net, base, rank=2, alpha=2., updates=2)
    loaded, _, _, _ = load_prior(base, device='cpu', grid_spacing_m=15., adapter=adapter)
    x, t = torch.randn(1, 1, 8, 8), torch.tensor([3])
    assert torch.equal(loaded(x, t), net(x, t))
    other = tmp_path / 'other.pt'
    save_base(other, UNet(**tiny_config()), tiny_config(), dict(steps=10, beta_start=1e-4, beta_end=.02), metadata(), updates=2)
    with pytest.raises(ValueError, match='hash'):
        load_prior(other, device='cpu', grid_spacing_m=15., adapter=adapter)


def test_untrained_checkpoint_requires_explicit_opt_in(tmp_path):
    from experiments.diffusion_prior.model import UNet
    from experiments.diffusion_prior.io import save_base, load_prior
    p = tmp_path / 'random.pt'
    save_base(p, UNet(**tiny_config()), tiny_config(), dict(steps=10, beta_start=1e-4, beta_end=.02), metadata(), updates=0)
    with pytest.raises(ValueError, match='untrained'):
        load_prior(p, device='cpu', grid_spacing_m=15.)
    load_prior(p, device='cpu', grid_spacing_m=15., allow_untrained=True)


class DeltaPrior(nn.Module):
    """Exact epsilon predictor for a clean distribution concentrated at zero."""
    def __init__(self, ddpm):
        super().__init__()
        self.ddpm = ddpm
        self.seen = []

    def forward(self, x, t):
        self.seen.append(tuple(x.shape))
        sigma = (1 - self.ddpm.abar[t]).sqrt()[:, None, None, None]
        return x / sigma


def test_overlap_score_analytic_sign_units_and_no_resize():
    from experiments.diffusion_prior.model import DDPM
    from experiments.diffusion_prior.guidance import patch_direction
    d = DDPM(steps=10)
    net = DeltaPrior(d)
    v = 3000 + torch.arange(11 * 19).reshape(1, 11, 19).float()
    noise = torch.randn_like(v)
    got, info = patch_direction(v, net, d, metadata(), timestep=7, noise=noise, stride=5, batch_size=3, weighting='kl')
    expected = d.abar[7] / (1 - d.abar[7]) * (v - 3000) / (1500 ** 2 * v.numel())
    assert torch.allclose(got, expected, atol=2e-11, rtol=2e-5)
    assert info['patches'] > 1 and info['coverage_min'] >= 1
    assert all(shape[1:] == (1, 8, 8) for shape in net.seen)
    assert not got.requires_grad
    got2, _ = patch_direction(v, net, d, metadata(), timestep=7, noise=noise, stride=5, batch_size=1, weighting='kl')
    assert torch.allclose(got, got2, atol=2e-11, rtol=2e-5)


def test_invalid_patches_and_noise():
    from experiments.diffusion_prior.model import DDPM
    from experiments.diffusion_prior.guidance import patch_direction
    d = DDPM(steps=10)
    net = DeltaPrior(d)
    with pytest.raises(ValueError, match='patch'):
        patch_direction(torch.ones(1, 4, 10), net, d, metadata(), 3, torch.zeros(1, 4, 10), 4, 1, 'kl')
    with pytest.raises(ValueError):
        patch_direction(torch.ones(1, 8, 8), net, d, metadata(), 3, torch.zeros(1, 8, 8), 9, 1, 'kl')


def test_parameter_gradient_ratio_and_descent_bound():
    from experiments.diffusion_prior.guidance import add_guidance
    class Velocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor([1., 2.]))
        def forward(self):
            return self.p.reshape(1, 1, 2) * torch.tensor([3., 5.])
    m = Velocity()
    m.p.grad = torch.tensor([2., -1.])
    old = m.p.grad.clone()
    report = add_guidance(m, torch.tensor([[[-2., 7.]]]), ratio=.1)
    correction = m.p.grad - old
    assert torch.allclose(correction.norm(), .1 * old.norm())
    assert torch.dot(old, m.p.grad) >= .9 * old.square().sum() - 1e-6
    assert abs(report['effective_ratio'] - .1) < 1e-6


def test_zero_physical_gradient_does_not_inject_prior():
    from experiments.diffusion_prior.guidance import add_guidance
    m = nn.Linear(2, 1)
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    # No forward should be needed in this case.
    result = add_guidance(m, torch.ones(1, 1, 2), ratio=.1)
    assert result['effective_ratio'] == 0
    assert all(torch.equal(p.grad, torch.zeros_like(p)) for p in m.parameters())


def test_guidance_schedule_and_random_isolation(tmp_path):
    from experiments.diffusion_prior.model import UNet
    from experiments.diffusion_prior.io import save_base
    from experiments.diffusion_prior.guidance import build_prior
    p = tmp_path / 'base.pt'
    save_base(p, UNet(**tiny_config()), tiny_config(), dict(steps=100, beta_start=1e-4, beta_end=.02), metadata(), updates=1)
    cfg = dict(checkpoint=str(p), warmup=2, every=3, seed=17, eta_start=.2, eta_end=.02, anneal_updates=20, patch_stride=4, patch_batch_size=2)
    state = torch.get_rng_state().clone()
    prior = build_prior(cfg, 'cpu', 15.)
    assert not prior.active(2) and prior.active(3) and not prior.active(4) and prior.active(6)
    v = torch.full((1, 9, 13), 3200.)
    g1, a = prior.direction(v, 3)
    g2, b = prior.direction(v, 3)
    assert torch.equal(g1, g2) and a['timestep'] == b['timestep']
    assert torch.equal(state, torch.get_rng_state())
    assert prior.timestep(40) <= prior.timestep(3)


def test_resize_control_is_explicit_and_network_free():
    from experiments.diffusion_prior.guidance import build_prior
    p = build_prior(dict(mode='resize_only', resize_shape=[4, 4], warmup=0, every=1), 'cpu', 15.)
    v = torch.arange(100.).reshape(1, 10, 10) + 2000
    g, info = p.direction(v, 1)
    y = torch.nn.functional.interpolate(v[:, None], (4, 4), mode='bilinear', align_corners=True)
    y = torch.nn.functional.interpolate(y, (10, 10), mode='bilinear', align_corners=True)[:, 0]
    assert torch.allclose(g, (v-y)/(1500.**2*v.numel()))
    assert info['unet_calls'] == 0


def test_disabled_build_does_not_consume_rng():
    from experiments.diffusion_prior.guidance import build_prior
    before = torch.get_rng_state().clone()
    assert build_prior(None, 'cpu', 15.) is None
    assert build_prior(dict(enabled=False), 'cpu', 15.) is None
    assert torch.equal(before, torch.get_rng_state())


def test_fixed_weight_can_act_at_zero_physical_gradient():
    from experiments.diffusion_prior.guidance import add_guidance
    class Velocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1, 2, 2))
        def forward(self):
            return self.p
    m = Velocity()
    m.p.grad = torch.zeros_like(m.p)
    info = add_guidance(m, torch.ones_like(m.p), fixed_weight=.2)
    assert torch.equal(m.p.grad, torch.full_like(m.p, .2))
    assert info['effective_ratio'] is None and info['gradient_cosine'] is None


@pytest.mark.parametrize('schedule', [dict(steps=1000,beta_start=.5,beta_end=.9), dict(steps=10,beta_start=1e-20,beta_end=1e-19)])
def test_unrepresentable_float32_schedule_is_rejected(schedule):
    from experiments.diffusion_prior.model import DDPM
    with pytest.raises(ValueError, match='represent'):
        DDPM(**schedule)

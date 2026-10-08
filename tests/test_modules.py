import torch

from g2ifwi.diffusion import DDPM, UNet, make_well_guidance, red_denoise
from g2ifwi.hashgrid import HashEncoder2D
from g2ifwi.inr import HashSirenVelocity
from g2ifwi.velocity_gen import generate_model


def test_hash_encoder_interpolates_and_is_differentiable():
    enc = HashEncoder2D(4, 2, 8, 32, 12, init_scale=0.5)
    c = torch.rand(50, 2)
    out = enc(c)
    assert out.shape == (50, 8)
    out.sum().backward()
    assert enc.tables.grad.abs().sum() > 0
    # continuity: nearby points give nearby features
    assert (enc(c + 1e-5) - out).abs().max() < 1e-3


def test_inr_starts_at_v_init():
    v0 = torch.full((10, 20), 3000.0)
    net = HashSirenVelocity(10, 20, v0, dict(n_levels=2, n_features=1, base_res=4, finest_res=8, log2_hashmap_size=10),
                            dict(hidden=16, layers=2, omega=30))
    assert torch.equal(net(), v0)


def test_generator_range():
    v = generate_model(rng=__import__("numpy").random.default_rng(1))
    assert v.shape == (152, 708) and 1500 <= v.min() and v.max() <= 4500


def test_unet_and_prior_step_shapes():
    net = UNet(channels=32, mult=(1, 2, 4), n_res=1).eval()
    assert net(torch.randn(2, 1, 64, 64), torch.tensor([3, 4])).shape == (2, 1, 64, 64)
    ddpm = DDPM()
    v = 2000 + 1500 * torch.rand(32, 100)
    out = red_denoise(net, ddpm, v, t_start=3, n_steps=3, size=(64, 64))
    assert out.shape == v.shape and torch.isfinite(out).all()
    W, M = make_well_guidance([(10, torch.full((32,), 3000.)), (80, torch.full((32,), 4000.))], 32, 100, 5)
    g = red_denoise(net, ddpm, v, t_start=3, n_steps=3, size=(64, 64), well=(W, M), gamma=1e-2)
    assert g.shape == v.shape

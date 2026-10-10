"""FR (Fourier reparameterized) hidden layers and band errors; no FD, no training."""
import copy
import importlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

SMALL = {"high_freq_num": 4, "low_freq_num": 4, "phi_num": 8, "alpha": .01, "lambda_init": "fr_inr"}


def fr():
    return importlib.import_module("experiments.fourier_modules")


def api():
    return importlib.import_module("experiments.residual_ifwi_experiment")


def upstream_bases(in_features, high, low, phi_num, alpha):
    """Line-by-line FR-INR sin_fr_layer.init_bases."""
    phi_set = [2 * math.pi * i / phi_num for i in range(phi_num)]
    high_freq = [i + 1 for i in range(high)]
    low_freq = [(i + 1) / low for i in range(low)]
    t_max = 2 * math.pi / low_freq[0] if low_freq else 2 * math.pi / min(high_freq)
    points = np.linspace(-t_max / 2, t_max / 2, in_features)
    rows = [[math.cos(f * x + p) for x in points] for f in low_freq + high_freq for p in phi_set]
    return alpha * torch.tensor(rows, dtype=torch.float32)


@pytest.mark.parametrize("high,low", [(4, 4), (6, 0), (0, 3)])
def test_bases_match_fr_inr_construction(high, low):
    ours = fr().fourier_bases(16, high, low, 8, .01)
    assert ours.shape == ((high + low) * 8, 16)
    assert torch.allclose(ours, upstream_bases(16, high, low, 8, .01), atol=1e-6)


def test_two_input_first_layer_would_be_rank_one():
    """Why the 2 -> hidden layer stays plain: x and z basis columns coincide."""
    bases = fr().fourier_bases(2, 128, 128, 32, .01)
    assert torch.allclose(bases[:, 0], bases[:, 1], atol=1e-6)


def test_only_hidden_layers_are_reparameterized_and_rank_checked():
    from ifwi_modules import IRN
    net = IRN(neuron=[2, 8, 8, 8, 1], omega_0=30., outermost_linear=True)
    assert fr().reparameterize_irn(net, SMALL) == [1, 2]
    assert isinstance(net.linear[0], torch.nn.Linear) and isinstance(net.linear[-1], torch.nn.Linear)
    assert all(isinstance(net.linear[i], fr().FRLinear) for i in (1, 2))
    assert all(net.linear[i].rank() == 8 for i in (1, 2))


def test_fr_inr_lambda_init_bounds_and_zero_bias():
    torch.manual_seed(0)
    layer = fr().FRLinear(8, 8, 4, 4, 8, .01)
    layer.init_fr_inr(30.)
    m = layer.bases.shape[0]
    bound = np.sqrt(6 / m) / torch.linalg.vector_norm(layer.bases, dim=1) / 30.
    assert (layer.lamb.abs() <= bound[None] + 1e-7).all()
    assert torch.equal(layer.bias, torch.zeros(8))


def test_pinv_init_reproduces_seed_matched_original_network():
    from ifwi_modules import IRN
    coords = torch.rand(1, 5, 7, 2)
    torch.manual_seed(3)
    plain = IRN(neuron=[2, 8, 8, 8, 1], omega_0=30., outermost_linear=True)
    torch.manual_seed(3)
    net = IRN(neuron=[2, 8, 8, 8, 1], omega_0=30., outermost_linear=True)
    fr().reparameterize_irn(net, {**SMALL, "lambda_init": "pinv"})
    for a, b in zip(fr().merged_weights(plain), fr().merged_weights(net)):
        assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(plain(coords)[0], net(coords)[0], atol=1e-4)


@pytest.mark.parametrize("lambda_init", ["fr_inr", "pinv"])
def test_residual_fr_starts_at_background_and_trains_only_lambda(lambda_init):
    initial = np.linspace(1500, 4500, 6 * 9, dtype=np.float32).reshape(6, 9)
    model = api().VelocityParameterization(initial, 15, [2, 8, 8, 1], fourier={**SMALL, "lambda_init": lambda_init})
    assert model.fourier_layers == [1]
    with torch.no_grad():
        assert torch.equal(model()[0], torch.from_numpy(initial))
    names = {name for name, _ in model.named_parameters()}
    assert "net.linear.1.lamb" in names and "net.linear.1.weight" not in names
    assert not any("bases" in key for key in model.state_dict())
    # Zero last layer: first gradient reaches the output layer, not lamb (as for plain IRN).
    model().sum().backward()
    assert model.net.linear[-1].weight.grad.abs().sum() > 0
    assert model.net.linear[1].bases.grad is None


def test_fr_state_dict_round_trip_is_strict():
    initial = np.full((5, 6), 3000., np.float32)
    first = api().VelocityParameterization(initial, 15, [2, 8, 8, 1], fourier=SMALL)
    second = api().VelocityParameterization(initial, 15, [2, 8, 8, 1], fourier=SMALL)
    second.load_state_dict(first.state_dict(), strict=True)
    assert torch.equal(first.net.linear[1].weight, second.net.linear[1].weight)


def test_fourier_config_validation_and_resume_compatibility():
    cfg = api().default_config()
    api().validate_config(cfg)
    cfg["model"]["fourier"] = None
    api().validate_config(cfg)
    assert api()._resume_configuration(cfg) == api()._resume_configuration(api().default_config())
    cfg["model"]["fourier"] = fr().default_fourier()
    api().validate_config(cfg)
    for key, value in (("lambda_init", "eye"), ("alpha", 0), ("phi_num", 0), ("extra", 1)):
        bad = copy.deepcopy(cfg)
        bad["model"]["fourier"][key] = value
        with pytest.raises(ValueError):
            api().validate_config(bad)
    shallow = copy.deepcopy(cfg)
    shallow["model"]["neurons"] = [2, 8, 1]
    with pytest.raises(ValueError):
        api().validate_config(shallow)


@pytest.mark.parametrize("name", ["residual_fr_ifwi_baseline1000", "residual_fr_ifwi_pinv_baseline1000"])
def test_fr_configs_change_only_name_and_fourier(name):
    import yaml
    folder = ROOT / "experiments/configs"
    base = yaml.safe_load((folder / "residual_ifwi_baseline1000.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((folder / f"{name}.yaml").read_text(encoding="utf-8"))
    api().validate_config(cfg)
    assert cfg["experiment_name"] == name
    fourier = cfg["model"].pop("fourier")
    assert {k: v for k, v in fourier.items() if k != "lambda_init"} == {
        k: v for k, v in fr().default_fourier().items() if k != "lambda_init"}
    base["experiment_name"] = name
    assert cfg == base


def test_band_rmse_parseval_and_background_ratio():
    sm = importlib.import_module("experiments.spectral_metrics")
    rng = np.random.default_rng(0)
    truth = rng.normal(3000, 300, (20, 30))
    background = truth + rng.normal(0, 50, truth.shape)
    prediction = truth + rng.normal(0, 20, truth.shape)
    result = sm.band_errors(prediction, truth, background, 15.)
    total = np.sqrt(np.mean((prediction - truth) ** 2))
    assert np.isclose(np.sqrt(sum(v ** 2 for v in result["rmse_mps"].values())), total)
    same = sm.band_errors(background, truth, background, 15.)
    assert all(np.isclose(v, 1.) for v in same["ratio_to_background"].values() if v is not None)


def test_band_rmse_places_a_known_wavenumber():
    sm = importlib.import_module("experiments.spectral_metrics")
    nz, nx, d = 40, 60, 15.
    m = 12  # DCT index along x -> k = m / (2 nx d) = 6.67 c/km, inside 3-8
    x = (np.arange(nx) + .5) / nx
    field = np.tile(np.cos(np.pi * m * x), (nz, 1)) * 100
    bands = sm.band_rmse(field, d)
    assert max(bands, key=bands.get) == "3-8cpkm"
    assert bands["3-8cpkm"] ** 2 > .999 * sum(v ** 2 for v in bands.values())

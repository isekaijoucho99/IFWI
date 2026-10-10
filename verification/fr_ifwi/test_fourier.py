"""FR equations, initialization and integration checks (no training labels)."""
import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)


def fr():
    name = "experiments.fr_siren"
    assert importlib.util.find_spec(name) is not None, "Fourier implementation is missing"
    return importlib.import_module(name)


def runner():
    return importlib.import_module("experiments.residual_ifwi_experiment")


def options(**overrides):
    return dict(low_freq_num=2, high_freq_num=2, phi_num=4, alpha=.01,
                target_layers="hidden_only", learnable_omega=False, **overrides)


def tiny_config(tmp_path=None, learnable=False):
    cfg = runner().default_config()
    cfg["model"].update(neurons=[2, 8, 8, 8, 1], architecture="fr_siren", fourier=options())
    cfg["model"]["fourier"]["learnable_omega"] = learnable
    cfg["acquisition"].update(num_shots=5, source_x_indices=[1, 3, 5, 7, 9])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60., npad=3)
    cfg["training"].update(epochs=4, checkpoint_interval=2, shot_batch_size=2,
                           shots_per_update=3, full_eval_interval=2)
    z, x = np.indices((9, 11))
    truth = (2300 + 70*z + 25*x + 200*(x >= 6)).astype(np.float32)
    if tmp_path is not None:
        source = tmp_path / "truth.csv"
        np.savetxt(source, truth, delimiter=",")
        cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
        cfg["initialization"]["sigma"] = 1.
    return cfg, truth


def make_model(truth, cfg):
    return runner().VelocityParameterization(truth * np.float32(.95), 15.,
        parameterization="residual", **cfg["model"])


def assert_same_state(a, b):
    assert a.keys() == b.keys()
    for name in a:
        if isinstance(a[name], torch.Tensor):
            assert torch.equal(a[name], b[name]), name
        elif isinstance(a[name], dict):
            assert_same_state(a[name], b[name])
        else:
            assert a[name] == b[name], name


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    # Do not initialize or modify a concurrently used local CUDA device.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_basis_matches_independent_cosine_formula():
    layer = fr().FourierLinear(11, 5, low_freq_num=2, high_freq_num=3,
                               phi_num=4, alpha=.01, omega0=30., dtype=torch.float64)
    frequencies = [.5, 1., 1., 2., 3.]
    points = np.linspace(-2*math.pi, 2*math.pi, 11)
    expected = [[.01*math.cos(w*z + 2*math.pi*p/4) for z in points]
                for w in frequencies for p in range(4)]
    torch.testing.assert_close(layer.B, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-14)


def test_high_only_basis_has_defined_sampling_interval():
    layer = fr().FourierLinear(8, 3, low_freq_num=0, high_freq_num=2, phi_num=4)
    assert layer.B.shape == (8, 8)
    assert torch.isfinite(layer.coeff).all()


def test_linear_output_and_coefficient_gradient_equal_dense_chain_rule():
    layer = fr().FourierLinear(8, 5, low_freq_num=2, high_freq_num=2, phi_num=4,
                               dtype=torch.float64)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    y = layer(x)
    dense_weight = (layer.coeff @ layer.B).detach().requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    bias_ref = layer.bias.detach().clone().requires_grad_(True)
    expected = F.linear(x_ref, dense_weight, bias_ref)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    target = torch.randn_like(y)
    (y*target).sum().backward()
    (expected*target).sum().backward()
    torch.testing.assert_close(layer.coeff.grad, dense_weight.grad @ layer.B.T)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(layer.bias.grad, bias_ref.grad)


def test_buffer_fixed_optimizer_excludes_basis_and_state_roundtrip():
    layer = fr().FourierLinear(8, 5, low_freq_num=2, high_freq_num=2, phi_num=4)
    assert "B" in dict(layer.named_buffers())
    assert "B" not in dict(layer.named_parameters())
    assert not layer.B.requires_grad
    basis = layer.B.clone()
    opt = torch.optim.Adam(layer.parameters(), lr=1e-4)
    layer(torch.randn(7, 8)).square().mean().backward()
    opt.step()
    assert torch.equal(basis, layer.B)
    other = fr().FourierLinear(8, 5, low_freq_num=2, high_freq_num=2, phi_num=4)
    other.load_state_dict(layer.state_dict(), strict=True)
    assert_same_state(layer.state_dict(), other.state_dict())
    assert layer.double().B.dtype == torch.float64


def test_coefficient_initialization_uses_per_basis_norm_and_siren_scale():
    torch.manual_seed(3)
    layer = fr().FourierLinear(128, 128)
    bound = math.sqrt(6/layer.B.shape[0]) / layer.B.norm(dim=1) / 30
    assert torch.all(layer.coeff.abs() <= bound[None] * (1+1e-6))
    assert torch.count_nonzero(layer.bias) == 0
    assert layer.B.shape == (512, 128)
    diag = layer.diagnostics()
    assert 0 < diag["numerical_rank"] < 128  # endpoint/phase redundancy is reported, not hidden
    assert diag["condition_number"] is None
    assert len(diag["basis_sha256"]) == 64
    json.dumps(diag, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("low_freq_num", -1), ("low_freq_num", True), ("high_freq_num", 1.5),
    ("phi_num", 0), ("phi_num", False), ("alpha", 0),
    ("alpha", float("nan")), ("omega0", -30), ("omega0", float("inf")),
])
def test_invalid_core_settings_fail_before_allocation(field, value):
    kwargs = dict(low_freq_num=2, high_freq_num=2, phi_num=4, alpha=.01, omega0=30.)
    kwargs[field] = value
    with pytest.raises(ValueError):
        fr().FourierLinear(8, 8, **kwargs)


def test_empty_basis_and_degenerate_rows_rejected():
    with pytest.raises(ValueError):
        fr().FourierLinear(8, 8, low_freq_num=0, high_freq_num=0)
    with pytest.raises(ValueError, match="degenerate"):
        fr().FourierLinear(3, 8, low_freq_num=1, high_freq_num=1, phi_num=4)


def test_three_hidden_layers_replaced_without_double_sine():
    from ifwi_modules import IRN
    torch.manual_seed(3)
    vanilla = IRN(neuron=[2, 128, 128, 128, 128, 1], omega_0=30,
                  outermost_linear=True, dropout=False, bias=True)
    torch.manual_seed(3)
    net = fr().FourierIRN([2, 128, 128, 128, 128, 1], omega_0=30)
    assert isinstance(net.linear[0], torch.nn.Linear)
    assert isinstance(net.linear[-1], torch.nn.Linear)
    assert all(isinstance(x, fr().FourierLinear) for x in net.linear[1:-1])
    assert torch.equal(vanilla.linear[0].weight, net.linear[0].weight)
    assert torch.equal(vanilla.linear[-1].weight, net.linear[-1].weight)
    assert sum(p.numel() for p in net.parameters()) == 197505
    coords = torch.randn(2, 4, 2)
    manual = torch.sin(30 * net.linear[0](coords))
    for layer in net.linear[1:-1]:
        manual = torch.sin(30 * F.linear(manual, layer.coeff @ layer.B, layer.bias))
    manual = net.linear[-1](manual)
    torch.testing.assert_close(net(coords)[0], manual, rtol=0, atol=0)


def test_numpy_sampling_rng_untouched_by_fourier_initialization():
    np.random.seed(3)
    state = np.random.get_state()
    fr().FourierIRN([2, 128, 128, 128, 128, 1])
    drawn = np.random.choice(49, 8, replace=False)
    np.random.set_state(state)
    assert np.array_equal(drawn, np.random.choice(49, 8, replace=False))


@pytest.mark.parametrize("learnable", [False, True])
def test_zero_residual_and_real_fd_gradients(learnable):
    cfg, truth = tiny_config(learnable=learnable)
    runner().validate_config(cfg)
    model = make_model(truth, cfg)
    assert torch.equal(model(), torch.from_numpy(truth*np.float32(.95))[None])
    problem = runner().build_forward_problem(truth, cfg, "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    coeff = model.net.linear[1].coeff.detach().clone()
    basis = model.net.linear[1].B.clone()
    init = model.fixed_init_mps.clone()
    runner().train_update(model, problem, opt)
    assert torch.equal(coeff, model.net.linear[1].coeff)
    runner().train_update(model, problem, opt)
    assert not torch.equal(coeff, model.net.linear[1].coeff)
    assert torch.equal(basis, model.net.linear[1].B)
    assert torch.equal(init, model.fixed_init_mps)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if learnable:
        assert model.net.activation_omega.shape == (3,)
        assert model.net.activation_omega.grad.abs().max() > 0


@pytest.mark.parametrize("field,value", [
    ("target_layers", "all"), ("learnable_omega", 1), ("high_freq_num", -1),
    ("phi_num", 0), ("alpha", 0), ("unknown_option", True),
])
def test_invalid_fourier_config_rejected(field, value):
    cfg, _ = tiny_config()
    cfg["model"]["fourier"][field] = value
    with pytest.raises(ValueError):
        runner().validate_config(cfg)


def test_fourier_options_never_silently_ignored_for_siren():
    cfg, _ = tiny_config()
    cfg["model"]["architecture"] = "siren"
    with pytest.raises(ValueError):
        runner().validate_config(cfg)


def test_legacy_model_state_is_unchanged_with_default_arguments():
    cfg, truth = tiny_config()
    torch.manual_seed(3)
    a = runner().VelocityParameterization(truth, 15., [2, 8, 8, 1])
    torch.manual_seed(3)
    b = runner().VelocityParameterization(truth, 15., [2, 8, 8, 1], architecture="siren")
    assert_same_state(a.state_dict(), b.state_dict())
    assert all("coeff" not in k and not k.endswith(".B") for k in a.state_dict())


def test_microbatch_accumulation_keeps_selected_objective_and_update():
    cfg, truth = tiny_config()
    cfg["training"]["shots_per_update"] = None
    cfg["training"]["shot_batch_size"] = None
    full = runner().build_forward_problem(truth, cfg, "cpu")
    cfg["training"]["shot_batch_size"] = 2
    batched = runner().build_forward_problem(truth, cfg, "cpu")
    # Keep the same target bytes even if a CPU backend changes batch arithmetic.
    batched["observed"] = full["observed"].clone()
    a = make_model(truth, cfg)
    b = copy.deepcopy(a)
    oa = torch.optim.Adam(a.parameters(), lr=1e-4)
    ob = torch.optim.Adam(b.parameters(), lr=1e-4)
    for _ in range(2):
        ra = runner().train_update(a, full, oa)
        rb = runner().train_update(b, batched, ob)
        assert ra["data_mse"] == pytest.approx(rb["data_mse"], rel=3e-5)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("learnable", [False, True])
def test_resume_matches_continuous_weights_adam_shots_and_metrics(tmp_path, learnable):
    cfg, _ = tiny_config(tmp_path, learnable=learnable)
    full = runner().run_experiment(cfg, tmp_path/"full", "cpu", save_figures=False)
    short = copy.deepcopy(cfg)
    short["training"]["epochs"] = 2
    first = runner().run_experiment(short, tmp_path/"short", "cpu", save_figures=False)
    resumed = runner().run_experiment(cfg, tmp_path/"resume", "cpu", resume=first/"checkpoint.pt", save_figures=False)
    a = torch.load(full/"checkpoint.pt", weights_only=False)
    b = torch.load(resumed/"checkpoint.pt", weights_only=False)
    assert_same_state(a["model"], b["model"])
    assert_same_state(a["optimizer"], b["optimizer"])
    assert [h["shot_indices"] for h in a["history"]] == [h["shot_indices"] for h in b["history"]]
    assert a["full_evaluation_history"] == b["full_evaluation_history"]
    assert a["fr_implementation_sha256"] == b["fr_implementation_sha256"]
    metadata = json.loads((full/"fourier_diagnostics.json").read_text())
    assert len(metadata["layers"]) == 2
    assert metadata["learnable_omega"] is learnable
    assert "experiments/fr_siren.py" in json.loads((full/"experiment_source_hashes.json").read_text())
    assert np.array_equal(np.load(full/"final_velocity.npy"), np.load(resumed/"final_velocity.npy"))
    corrupted = torch.load(first/"checkpoint.pt", weights_only=False)
    corrupted["fr_implementation_sha256"] = {"bad": "hash"}
    bad = tmp_path/"changed_implementation.pt"
    torch.save(corrupted, bad)
    with pytest.raises(ValueError, match="Fourier implementation"):
        runner().run_experiment(cfg, tmp_path/"bad", "cpu", resume=bad, save_figures=False)


def test_preset_changes_only_network_and_label():
    import yaml
    baseline = yaml.safe_load((ROOT/"experiments/configs/residual_ifwi_baseline1000.yaml").read_text())
    target = ROOT/"experiments/configs/residual_ifwi_fr1000.yaml"
    assert target.exists(), "Fourier GPU preset is missing"
    candidate = yaml.safe_load(target.read_text())
    runner().validate_config(candidate)
    for key in baseline:
        if key not in {"model", "experiment_name"}:
            assert candidate[key] == baseline[key], key
    for key in baseline["model"]:
        assert candidate["model"][key] == baseline["model"][key]
    assert candidate["model"]["architecture"] == "fr_siren"


def test_full_marmousi_preflight_has_no_fd_and_exact_zero_residual():
    from scripts.check_fr_ifwi import check
    result = check(ROOT/"experiments/configs/residual_ifwi_fr1000.yaml")
    assert result["passed"] and not result["finite_difference_executed"]
    assert result["velocity_shape"] == [1, 94, 288]
    assert result["initial_residual_exactly_zero"]
    assert result["model"]["trainable_parameters"] == 197505
    assert [x["numerical_rank"] for x in result["model"]["layers"]] == [126, 126, 126]


@pytest.mark.parametrize("architecture", [None, "fourier_encoder", True])
def test_unknown_architecture_is_rejected(architecture):
    cfg, _ = tiny_config()
    cfg["model"]["architecture"] = architecture
    with pytest.raises(ValueError):
        runner().validate_config(cfg)

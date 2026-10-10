"""Physical-unit, original-FD and reproducibility tests for the isolated entry."""
import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)


def api():
    return importlib.import_module("experiments.residual_ifwi_experiment")


def small_config():
    cfg = api().default_config()
    cfg["model"]["neurons"] = [2, 8, 8, 1]
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["training"].update(epochs=3, checkpoint_interval=2)
    return cfg


def fields():
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    initial = truth * .95
    return truth, initial


def test_zero_residual_exactly_preserves_background_and_units():
    _, initial = fields()
    model = api().VelocityParameterization(initial, 15, [2, 8, 8, 1], parameterization="residual")
    assert torch.equal(model(), torch.from_numpy(initial)[None])
    assert model.fixed_init_mps.requires_grad is False
    assert "fixed_init_mps" in dict(model.named_buffers())
    assert "fixed_init_mps" not in dict(model.named_parameters())
    assert model.coords.shape == (1, 8, 9, 2)
    assert model.coords[0, 2, 3].tolist() == pytest.approx([.045, .030])
    assert model().shape == (1, 8, 9)


def test_residual_has_no_extra_mean_and_uses_configured_kmps_scale():
    _, initial = fields()
    model = api().VelocityParameterization(initial, 15, [2, 8, 1], std_kmps=2, mean_kmps=4)
    with torch.no_grad():
        model.net.linear[-1].bias.fill_(.125)
    assert torch.allclose(model(), torch.from_numpy(initial)[None] + 250)


def test_original_absolute_parameterization_is_preserved():
    _, initial = fields()
    model = api().VelocityParameterization(initial, 15, [2, 8, 1], parameterization="absolute")
    raw, _ = model.net(model.coords)
    assert torch.equal(model(), (raw.squeeze(-1) + 3) * 1000)


def test_real_fd_shape_and_observations_are_detached():
    truth, _ = fields()
    problem = api().build_forward_problem(truth, small_config(), "cpu")
    assert problem["observed"].shape == (1, 2, 32, 9)
    assert problem["observed"].requires_grad is False
    assert torch.isfinite(problem["observed"]).all()
    assert problem["observed"].abs().max() > 0
    assert "truth" not in problem


def test_real_fd_backward_updates_last_then_hidden_layers_without_updating_init():
    truth, initial = fields()
    cfg = small_config()
    problem = api().build_forward_problem(truth, cfg, "cpu")
    torch.manual_seed(3)
    model = api().VelocityParameterization(initial, 15, cfg["model"]["neurons"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    initial_buffer = model.fixed_init_mps.clone()
    hidden_before = model.net.linear[0].weight.detach().clone()
    last_before = model.net.linear[-1].weight.detach().clone()
    report = api().train_update(model, problem, opt)
    assert np.isfinite(report["data_mse"]) and report["data_mse"] > 0
    assert report["gradient_norm"] > 0 and report["parameter_update_norm"] > 0
    assert torch.equal(model.net.linear[0].weight, hidden_before)
    assert not torch.equal(model.net.linear[-1].weight, last_before)
    api().train_update(model, problem, opt)
    assert not torch.equal(model.net.linear[0].weight, hidden_before)
    assert torch.equal(model.fixed_init_mps, initial_buffer)
    assert model.fixed_init_mps.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_absolute_step_matches_original_ifwi_training():
    from ifwi_modules import IFWI2D
    truth, initial = fields()
    cfg = small_config()
    problem = api().build_forward_problem(truth, cfg, "cpu")
    geometry = problem["geometry_tensors"]
    torch.manual_seed(3)
    original = IFWI2D(**geometry, dz=15, dt=.001, npad=3, order=2, vmax=float(truth.max()),
        mean=3, std=1, neuron=cfg["model"]["neurons"], omega_0=30, activation="sine",
        bias=True, dropout=False, outermost_linear=True, segment_size=32, device="cpu")
    torch.manual_seed(3)
    control = api().VelocityParameterization(initial, 15, cfg["model"]["neurons"], parameterization="absolute")
    assert all(torch.equal(a, b) for a, b in zip(original.vel_net.parameters(), control.net.parameters()))
    original.params = original.vel_net.parameters()
    original.clip = .25
    opt_original = torch.optim.Adam(original.params, lr=1e-4)
    opt_control = torch.optim.Adam(control.parameters(), lr=1e-4)
    _, losses = original.train_one_epoch(opt_original, None, problem["wavelet"], problem["observed"], 0, 0)
    report = api().train_update(control, problem, opt_control)
    assert report["data_mse"] == pytest.approx(losses[1], rel=1e-6)
    assert all(torch.allclose(a, b, rtol=1e-6, atol=1e-8) for a, b in zip(original.vel_net.parameters(), control.net.parameters()))


def test_shot_accumulation_preserves_full_objective_and_adam_update():
    truth, initial = fields()
    cfg = small_config()
    full = api().build_forward_problem(truth, cfg, "cpu")
    cfg["training"]["shot_batch_size"] = 1
    batches = api().build_forward_problem(truth, cfg, "cpu")
    assert torch.equal(full["observed"], batches["observed"])
    torch.manual_seed(3)
    first = api().VelocityParameterization(initial, 15, [2, 8, 8, 1])
    second = copy.deepcopy(first)
    a = api().train_update(first, full, torch.optim.Adam(first.parameters(), lr=1e-4))
    b = api().train_update(second, batches, torch.optim.Adam(second.parameters(), lr=1e-4))
    assert a["data_mse"] == pytest.approx(b["data_mse"], rel=1e-6)
    assert all(torch.allclose(x, y, rtol=1e-5, atol=1e-7) for x, y in zip(first.parameters(), second.parameters()))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 12000])
def test_invalid_live_velocity_fails_without_clipping(value):
    cfg = small_config()
    v = torch.full((1, 8, 9), 2500.)
    v[0, 0, 0] = value
    before = v.clone()
    with pytest.raises((ValueError, FloatingPointError)):
        api().check_velocity(v, cfg)
    assert torch.equal(v, before) or (torch.isnan(v) == torch.isnan(before)).all()


@pytest.mark.parametrize("section,key,value", [
    ("model", "attention", True), ("training", "epochs", 0),
    ("initialization", "sigma", -1), ("training", "learning_rate", float("nan")),
    ("forward", "npad", 0), ("forward", "order", 4),
])
def test_invalid_or_out_of_scope_configuration_is_rejected(section, key, value):
    cfg = small_config()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        api().validate_config(cfg)


def test_waveform_mse_has_no_velocity_supervision():
    prediction = torch.tensor([[[[1., 3.]]]], requires_grad=True)
    observations = torch.tensor([[[[2., 2.]]]])
    loss = api().waveform_mse(prediction, observations)
    assert loss.item() == 1
    loss.backward()
    assert prediction.grad.flatten().tolist() == [-1, 1]


def test_saved_final_model_loss_and_resume_match_continuous_training(tmp_path):
    truth, _ = fields()
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = small_config()
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1
    cfg["training"]["epochs"] = 3
    full = api().run_experiment(cfg, tmp_path / "full", "cpu", save_figures=False)
    partial_cfg = copy.deepcopy(cfg)
    partial_cfg["training"]["epochs"] = 1
    first = api().run_experiment(partial_cfg, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    assert np.array_equal(np.load(full / "final_velocity.npy"), np.load(resumed / "final_velocity.npy"))
    resumed_metrics = json.loads((resumed / "metrics.json").read_text())
    assert resumed_metrics["resumed_from_updates"] == 1
    saved = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["completed_updates"] == 3
    assert saved["fixed_init_units"] == "m/s"
    stats = json.loads((full / "metrics.json").read_text())
    problem = api().build_forward_problem(truth, cfg, "cpu")
    model = api().VelocityParameterization(np.load(full / "initial_velocity.npy"), 15, cfg["model"]["neurons"])
    model.load_state_dict(saved["model"])
    actual = api().evaluate_data_mse(model, problem)
    assert stats["final"]["data_mse"] == pytest.approx(actual, rel=1e-7)
    assert stats["final"]["rmse_mps"] >= 0
    assert stats["final"]["mae_mps"] >= 0
    assert np.isfinite(stats["final"]["ssim"])
    assert stats["timing"]["training_seconds"] > 0
    assert torch.equal(saved["model"]["fixed_init_mps"], torch.from_numpy(np.load(full / "initial_velocity.npy"))[None])


def test_repeat_run_does_not_overwrite_inputs_or_previous_results(tmp_path):
    truth, _ = fields()
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    before = source.read_bytes()
    cfg = small_config()
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1
    cfg["training"]["epochs"] = 1
    first = api().run_experiment(cfg, tmp_path / "runs", "cpu", save_figures=False)
    marker = (first / "metrics.json").read_bytes()
    second = api().run_experiment(cfg, tmp_path / "runs", "cpu", save_figures=False)
    assert first != second
    assert source.read_bytes() == before
    assert (first / "metrics.json").read_bytes() == marker
    assert np.array_equal(np.load(first / "final_velocity.npy"), np.load(second / "final_velocity.npy"))


def test_failed_adam_update_rolls_back_parameters_and_optimizer_state():
    truth, initial = fields()
    cfg = small_config()
    problem = api().build_forward_problem(truth, cfg, "cpu")
    model = api().VelocityParameterization(initial, 15, [2, 8, 8, 1])
    opt = torch.optim.Adam(model.parameters(), lr=100.)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises((ValueError, FloatingPointError)):
        api().train_update(model, problem, opt)
    assert opt.state_dict()["state"] == {}
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_metrics_failure_after_adam_keeps_checkpoint_consistent(tmp_path, monkeypatch):
    truth, _ = fields()
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = small_config()
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    module = api()
    real_metrics, real_update = module.velocity_metrics, module.train_update
    updated = False

    def update(*args):
        nonlocal updated
        result = real_update(*args)
        updated = True
        return result

    def metrics(*args):
        if updated:
            raise OSError("Injected reporting failure")
        return real_metrics(*args)

    monkeypatch.setattr(module, "train_update", update)
    monkeypatch.setattr(module, "velocity_metrics", metrics)
    with pytest.raises(OSError, match="Injected reporting failure"):
        module.run_experiment(cfg, tmp_path / "runs", "cpu", save_figures=False)
    output = next((tmp_path / "runs").iterdir())
    saved = torch.load(output / "interrupted.pt", map_location="cpu", weights_only=False)
    assert saved["completed_updates"] == 0
    assert saved["history"] == []
    assert saved["optimizer"]["state"] == {}
    assert torch.count_nonzero(saved["model"]["net.linear.2.weight"]) == 0


def test_full_grid_coordinates_match_original_ifwi_exactly():
    from ifwi_modules import IFWI2D
    nz, nx = 94, 288
    xs = torch.tensor([[20]])
    zs = torch.tensor([[1]])
    xr = torch.arange(nx)[None, None]
    zr = torch.full_like(xr, 2)
    original = IFWI2D(nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr, dz=15.,
                      dt=.0019, npad=15, order=2, vmax=5500., device="cpu")
    model = api().VelocityParameterization(np.full((nz, nx), 3000., np.float32), 15., [2, 8, 1])
    assert torch.equal(original.coords, model.coords)

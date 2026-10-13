"""Low-rank residual (U C V^T), its composition with FR / trainable omega, and its reports; tiny CPU FD only."""
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

SMALL_FR = {"high_freq_num": 8, "low_freq_num": 8, "phi_num": 8, "alpha": .01, "lambda_init": "fr_inr"}
SMALL_LR = {"rank_z": 3, "rank_x": 4, "hidden": [8, 8], "coord_scale": .5, "output_gain": 1.}


def mod(name):
    return importlib.import_module(f"experiments.{name}")


def api():
    return mod("residual_ifwi_experiment")


def background(nz=6, nx=9):
    return np.linspace(1500, 4500, nz * nx, dtype=np.float32).reshape(nz, nx)


def lr_model(fourier=None, trainable_omega=False, setting=SMALL_LR, initial=None):
    initial = background() if initial is None else initial
    return api().VelocityParameterization(initial, 15, None, fourier=fourier,
                                          trainable_omega=trainable_omega, low_rank=setting)


def lr_config(**model):
    cfg = api().default_config()
    cfg["model"].update(neurons=None, low_rank=copy.deepcopy(SMALL_LR), **model)
    return cfg


@pytest.mark.parametrize("fourier,trainable_omega", [(None, False), (SMALL_FR, False), (SMALL_FR, True), (None, True)])
def test_starts_exactly_at_background_with_zero_core(fourier, trainable_omega):
    model = lr_model(fourier, trainable_omega)
    assert torch.equal(model(), model.fixed_init_mps)
    assert model.low_rank.core.count_nonzero() == 0
    assert model.net is None and not hasattr(model, "coords")
    assert model.low_rank.core.shape == (3, 4)
    if fourier is not None:
        assert model.fourier_layers == {"z": [1], "x": [1]}
    if trainable_omega:
        assert model.omega_layers == {"z": [0, 1], "x": [0, 1]}
        assert mod("trainable_omega").omega_values(model) == [30.] * 4


def test_output_is_gain_times_u_core_vt_with_bounded_rank():
    torch.manual_seed(0)
    model = lr_model(setting={**SMALL_LR, "output_gain": .25})
    with torch.no_grad():
        model.low_rank.core.normal_()
        u, v = model.low_rank.factors()
        assert u.shape == (6, 3) and v.shape == (9, 4)
        expected = model.fixed_init_mps[0] + 1000 * .25 * (u @ model.low_rank.core @ v.T)
        assert torch.allclose(model()[0], expected, rtol=0, atol=1e-3)
        # Rank of the product itself: v - v_init carries float32 rounding of the ~3000 m/s background.
        singular = torch.linalg.svdvals(model.low_rank().double())
        assert (singular > 1e-4 * singular[0]).sum() <= 3


def test_factors_see_one_axis_each_with_scaled_km_coordinates():
    model = lr_model()
    lr = model.low_rank
    assert torch.equal(lr.z_coords[:, 0], torch.from_numpy((np.arange(6) * 15 / 1000 * .5).astype(np.float32)))
    assert torch.equal(lr.x_coords[:, 0], torch.from_numpy((np.arange(9) * 15 / 1000 * .5).astype(np.float32)))
    # IRN draws 1-input first-layer weights from U(-1, 1); coord_scale 0.5 halves the effective range.
    for net in (lr.net_z, lr.net_x):
        assert net.linear[0].in_features == 1 and net.linear[0].weight.abs().max() <= 1
    assert lr.net_z.linear[-1].out_features == 3 and lr.net_x.linear[-1].out_features == 4


def test_first_gradient_reaches_only_the_core_then_the_factors():
    model = lr_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gradient = torch.randn_like(model.fixed_init_mps)
    model().backward(gradient)
    assert model.low_rank.core.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if name != "low_rank.core":
            assert parameter.grad is not None and parameter.grad.count_nonzero() == 0, name
    optimizer.step()
    optimizer.zero_grad()
    model().backward(gradient)
    assert model.low_rank.net_z.linear[-1].weight.grad.abs().sum() > 0
    assert model.low_rank.net_x.linear[0].weight.grad.abs().sum() > 0


def test_trainable_omega_moves_only_after_the_core_and_keeps_its_group():
    model = lr_model(SMALL_FR, True)
    groups = mod("trainable_omega").parameter_groups(model, 1e-4, {"learning_rate": 1e-2})
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    assert len(groups[1]["params"]) == 4 and any(p is model.low_rank.core for p in groups[0]["params"])
    model().backward(torch.randn_like(model.fixed_init_mps))
    assert all(p.grad.count_nonzero() == 0 for p in groups[1]["params"])


def test_seed_matched_construction_is_deterministic():
    torch.manual_seed(3)
    first = lr_model(SMALL_FR)
    torch.manual_seed(3)
    second = lr_model(SMALL_FR)
    for (name, a), b in zip(first.state_dict().items(), second.state_dict().values()):
        assert torch.equal(a, b), name


def test_validation_and_resume_compatibility():
    cfg = lr_config()
    api().validate_config(cfg)
    api().validate_config(lr_config(fourier=SMALL_FR, trainable_omega={"learning_rate": 1e-2}))
    plain = api().default_config()
    assert api()._resume_configuration(plain) == api()._resume_configuration(
        {**plain, "model": {**plain["model"], "low_rank": None}})
    bad_settings = [{**SMALL_LR, "rank_z": 0}, {**SMALL_LR, "rank_x": 2.}, {**SMALL_LR, "hidden": []},
                    {**SMALL_LR, "hidden": [8, 0]}, {**SMALL_LR, "coord_scale": 0}, {**SMALL_LR, "output_gain": True},
                    {**SMALL_LR, "extra": 1}, {k: v for k, v in SMALL_LR.items() if k != "rank_x"}, 3]
    for setting in bad_settings:
        bad = copy.deepcopy(cfg)
        bad["model"]["low_rank"] = setting
        with pytest.raises(ValueError):
            api().validate_config(bad)
    leftover = copy.deepcopy(cfg)
    leftover["model"]["neurons"] = [2, 8, 1]
    with pytest.raises(ValueError, match="neurons must be null"):
        api().validate_config(leftover)
    # Absolute (LR-IFWI paper mode) is allowed; a residual model must keep the zero core.
    absolute = copy.deepcopy(cfg)
    absolute["parameterization"] = "absolute"
    absolute["model"]["low_rank"].update(coordinates="index", core_init="random")
    api().validate_config(absolute)
    random_residual = copy.deepcopy(cfg)
    random_residual["model"]["low_rank"]["core_init"] = "random"
    with pytest.raises(ValueError, match="core_init zero"):
        api().validate_config(random_residual)
    for key, value in (("coordinates", "grid"), ("core_init", "normal")):
        bad = copy.deepcopy(absolute)
        bad["model"]["low_rank"][key] = value
        with pytest.raises(ValueError):
            api().validate_config(bad)
    shallow = lr_config(fourier=SMALL_FR)
    shallow["model"]["low_rank"]["hidden"] = [8]
    with pytest.raises(ValueError, match="hidden-to-hidden"):
        api().validate_config(shallow)
    no_lr = api().default_config()
    no_lr["model"]["neurons"] = None
    with pytest.raises(ValueError):
        api().validate_config(no_lr)


def tiny_config(tmp_path, **model):
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    truth[5:7, 3:6] = 3400.
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = lr_config(**model)
    cfg["model"]["low_rank"]["output_gain"] = 5.
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1
    cfg["training"].update(epochs=3, checkpoint_interval=1)
    return cfg


@pytest.mark.parametrize("model", [{}, {"fourier": SMALL_FR, "trainable_omega": {"learning_rate": 1e-2}}])
def test_run_starts_at_background_and_resumes_exactly(tmp_path, model):
    cfg = tiny_config(tmp_path, **model)
    full = api().run_experiment(cfg, tmp_path / "full", "cpu", save_figures=False)
    checks = json.loads((full / "initial_checks.json").read_text())
    assert checks["max_residual_at_run_start_mps"] == 0 and checks["low_rank_core_zero_at_start"]
    assert checks["coordinates_order"] == ["z", "x"]
    notes = json.loads((full / "comparison_notes.json").read_text())
    assert any("low-rank residual" in change for change in notes["changes_vs_original_random"])
    assert notes["initialization"].endswith("zero low-rank core C")
    final = np.load(full / "final_velocity.npy")
    assert not np.array_equal(final, np.load(full / "initial_velocity.npy"))
    partial = copy.deepcopy(cfg)
    partial["training"]["epochs"] = 1
    first = api().run_experiment(partial, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    assert np.array_equal(final, np.load(resumed / "final_velocity.npy"))


def test_resume_rejects_a_changed_low_rank_setting(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["training"]["epochs"] = 1
    first = api().run_experiment(cfg, tmp_path / "first", "cpu", save_figures=False)
    changed = copy.deepcopy(cfg)
    changed["training"]["epochs"] = 2
    changed["model"]["low_rank"]["rank_x"] = 5
    with pytest.raises(ValueError, match="configuration differs"):
        api().run_experiment(changed, tmp_path / "changed", "cpu", resume=first / "checkpoint.pt", save_figures=False)


def test_calibrated_gain_matches_reference_first_step(tmp_path):
    cfg = tiny_config(tmp_path)
    reference = copy.deepcopy(cfg)
    reference["experiment_name"] = "reference"
    reference["model"] = api().default_config()["model"]
    reference["model"]["neurons"] = [2, 8, 8, 1]
    result = mod("low_rank_calibration").calibrate(cfg, reference, "cpu")
    assert result["reference_first_step"]["rms_mps"] > 0 and result["low_rank_first_step"]["rms_mps"] > 0
    # Step 1 moves only C by ~lr*sign(g), so the velocity change is linear in the gain.
    assert result["recommended_first_step"]["rms_ratio_to_reference"] == pytest.approx(1, rel=1e-3)
    other = copy.deepcopy(reference)
    other["forward"]["nt"] = 30
    with pytest.raises(ValueError, match="forward"):
        mod("low_rank_calibration").calibrate(cfg, other, "cpu")


def test_step_report_rebuilds_low_rank_checkpoints(tmp_path):
    report = mod("residual_step_report")
    cfg = lr_config(fourier=SMALL_FR, trainable_omega={"learning_rate": 1e-2})
    torch.manual_seed(0)
    model = lr_model(SMALL_FR, True, initial=np.full((5, 6), 3000., np.float32))
    base = copy.deepcopy(model.state_dict())
    direction = {n: torch.randn_like(p) for n, p in model.named_parameters() if not n.endswith("omega")}
    (tmp_path / "checkpoints").mkdir()
    for index, scale in enumerate([0., .1, .3], start=1):
        state = copy.deepcopy(base)
        for name, d in direction.items():
            state[name] = base[name] + scale * d
        torch.save({"config": cfg, "model": state}, tmp_path / "checkpoints" / f"update_{50 * index:04d}.pt")
    rows, config = report.displacement_series([tmp_path])
    names = report.layer_labels(config)
    assert names == ["z0", "z1", "z2", "x0", "x1", "x2", "core"]
    assert len(rows[0]["displacement"]) == len(names)
    assert rows[1]["displacement"][-1] == pytest.approx(2 * rows[0]["displacement"][-1])
    payload = torch.load(tmp_path / "checkpoints" / "update_0050.pt", weights_only=False)
    weights = report.preactivation_weights(report.build_model(payload["config"], payload["model"]))
    assert np.allclose(weights[-1], payload["model"]["low_rank.core"].double().numpy())
    plain = [{"start": 50, "end": 100, "displacement": [1., 1., 1.], "norm_start": [1., 1., 1.]}]
    with pytest.raises(ValueError, match="layer structures"):
        report.compare([plain, rows], ["plain", "lr"])


def test_lowrank_report_spectra_ranks_and_relative_change(tmp_path):
    report = mod("residual_lowrank_report")
    rng = np.random.default_rng(0)
    background = np.full((10, 12), 3000.)
    truth = background + rng.normal(0, 100, background.shape)
    rank_one = np.outer(np.linspace(1, 2, 10), np.linspace(-1, 1, 12)) * 50
    rank_two = rank_one + np.outer(np.sin(np.arange(10)), np.cos(np.arange(12))) * 20
    (tmp_path / "checkpoints").mkdir()
    np.save(tmp_path / "true_velocity.npy", truth)
    np.save(tmp_path / "initial_velocity.npy", background)
    np.save(tmp_path / "checkpoints" / "velocity_0050.npy", background + rank_one)
    np.save(tmp_path / "checkpoints" / "velocity_0100.npy", background + rank_two)
    result = report.run_series([tmp_path])
    rows = result["rows"]
    assert [r["update"] for r in rows] == [0, 50, 100]
    assert rows[0]["entropy_rank"] is None and rows[0]["frobenius_mps"] == 0
    assert rows[1]["r99"] == 1 and rows[1]["entropy_rank"] == pytest.approx(1)
    assert rows[2]["r99"] == 2 and 1 < rows[2]["entropy_rank"] <= 2
    assert result["truth_residual"]["r99"] > 2
    expected = np.square(rank_two - rank_one).sum() / np.square(background + rank_one).sum()
    assert rows[2]["window_start"] == 50 and rows[2]["relative_change"] == pytest.approx(expected)


@pytest.mark.parametrize("name,base", [("paper13_lr_fr_ifwi", "paper13_fr_ifwi"), ("paper13_lr_ifwi", "paper13_ifwi")])
def test_paper13_low_rank_configs_change_only_the_residual_network(name, base):
    yaml = pytest.importorskip("yaml")
    folder = ROOT / "experiments/configs"
    cfg, ref = (yaml.safe_load((folder / f"{n}.yaml").read_text(encoding="utf-8")) for n in (name, base))
    api().validate_config(cfg)
    # The calibration script requires every non-model section to match its reference.
    mod("low_rank_calibration").check_comparable(cfg, ref)
    assert cfg["model"].pop("low_rank") == {**mod("low_rank_modules").default_low_rank(), "output_gain": 1.}
    assert cfg["model"].pop("neurons") is None and ref["model"].pop("neurons") == [2, 128, 128, 128, 128, 1]
    assert cfg["model"] == ref["model"]  # FR and trainable omega kept exactly as in the base run
    assert cfg["experiment_name"] == name

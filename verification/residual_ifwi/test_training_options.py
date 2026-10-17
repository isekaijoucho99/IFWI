"""Adam betas/eps, late-arrival time gain and EMA / lowest-misfit estimates; tiny CPU FD runs."""
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

STAGES = [{"max_hz": 20.0, "updates": 1}, {"max_hz": None}]


def opts():
    return importlib.import_module("experiments.training_options")


def api():
    return importlib.import_module("experiments.residual_ifwi_experiment")


def tiny_config(tmp_path, **training):
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    truth[5:7, 3:6] = 3400.
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = api().default_config()
    cfg["model"]["neurons"] = [2, 8, 8, 8, 1]
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1
    cfg["training"].update(epochs=4, checkpoint_interval=1, **copy.deepcopy(training))
    return cfg


def test_adam_betas_and_eps_are_optional_and_validated():
    muon = importlib.import_module("experiments.muon_optimizer")
    model = api().VelocityParameterization(np.full((5, 6), 3000., np.float32), 15, [2, 8, 8, 1])
    cfg = api().default_config()
    default = muon.build_optimizer(model, cfg)
    assert all(g["betas"] == (.9, .999) and g["eps"] == 1e-8 for g in default.param_groups)
    cfg["training"]["optimizer"] = {"name": "adam", "betas": [.9, .99], "eps": 1e-7}
    api().validate_config(cfg)
    tuned = muon.build_optimizer(model, cfg)
    assert all(g["betas"] == (.9, .99) and g["eps"] == 1e-7 and g["lr"] == 1e-4 for g in tuned.param_groups)
    for bad in ({"name": "adam", "betas": [.9]}, {"name": "adam", "betas": [.9, 1.]}, {"name": "adam", "eps": 0},
                {"name": "adam", "betas": [.9, True]}, {"name": "adam", "learning_rate": 1e-3}):
        broken = copy.deepcopy(cfg)
        broken["training"]["optimizer"] = bad
        with pytest.raises(ValueError):
            api().validate_config(broken)
    # A changed Adam setting is a different run for resuming.
    plain = api().default_config()
    assert api()._resume_configuration(plain) != api()._resume_configuration(cfg)


def test_time_gain_weights_and_energy_scale():
    records = torch.ones(1, 2, 11, 3, dtype=torch.float64)
    gained = opts().time_gain(records, 2., .1)
    assert torch.allclose(gained[0, 0, :, 0], (torch.arange(11, dtype=torch.float64) / 10) ** 2)
    observed = torch.randn(1, 2, 64, 3, generator=torch.Generator().manual_seed(0))
    scale = opts().gain_scale(observed, None, 1., .001)
    zero = torch.zeros_like(observed)
    full = api().waveform_mse(zero, observed)
    assert float(scale * api().waveform_mse(zero, opts().time_gain(observed, 1., .001))) == pytest.approx(float(full), rel=1e-5)
    assert [opts().time_power({"power": 1.5, "start_update": 3}, u) for u in (1, 2, 3, 9)] == [None, None, 1.5, 1.5]
    assert opts().time_power(None, 5) is None
    for bad in ({"power": 0., "start_update": 1}, {"power": 1., "start_update": 0}, {"power": 1.}, {"power": True, "start_update": 1}):
        with pytest.raises(ValueError):
            opts().validate_time_gain(bad)


def test_time_gain_run_logs_its_objective_combines_with_stages_and_resumes(tmp_path):
    gain = {"power": 1.0, "start_update": 2}
    cfg = tiny_config(tmp_path, time_gain=gain, frequency_schedule=copy.deepcopy(STAGES))
    full = api().run_experiment(cfg, tmp_path / "full", "cpu", save_figures=False)
    history = json.loads((full / "history.json").read_text())
    assert [h.get("time_gain_power") for h in history[:4]] == [None, 1., 1., 1.]
    assert [h.get("max_hz") for h in history[:4]] == [20., None, None, None]
    assert all(h["objective"] > 0 for h in history[:4])
    plain = api().run_experiment(tiny_config(tmp_path, frequency_schedule=copy.deepcopy(STAGES)), tmp_path / "plain", "cpu",
                                 save_figures=False)
    plain_history = json.loads((plain / "history.json").read_text())
    assert history[0] == {**plain_history[0], "step_seconds": history[0]["step_seconds"]}     # identical until the gain starts
    assert history[1]["data_mse"] == plain_history[1]["data_mse"]                         # same state, plain MSE logged
    assert not np.array_equal(np.load(full / "final_velocity.npy"), np.load(plain / "final_velocity.npy"))
    notes = json.loads((full / "comparison_notes.json").read_text())
    assert notes["time_gain"] == gain and any("time gain" in c for c in notes["changes_vs_original_random"])
    partial = copy.deepcopy(cfg)
    partial["training"]["epochs"] = 1
    first = api().run_experiment(partial, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    assert np.array_equal(np.load(full / "final_velocity.npy"), np.load(resumed / "final_velocity.npy"))
    changed = copy.deepcopy(cfg)
    changed["training"]["time_gain"]["power"] = 2.
    with pytest.raises(ValueError, match="configuration differs"):
        api().run_experiment(changed, tmp_path / "wrong", "cpu", resume=first / "checkpoint.pt", save_figures=False)


def test_report_estimates_ema_and_lowest_misfit_without_changing_training(tmp_path):
    setting = {"ema_decay": .5, "ema_start": 2}
    cfg = tiny_config(tmp_path, report_estimates=setting)
    run = api().run_experiment(cfg, tmp_path / "run", "cpu", save_figures=False)
    plain = api().run_experiment(tiny_config(tmp_path), tmp_path / "plain", "cpu", save_figures=False)
    assert np.array_equal(np.load(run / "final_velocity.npy"), np.load(plain / "final_velocity.npy"))   # training unchanged
    assert not (plain / "final_velocity_ema.npy").exists() and "estimates" not in json.loads((plain / "metrics.json").read_text())
    # EMA over post-update velocities of updates 2, 3, 4 (checkpoint snapshots), decay 0.5.
    snaps = {u: np.load(run / "checkpoints" / f"velocity_{u:04d}.npy").astype(np.float64) for u in (1, 2, 3, 4)}
    expected = snaps[2]
    for u in (3, 4):
        expected = .5 * expected + .5 * snaps[u]
    assert np.allclose(np.load(run / "final_velocity_ema.npy"), expected, rtol=0, atol=1e-3)
    metrics = json.loads((run / "metrics.json").read_text())["estimates"]
    assert metrics["ema"]["averaged_updates"] == 3 and metrics["ema"]["data_mse"] > 0
    # Lowest misfit among the pre-update evaluations (states 0-3) and the final post-update evaluation (state 4).
    history = json.loads((run / "history.json").read_text())
    evaluated = {h["evaluated_updates"]: h["data_mse"] for h in history}
    best_state = min(evaluated, key=evaluated.get)
    assert metrics["lowest_misfit"]["completed_updates"] == best_state
    assert metrics["lowest_misfit"]["data_mse"] == evaluated[best_state]
    states = {0: np.load(run / "run_start_velocity.npy"), **{u: snaps[u].astype(np.float32) for u in snaps}}
    assert np.array_equal(np.load(run / "lowest_misfit_velocity.npy"), states[best_state])
    # Resuming restores both estimates exactly.
    partial = copy.deepcopy(cfg)
    partial["training"]["epochs"] = 2
    first = api().run_experiment(partial, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    for name in ("final_velocity_ema.npy", "lowest_misfit_velocity.npy"):
        assert np.array_equal(np.load(run / name), np.load(resumed / name))
    for bad in ({"ema_decay": 1., "ema_start": 1}, {"ema_decay": 0., "ema_start": 1}, {"ema_decay": .9, "ema_start": 0},
                {"ema_decay": .9}):
        with pytest.raises(ValueError):
            opts().validate_report_estimates(bad)


def test_round_two_configs_change_one_setting_each():
    yaml = pytest.importorskip("yaml")
    folder = ROOT / "experiments/configs"
    load = lambda n: yaml.safe_load((folder / f"{n}.yaml").read_text(encoding="utf-8"))
    base = load("paper13_fr_ifwi_lrsched")
    assert base["training"]["report_estimates"] == {"ema_decay": .99, "ema_start": 501}
    changes = {"paper13_fr_ifwi_lrsched_beta2": lambda c: c["training"].pop("optimizer") == {"name": "adam", "betas": [.9, .99]},
               "paper13_fr_ifwi_lrsched_tgain": lambda c: c["training"].pop("time_gain") == {"power": 1., "start_update": 301},
               "paper13_fr_ifwi_lrsched_alpha005": lambda c: c["model"]["fourier"].update(alpha=.01) is None
                                                              and load("paper13_fr_ifwi_lrsched_alpha005")["model"]["fourier"]["alpha"] == .005}
    for name, undo in changes.items():
        cfg = load(name)
        api().validate_config(cfg)
        assert undo(cfg)
        cfg["experiment_name"] = base["experiment_name"]
        assert cfg == base

"""Per-layer trainable omega and paper-protocol configs; tiny CPU FD only."""
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


def api():
    return importlib.import_module("experiments.residual_ifwi_experiment")


def om():
    return importlib.import_module("experiments.trainable_omega")


@pytest.mark.parametrize("fourier", [None, SMALL_FR])
def test_trainable_omega_starts_identical_to_fixed_omega(fourier):
    initial = np.linspace(1500, 4500, 6 * 9, dtype=np.float32).reshape(6, 9)
    torch.manual_seed(3)
    fixed = api().VelocityParameterization(initial, 15, [2, 8, 8, 8, 1], fourier=fourier)
    torch.manual_seed(3)
    trainable = api().VelocityParameterization(initial, 15, [2, 8, 8, 8, 1], fourier=fourier, trainable_omega=True)
    assert trainable.omega_layers == [0, 1, 2]
    assert om().omega_values(trainable) == [30.] * 3
    with torch.no_grad():
        for model in (fixed, trainable):
            model.net.linear[-1].weight.fill_(.01)
        assert torch.equal(fixed(), trainable())
    names = {name for name, _ in trainable.named_parameters()}
    assert {"net.linear.0.omega", "net.linear.1.omega", "net.linear.2.omega"} <= names
    assert not any(name.startswith("net.linear.3.") and name.endswith("omega") for name in names)


def test_omega_receives_gradient_and_has_its_own_learning_rate():
    initial = np.full((5, 6), 3000., np.float32)
    model = api().VelocityParameterization(initial, 15, [2, 8, 8, 1], fourier=SMALL_FR, trainable_omega=True)
    with torch.no_grad():
        model.net.linear[-1].weight.fill_(.01)
    model().sum().backward()
    assert all(model.net.linear[i].omega.grad.abs() > 0 for i in (0, 1))
    groups = om().parameter_groups(model, 1e-4, {"learning_rate": 1e-2})
    assert [g["lr"] for g in groups] == [1e-4, 1e-2]
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    optimizer = torch.optim.Adam(groups)
    before = om().omega_values(model)
    optimizer.step()
    # Adam's first step moves each parameter by about its group lr.
    assert all(abs(abs(a - b) - 1e-2) < 1e-4 for a, b in zip(om().omega_values(model), before))


def test_plain_groups_match_original_adam_construction():
    initial = np.full((5, 6), 3000., np.float32)
    model = api().VelocityParameterization(initial, 15, [2, 8, 8, 1])
    groups = om().parameter_groups(model, 1e-4, None)
    assert len(groups) == 1 and groups[0]["params"] == list(model.parameters())


def test_trainable_omega_validation_and_resume_compatibility():
    cfg = api().default_config()
    assert api()._resume_configuration(cfg) == api()._resume_configuration({**cfg, "model": {**cfg["model"], "trainable_omega": None}})
    cfg["model"]["trainable_omega"] = {"learning_rate": 1e-2}
    api().validate_config(cfg)
    for bad_value in ({"learning_rate": 0}, {"learning_rate": True}, {"lr": 1e-2}, {"learning_rate": 1e-2, "x": 1}, 1e-2):
        bad = copy.deepcopy(cfg)
        bad["model"]["trainable_omega"] = bad_value
        with pytest.raises(ValueError):
            api().validate_config(bad)


def test_trainable_omega_run_logs_omega_and_resumes_exactly(tmp_path):
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = api().default_config()
    cfg["model"].update(neurons=[2, 8, 8, 1], fourier=SMALL_FR, trainable_omega={"learning_rate": 1e-2})
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1
    cfg["training"].update(epochs=3, checkpoint_interval=1)
    full = api().run_experiment(cfg, tmp_path / "full", "cpu", save_figures=False)
    history = json.loads((full / "history.json").read_text())
    assert len(history[0]["omega_after"]) == 2
    # Zero last layer: omega gets no gradient on update 1, then starts to move.
    assert history[0]["omega_after"] == [30., 30.]
    assert history[-2]["omega_after"] != [30., 30.]
    partial = copy.deepcopy(cfg)
    partial["training"]["epochs"] = 1
    first = api().run_experiment(partial, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    assert np.array_equal(np.load(full / "final_velocity.npy"), np.load(resumed / "final_velocity.npy"))


def _yaml(name):
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / "experiments/configs" / f"{name}.yaml").read_text(encoding="utf-8"))


def test_omega_baseline1000_config_adds_only_trainable_omega():
    cfg, base = _yaml("residual_fr_ifwi_omega_baseline1000"), _yaml("residual_fr_ifwi_baseline1000")
    api().validate_config(cfg)
    assert cfg["model"].pop("trainable_omega") == {"learning_rate": .01}
    base["experiment_name"] = cfg["experiment_name"]
    assert cfg == base


def test_paper13_configs_follow_paper_protocol_and_differ_only_in_network():
    ifwi, fr_ifwi, stage1 = _yaml("paper13_ifwi"), _yaml("paper13_fr_ifwi"), _yaml("residual_ifwi_stage1")
    for cfg in (ifwi, fr_ifwi):
        api().validate_config(cfg)
    # Sec. 3.1: 13 shots at 300 m (20 cells), 1000 samples at 1.9 ms, 8 Hz, every grid point a receiver.
    assert ifwi["acquisition"]["num_shots"] == 13 and ifwi["acquisition"]["source_spacing_index"] * 15 == 300
    assert ifwi["forward"]["nt"] == 1000 and ifwi["forward"]["dt_s"] == .0019 and ifwi["forward"]["frequency_hz"] == 8
    assert ifwi["acquisition"]["receiver_stride"] == 1
    assert ifwi["training"].get("shots_per_update") is None  # all shots in every update
    assert ifwi["training"]["epochs"] == 1000
    assert "fourier" not in ifwi["model"] and "trainable_omega" not in ifwi["model"]
    assert fr_ifwi["model"].pop("trainable_omega") == {"learning_rate": .01}
    assert fr_ifwi["model"].pop("fourier") == importlib.import_module("experiments.fourier_modules").default_fourier()
    fr_ifwi["experiment_name"] = ifwi["experiment_name"]
    assert fr_ifwi == ifwi
    stage1.update(experiment_name=ifwi["experiment_name"])
    stage1["training"]["epochs"] = 1000
    assert stage1 == ifwi

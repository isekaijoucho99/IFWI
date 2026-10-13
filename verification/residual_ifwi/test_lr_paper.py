"""LR-IFWI paper mode: absolute low-rank model, index coordinates, random core, Stage-1 pretraining."""
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

PAPER_LR = {"rank_z": 3, "rank_x": 4, "hidden": [8, 8], "coordinates": "index", "coord_scale": 1.,
            "core_init": "random", "output_gain": 1.}


def mod(name):
    return importlib.import_module(f"experiments.{name}")


def api():
    return mod("residual_ifwi_experiment")


def paper_model(setting=PAPER_LR, seed=3, initial=None):
    torch.manual_seed(seed)
    initial = np.full((6, 9), 3000., np.float32) if initial is None else initial
    return api().VelocityParameterization(initial, 15, None, parameterization="absolute", low_rank=dict(setting))


def test_index_coordinates_and_random_core_after_seed_matched_factors():
    model = paper_model().low_rank
    assert model.z_coords[:, 0].tolist() == list(range(1, 7))
    assert model.x_coords[:, 0].tolist() == list(range(1, 10))
    assert model.core.abs().max() > 0 and model.core.abs().max() <= 1 / np.sqrt(4)
    zero = paper_model({**PAPER_LR, "core_init": "zero"}).low_rank
    # The core is drawn after both factor networks, so their weights match the zero-core model.
    assert all(torch.equal(a, b) for a, b in zip(model.net_z.parameters(), zero.net_z.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(model.net_x.parameters(), zero.net_x.parameters()))
    assert torch.count_nonzero(zero.core) == 0


def test_absolute_low_rank_velocity_is_mean_plus_std_times_ucvt():
    model = paper_model({**PAPER_LR, "output_gain": 2.})
    model.std_kmps, model.mean_kmps = 1.5, 3.
    with torch.no_grad():
        u, v = model.low_rank.factors()
        expected = (2. * (u @ model.low_rank.core @ v.T) * 1.5 + 3.) * 1000
        assert torch.allclose(model()[0], expected)


def test_pretraining_fits_the_initial_model_and_keeps_the_lowest_loss_state():
    initial = np.linspace(1800, 3600, 6 * 9, dtype=np.float32).reshape(6, 9)
    model = paper_model(initial=initial)
    with torch.no_grad():
        before = float((model()[0] - torch.from_numpy(initial)).square().mean().sqrt())
    history, best = api().pretrain_to_initial(model, initial, {"updates": 40, "learning_rate": 1e-2}, None, 3., 1., emit=lambda _: None)
    assert [row["completed_updates"] for row in history] == list(range(41))
    losses = [row["normalized_mse"] for row in history]
    assert best == int(np.argmin(losses))
    with torch.no_grad():
        after = float((model()[0] - torch.from_numpy(initial)).square().mean().sqrt())
    assert after == pytest.approx(history[best]["rmse_vs_initial_mps"], rel=1e-4)
    assert after < .5 * before


def test_pretrain_validation():
    cfg = api().default_config()
    cfg["parameterization"] = "absolute"
    cfg["initialization"]["pretrain"] = {"updates": 10, "learning_rate": 1e-4}
    api().validate_config(cfg)
    residual = copy.deepcopy(cfg)
    residual["parameterization"] = "residual"
    api().validate_config(residual)  # pretrained features, head zeroed afterwards
    for bad in ({"updates": 0, "learning_rate": 1e-4}, {"updates": 10}, {"updates": 10, "learning_rate": -1},
                {"updates": 10.5, "learning_rate": 1e-4}, {"updates": 10, "learning_rate": 1e-4, "x": 1}):
        broken = copy.deepcopy(cfg)
        broken["initialization"]["pretrain"] = bad
        with pytest.raises(ValueError):
            api().validate_config(broken)
    plain = api().default_config()
    assert api()._resume_configuration(plain) == api()._resume_configuration(
        {**plain, "initialization": {**plain["initialization"], "pretrain": None}})


def tiny_paper_config(tmp_path, low_rank=True):
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    truth[5:7, 3:6] = 3400.
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = api().default_config()
    cfg["parameterization"] = "absolute"
    if low_rank:
        cfg["model"].update(neurons=None, low_rank=dict(PAPER_LR))
    else:
        cfg["model"]["neurons"] = [2, 8, 8, 1]
    cfg["initialization"].update(sigma=1, pretrain={"updates": 30, "learning_rate": 1e-2})
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6], source_depth_index=2, receiver_depth_index=0)
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["training"].update(epochs=3, checkpoint_interval=1)
    return cfg


@pytest.mark.parametrize("low_rank", [True, False])
def test_paper_run_pretrains_records_and_resumes_without_repeating_pretraining(tmp_path, low_rank):
    cfg = tiny_paper_config(tmp_path, low_rank)
    full = api().run_experiment(cfg, tmp_path / "full", "cpu", save_figures=False)
    info = json.loads((full / "pretraining.json").read_text())
    assert info["updates"] == 30 and 0 <= info["selected_update"] <= 30
    rows = (full / "pretraining_history.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 31
    start, initial = np.load(full / "run_start_velocity.npy"), np.load(full / "initial_velocity.npy")
    assert np.sqrt(np.mean((start - initial) ** 2)) == pytest.approx(info["rmse_vs_initial_mps"], rel=1e-5)
    # Surface receivers record a nonzero wavefield under the free surface.
    assert np.abs(np.load(full / "observed.npy")).max() > 0
    partial = copy.deepcopy(cfg)
    partial["training"]["epochs"] = 1
    first = api().run_experiment(partial, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    assert not (resumed / "pretraining.json").exists()
    assert np.array_equal(np.load(full / "final_velocity.npy"), np.load(resumed / "final_velocity.npy"))


def test_lrpaper13_configs_follow_the_paper_and_differ_only_in_the_network():
    yaml = pytest.importorskip("yaml")
    folder = ROOT / "experiments/configs"
    lr, ifwi = (yaml.safe_load((folder / f"{n}.yaml").read_text(encoding="utf-8")) for n in ("lrpaper13_lr_ifwi", "lrpaper13_ifwi"))
    for cfg in (lr, ifwi):
        api().validate_config(cfg)
        assert cfg["parameterization"] == "absolute"
        assert cfg["initialization"]["pretrain"] == {"updates": 5000, "learning_rate": 1e-4}
        assert cfg["model"]["mean_kmps"] == 3 and cfg["model"]["std_kmps"] == 1 and cfg["model"]["omega_0"] == 30
        acq, fd, training = cfg["acquisition"], cfg["forward"], cfg["training"]
        # Sec. III-A: 13 shots at 300 m, receivers every 15 m; depths follow the original IFWI code
        # (sources 15 m, receivers 30 m) like all earlier 13-shot runs, not the papers' text.
        assert acq["num_shots"] == 13 and acq["source_spacing_index"] * 15 == 300 and acq["receiver_stride"] == 1
        assert acq["source_depth_index"] == 1 and acq["receiver_depth_index"] == 2
        assert fd["frequency_hz"] == 8 and fd["dt_s"] == .0019 and fd["nt"] == 1000 and fd["free_surface"] is True
        assert training["learning_rate"] == 1e-4 and training["epochs"] == 1000 and "shots_per_update" not in training
    assert lr["model"].pop("low_rank") == {"rank_z": 50, "rank_x": 100, "hidden": [128, 128, 128], "coordinates": "index",
                                           "coord_scale": 1.0, "core_init": "random", "output_gain": 1.0}
    assert lr["model"].pop("neurons") is None and ifwi["model"].pop("neurons") == [2, 128, 128, 128, 128, 1]
    lr["experiment_name"] = ifwi["experiment_name"]
    assert lr == ifwi
    paper13 = yaml.safe_load((folder / "paper13_ifwi.yaml").read_text(encoding="utf-8"))
    for section in ("data", "acquisition", "forward", "training"):
        assert ifwi[section] == paper13[section]  # comparable with the earlier 13-shot runs


def test_factorial_design_varies_only_the_intended_factors():
    """Residual vs paper mode x 2-D vs low-rank x no FR vs FR + omega share all non-model settings."""
    yaml = pytest.importorskip("yaml")
    folder = ROOT / "experiments/configs"
    load = lambda n: yaml.safe_load((folder / f"{n}.yaml").read_text(encoding="utf-8"))
    paper = {n: load(n) for n in ("lrpaper13_ifwi", "lrpaper13_fr_ifwi", "lrpaper13_lr_ifwi", "lrpaper13_lr_fr_ifwi")}
    residual = {n: load(n) for n in ("paper13_ifwi", "paper13_fr_ifwi", "paper13_lr_ifwi", "paper13_lr_fr_ifwi")}
    for cfg in [*paper.values(), *residual.values()]:
        api().validate_config(cfg)
        for section in ("data", "acquisition", "forward", "training"):
            assert cfg[section] == residual["paper13_ifwi"][section]
    fr = residual["paper13_fr_ifwi"]["model"]
    for plain, with_fr in (("lrpaper13_ifwi", "lrpaper13_fr_ifwi"), ("lrpaper13_lr_ifwi", "lrpaper13_lr_fr_ifwi")):
        a, b = copy.deepcopy(paper[plain]), copy.deepcopy(paper[with_fr])
        # FR settings are exactly those of the 13-shot FR run.
        assert b["model"].pop("fourier") == fr["fourier"]
        assert b["model"].pop("trainable_omega") == fr["trainable_omega"]
        b["experiment_name"] = a["experiment_name"]
        assert a == b


def test_trainable_omega_also_moves_during_pretraining():
    torch.manual_seed(3)
    initial = np.linspace(1800, 3600, 6 * 9, dtype=np.float32).reshape(6, 9)
    fr = {"high_freq_num": 8, "low_freq_num": 8, "phi_num": 8, "alpha": .01, "lambda_init": "fr_inr"}
    model = api().VelocityParameterization(initial, 15, None, parameterization="absolute",
                                           fourier=fr, trainable_omega=True, low_rank=dict(PAPER_LR))
    api().pretrain_to_initial(model, initial, {"updates": 20, "learning_rate": 1e-2},
                              {"learning_rate": 1e-2}, 3., 1., emit=lambda _: None)
    omegas = mod("trainable_omega").omega_values(model)
    assert len(omegas) == 4 and any(abs(w - 30.) > 1e-3 for w in omegas)


@pytest.mark.parametrize("low_rank", [True, False])
def test_residual_pretraining_matches_absolute_then_starts_exactly_at_the_background(low_rank):
    initial = np.linspace(1800, 3600, 6 * 9, dtype=np.float32).reshape(6, 9)
    setting = {"updates": 15, "learning_rate": 1e-2}
    models = {}
    for mode in ("absolute", "residual"):
        torch.manual_seed(3)
        models[mode] = api().VelocityParameterization(initial, 15, None if low_rank else [2, 8, 8, 1], parameterization=mode,
                                                      low_rank=dict(PAPER_LR) if low_rank else None, zero_residual_head=False)
        _, best = api().pretrain_to_initial(models[mode], initial, setting, None, 3., 1., emit=lambda _: None)
    absolute, residual = (dict(models[m].state_dict()) for m in ("absolute", "residual"))
    head = {"low_rank.core"} if low_rank else {"net.linear.2.weight", "net.linear.2.bias"}
    # Identical pretraining; only the residual head is zeroed afterwards.
    assert all(torch.equal(absolute[k], residual[k]) for k in absolute if k not in head)
    assert all(torch.count_nonzero(residual[k]) == 0 for k in head)
    assert all(torch.count_nonzero(absolute[k]) > 0 for k in head)
    with torch.no_grad():
        assert torch.equal(models["residual"]()[0], torch.from_numpy(initial))


def test_residual_pretrained_run_starts_at_the_background_and_resumes(tmp_path):
    cfg = tiny_paper_config(tmp_path, low_rank=True)
    cfg["parameterization"] = "residual"
    full = api().run_experiment(cfg, tmp_path / "full", "cpu", save_figures=False)
    info = json.loads((full / "pretraining.json").read_text())
    assert info["output_head_zeroed_after_pretraining"] is True and info["rmse_vs_initial_mps"] == 0
    assert info["fit_rmse_vs_initial_mps"] > 0
    assert np.array_equal(np.load(full / "run_start_velocity.npy"), np.load(full / "initial_velocity.npy"))
    partial = copy.deepcopy(cfg)
    partial["training"]["epochs"] = 1
    first = api().run_experiment(partial, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    assert not (resumed / "pretraining.json").exists()
    assert np.array_equal(np.load(full / "final_velocity.npy"), np.load(resumed / "final_velocity.npy"))


def test_pretrained_residual_configs_differ_by_one_factor():
    yaml = pytest.importorskip("yaml")
    folder = ROOT / "experiments/configs"
    load = lambda n: yaml.safe_load((folder / f"{n}.yaml").read_text(encoding="utf-8"))
    for new, paper in (("paper13_pretrained_ifwi", "lrpaper13_ifwi"), ("paper13_pretrained_lr_ifwi", "lrpaper13_lr_ifwi")):
        a, b = load(new), load(paper)
        api().validate_config(a)
        assert (a.pop("parameterization"), b.pop("parameterization")) == ("residual", "absolute")
        a["experiment_name"] = b["experiment_name"]
        assert a == b
    pretrained, ifwi13 = load("paper13_pretrained_ifwi"), load("paper13_ifwi")
    assert pretrained["initialization"].pop("pretrain") == {"updates": 5000, "learning_rate": 1e-4}
    pretrained["experiment_name"] = ifwi13["experiment_name"]
    assert pretrained == ifwi13

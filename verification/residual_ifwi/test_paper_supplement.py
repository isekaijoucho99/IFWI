"""Paper supplement: spec generation, data preparation, selection and tiny CPU core_runner runs."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def ps():
    return importlib.import_module("paper_supplement")


def runner():
    return importlib.import_module("core_runner")


def api():
    return importlib.import_module("experiments.residual_ifwi_experiment")


def strip(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop("experiment_name")
    return cfg


def test_every_spec_validates_and_differs_from_its_base_only_where_intended():
    queues = ps().build_specs({"C_grid_lr": 10.0, "D_E2_lr": 3e-5, "D_E3_lr": 1e-4, "H_scale": 3.0})
    specs = {}
    for item in (s for items in queues.values() for s in items):
        assert specs.setdefault(item["name"], item) == item                     # shared baselines are identical
    for item in specs.values():
        runner().check_spec(item)
        api().validate_config(item["experiment"])
    base = lambda method, condition="clean", seed=3: strip(ps().experiment(method, condition, seed))
    for seed in (0, 1, 3):
        assert strip(specs[f"A_E3_s{seed}"]["experiment"]) == base("E3", seed=seed)
        assert specs[f"A_E4small5_s{seed}"]["control"] == {"kind": "small_output", "rms_mps": 5.0}
        ms = strip(specs[f"A_E2MS_s{seed}"]["experiment"])
        assert ms["training"].pop("frequency_schedule") == ps().MULTISCALE and ms == base("E2", seed=seed)
    for condition in ("sigma2", "sigma4"):
        item = specs[f"B_E2MS_{condition}_s3"]
        assert item["robustness"]["condition"] == f"marmousi_noise_{condition}" and item["robustness"]["noise_seed"] == 3
    grid = specs["C_gridMS_overthrust_lr10_s3"]
    assert grid["model"] == {"kind": "grid"} and grid["experiment"]["training"]["learning_rate"] == 10.
    assert grid["experiment"]["training"]["shot_batch_size"] == 5 and grid["robustness"]["condition"] == "overthrust"
    direct = specs["E_directFR_prefit_s3"]
    assert direct["experiment"]["parameterization"] == "absolute" and direct["control"]["kind"] == "prefit_background"
    assert direct["experiment"]["model"]["fourier"] == ps().FOURIER and "trainable_omega" not in direct["experiment"]["model"]
    assert specs["G2_direct_keepadam_s3"]["control"]["keep_optimizer_state"] is True
    assert specs["G3_direct_lr1em05_s3"]["experiment"]["training"]["learning_rate"] == 1e-5
    assert specs["J_E4_sigma4_beta2_s3"]["experiment"]["training"]["optimizer"] == {"name": "adam", "betas": [.9, .99]}
    assert specs["K_E3_lineardepth_s3"]["robustness"]["condition"] == "marmousi_clean_lineardepth"
    assert specs["D_screen_E2_sv300_lr1em04"]["experiment"]["model"]["std_kmps"] == .3
    assert all(s["experiment"]["training"]["epochs"] == 200 for n, s in specs.items() if "_screen_" in n)
    assert specs["F_E2_first5_s3"]["experiment"]["training"]["epochs"] == 5
    assert "q5_C_formal" not in ps().build_specs()                              # needs a selection first


def test_overthrust_csv_is_reproduced_bit_for_bit(tmp_path):
    path, digest = ps().prepare_data(tmp_path)
    assert digest == ps().OVERTHRUST_CSV_SHA256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_selection_uses_the_lowest_waveform_mse(tmp_path):
    runs = []
    for lr, mse in ((3., .02), (10., .01), (30., .03)):
        run = tmp_path / f"lr{lr:g}"
        run.mkdir()
        cfg = ps().short(ps().experiment("E2"))
        cfg["training"]["learning_rate"] = lr
        (run / "run_spec.json").write_text(json.dumps({"name": run.name, "experiment": cfg, "model": {"kind": "grid"}}))
        (run / "evaluation_history.json").write_text(json.dumps([{"evaluated_updates": 0, "data_mse": 1.},
                                                                {"evaluated_updates": 200, "data_mse": mse}]))
        runs.append(run)
    best = ps().select(runs, "C_grid_lr", tmp_path / "selection.json")
    record = json.loads((tmp_path / "selection.json").read_text())
    assert best["value"] == 10. and record["C_grid_lr"] == 10. and len(record["_candidates"]["C_grid_lr"]["candidates"]) == 3


def tiny_spec(tmp_path, name, method="E2", **changes):
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    truth[5:7, 3:6] = 3400.
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = ps().experiment(method)
    cfg["model"]["neurons"] = [2, 8, 8, 8, 1]
    if "fourier" in cfg["model"]:
        cfg["model"]["fourier"] = {"high_freq_num": 8, "low_freq_num": 8, "phi_num": 8, "alpha": .01, "lambda_init": "fr_inr"}
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1
    cfg["training"].update(epochs=3, checkpoint_interval=1)
    if "frequency_schedule" in cfg["training"]:
        cfg["training"]["frequency_schedule"] = [{"max_hz": 20.0, "updates": 1}, {"max_hz": None}]
    item = ps().spec(name, cfg)
    item["robustness"]["condition"] = "tiny_clean"
    for key, value in changes.items():
        if key == "training":
            item["experiment"]["training"].update(value)
        else:
            item[key] = value
    return item


def run(tmp_path, item):
    out = runner().run(item, tmp_path / item["name"], "cpu", save_figures=False, shared_dir=tmp_path / "shared")
    return out, json.loads((out / "history.json").read_text())


def test_core_runner_multiscale_lr_schedule_and_shared_observations(tmp_path):
    out, history = run(tmp_path, tiny_spec(tmp_path, "ms", "E2MS", training={"lr_schedule": {
        "warmup_updates": 2, "decay_start": 1, "decay_updates": 1, "final_factor": 1.0}}))
    assert [h.get("max_hz") for h in history] == [20., None, None]
    assert [h["lr_factor"] for h in history] == [.5, .5, 1.]                 # warmup restarts at the full-band stage
    assert json.loads((out / "frequency_scales.json").read_text())["20.0"] > 1
    assert all(h["cumulative_seconds"] >= 0 and "velocity_delta_rms_mps" in h for h in history)
    # The second run of the condition reads the same shared observations (hash-checked).
    _, again = run(tmp_path, tiny_spec(tmp_path, "plain"))
    assert again[0]["data_mse"] == history[0]["data_mse"]
    noisy = tiny_spec(tmp_path, "noisy")
    noisy["robustness"] = {"condition": "tiny_noise2", "noise_sigma": 2.0, "noise_seed": 3}
    out, noisy_history = run(tmp_path, noisy)
    manifest = json.loads((out / "observation_manifest.json").read_text())
    assert manifest["observed_dtype"] == "float64" and noisy_history[0]["data_mse"] > history[0]["data_mse"]


def test_core_runner_grid_and_fourier_feature_models(tmp_path):
    out, history = run(tmp_path, tiny_spec(tmp_path, "grid", training={"learning_rate": 10.}, model={"kind": "grid"}))
    start, final = np.load(out / "run_start_velocity.npy"), np.load(out / "final_velocity.npy")
    assert np.array_equal(start, np.load(out / "initial_velocity.npy"))      # dv starts at zero
    assert 0 < np.abs(final - start).max() <= 3 * 10. * 1.05                  # Adam steps of ~10 m/s
    mechanisms = [json.loads(line) for line in (out / "mechanisms.jsonl").read_text().splitlines()]
    assert mechanisms and all(m["layer_changes"] == [] for m in mechanisms)
    out, history = run(tmp_path, tiny_spec(tmp_path, "ff", model={"kind": "fourier_features", "scale_cpkm": 3.0,
                                                                  "features": 4, "hidden": [8, 8]}))
    assert np.array_equal(np.load(out / "run_start_velocity.npy"), np.load(out / "initial_velocity.npy"))
    assert not np.array_equal(np.load(out / "final_velocity.npy"), np.load(out / "initial_velocity.npy"))
    model = importlib.import_module("core_models").FourierFeatureVelocity(np.full((4, 5), 3000., np.float32), 15., 3., 4, (8,), seed=7)
    assert sum(p.numel() for p in model.parameters()) == 8 * 8 + 8 + 8 + 1


def test_core_runner_prefit_keep_adam_and_linear_background(tmp_path):
    prefit = {"kind": "prefit_background", "updates": 4, "learning_rate": 1e-4, "selection": "best_background_fit"}
    keep = tiny_spec(tmp_path, "keep", "direct", control={**prefit, "keep_optimizer_state": True})
    out, history = run(tmp_path, keep)
    report = json.loads((out / "initialization_control.json").read_text())
    assert report["optimizer_reset_for_inversion"] is False and report["inversion_started_with_prefit_adam_moments"]
    saved = torch.load(out / "checkpoint.pt", weights_only=False)
    steps = {float(s["step"]) for s in saved["optimizer"]["state"].values()}
    assert steps == {4. + 3.}                                                 # 4 prefit steps carried on, + 3 updates
    assert [g["lr"] for g in saved["optimizer"]["param_groups"]] == [1e-4]
    reset = tiny_spec(tmp_path, "reset", "direct", control=prefit)
    out, _ = run(tmp_path, reset)
    saved = torch.load(out / "checkpoint.pt", weights_only=False)
    assert {float(s["step"]) for s in saved["optimizer"]["state"].values()} == {3.}
    linear = tiny_spec(tmp_path, "linear", background={"kind": "linear_depth", "top_mps": 1500., "gradient_per_s": 2.})
    linear["robustness"]["condition"] = "tiny_linear"
    out, _ = run(tmp_path, linear)
    start = np.load(out / "run_start_velocity.npy")
    assert np.allclose(start[:, 0], 1500. + 2. * np.arange(8) * 15.) and np.allclose(start, start[:, :1])
    assert json.loads((out / "initial_model_provenance.json").read_text())["background_override"]["top_mps"] == 1500.
    with pytest.raises(ValueError, match="spec.background"):
        runner().check_spec({**linear, "background": {"kind": "linear_depth", "top_mps": 1500.}})
    with pytest.raises(ValueError, match="does not support"):
        runner().check_spec(tiny_spec(tmp_path, "gain", training={"time_gain": {"power": 1., "start_update": 1}}))


def test_traveltime_warp_keeps_ranges_and_starts_at_background(tmp_path):
    models = importlib.import_module("core_models")
    v0 = np.linspace(1500, 4500, 20, dtype=np.float32)[:, None] + np.zeros((20, 30), np.float32)
    coords = models.traveltime_coordinates(v0, 15., "z")
    assert np.allclose(coords[..., 0][0], np.arange(30) * .015)               # x unchanged
    assert np.isclose(coords[-1, :, 1].mean(), 19 * .015, rtol=1e-5)          # mean depth range kept
    stretch = np.diff(coords[:, 0, 1]) / .015
    assert stretch[0] > 1 > stretch[-1]                                        # finer where slow
    item = tiny_spec(tmp_path, "warp", "E3", model={"kind": "siren", "coordinate_warp": {"kind": "traveltime", "axes": "xz"}})
    out, history = run(tmp_path, item)
    assert np.array_equal(np.load(out / "run_start_velocity.npy"), np.load(out / "initial_velocity.npy"))
    assert json.loads((out / "coordinate_warp.json").read_text())["axes"] == "xz" and len(history) == 3
    with pytest.raises(ValueError, match="coordinate_warp"):
        runner().check_spec({**item, "model": {"kind": "siren", "coordinate_warp": {"kind": "traveltime", "axes": "x"}}})


def test_queue_dry_run_lists_every_spec(tmp_path, capsys):
    queue = importlib.import_module("paper_queue")
    folder = tmp_path / "supplement"
    ps().write_specs(None, folder)
    sys.argv = ["paper_queue.py", str(folder / "queues" / "q3_B_E.json"), "--out-root", str(tmp_path / "runs"), "--dry-run"]
    queue.main()
    printed = capsys.readouterr().out
    assert printed.count("core_runner.py") == 3 and "B_E2MS_sigma2_s3.json" in printed
    assert (folder / "configs" / "I_overthrust.json").exists()

"""Learning-rate factor per update (warmup per frequency stage, cosine decay); tiny CPU FD."""
import copy
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

STAGES = [{"max_hz": 20.0, "updates": 2}, {"max_hz": 40.0, "updates": 2}, {"max_hz": None}]
LRS = {"warmup_updates": 2, "decay_start": 5, "decay_updates": 2, "final_factor": .1}


def mod():
    return importlib.import_module("experiments.lr_schedule")


def api():
    return importlib.import_module("experiments.residual_ifwi_experiment")


def test_factor_combines_stage_warmup_and_cosine_decay():
    factors = [mod().lr_factor(LRS, STAGES, u) for u in range(1, 9)]
    # updates 1-2: first stage, warmup 1/2 then 1; 3-4: next stage, warmup restarts;
    # 5-6: full band, warmup restarts while the decay runs 1 -> 0.55; afterwards final 0.1.
    expected = [.5, 1., .5, 1., .5, .55, .1, .1]
    assert factors == pytest.approx(expected)
    plain = {**LRS, "decay_start": 3, "decay_updates": 4}
    no_stages = [mod().lr_factor(plain, None, u) for u in range(1, 9)]
    decay = lambda u: .1 + .9 * .5 * (1 + math.cos(math.pi * (u - 3) / 4))
    assert no_stages == pytest.approx([.5, 1., decay(3), decay(4), decay(5), decay(6), .1, .1])
    assert mod().lr_factor(None, STAGES, 1) == 1.
    assert [mod().stage_start(STAGES, u) for u in (1, 2, 3, 4, 5, 99)] == [1, 1, 3, 3, 5, 5]


def test_validation():
    mod().validate_lr_schedule(None)
    mod().validate_lr_schedule(LRS)
    for bad in ({k: v for k, v in LRS.items() if k != "final_factor"}, {**LRS, "extra": 1},
                {**LRS, "warmup_updates": -1}, {**LRS, "warmup_updates": 1.5}, {**LRS, "decay_start": 0},
                {**LRS, "decay_updates": 0}, {**LRS, "final_factor": 0.}, {**LRS, "final_factor": 1.5},
                {**LRS, "final_factor": True}, [1, 2]):
        with pytest.raises(ValueError):
            mod().validate_lr_schedule(bad)


def tiny_config(tmp_path, lr_schedule=LRS, stages=STAGES):
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    truth[5:7, 3:6] = 3400.
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = api().default_config()
    cfg["model"].update(neurons=[2, 8, 8, 8, 1], trainable_omega={"learning_rate": 1e-2},
                        fourier={"high_freq_num": 8, "low_freq_num": 8, "phi_num": 8, "alpha": .01, "lambda_init": "fr_inr"})
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1
    cfg["training"].update(epochs=6, checkpoint_interval=1)
    if stages is not None:
        cfg["training"]["frequency_schedule"] = copy.deepcopy(stages)
    if lr_schedule is not None:
        cfg["training"]["lr_schedule"] = dict(lr_schedule)
    return cfg


def test_runner_scales_every_group_logs_the_factor_and_resumes_exactly(tmp_path):
    cfg = tiny_config(tmp_path)
    full = api().run_experiment(cfg, tmp_path / "full", "cpu", save_figures=False)
    history = json.loads((full / "history.json").read_text())
    assert [h["lr_factor"] for h in history[:6]] == pytest.approx([.5, 1., .5, 1., .5, .55])
    saved = torch.load(full / "checkpoint.pt", weights_only=False)
    assert [g["lr"] for g in saved["optimizer"]["param_groups"]] == pytest.approx([1e-4 * .55, 1e-2 * .55])  # network, omega
    notes = json.loads((full / "comparison_notes.json").read_text())
    assert notes["lr_schedule"] == LRS
    assert any("learning-rate factor" in change for change in notes["changes_vs_original_random"])
    # Resuming after a stage switch rebuilds base rates from the configuration, not the saved scaled ones.
    partial = copy.deepcopy(cfg)
    partial["training"]["epochs"] = 3
    first = api().run_experiment(partial, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    assert np.array_equal(np.load(full / "final_velocity.npy"), np.load(resumed / "final_velocity.npy"))
    changed = copy.deepcopy(cfg)
    changed["training"]["lr_schedule"]["final_factor"] = .2
    with pytest.raises(ValueError, match="configuration differs"):
        api().run_experiment(changed, tmp_path / "wrong", "cpu", resume=first / "checkpoint.pt", save_figures=False)


def test_without_a_schedule_nothing_is_logged_or_scaled(tmp_path):
    run = api().run_experiment(tiny_config(tmp_path, None), tmp_path / "plain", "cpu", save_figures=False)
    history = json.loads((run / "history.json").read_text())
    assert all("lr_factor" not in h for h in history)
    assert "lr_schedule" not in json.loads((run / "comparison_notes.json").read_text())
    saved = torch.load(run / "checkpoint.pt", weights_only=False)
    assert [g["lr"] for g in saved["optimizer"]["param_groups"]] == [1e-4, 1e-2]


def test_lr_schedule_configs_change_only_the_schedule_and_stage_lengths():
    yaml = pytest.importorskip("yaml")
    folder = ROOT / "experiments/configs"
    load = lambda n: yaml.safe_load((folder / f"{n}.yaml").read_text(encoding="utf-8"))
    late = {"warmup_updates": 20, "decay_start": 201, "decay_updates": 800, "final_factor": .1}
    for name, base, schedule in (("paper13_fr_ifwi_multiscale_lrsched", "paper13_fr_ifwi_multiscale", late),
                                 ("paper13_fr_ifwi_lrsched", "paper13_fr_ifwi", late),
                                 ("paper13_ifwi_multiscale_lrsched", "paper13_ifwi_multiscale", late),
                                 ("paper13_fr_ifwi_multiscale_short_lrsched", "paper13_fr_ifwi_multiscale",
                                  {**late, "decay_start": 101, "decay_updates": 900})):
        cfg, ref = load(name), load(base)
        api().validate_config(cfg)
        assert cfg["training"].pop("lr_schedule") == schedule
        assert cfg["training"].pop("report_estimates") == {"ema_decay": .99, "ema_start": 501}   # evaluation only
        if name.endswith("short_lrsched"):
            assert [s.get("updates") for s in cfg["training"]["frequency_schedule"]] == [50, 50, None]
            for stage in cfg["training"]["frequency_schedule"][:-1]:
                stage["updates"] = 100
        cfg["experiment_name"] = ref["experiment_name"]
        assert cfg == ref

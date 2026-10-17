"""Fixed-gradient update-response diagnostic; tiny CPU FD for the one gradient."""
import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)


def mod():
    return importlib.import_module("experiments.update_response_report")


def test_stripe_shares_identify_stripe_orientation():
    x = np.arange(64)
    vertical = np.tile(np.cos(np.pi * 30 * (x + .5) / 64), (32, 1))      # varies along x only, high kx
    shares = mod().stripe_shares(vertical, 15.)
    assert shares["vertical_stripe_share"] > .99 and shares["horizontal_stripe_share"] < 1e-6
    shares = mod().stripe_shares(vertical.T.copy(), 15.)
    assert shares["horizontal_stripe_share"] > .99 and shares["vertical_stripe_share"] < 1e-6
    smooth = np.outer(np.linspace(0, 1, 32), np.ones(64))
    assert mod().stripe_shares(smooth, 15.)["high_k_share"] < .01


def tiny_config(tmp_path, low_rank):
    api = importlib.import_module("experiments.residual_ifwi_experiment")
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    truth[5:7, 3:6] = 3400.
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = api.default_config()
    cfg["parameterization"] = "absolute"
    cfg["initialization"].update(sigma=1, pretrain={"updates": 10, "learning_rate": 1e-2})
    if low_rank:
        cfg["model"].update(neurons=None, low_rank={"rank_z": 3, "rank_x": 4, "hidden": [8, 8], "coordinates": "normalized",
                                                    "coord_scale": 1., "core_init": "random", "output_gain": 1.})
    else:
        cfg["model"]["neurons"] = [2, 8, 8, 1]
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    return cfg


def test_report_runs_each_model_from_one_shared_gradient(tmp_path):
    configs = [tiny_config(tmp_path, False), tiny_config(tmp_path, True)]
    configs[1]["experiment_name"] = "tiny_lr"
    rows = mod().report(configs, [1, 3])
    assert [(r["experiment_name"], r["step"]) for r in rows] == [
        ("residual_ifwi_stage1", 1), ("residual_ifwi_stage1", 3), ("tiny_lr", 1), ("tiny_lr", 3)]
    for row in rows:
        assert row["rms_dm_mps"] > 0 and -1 <= row["cos_dm_descent"] <= 1
    mismatched = copy.deepcopy(configs)
    mismatched[1]["initialization"]["sigma"] = 2
    with pytest.raises(ValueError, match="must match"):
        mod().report(mismatched, [1])

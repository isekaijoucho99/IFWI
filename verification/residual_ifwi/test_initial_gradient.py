"""Offline first-gradient diagnostic for candidate initial models; tiny CPU FD."""
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
    return importlib.import_module("experiments.initial_gradient_report")


def test_alignment_signs_shares_and_band_split():
    rng = np.random.default_rng(0)
    truth = rng.normal(3000, 200, (20, 30))
    initial = np.full_like(truth, 3000.)
    update = truth - initial
    toward = mod().alignment(-update, truth, initial, 15.)   # gradient = -(v_true - v_init)
    away = mod().alignment(update, truth, initial, 15.)
    for key, value in toward.items():
        if key.startswith("cos_"):
            assert value == pytest.approx(1.) and away[key] == pytest.approx(-1.)
    shallow = np.zeros_like(truth)
    shallow[:5] = 1.                                          # rows 0..4 -> depth < 75 m
    shares = mod().alignment(shallow, truth, initial, 15., shallow_m=150.)
    assert shares["grad_energy_share_z_lt_150m"] == pytest.approx(1.)
    # Bands partition the field: band-passed parts sum back to it.
    parts = sum(mod().band_pass(update, 15., lo, hi) for lo, hi in mod().DEFAULT_BANDS_CPKM)
    assert np.allclose(parts, update)


def test_report_runs_the_configured_forward_problem(tmp_path):
    api = importlib.import_module("experiments.residual_ifwi_experiment")
    truth = np.linspace(2300, 3200, 8, dtype=np.float32)[:, None] + np.zeros((8, 9), np.float32)
    truth[5:7, 3:6] = 3400.
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = api.default_config()
    cfg["model"]["neurons"] = [2, 8, 1]
    cfg["acquisition"].update(num_shots=2, source_x_indices=[2, 6])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60, npad=3)
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    rows = mod().report(cfg, [.5, 1.], "cpu")
    assert [row["sigma"] for row in rows] == [.5, 1.]
    assert rows[0]["initial_rmse_mps"] < rows[1]["initial_rmse_mps"]
    for row in rows:
        assert row["initial_data_mse"] > 0
        assert all(-1 <= v <= 1 for k, v in row.items() if k.startswith("cos_") and v is not None)
        assert 0 <= row["grad_energy_share_z_lt_150m"] <= 1

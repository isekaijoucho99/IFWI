"""Offline region errors and checkpoint step-displacement reports; no FD, no training."""
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


def mod(name):
    return importlib.import_module(f"experiments.{name}")


def fields():
    rng = np.random.default_rng(0)
    truth = np.full((10, 12), 3000.)
    truth[6:8, 2:9] = 5500.  # a thin high-velocity layer, 14 cells
    background = truth + rng.normal(0, 200, truth.shape)
    prediction = truth + rng.normal(0, 100, truth.shape)
    return prediction, truth, background


def test_region_shares_partition_and_recover_total_rmse():
    prediction, truth, background = fields()
    result = mod("region_metrics").region_errors(prediction, truth, background, 100., depth_edges_m=(0., 300., 600., np.inf))
    depth = ["z0-300m", "z300-600m", "z>600m"]
    assert sum(result["sse_share"][n] for n in depth) == pytest.approx(1)
    assert result["sse_share"]["v>=5000"] + result["sse_share"]["v<5000"] == pytest.approx(1)
    assert result["cell_fraction"]["v>=5000"] == pytest.approx(14 / 120)
    total = np.sqrt(np.mean((prediction - truth) ** 2))
    assert np.sqrt(sum(result["cell_fraction"][n] * result["rmse_mps"][n] ** 2 for n in depth)) == pytest.approx(total)
    mask = truth >= 5000
    assert result["rmse_mps"]["v>=5000"] == pytest.approx(np.sqrt(np.mean((prediction - truth)[mask] ** 2)))
    same = mod("region_metrics").region_errors(background, truth, background, 100.)
    assert same["rmse_mps"] == same["background_rmse_mps"]


def write_run(folder, truth, background, snapshots, completed=None):
    (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
    np.save(folder / "true_velocity.npy", truth)
    np.save(folder / "initial_velocity.npy", background)
    for update, velocity in snapshots.items():
        np.save(folder / "checkpoints" / f"velocity_{update:04d}.npy", velocity)
    if completed is not None:
        np.save(folder / "final_velocity.npy", snapshots[completed])
        (folder / "metrics.json").write_text(json.dumps({"completed_updates": completed}))


def test_band_report_joins_resumed_directories(tmp_path):
    prediction, truth, background = fields()
    write_run(tmp_path / "a", truth, background, {50: prediction})
    write_run(tmp_path / "b", truth, background, {100: prediction * .5 + truth * .5}, completed=100)
    report = mod("residual_band_report")
    bands, regions = report.run_series(f"{tmp_path / 'a'},{tmp_path / 'b'}", 15., report.DEFAULT_EDGES_CPKM)
    assert [r["update"] for r in bands] == [0, 50, 100] == [r["update"] for r in regions]
    assert regions[0]["rmse_mps"] == regions[0]["background_rmse_mps"]
    assert regions[2]["rmse_mps"]["v>=5000"] < regions[1]["rmse_mps"]["v>=5000"]
    other = background.copy()
    other[0, 0] += 1
    write_run(tmp_path / "c", truth, other, {100: prediction})
    with pytest.raises(ValueError, match="different truth or background"):
        report.run_series(f"{tmp_path / 'a'},{tmp_path / 'c'}", 15., report.DEFAULT_EDGES_CPKM)


def small_config(fourier=None, trainable_omega=None):
    cfg = mod("residual_ifwi_experiment").default_config()
    cfg["model"]["neurons"] = [2, 8, 8, 8, 1]
    if fourier is not None:
        cfg["model"]["fourier"] = fourier
    if trainable_omega is not None:
        cfg["model"]["trainable_omega"] = trainable_omega
    return cfg


def save_checkpoints(folder, cfg, scales, seed=0):
    """update_0050, 0100, ...: every trainable tensor shifted by scale * a fixed direction."""
    api = mod("residual_ifwi_experiment")
    torch.manual_seed(seed)
    model = api.VelocityParameterization(np.full((5, 6), 3000., np.float32), 15, cfg["model"]["neurons"],
        fourier=cfg["model"].get("fourier"), trainable_omega=cfg["model"].get("trainable_omega") is not None)
    direction = {n: torch.randn_like(p) for n, p in model.named_parameters() if not n.endswith("omega")}
    base = copy.deepcopy(model.state_dict())
    (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
    for index, scale in enumerate(scales, start=1):
        state = copy.deepcopy(base)
        for name, d in direction.items():
            state[name] = base[name] + scale * d
        torch.save({"config": cfg, "model": state}, folder / "checkpoints" / f"update_{50 * index:04d}.pt")
    return model


def test_step_displacement_matches_merged_preactivation_weights(tmp_path):
    report = mod("residual_step_report")
    cfg = small_config(SMALL_FR, {"learning_rate": .01})
    save_checkpoints(tmp_path / "fr", cfg, [0., .1, .3])
    rows, _ = report.displacement_series([tmp_path / "fr"])
    assert [(r["start"], r["end"]) for r in rows] == [(50, 100), (100, 150)]
    weights = [report.load_weights(tmp_path / "fr" / "checkpoints" / f"update_{u:04d}.pt")[0] for u in (50, 100, 150)]
    assert len(weights[0]) == 4
    for row, (a, b) in zip(rows, zip(weights[:-1], weights[1:])):
        assert row["displacement"] == pytest.approx([np.linalg.norm(y - x) for x, y in zip(a, b)])
    # Constant shift direction in lambda -> displacement of FR layers doubles in the second window.
    assert rows[1]["displacement"][1] == pytest.approx(2 * rows[0]["displacement"][1])
    # Pre-activation weight includes omega (30 here) for activated layers, not for the output layer.
    payload = torch.load(tmp_path / "fr" / "checkpoints" / "update_0050.pt", weights_only=False)
    net = report.build_model(payload["config"], payload["model"])
    assert np.allclose(weights[0][1], 30 * net.net.linear[1].weight.detach().double().numpy())
    assert np.allclose(weights[0][-1], net.net.linear[-1].weight.detach().double().numpy())


def test_step_ratio_against_reference_and_joined_resume(tmp_path):
    report = mod("residual_step_report")
    plain = small_config()
    save_checkpoints(tmp_path / "base", plain, [0., .1, .2])
    # Same parameter shifts, scaled 3x: displacement ratio 3 in every layer.
    save_checkpoints(tmp_path / "big", plain, [0., .3, .6])
    first, _ = report.displacement_series([tmp_path / "base"])
    second, _ = report.displacement_series([tmp_path / "big"])
    ratios = report.compare([first, second], ["base", "big"])["big"]
    assert all(r == pytest.approx(3) for row in ratios for r in row["ratio"])
    # A resumed run continues update numbering in a second directory.
    resumed = tmp_path / "resumed" / "checkpoints"
    resumed.mkdir(parents=True)
    (tmp_path / "base" / "checkpoints" / "update_0150.pt").rename(resumed / "update_0150.pt")
    joined, _ = report.displacement_series(report.split_dirs(f"{tmp_path / 'base'},{tmp_path / 'resumed'}"))
    assert [(r["start"], r["end"]) for r in joined] == [(50, 100), (100, 150)]
    assert joined[1]["displacement"] == pytest.approx(first[1]["displacement"])

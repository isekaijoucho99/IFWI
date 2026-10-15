"""Frequency continuation of the misfit (low-pass stages) and the saved-state gradient report; tiny CPU FD."""
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

SCHEDULE = [{"max_hz": 3.0, "updates": 100}, {"max_hz": 5.0, "updates": 100}, {"max_hz": None}]


def fc():
    return importlib.import_module("experiments.frequency_continuation")


def api():
    return importlib.import_module("experiments.residual_ifwi_experiment")


def records(*pulses, nt=500, dt=.002):
    """[1, len(pulses), nt, 1] float64 records of Gaussian-windowed cosines (center s, width s, Hz)."""
    t = torch.arange(nt, dtype=torch.float64) * dt
    traces = [torch.exp(-((t - c) / w) ** 2) * torch.cos(2 * np.pi * f * (t - c)) for c, w, f in pulses]
    return torch.stack(traces)[None, :, :, None]


def test_lowpass_is_zero_phase_keeps_the_passband_and_removes_the_stopband():
    x = records((2., .4, 0.), (2., .05, 30.), nt=2000)     # < 2 Hz pulse; 30 Hz wavelet; 4 s records
    y = fc().lowpass(x, 5., .002)
    assert torch.allclose(y[:, 0], x[:, 0], rtol=0, atol=1e-4 * x[:, 0].abs().max())
    assert y[:, 1].abs().max() < 1e-4 * x[:, 1].abs().max()
    pulse = fc().lowpass(records((2., .05, 8.), nt=2000), 5., .002)[0, 0, :, 0]
    assert torch.allclose(pulse[500:1001], pulse[1000:1501].flip(0), atol=1e-9 * pulse.abs().max())  # no shift
    assert fc().lowpass(x, None, .002) is x
    small = x[:, :, ::25].clone().requires_grad_(True)     # linear and differentiable through the FFT
    assert torch.autograd.gradcheck(lambda r: fc().lowpass(r, 5., .05), (small,))


def test_lowpass_pads_the_time_axis_against_wraparound():
    late = records((.97, .01, 4.))                          # energy near the end of a 1 s record
    filtered = fc().lowpass(late, 5., .002)

    def reference(n):                                       # same response with another FFT length
        freqs = torch.fft.rfftfreq(n, .002, dtype=torch.float64)
        response = 1 / (1 + (freqs / 5.) ** 12)
        return torch.fft.irfft(torch.fft.rfft(late, n=n, dim=2) * response[None, None, :, None], n=n, dim=2)[:, :, :500]

    peak = reference(16 * 500).abs().max()
    assert (filtered - reference(16 * 500)).abs().max() < 1e-3 * peak     # 2 nt padding ~ linear filtering
    assert (reference(500) - reference(16 * 500)).abs().max() > 1e-2 * peak  # no padding would wrap around


def test_stages_cutoffs_and_validation():
    assert [fc().stage_cutoff(SCHEDULE, u) for u in (1, 100, 101, 200, 201, 5000)] == [3., 3., 5., 5., None, None]
    assert fc().stage_cutoff(None, 1) is None
    assert fc().stage_boundaries(SCHEDULE) == [(1, 100, 3.), (101, 200, 5.), (201, None, None)]
    fc().validate_frequency_schedule(SCHEDULE, .0019)
    fc().validate_frequency_schedule(None, .0019)
    for bad in ([{"max_hz": None}],                                            # one stage
                [{"max_hz": 3., "updates": 10}, {"max_hz": 5.}],                # last not full band
                [{"max_hz": 5., "updates": 10}, {"max_hz": 3., "updates": 10}, {"max_hz": None}],
                [{"max_hz": 300., "updates": 10}, {"max_hz": None}],          # above Nyquist (263 Hz)
                [{"max_hz": 0., "updates": 10}, {"max_hz": None}],
                [{"max_hz": 3., "updates": 0}, {"max_hz": None}],
                [{"max_hz": 3., "updates": 1.5}, {"max_hz": None}],
                [{"max_hz": 3.}, {"max_hz": None}],
                [{"max_hz": 3., "updates": 10}, {"max_hz": None, "updates": 5}]):
        with pytest.raises(ValueError):
            fc().validate_frequency_schedule(bad, .0019)


def test_energy_scale_puts_a_band_objective_on_the_full_band_scale():
    observed = records((.3, .05, 8.), (.6, .08, 6.)).float()
    scale = fc().energy_scale(observed, 4., .002)
    assert scale > 1
    filtered = fc().lowpass(observed, 4., .002)
    zero = torch.zeros_like(observed)
    full = api().waveform_mse(zero, observed)
    band = scale * api().waveform_mse(zero, filtered)
    assert float(band) == pytest.approx(float(full), rel=1e-5)
    assert fc().energy_scales(observed, [{"max_hz": 4., "updates": 1}, {"max_hz": None}], .002) == {4.: scale}
    assert fc().energy_scale(observed, None, .002) == 1.


def tiny_config(tmp_path, schedule=None):
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
    cfg["training"].update(epochs=3, checkpoint_interval=1)
    if schedule is not None:
        cfg["training"]["frequency_schedule"] = copy.deepcopy(schedule)
    return cfg


TINY = [{"max_hz": 20.0, "updates": 1}, {"max_hz": 40.0, "updates": 1}, {"max_hz": None}]


def test_multiscale_run_follows_the_stages_logs_full_band_mse_and_resumes_exactly(tmp_path):
    cfg = tiny_config(tmp_path, TINY)
    full = api().run_experiment(cfg, tmp_path / "full", "cpu", save_figures=False)
    history = json.loads((full / "history.json").read_text())
    assert [h.get("max_hz") for h in history[:3]] == [20., 40., None]
    assert all(h["objective"] > 0 for h in history[:2]) and "objective" not in history[2]
    stages = json.loads((full / "frequency_schedule.json").read_text())["stages"]
    assert [(s["first_update"], s["last_update"], s["max_hz"]) for s in stages] == [(1, 1, 20.), (2, 2, 40.), (3, None, None)]
    assert stages[0]["energy_scale"] > stages[1]["energy_scale"] > 1 == stages[2]["energy_scale"]
    notes = json.loads((full / "comparison_notes.json").read_text())
    assert notes["frequency_schedule"] == TINY
    assert any("frequency continuation" in change for change in notes["changes_vs_original_random"])
    # data_mse is the full-band MSE of the same forward pass: equal to a full-band run's first value.
    plain = api().run_experiment(tiny_config(tmp_path), tmp_path / "plain", "cpu", save_figures=False)
    plain_history = json.loads((plain / "history.json").read_text())
    assert history[0]["data_mse"] == plain_history[0]["data_mse"]
    assert "max_hz" not in plain_history[0] and "objective" not in plain_history[0]
    assert "frequency_schedule" not in json.loads((plain / "comparison_notes.json").read_text())
    assert not (plain / "frequency_schedule.json").exists()
    # Resuming inside the schedule continues the same stages bit for bit.
    partial = copy.deepcopy(cfg)
    partial["training"]["epochs"] = 1
    first = api().run_experiment(partial, tmp_path / "partial", "cpu", save_figures=False)
    resumed = api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    assert np.array_equal(np.load(full / "final_velocity.npy"), np.load(resumed / "final_velocity.npy"))
    changed = copy.deepcopy(cfg)
    changed["training"]["frequency_schedule"][0]["max_hz"] = 25.
    with pytest.raises(ValueError, match="configuration differs"):
        api().run_experiment(changed, tmp_path / "wrong", "cpu", resume=first / "checkpoint.pt", save_figures=False)


def test_training_objective_is_the_band_mse_on_the_full_band_scale(tmp_path):
    cfg = tiny_config(tmp_path, TINY)
    truth = np.loadtxt(tmp_path / "truth.csv", delimiter=",", dtype=np.float32)
    problem = api().build_forward_problem(truth, cfg, "cpu")
    problem["frequency_scales"] = fc().energy_scales(problem["observed"], TINY, .001)
    zero = torch.zeros_like(problem["observed"])
    full = api().training_objective(zero, problem["observed"], 2, None, problem)
    assert torch.equal(full, api().waveform_mse(zero, problem["observed"], 2))
    for cutoff in (20., 40.):
        assert float(api().training_objective(zero, problem["observed"], 2, cutoff, problem)) == pytest.approx(float(full), rel=1e-5)


def test_gradient_report_matches_the_training_objective_and_reads_run_directories(tmp_path):
    report = importlib.import_module("experiments.gradient_report")
    cfg = tiny_config(tmp_path, TINY)
    run = api().run_experiment(cfg, tmp_path / "run", "cpu", save_figures=False)
    states = report.state_list("initial") + report.state_list(str(run))
    assert [s[1] for s in states] == [0, 0, 1, 2, 3]
    rows, residuals = report.report(cfg, states, [None, 20.], "cpu")
    assert [(r["state"], r["objective"]) for r in rows[:2]] == [("initial", "full"), ("initial", "<=20Hz")]
    assert len(rows) == 10 and len(residuals) == 5
    first = json.loads((run / "history.json").read_text())[0]
    start = [r for r in rows if r["state"] == "run_start"]
    assert start[1]["objective_value"] == pytest.approx(first["objective"], rel=1e-6)   # same objective as training
    assert start[0]["data_mse"] == pytest.approx(first["data_mse"], rel=1e-6)
    for row in rows:
        assert all(-1 <= v <= 1 for k, v in row.items() if k.startswith("cos_") and v is not None)
        shares = [row[f"grad_energy_{n}"] for n in ("shallow", "middle", "deep")]
        assert sum(shares) == pytest.approx(1.)
    for row in residuals:
        assert all(v >= 0 for k, v in row.items() if k.startswith("rel_") and v is not None)
    # A gradient pointing straight at the truth has every cosine = 1.
    truth = np.loadtxt(tmp_path / "truth.csv", delimiter=",")
    velocity = truth + 50.
    aligned = report.alignment(velocity - truth, truth, velocity, 15.)
    assert all(v == pytest.approx(1.) for k, v in aligned.items() if k.startswith("cos_") and v is not None)


def test_multiscale_configs_change_only_the_schedule():
    yaml = pytest.importorskip("yaml")
    folder = ROOT / "experiments/configs"
    load = lambda n: yaml.safe_load((folder / f"{n}.yaml").read_text(encoding="utf-8"))
    for name, base in (("paper13_fr_ifwi_multiscale", "paper13_fr_ifwi"), ("paper13_ifwi_multiscale", "paper13_ifwi")):
        cfg, ref = load(name), load(base)
        api().validate_config(cfg)
        assert cfg["training"].pop("frequency_schedule") == SCHEDULE
        cfg["experiment_name"] = ref["experiment_name"]
        assert cfg == ref

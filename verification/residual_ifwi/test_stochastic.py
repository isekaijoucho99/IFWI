"""CPU regression tests for shot-sampled original-FD residual IFWI."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def cpu_only_runner(monkeypatch):
    # The real runner otherwise seeds/serializes CUDA RNG even with device=cpu.
    # Keep this CPU suite independent of the concurrently running GPU experiment.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def experiment_api():
    return importlib.import_module("experiments.residual_ifwi_experiment")


def stochastic_api():
    name = "experiments.residual_ifwi_stochastic"
    assert importlib.util.find_spec(name) is not None, "Shot-sampling implementation is missing"
    return importlib.import_module(name)


def tiny_config():
    cfg = experiment_api().default_config()
    cfg["model"]["neurons"] = [2, 8, 8, 1]
    cfg["acquisition"].update(num_shots=5, source_x_indices=[1, 3, 5, 7, 9])
    cfg["forward"].update(dt_s=.001, nt=32, frequency_hz=60., npad=3)
    cfg["training"].update(epochs=4, checkpoint_interval=2, shot_batch_size=2)
    return cfg


def tiny_fields():
    z, x = np.indices((9, 11))
    truth = (2300. + 70. * z + 25. * x + 200. * (x >= 6)).astype(np.float32)
    return truth, truth * np.float32(.95)


def new_model(initial, cfg):
    torch.manual_seed(3)
    return experiment_api().VelocityParameterization(initial, 15., cfg["model"]["neurons"])


def direct_problem(truth, cfg, sources):
    cfg = copy.deepcopy(cfg)
    cfg["acquisition"].update(num_shots=len(sources), source_x_indices=list(sources))
    cfg["training"]["shot_batch_size"] = None
    cfg["training"].pop("shots_per_update", None)
    cfg["training"].pop("full_eval_interval", None)
    return experiment_api().build_forward_problem(truth, cfg, "cpu")


def direct_prediction(model, problem):
    """Independent full-shot reference, bypassing selection and MSE helpers."""
    assert len(problem["propagators"]) == 1
    solver = problem["propagators"][0]["solver"]
    with torch.no_grad():
        return solver(vmodel=model(), segment_wavelet=problem["wavelet"], option=0)[2]


def assert_optimizer_equal(first, second, *, exact):
    a, b = first["state"], second["state"]
    assert first["param_groups"] == second["param_groups"]
    assert a.keys() == b.keys()
    for index in a:
        assert a[index].keys() == b[index].keys()
        for key in a[index]:
            if isinstance(a[index][key], torch.Tensor):
                if exact:
                    assert torch.equal(a[index][key], b[index][key]), (index, key)
                else:
                    torch.testing.assert_close(a[index][key], b[index][key], rtol=2e-5, atol=1e-8)
            else:
                assert a[index][key] == b[index][key]


def test_seeded_sampling_is_without_replacement_and_reaches_all_49_shots():
    sampler = stochastic_api().sample_shot_indices
    np.random.seed(23)
    first = [sampler(49, 8) for _ in range(40)]
    np.random.seed(23)
    repeated = [sampler(49, 8) for _ in range(40)]
    assert first == repeated
    for selected in first:
        assert isinstance(selected, list)
        assert len(selected) == len(set(selected)) == 8
        assert all(isinstance(index, int) and 0 <= index < 49 for index in selected)
    assert set().union(*(set(selected) for selected in first)) == set(range(49))
    assert len({tuple(selected) for selected in first}) > 1


def test_sampling_obeys_restored_global_numpy_state():
    sampler = stochastic_api().sample_shot_indices
    np.random.seed(41)
    sampler(49, 8)
    state = np.random.get_state()
    expected = [sampler(49, 8) for _ in range(3)]
    np.random.seed(999)
    np.random.set_state(state)
    assert [sampler(49, 8) for _ in range(3)] == expected


@pytest.mark.parametrize("count", [0, -1, 50, 1.5, True])
def test_rejects_invalid_shots_per_update(count):
    cfg = experiment_api().default_config()
    cfg["acquisition"].update(num_shots=49, source_x_indices=list(range(20, 261, 5)))
    cfg["training"]["shots_per_update"] = count
    with pytest.raises(ValueError, match="shots_per_update"):
        experiment_api().validate_config(cfg)


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_rejects_invalid_full_evaluation_interval(interval):
    cfg = tiny_config()
    cfg["training"]["full_eval_interval"] = interval
    with pytest.raises(ValueError, match="full_eval_interval"):
        experiment_api().validate_config(cfg)


def test_sampling_settings_are_optional_without_changing_full_shot_default():
    cfg = experiment_api().default_config()
    assert cfg["training"].get("shots_per_update") is None
    assert experiment_api().validate_config(cfg) == cfg
    cfg["acquisition"].update(num_shots=49, source_x_indices=list(range(20, 261, 5)))
    cfg["training"].update(shots_per_update=8, full_eval_interval=25, shot_batch_size=4)
    assert experiment_api().validate_config(cfg) == cfg


def test_selected_sources_and_real_observations_keep_requested_order():
    truth, _ = tiny_fields()
    cfg = tiny_config()
    original = experiment_api().build_forward_problem(truth, cfg, "cpu")
    original_observations = original["observed"].clone()
    selected = stochastic_api().select_shot_problem(original, [4, 1, 3])
    assert selected["geometry"]["source_x_indices"] == [9, 3, 7]
    assert selected["geometry"]["num_shots"] == 3
    assert selected["geometry_tensors"]["xs"].tolist() == [[9, 3, 7]]
    assert torch.equal(selected["observed"], original_observations[:, [4, 1, 3]])
    assert selected["observed"].shape == (1, 3, 32, 11)
    assert selected["observed"].requires_grad is False
    assert [(item["first"], item["last"]) for item in selected["propagators"]] == [(0, 2), (2, 3)]
    old_solvers = [item["solver"] for item in original["propagators"]]
    assert all(item["solver"] is not old for item in selected["propagators"] for old in old_solvers)
    assert all(item["solver"].fd.vmax == float(truth.max()) for item in selected["propagators"])

    exact_truth = new_model(truth, cfg)
    assert experiment_api().evaluate_data_mse(exact_truth, selected) == pytest.approx(0., abs=1e-12)
    wrong_order = dict(selected, observed=selected["observed"].flip(1))
    assert experiment_api().evaluate_data_mse(exact_truth, wrong_order) > 1e-8
    assert original["geometry"]["source_x_indices"] == [1, 3, 5, 7, 9]
    assert torch.equal(original["observed"], original_observations)


def test_sampled_update_normalizes_by_selected_shots_and_reports_their_sources():
    truth, initial = tiny_fields()
    cfg = tiny_config()
    cfg["training"].update(shots_per_update=3, full_eval_interval=2)
    problem = experiment_api().build_forward_problem(truth, cfg, "cpu")
    sampler = stochastic_api().sample_shot_indices
    np.random.seed(23)
    expected_indices = sampler(5, 3)
    expected_sources = [cfg["acquisition"]["source_x_indices"][i] for i in expected_indices]
    independent = direct_problem(truth, cfg, expected_sources)
    model = new_model(initial, cfg)
    expected_loss = (direct_prediction(model, independent) - independent["observed"]).square().mean().item()
    assert expected_loss > 0
    np.random.seed(23)
    report = experiment_api().train_update(model, problem, torch.optim.Adam(model.parameters(), lr=1e-4))
    assert report["shot_indices"] == expected_indices
    assert report["source_x_indices"] == expected_sources
    assert report["data_mse"] == pytest.approx(expected_loss, rel=2e-6)
    assert report["gradient_norm"] > 0
    assert problem["observed"].shape[1] == 5


def test_selected_microbatches_preserve_full_selected_gradients_and_adam_state():
    truth, initial = tiny_fields()
    cfg = tiny_config()
    original = experiment_api().build_forward_problem(truth, cfg, "cpu")
    accumulated = stochastic_api().select_shot_problem(original, [4, 1, 3])
    independent = direct_problem(truth, cfg, [9, 3, 7])
    assert torch.equal(accumulated["observed"], independent["observed"])
    first = new_model(initial, cfg)
    second = copy.deepcopy(first)
    opt_first = torch.optim.Adam(first.parameters(), lr=1e-4)
    opt_second = torch.optim.Adam(second.parameters(), lr=1e-4)
    for _ in range(2):
        a = experiment_api().train_update(first, accumulated, opt_first)
        b = experiment_api().train_update(second, independent, opt_second)
        assert a["data_mse"] == pytest.approx(b["data_mse"], rel=2e-6)
        for x, y in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=1e-8)
            torch.testing.assert_close(x, y, rtol=2e-5, atol=1e-8)
        assert_optimizer_equal(opt_first.state_dict(), opt_second.state_dict(), exact=False)
    assert first.net.linear[0].weight.grad.norm() > 0


def test_short_window_evaluation_remains_full_data_and_does_not_draw_shots():
    truth, initial = tiny_fields()
    cfg = tiny_config()
    cfg["training"].update(shots_per_update=3, full_eval_interval=2)
    problem = experiment_api().build_forward_problem(truth, cfg, "cpu")
    independent = direct_problem(truth, cfg, [1, 3, 5, 7, 9])
    model = new_model(initial, cfg)
    expected_loss = (direct_prediction(model, independent) - independent["observed"]).square().mean().item()
    np.random.seed(19)
    before = np.random.get_state()
    report = stochastic_api().evaluate_windows(model, problem)
    after = np.random.get_state()
    assert report["data_mse"] == pytest.approx(expected_loss, rel=2e-6)
    assert report["common_1p9s_data_mse"] == pytest.approx(expected_loss, rel=2e-6)
    assert report.get("baseline_13shots_1p9s_data_mse") is None
    assert before[0] == after[0] and before[2:] == after[2:]
    assert np.array_equal(before[1], after[1])


def test_long_record_evaluation_uses_first_1000_samples_and_baseline_source_locations():
    # Baseline shots are deliberately reversed and preceded by an extra shot.
    # Prefix position is therefore an incorrect way to identify the 13-shot set.
    cfg = tiny_config()
    cfg["acquisition"].update(num_shots=14, source_x_indices=[30, *range(260, 19, -20)])
    cfg["forward"].update(dt_s=.0019, nt=1002, frequency_hz=8.)
    cfg["training"]["shot_batch_size"] = 4
    truth = np.full((3, 261), 2400., dtype=np.float32)
    problem = experiment_api().build_forward_problem(truth, cfg, "cpu")
    model = new_model(truth, cfg)
    problem["observed"] = problem["observed"].clone()
    problem["observed"][:, 0] += 3.
    problem["observed"][:, 1:] += 1.
    problem["observed"][:, :, 1000:] += 2.
    report = stochastic_api().evaluate_windows(model, problem)
    # 13 baseline shots have residual 1, extra shot residual 3; the last two
    # samples have residuals 3 and 5 respectively. All receivers are retained.
    # Offsets added to float32 FD records incur subtraction rounding.
    assert report["common_1p9s_data_mse"] == pytest.approx(22. / 14., rel=1e-5)
    assert report["baseline_13shots_1p9s_data_mse"] == pytest.approx(1., rel=1e-5)
    assert report["data_mse"] == pytest.approx((1000. * 22. + 2. * 142.) / (14. * 1002.), rel=1e-5)


def test_sampling_resume_keeps_exact_shot_sequence_weights_and_full_evaluation_history(tmp_path):
    truth, _ = tiny_fields()
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = tiny_config()
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1.
    cfg["training"].update(shots_per_update=3, full_eval_interval=2)
    continuous = experiment_api().run_experiment(cfg, tmp_path / "continuous", "cpu", save_figures=False)
    partial_cfg = copy.deepcopy(cfg)
    partial_cfg["training"]["epochs"] = 2
    first = experiment_api().run_experiment(partial_cfg, tmp_path / "partial", "cpu", save_figures=False)
    resumed = experiment_api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=first / "checkpoint.pt", save_figures=False)
    full_saved = torch.load(continuous / "checkpoint.pt", map_location="cpu", weights_only=False)
    resume_saved = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    full_sequence = [row["shot_indices"] for row in full_saved["history"]]
    resume_sequence = [row["shot_indices"] for row in resume_saved["history"]]
    assert len(full_sequence) == 4
    assert full_sequence == resume_sequence
    assert len({tuple(selected) for selected in full_sequence}) > 1
    for row in full_saved["history"]:
        assert len(row["shot_indices"]) == len(set(row["shot_indices"])) == 3
        assert row["source_x_indices"] == [[1, 3, 5, 7, 9][i] for i in row["shot_indices"]]
    for name in full_saved["model"]:
        assert torch.equal(full_saved["model"][name], resume_saved["model"][name]), name
    assert_optimizer_equal(full_saved["optimizer"], resume_saved["optimizer"], exact=True)
    np.testing.assert_array_equal(np.load(continuous / "final_velocity.npy"), np.load(resumed / "final_velocity.npy"))
    assert torch.equal(full_saved["rng"]["torch"], resume_saved["rng"]["torch"])
    assert np.array_equal(full_saved["rng"]["numpy"][1], resume_saved["rng"]["numpy"][1])
    assert full_saved["rng"]["numpy"][2:] == resume_saved["rng"]["numpy"][2:]
    for out, saved in ((continuous, full_saved), (resumed, resume_saved)):
        full_history = json.loads((out / "full_evaluation_history.json").read_text())
        assert full_history == saved["full_evaluation_history"]
        assert [row["evaluated_updates"] for row in full_history] == [0, 2, 4]
        assert all(np.isfinite(row["data_mse"]) and row["data_mse"] >= 0 for row in full_history)
        assert all(row.get("shot_indices") is None for row in full_history)
        metrics = json.loads((out / "metrics.json").read_text())
        assert full_history[-1]["data_mse"] == metrics["final"]["data_mse"]


@pytest.mark.parametrize("plotting_boundary", ["save_plots", "save_sampling_plot"])
def test_resume_after_final_plot_failure_finalizes_without_extra_adam_updates(tmp_path, monkeypatch, plotting_boundary):
    truth, _ = tiny_fields()
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = tiny_config()
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1.
    cfg["training"].update(shots_per_update=3, full_eval_interval=2)
    module = experiment_api()

    def plotting_failure(*args, **kwargs):
        raise OSError("Injected final plotting failure")

    with monkeypatch.context() as patch:
        if plotting_boundary == "save_sampling_plot":
            patch.setattr(module, "save_plots", lambda *args, **kwargs: None)
        patch.setattr(module, plotting_boundary, plotting_failure)
        with pytest.raises(OSError, match="Injected final plotting failure"):
            module.run_experiment(cfg, tmp_path / "failed", "cpu", save_figures=True)
    failed = next((tmp_path / "failed").iterdir())
    interrupted = torch.load(failed / "interrupted.pt", map_location="cpu", weights_only=False)
    assert interrupted["completed_updates"] == 4
    assert len(interrupted["history"]) == 4
    resumed = module.run_experiment(cfg, tmp_path / "resumed", "cpu", resume=failed / "interrupted.pt", save_figures=False)
    restored = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert restored["completed_updates"] == 4
    assert restored["history"] == interrupted["history"]
    for name in interrupted["model"]:
        assert torch.equal(interrupted["model"][name], restored["model"][name]), name
    assert_optimizer_equal(interrupted["optimizer"], restored["optimizer"], exact=True)
    assert [row["evaluated_updates"] for row in restored["full_evaluation_history"]] == [0, 2, 4]
    assert restored["full_evaluation_history"] == interrupted["full_evaluation_history"]
    assert json.loads((resumed / "status.json").read_text())["state"] == "completed"
    assert json.loads((resumed / "metrics.json").read_text())["completed_updates"] == 4


def test_full_shot_run_keeps_original_completed_budget_resume_rejection(tmp_path):
    truth, _ = tiny_fields()
    source = tmp_path / "truth.csv"
    np.savetxt(source, truth, delimiter=",")
    cfg = tiny_config()
    cfg["data"].update(model_file=str(source), downsample=1, csv_header="none")
    cfg["initialization"]["sigma"] = 1.
    cfg["training"]["epochs"] = 1
    completed = experiment_api().run_experiment(cfg, tmp_path / "completed", "cpu", save_figures=False)
    with pytest.raises(ValueError, match="exceed completed updates"):
        experiment_api().run_experiment(cfg, tmp_path / "resumed", "cpu", resume=completed / "checkpoint.pt", save_figures=False)

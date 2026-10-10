"""CPU supervisor tests: real artifact chains, fake external GPU jobs."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def api():
    name = "experiments.run_residual_ifwi_to_convergence"
    assert importlib.util.find_spec(name) is not None, "Automatic continuation supervisor is missing"
    return importlib.import_module(name)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(path, config, *, previous=None, plateau=True):
    """Generate a complete checkpoint chain without running a wave equation."""
    path.mkdir(parents=True)
    (path / "checkpoints").mkdir()
    count = config["training"]["epochs"]
    resumed = 0 if previous is None else json.loads((previous / "status.json").read_text())["completed_updates"]
    rng = np.random.RandomState(3)
    history = []
    for update in range(1, count + 1):
        indices = rng.choice(49, 8, replace=False).tolist()
        history.append({"update": update, "evaluated_updates": update - 1,
                        "shot_indices": indices, "source_x_indices": [20 + 5 * i for i in indices],
                        "data_mse": 1., "step_seconds": .01})
    full = [{"update": i, "evaluated_updates": i,
             "data_mse": 1. if plateau else float(np.exp(-i / 500.)),
             "rmse_mps": 123., "mae_mps": 100., "ssim": .5,
             "common_1p9s_data_mse": 1., "baseline_13shots_1p9s_data_mse": 1.,
             "loss_scope": "all_available_shots", "timing": "postupdate_evaluation"}
            for i in range(0, count + 1, 50)]
    if full[-1]["evaluated_updates"] != count:
        full.append({**full[-1], "update": count, "evaluated_updates": count})
    if previous is not None:
        old_full = json.loads((previous / "full_evaluation_history.json").read_text())
        full = old_full + [row for row in full if row["evaluated_updates"] > resumed]
    field = np.full((8, 8), 2000., dtype=np.float32)
    for name, value in (("true_velocity", field), ("initial_velocity", field),
                        ("run_start_velocity", field), ("final_velocity", field),
                        ("final_residual", np.zeros_like(field)),
                        ("observed", np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4))):
        np.save(path / f"{name}.npy", value)
    for update in range((resumed // 50 + 1) * 50, count + 1, 50):
        np.save(path / "checkpoints" / f"velocity_{update:04d}.npy", field)
    original = json.loads((ROOT / "source_manifest.json").read_text())
    original = {name: original[name] for name in ("ifwi_modules.py", "rnn_fd.py", "generator.py", "plot_functions.py")}
    names = ("residual_ifwi.py", "experiments/residual_ifwi_experiment.py",
             "experiments/residual_ifwi_stochastic.py", "experiments/residual_ifwi_data.py",
             "experiments/residual_ifwi_report.py")
    source_hashes = {name: digest(ROOT / name) for name in names}
    payload = {"schema": "residual_ifwi_stage1_v1", "model": {"weight": torch.zeros(1)},
               "optimizer": {"state": {0: {"step": torch.tensor(float(count)),
                                "exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)}},
                             "param_groups": [{"lr": .0001, "params": [0]}]},
               "config": config, "completed_updates": count, "history": history,
               "full_evaluation_history": full, "provenance": {"test": "fixed"},
               "original_source_hashes": original, "fixed_init_units": "m/s",
               "rng": {"python": random.getstate(), "numpy": rng.get_state(),
                       "torch": torch.get_rng_state(), "cuda": []}}
    torch.save(payload, path / "checkpoint.pt")
    write_json(path / "config.json", config)
    write_json(path / "status.json", {"state": "completed", "completed_updates": count})
    write_json(path / "metrics.json", {"completed_updates": count,
                                      "resumed_from_updates": resumed, "final": full[-1],
                                      "timing": {"training_seconds": .01 * (count - resumed)}})
    write_json(path / "original_source_hashes.json", original)
    write_json(path / "experiment_source_hashes.json", source_hashes)
    write_json(path / "full_evaluation_history.json", full)
    write_json(path / "history.json", full)
    write_json(path / "sampled_training_history.json", history)
    (path / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    return path


@pytest.fixture
def completed_run(tmp_path):
    config = yaml.safe_load((ROOT / "experiments/configs/residual_ifwi_49shots_5s_800.yaml").read_text())
    return make_run(tmp_path / "original_800", config)


def fake_jobs(monkeypatch, *, plateau=True, training_code=0, audit_code=0, corrupt=None, plot_code=0):
    module = api()
    commands = []

    def execute(command, *, cwd, stdout_path, stderr_path, on_started=None):
        commands.append(list(command))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("test child stdout\n")
        stderr_path.write_text("")
        if on_started is not None:
            on_started(4321)
        script = Path(command[command.index("-B") + 1]).name
        if script == "residual_ifwi.py":
            if training_code:
                return training_code
            previous = Path(command[command.index("--resume") + 1]).parent
            config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
            config["training"]["epochs"] = int(command[command.index("--epochs") + 1])
            importlib.import_module("experiments.residual_ifwi_experiment").validate_config(config)
            output = Path(command[command.index("--output-dir") + 1]) / "fake_run"
            make_run(output, config, previous=previous, plateau=plateau)
            if corrupt is not None:
                corrupt(output)
            return 0
        if script == "audit_stochastic_run.py":
            write_json(Path(command[command.index("--output") + 1]), {"passed": not audit_code,
                                                                       "failed_checks": ["test"] if audit_code else []})
            return audit_code
        if script == "plot_residual_ifwi_paper.py":
            return plot_code
        raise AssertionError(f"Unexpected external job {command}")

    monkeypatch.setattr(module, "_execute", execute)
    return commands


def test_supervisor_is_available():
    assert callable(api().run_until_convergence)


def test_existing_output_is_rejected_without_touching_it(completed_run, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises((ValueError, FileExistsError)):
        api().run_until_convergence(completed_run, output)
    assert marker.read_text() == "keep"
    assert list(output.iterdir()) == [marker]


def test_plateau_stops_and_preserves_original_artifacts(completed_run, tmp_path, monkeypatch):
    before = {p.relative_to(completed_run): digest(p) for p in completed_run.rglob("*") if p.is_file()}
    commands = fake_jobs(monkeypatch)
    output = tmp_path / "continuation"
    final = api().run_until_convergence(completed_run, output, device="cpu")
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == "converged"
    assert Path(state["final_output"]) == final
    assert state["completed_updates"] >= 1200
    assert state["completed_updates"] <= 1400
    training = [cmd for cmd in commands if "residual_ifwi.py" in [Path(item).name for item in cmd]]
    targets = [int(cmd[cmd.index("--epochs") + 1]) for cmd in training]
    assert targets == list(range(1000, state["completed_updates"] + 1, 200))
    assert Path(training[0][training[0].index("--resume") + 1]) == completed_run / "checkpoint.pt"
    for first, second in zip(training, training[1:]):
        previous_parent = Path(first[first.index("--output-dir") + 1]) / "fake_run"
        assert Path(second[second.index("--resume") + 1]) == previous_parent / "checkpoint.pt"
    original_config = json.loads((completed_run / "config.json").read_text())
    for cmd in training:
        saved = yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())
        assert saved == original_config
    assert {p.relative_to(completed_run): digest(p) for p in completed_run.rglob("*") if p.is_file()} == before


def test_improving_loss_continues_until_explicit_limit(completed_run, tmp_path, monkeypatch):
    fake_jobs(monkeypatch, plateau=False)
    output = tmp_path / "limit"
    final = api().run_until_convergence(completed_run, output, device="cpu", max_updates=1250)
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == "update_limit_reached"
    assert state["completed_updates"] == 1250
    assert Path(state["final_output"]) == final
    assert not json.loads((output / "convergence.json").read_text())["converged"]


@pytest.mark.parametrize("which", ["training", "audit"])
def test_external_failure_never_claims_convergence(completed_run, tmp_path, monkeypatch, which):
    commands = fake_jobs(monkeypatch, training_code=7 if which == "training" else 0,
                         audit_code=9 if which == "audit" else 0)
    output = tmp_path / which
    with pytest.raises(RuntimeError):
        api().run_until_convergence(completed_run, output, device="cpu")
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == "failed"
    assert which in state["error"].lower()
    assert len([cmd for cmd in commands if Path(cmd[cmd.index("-B") + 1]).name == "residual_ifwi.py"]) == 1
    assert not (output / "convergence.json").exists()


@pytest.mark.parametrize("filename", ["run_start_velocity.npy", "observed.npy"])
def test_broken_artifact_chain_is_rejected(completed_run, tmp_path, monkeypatch, filename):
    def corrupt(output):
        value = np.load(output / filename)
        value.flat[0] += 1
        np.save(output / filename, value)
    fake_jobs(monkeypatch, corrupt=corrupt)
    output = tmp_path / "bad_chain"
    with pytest.raises((ValueError, RuntimeError)):
        api().run_until_convergence(completed_run, output, device="cpu")
    assert json.loads((output / "state.json").read_text())["state"] == "failed"
    assert not (output / "convergence.json").exists()


def test_changed_history_prefix_is_rejected(completed_run, tmp_path, monkeypatch):
    def corrupt(output):
        path = output / "checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["history"][0]["shot_indices"] = [0] * 8
        torch.save(checkpoint, path)
    fake_jobs(monkeypatch, corrupt=corrupt)
    output = tmp_path / "bad_prefix"
    with pytest.raises((ValueError, RuntimeError)):
        api().run_until_convergence(completed_run, output, device="cpu")
    assert json.loads((output / "state.json").read_text())["state"] == "failed"


def test_plot_failure_keeps_successful_training_and_final_state(completed_run, tmp_path, monkeypatch):
    fake_jobs(monkeypatch, plot_code=5)
    output = tmp_path / "plot_failure"
    final = api().run_until_convergence(completed_run, output, device="cpu", max_updates=1000)
    assert final.is_dir()
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == "update_limit_reached"
    assert state["stages"][0]["plot_exit_code"] == 5


def test_maximum_at_start_never_launches_training(completed_run, tmp_path, monkeypatch):
    commands = fake_jobs(monkeypatch)
    output = tmp_path / "no_updates"
    final = api().run_until_convergence(completed_run, output, device="cpu", max_updates=800)
    assert final == completed_run
    assert not commands
    assert json.loads((output / "state.json").read_text())["state"] == "update_limit_reached"


def test_modified_prior_adam_checkpoint_is_never_used_for_next_segment(completed_run, tmp_path, monkeypatch):
    fake_jobs(monkeypatch)
    module = api()
    normal = module._execute

    def execute(command, **kwargs):
        code = normal(command, **kwargs)
        if Path(command[command.index("-B") + 1]).name == "plot_residual_ifwi_paper.py":
            run = Path(command[command.index("--run-dir") + 1])
            path = run / "checkpoint.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint["optimizer"]["state"][0]["exp_avg"].fill_(123.)
            torch.save(checkpoint, path)
        return code

    monkeypatch.setattr(module, "_execute", execute)
    output = tmp_path / "modified_adam"
    with pytest.raises((ValueError, RuntimeError)):
        module.run_until_convergence(completed_run, output, device="cpu")
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == "failed"
    assert len(state["stages"]) == 1


def test_unaligned_explicit_limit_is_not_reported_as_convergence(completed_run, tmp_path, monkeypatch):
    fake_jobs(monkeypatch)
    output = tmp_path / "partial_limit"
    final = api().run_until_convergence(completed_run, output, device="cpu", chunk_updates=500, max_updates=1223)
    state = json.loads((output / "state.json").read_text())
    assert state["state"] == "update_limit_reached"
    assert state["completed_updates"] == 1223
    assert Path(state["final_output"]) == final
    assert not json.loads((output / "convergence.json").read_text())["converged"]


def test_truthy_nonboolean_audit_result_is_rejected(completed_run, tmp_path, monkeypatch):
    fake_jobs(monkeypatch)
    module = api()
    normal = module._execute

    def execute(command, **kwargs):
        code = normal(command, **kwargs)
        if Path(command[command.index("-B") + 1]).name == "audit_stochastic_run.py":
            write_json(Path(command[command.index("--output") + 1]), {"passed": "true"})
        return code

    monkeypatch.setattr(module, "_execute", execute)
    output = tmp_path / "invalid_audit"
    with pytest.raises(RuntimeError):
        module.run_until_convergence(completed_run, output, device="cpu")
    assert json.loads((output / "state.json").read_text())["state"] == "failed"
    assert not (output / "convergence.json").exists()


def test_generated_config_passes_real_cli_validation(completed_run, tmp_path):
    output = tmp_path / "exported_config"
    api().run_until_convergence(completed_run, output, device="cpu", max_updates=800)
    config_path = output / "saved_config.yaml"
    never_created = tmp_path / "dry_run_output"
    import os
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "residual_ifwi.py"),
                             "--config", str(config_path), "--resume", str(completed_run / "checkpoint.pt"),
                             "--epochs", "1000", "--device", "cpu", "--dry-run",
                             "--output-dir", str(never_created)], cwd=ROOT, capture_output=True,
                            text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1"))
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["forward"]["pml_reflection"] == 1e-6
    assert parsed["training"]["epochs"] == 1000
    assert not never_created.exists()


def test_started_callback_failure_terminates_and_reaps_real_child(tmp_path, monkeypatch):
    module = api()
    actual_popen = module.subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        process = actual_popen(*args, **kwargs)
        processes.append(process)
        return process

    def callback(pid):
        raise OSError("state write failed")

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    try:
        with pytest.raises(OSError, match="state write failed"):
            module._execute([sys.executable, "-B", "-c", "import time; time.sleep(60)"], cwd=ROOT,
                            stdout_path=tmp_path / "stdout.log", stderr_path=tmp_path / "stderr.log",
                            on_started=callback)
        assert len(processes) == 1
        assert processes[0].poll() is not None, "Callback failure left the child running"
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)

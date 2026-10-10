"""Continue an immutable residual IFWI run in audited chunks until a plateau.

The existing training CLI restores Adam and all RNG states. This supervisor
changes only its cumulative update budget, never the waveform objective.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.residual_ifwi_convergence import ConvergencePolicy, analyze_convergence

TRAINING_FILES = ("residual_ifwi.py", "experiments/residual_ifwi_experiment.py",
                  "experiments/residual_ifwi_stochastic.py", "experiments/residual_ifwi_data.py",
                  "experiments/residual_ifwi_report.py")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(run):
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    if saved.get("schema") != "residual_ifwi_stage1_v1":
        raise ValueError("Unsupported checkpoint schema")
    if not all(torch.isfinite(value).all().item() for value in saved["model"].values()):
        raise ValueError("Nonfinite checkpoint model")
    if not {"python", "numpy", "torch", "cuda"}.issubset(saved["rng"]):
        raise ValueError("Incomplete checkpoint RNG state")
    count = saved["completed_updates"]
    states = list(saved["optimizer"]["state"].values())
    if not states or any(float(state["step"]) != count for state in states):
        raise ValueError("Adam checkpoint step differs from completed updates")
    return saved


def _completed_run(run, expected=None):
    status, config, metrics = (_read(run / name) for name in ("status.json", "config.json", "metrics.json"))
    saved = _checkpoint(run)
    count = saved["completed_updates"]
    if (status["state"] != "completed" or count != status["completed_updates"]
            or count != metrics["completed_updates"] or count != config["training"]["epochs"]
            or count != len(saved["history"]) or saved["config"] != config
            or (expected is not None and count != expected)):
        raise ValueError("Run is not complete at the requested cumulative budget")
    if saved["history"] != _read(run / "sampled_training_history.json"):
        raise ValueError("Checkpoint sampled history differs from saved history")
    if saved["full_evaluation_history"] != _read(run / "full_evaluation_history.json"):
        raise ValueError("Checkpoint full evaluation history differs from saved history")
    return config, metrics, saved


def _freeze(start):
    training = _read(start / "experiment_source_hashes.json")
    if set(training) != set(TRAINING_FILES):
        raise ValueError("Initial run has an incomplete training source manifest")
    core = _read(start / "original_source_hashes.json")
    for names in (training, core, _read(ROOT / "source_manifest.json")):
        for name, expected in names.items():
            if _sha(ROOT / name) != expected:
                raise ValueError(f"Frozen source identity differs: {name}")
    current = {**training, **_read(ROOT / "source_manifest.json")}
    for name in ("experiments/run_residual_ifwi_to_convergence.py",
                 "experiments/residual_ifwi_convergence.py",
                 "verification/residual_ifwi/audit_stochastic_run.py",
                 "experiments/plot_residual_ifwi_paper.py", "source_manifest.json"):
        current[name] = _sha(ROOT / name)
    return {"start_run": str(start),
            "original_run_files": {str(path): _sha(path) for path in start.iterdir() if path.is_file()},
            "source_files": {str(ROOT / name): digest for name, digest in current.items()},
            "training_source_hashes": training, "original_source_hashes": core,
            "start_checkpoint_sha256": _sha(start / "checkpoint.pt")}


def _verify_frozen(frozen):
    for group in ("original_run_files", "source_files"):
        for name, expected in frozen[group].items():
            if not Path(name).is_file() or _sha(name) != expected:
                raise ValueError(f"Frozen file changed: {name}")


def _same_array(first, second):
    a, b = (np.load(path, mmap_mode="r", allow_pickle=False) for path in (first, second))
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    # View bytes, rather than numeric ==, to include signed-zero differences.
    return bool(np.array_equal(a.view(np.uint8), b.view(np.uint8)))


def _validate_chain(previous, current, prior_saved, target, frozen):
    config, metrics, saved = _completed_run(current, target)
    previous_config = _read(previous / "config.json")
    expected_config = json.loads(json.dumps(previous_config))
    expected_config["training"]["epochs"] = target
    if config != expected_config:
        raise ValueError("Continuation changed a setting other than cumulative epochs")
    prior_count = prior_saved["completed_updates"]
    if metrics["resumed_from_updates"] != prior_count:
        raise ValueError("Continuation did not restore the previous update count")
    if saved["history"][:prior_count] != prior_saved["history"]:
        raise ValueError("Continuation rewrote the sampled history prefix")
    length = len(prior_saved["full_evaluation_history"])
    if saved["full_evaluation_history"][:length] != prior_saved["full_evaluation_history"]:
        raise ValueError("Continuation rewrote the full evaluation history prefix")
    if not _same_array(previous / "final_velocity.npy", current / "run_start_velocity.npy"):
        raise ValueError("Continuation run-start velocity differs from prior final velocity")
    for name in ("true_velocity.npy", "initial_velocity.npy", "observed.npy"):
        if not _same_array(previous / name, current / name):
            raise ValueError(f"Continuation changed {name}")
    if (_read(current / "experiment_source_hashes.json") != frozen["training_source_hashes"]
            or _read(current / "original_source_hashes.json") != frozen["original_source_hashes"]):
        raise ValueError("Continuation changed source identities")
    return metrics, saved


def _collect_velocities(run, velocities):
    for path in sorted((run / "checkpoints").glob("velocity_*.npy")):
        match = re.fullmatch(r"velocity_([0-9]+)\.npy", path.name)
        if match:
            update = int(match.group(1))
            array = np.load(path, allow_pickle=False)
            if update in velocities and not np.array_equal(velocities[update].view(np.uint8), array.view(np.uint8)):
                raise ValueError(f"Conflicting velocity snapshots at update {update}")
            velocities[update] = array
    metrics = _read(run / "metrics.json")
    for update, name in ((metrics["resumed_from_updates"], "run_start_velocity.npy"),
                         (metrics["completed_updates"], "final_velocity.npy")):
        array = np.load(run / name, allow_pickle=False)
        if update in velocities and not np.array_equal(velocities[update].view(np.uint8), array.view(np.uint8)):
            raise ValueError(f"Conflicting endpoint velocity at update {update}")
        velocities[update] = array


def _execute(command, *, cwd, stdout_path, stderr_path, on_started=None):
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                                   creationflags=flags, env=dict(os.environ, PYTHONUTF8="1"))
        try:
            if on_started is not None:
                on_started(process.pid)
            return process.wait()
        except BaseException:
            # Avoid an orphan GPU job if the supervisor is explicitly interrupted.
            process.terminate()
            process.wait()
            raise


@contextmanager
def _awake_and_locked(output):
    with (output / "supervisor.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import ctypes
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            kernel = ctypes.windll.kernel32
            guarded = bool(kernel.SetThreadExecutionState(0x80000001))
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                if guarded:
                    kernel.SetThreadExecutionState(0x80000000)
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def run_until_convergence(start_run: Path, output_parent: Path, *, device="cuda:0",
                          chunk_updates=200, policy: ConvergencePolicy | None = None,
                          max_updates: int | None = None, reference_dir: Path | None = None) -> Path:
    start, output = Path(start_run).resolve(), Path(output_parent).resolve()
    if output.exists():
        raise FileExistsError(f"Continuation output already exists: {output}")
    if output == start or start in output.parents:
        raise ValueError("Continuation output must be outside the immutable initial run")
    policy = ConvergencePolicy() if policy is None else policy
    if isinstance(chunk_updates, bool) or not isinstance(chunk_updates, int) or chunk_updates <= 0:
        raise ValueError("chunk_updates must be a positive integer")
    config, initial_metrics, saved = _completed_run(start)
    origin = saved["completed_updates"]
    interval = config["training"].get("full_eval_interval", config["training"]["checkpoint_interval"])
    if interval != policy.eval_interval or chunk_updates % interval:
        raise ValueError("Chunks and convergence policy must match the frozen full evaluation interval")
    if max_updates is not None and (isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates < origin):
        raise ValueError("max_updates must be an integer >= initial completed updates")
    if reference_dir is not None:
        reference_dir = Path(reference_dir).resolve()
        ref_config = _read(reference_dir / "config.json")
        if ref_config["acquisition"]["num_shots"] != 13 or ref_config["forward"]["nt"] != 1000:
            raise ValueError("Audit reference must be the original 13-shot, 1000-sample run")
    frozen = _freeze(start)
    output.mkdir(parents=True, exist_ok=False)
    (output / "stages").mkdir()
    (output / "logs").mkdir()
    config_path = output / "saved_config.yaml"
    config_json_path = output / "saved_config.json"
    # PyYAML 1.1 reads JSON's 1e-06 as a string; its own dumper emits 1.0e-06.
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _json(config_json_path, config)
    _json(output / "policy.json", asdict(policy))
    for path in (config_path, config_json_path, output / "policy.json"):
        frozen["source_files"][str(path)] = _sha(path)
    _json(output / "frozen_sources.json", frozen)
    state = {"state": "preparing", "supervisor_pid": os.getpid(), "start_run": str(start),
             "resumed_from_updates": origin, "completed_updates": origin,
             "chunk_updates": chunk_updates, "max_updates": max_updates,
             "final_output": str(start), "current_run": str(start), "active_output_parent": None,
             "stages": [], "selection": "last completed update; no truth-based selection"}
    began = time.perf_counter()
    velocities = {}
    current, metrics = start, initial_metrics
    current_checkpoint_sha256 = frozen["start_checkpoint_sha256"]

    def persist(**changes):
        state.update(changes)
        state["elapsed_supervisor_seconds"] = time.perf_counter() - began
        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _json(output / "state.json", state)

    def child(kind, command, stem):
        persist(state=kind, child_pid=None, command=command)
        return _execute(command, cwd=ROOT, stdout_path=output / "logs" / f"{stem}_stdout.log",
                        stderr_path=output / "logs" / f"{stem}_stderr.log",
                        on_started=lambda pid: persist(child_pid=pid))

    persist()
    try:
        with _awake_and_locked(output):
            _collect_velocities(start, velocities)
            while max_updates is None or saved["completed_updates"] < max_updates:
                _verify_frozen(frozen)
                if _sha(current / "checkpoint.pt") != current_checkpoint_sha256:
                    raise ValueError("Previously audited Adam/RNG checkpoint changed before continuation")
                previous_count = saved["completed_updates"]
                target = previous_count + chunk_updates
                if max_updates is not None:
                    target = min(target, max_updates)
                number = len(state["stages"]) + 1
                stage_parent = output / "stages" / f"stage_{number:04d}"
                stage = {"stage": number, "resumed_from_updates": previous_count,
                         "target_updates": target, "resume_checkpoint": str(current / "checkpoint.pt"),
                         "resume_checkpoint_sha256": _sha(current / "checkpoint.pt")}
                state["stages"].append(stage)
                persist(active_output_parent=str(stage_parent), current_run=None)
                command = [sys.executable, "-X", "utf8", "-u", "-B", str(ROOT / "residual_ifwi.py"),
                           "--config", str(config_path), "--resume", str(current / "checkpoint.pt"),
                           "--epochs", str(target), "--device", str(device), "--output-dir", str(stage_parent)]
                stage_started = time.perf_counter()
                code = child("training", command, f"stage_{number:04d}_training")
                stage["training_exit_code"] = code
                if code:
                    raise RuntimeError(f"Training failed with exit code {code}; inspect stage logs and saved checkpoints")
                candidates = sorted(path for path in stage_parent.iterdir() if path.is_dir() and (path / "status.json").is_file())
                if len(candidates) != 1:
                    raise ValueError("Expected exactly one new training result directory")
                next_run = candidates[0]
                persist(current_run=str(next_run))
                metrics, next_saved = _validate_chain(current, next_run, saved, target, frozen)
                _verify_frozen(frozen)
                audit_path = next_run / "audit.json"
                audit_command = [sys.executable, "-X", "utf8", "-B", str(ROOT / "verification/residual_ifwi/audit_stochastic_run.py"),
                                 "--run-dir", str(next_run), "--output", str(audit_path)]
                if reference_dir is not None:
                    audit_command += ["--reference-dir", str(reference_dir)]
                code = child("auditing", audit_command, f"stage_{number:04d}_audit")
                stage["audit_exit_code"] = code
                if code or _read(audit_path).get("passed") is not True:
                    raise RuntimeError(f"Audit failed with exit code {code}; continuation stopped")
                current, saved = next_run, next_saved
                _collect_velocities(current, velocities)
                history = saved["full_evaluation_history"]
                aligned = [row for row in history if row["evaluated_updates"] <= origin or (row["evaluated_updates"] - origin) % interval == 0]
                decision = analyze_convergence(aligned, velocities, resumed_from_updates=origin, policy=policy)
                if (target - origin) % interval:
                    decision.update(converged=False, state="unaligned_limit_endpoint", evaluated_updates=target)
                _json(output / "convergence.json", {**decision, "policy": asdict(policy), "original_resume_updates": origin})
                stage.update(output=str(current), completed_updates=target,
                             checkpoint_sha256=_sha(current / "checkpoint.pt"),
                             elapsed_stage_seconds=time.perf_counter() - stage_started,
                             training_seconds=metrics["timing"]["training_seconds"],
                             convergence=decision, final_metrics=metrics["final"])
                current_checkpoint_sha256 = stage["checkpoint_sha256"]
                cumulative_training = sum(item["training_seconds"] for item in state["stages"])
                persist(state="reporting", completed_updates=target, child_pid=None, final_output=str(current),
                        additional_training_seconds=cumulative_training,
                        cumulative_training_seconds=initial_metrics["timing"]["training_seconds"] + cumulative_training,
                        final_metrics=metrics["final"])
                plot_command = [sys.executable, "-X", "utf8", "-B", str(ROOT / "experiments/plot_residual_ifwi_paper.py"),
                                "--run-dir", str(current), "--output-dir", str(current / "paper_style_original_ifwi"),
                                "--style", "original_ifwi"]
                try:
                    stage["plot_exit_code"] = child("reporting", plot_command, f"stage_{number:04d}_plot")
                except Exception as error:
                    stage["plot_exit_code"] = None
                    stage["plot_error"] = f"{type(error).__name__}: {error}"
                _verify_frozen(frozen)
                if _sha(current / "checkpoint.pt") != current_checkpoint_sha256:
                    raise ValueError("Previously audited Adam/RNG checkpoint changed during reporting")
                if decision["converged"]:
                    persist(state="converged", child_pid=None, conclusion="Full-data MSE and saved velocity satisfy the frozen numerical plateau rule")
                    return current
                persist(state="continuing", child_pid=None)
            persist(state="update_limit_reached", final_output=str(current), child_pid=None,
                    conclusion="Explicit cumulative update limit reached; convergence was not established")
            return current
    except BaseException as error:
        persist(state="failed", child_pid=None, error=f"{type(error).__name__}: {error}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-updates", type=int, default=200)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--policy-json", type=Path)
    args = parser.parse_args(argv)
    policy = ConvergencePolicy(**_read(args.policy_json)) if args.policy_json else None
    final = run_until_convergence(args.start_run, args.output_dir, device=args.device,
                                 chunk_updates=args.chunk_updates, policy=policy,
                                 max_updates=args.max_updates, reference_dir=args.reference_dir)
    print(f"Final output: {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

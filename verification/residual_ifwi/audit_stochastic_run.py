"""Read-only CPU audit of a completed 49-shot residual IFWI run.

This audits saved artifacts and the sampling RNG, not the wave-equation physics.
No propagator is imported or run. By default the JSON report goes to stdout;
--output creates a new file exclusively and never replaces an existing report.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import torch
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ("ifwi_modules.py", "rnn_fd.py", "generator.py", "plot_functions.py")
EXPERIMENT_FILES = ("residual_ifwi.py", "experiments/residual_ifwi_experiment.py",
                    "experiments/residual_ifwi_stochastic.py", "experiments/residual_ifwi_data.py",
                    "experiments/residual_ifwi_report.py")
SOURCES = list(range(20, 261, 5))
BASELINE_SOURCES = list(range(20, 261, 20))
METRIC_KEYS = ("rmse_mps", "mae_mps", "ssim")
LOSS_KEYS = ("data_mse", "common_1p9s_data_mse", "baseline_13shots_1p9s_data_mse")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bitwise_equal(first, second):
    first, second = np.asarray(first), np.asarray(second)
    return (first.shape == second.shape and first.dtype == second.dtype
            and np.array_equal(first.view(np.uint8), second.view(np.uint8)))


def velocity_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2 or min(truth.shape) < 3:
        raise ValueError("Velocity metric inputs must be equal two-dimensional grids")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Velocity metric inputs must be finite")
    difference = prediction - truth
    window = min(7, min(truth.shape))
    window -= int(window % 2 == 0)
    return {
        "rmse_mps": float(np.sqrt(np.mean(difference ** 2))),
        "mae_mps": float(np.mean(np.abs(difference))),
        "ssim": float(structural_similarity(truth, prediction,
                      data_range=float(truth.max() - truth.min()) or 1., win_size=window)),
    }


def metrics_agree(first, second, keys):
    return all(key in first and key in second
               and isinstance(first[key], (int, float)) and not isinstance(first[key], bool)
               and np.isfinite(first[key]) and np.isfinite(second[key])
               and np.isclose(first[key], second[key], atol=1e-10, rtol=1e-12)
               for key in keys)


def cpu_model_class():
    """Load only the original IRN and parameterization AST, without FD imports."""
    namespace = {"np": np, "torch": torch, "math": math}
    for relative, names in (("ifwi_modules.py", {"IRN"}),
                            ("experiments/residual_ifwi_experiment.py", {"_positive", "VelocityParameterization"})):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        if {node.name for node in nodes} != names:
            raise ValueError(f"Missing model definition in {path}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["VelocityParameterization"]


def compare_observed(observed, acquisition, reference, reference_acquisition, atol, rtol):
    """Compare only shared source locations and the first 1000 samples, in chunks."""
    current_sources = acquisition["source_x_indices"]
    old_sources = reference_acquisition["source_x_indices"]
    if list(old_sources) != BASELINE_SOURCES or reference.shape != (1, 13, 1000, 288):
        raise ValueError("Reference must contain the original 13 shots and 1000-sample records")
    maximum = 0.
    mismatches = 0
    allclose = True
    per_shot = []
    for old_index, source in enumerate(old_sources):
        new_index = current_sources.index(source)
        shot_maximum, shot_mismatches = 0., 0
        for first in range(0, 1000, 128):
            last = min(first + 128, 1000)
            new = np.asarray(observed[0, new_index, first:last])
            old = np.asarray(reference[0, old_index, first:last])
            difference = np.abs(new.astype(np.float64) - old.astype(np.float64))
            shot_maximum = max(shot_maximum, float(difference.max()))
            shot_mismatches += int(np.count_nonzero(new != old))
            allclose = allclose and bool(np.allclose(new, old, atol=atol, rtol=rtol, equal_nan=False))
        maximum = max(maximum, shot_maximum)
        mismatches += shot_mismatches
        per_shot.append({"source_x_index": source, "current_shot_index": new_index,
                         "reference_shot_index": old_index, "max_abs_difference": shot_maximum,
                         "unequal_element_count": shot_mismatches})
    return {"exact_values": mismatches == 0, "allclose": allclose,
            "dtype_equal": str(observed.dtype) == str(reference.dtype),
            "max_abs_difference": maximum, "unequal_element_count": mismatches,
            "compared_shape": [1, 13, 1000, 288], "atol": atol, "rtol": rtol,
            "per_shot": per_shot}


def audit(args):
    out = args.run_dir.resolve()
    report = {"schema": "residual_ifwi_stochastic_cpu_audit_v1", "run_dir": str(out),
              "reference_dir": str(args.reference_dir.resolve()) if args.reference_dir else None,
              "device": "cpu", "checks": [], "details": {},
              "limitations": [
                  "No finite-difference simulation or waveform MSE is recomputed.",
                  "This checks saved artifact consistency; it does not establish physical correctness or convergence.",
                  "Nonzero late-time observations exclude an entirely zero-appended tail, not every possible data error.",
                  "CPU checkpoint evaluation uses declared tolerances because CUDA and CPU arithmetic can differ.",
              ]}

    def check(name, passed, **detail):
        report["checks"].append({"name": name, "passed": bool(passed), **detail})

    config = read_json(out / "config.json")
    acquisition = read_json(out / "acquisition.json")
    metrics = read_json(out / "metrics.json")
    initial_checks = read_json(out / "initial_checks.json")
    protocol = read_json(out / "sampling_protocol.json")
    status = read_json(out / "status.json")
    sampled = read_json(out / "sampled_training_history.json")
    full = read_json(out / "full_evaluation_history.json")
    history = read_json(out / "history.json")
    with (out / "history.jsonl").open("r", encoding="utf-8") as stream:
        jsonl = [json.loads(line) for line in stream if line.strip()]
    checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    epochs = config["training"]["epochs"]
    completed = checkpoint["completed_updates"]
    resumed = metrics["resumed_from_updates"]
    interval = config["training"].get("full_eval_interval", config["training"]["checkpoint_interval"])
    check("completed_update_counts", isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0
          and epochs == completed == metrics["completed_updates"] == status["completed_updates"] == len(sampled)
          and status["state"] == "completed", config_epochs=epochs, checkpoint_updates=completed,
          metrics_updates=metrics["completed_updates"], status_updates=status["completed_updates"],
          sampled_rows=len(sampled), status=status["state"])
    check("checkpoint_schema_config_units", checkpoint.get("schema") == "residual_ifwi_stage1_v1"
          and checkpoint["config"] == config and checkpoint.get("fixed_init_units") == "m/s"
          and metrics.get("metric_velocity_units") == "m/s")
    check("sampled_history_copies", sampled == jsonl == checkpoint["history"])
    check("full_history_copies", full == history == checkpoint.get("full_evaluation_history"))
    check("approved_49shot_protocol", config["parameterization"] == "residual" and config["seed"] == 3
          and config["acquisition"]["num_shots"] == 49
          and config["acquisition"]["source_x_indices"] == SOURCES
          and config["training"].get("shots_per_update") == 8
          and config["training"]["shot_batch_size"] == 4
          and config["forward"]["nt"] == 2632 and config["forward"]["dt_s"] == .0019
          and config["data"]["grid_spacing_m"] == 15. and config["data"]["downsample"] == 4
          and config["data"]["crop_shape"] is None and config["initialization"]["sigma"] == 15.
          and config["initialization"]["gaussian_mode"] == "reflect"
          and config["initialization"]["gaussian_truncate"] == 4.
          and config["model"] == {"neurons": [2, 128, 128, 128, 128, 1], "omega_0": 30., "mean_kmps": 3., "std_kmps": 1.}
          and config["training"]["learning_rate"] == 1e-4,
          total_updates_requested=epochs, is_800_update_run=epochs == 800,
          shot_microbatch_size=4, sampled_shots_per_update=8)
    check("sampling_metadata", protocol["shots_per_update"] == 8 and protocol["available_shots"] == 49
          and protocol["shot_microbatch_size"] == 4 and protocol["full_eval_interval"] == interval
          and protocol["nt"] == 2632 and protocol["dt_s"] == .0019
          and protocol["time_segmentation"] is False
          and np.isclose(protocol["record_sample_span_s"], 4.9989, rtol=0., atol=1e-12))
    check("acquisition_order_shape", acquisition["source_x_indices"] == SOURCES
          and acquisition["num_shots"] == 49 and acquisition["nz"] == 94 and acquisition["nx"] == 288
          and acquisition["num_receivers"] == 288 and acquisition["nt"] == 2632
          and acquisition["receiver_x_indices"] == list(range(288))
          and acquisition["source_z_index"] == 1 and acquisition["receiver_z_index"] == 2
          and acquisition["axis_order"] == ["z", "x"])

    rng = np.random.RandomState(config["seed"])
    invalid_samples, mismatched_draws, invalid_rows = [], [], []
    coverage = np.zeros(49, dtype=np.int64)
    diagnostic_keys = (*METRIC_KEYS, "data_mse", "gradient_norm", "parameter_update_norm", "step_seconds")
    for update, row in enumerate(sampled, 1):
        indices = row.get("shot_indices", [])
        valid = (isinstance(indices, list) and len(indices) == 8 and len(set(indices)) == 8
                 and all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < 49 for i in indices))
        if not valid or row.get("source_x_indices") != ([SOURCES[i] for i in indices] if valid else None):
            invalid_samples.append(update)
        if valid:
            coverage[indices] += 1
        expected = rng.choice(49, 8, replace=False).tolist()
        if indices != expected:
            mismatched_draws.append(update)
        diagnostics_ok = all(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
                             and np.isfinite(row[key]) for key in diagnostic_keys)
        if (row.get("update") != update or row.get("evaluated_updates") != update - 1
                or row.get("loss_scope") != "sampled_shots_mean" or not diagnostics_ok
                or (diagnostics_ok and (row["data_mse"] < 0 or row["gradient_norm"] < 0
                    or row["parameter_update_norm"] < 0 or row["step_seconds"] < 0))):
            invalid_rows.append(update)
    check("sampled_shot_indices_source_order", not invalid_samples, invalid_updates=invalid_samples)
    check("training_timing_finite_diagnostics", not invalid_rows, invalid_updates=invalid_rows)
    check("seed3_sampling_replay", not mismatched_draws, mismatched_updates=mismatched_draws)
    expected_state, saved_state = rng.get_state(), checkpoint["rng"]["numpy"]
    check("checkpoint_numpy_rng_state", expected_state[0] == saved_state[0]
          and np.array_equal(expected_state[1], saved_state[1])
          and expected_state[2:] == tuple(saved_state[2:]),
          expected_position=int(expected_state[2]), saved_position=int(saved_state[2]))
    report["details"]["sampling_coverage"] = {
        "shot_evaluations": int(coverage.sum()), "expected_shot_evaluations": 8 * completed,
        "mean_per_shot": float(coverage.mean()), "minimum_per_shot": int(coverage.min()),
        "maximum_per_shot": int(coverage.max()), "counts_in_acquisition_order": coverage.tolist()}

    full_counts = [row.get("evaluated_updates") for row in full]
    valid_counts = bool(full_counts) and all(isinstance(n, int) and not isinstance(n, bool) for n in full_counts)
    required = [0, *range(interval, completed + 1, interval), completed]
    if resumed:
        required = [0, resumed, *[n for n in range(interval, completed + 1, interval) if n > resumed], completed]
    required = sorted(set(required))
    endpoints_ok = valid_counts and full_counts == sorted(set(full_counts)) and full_counts[0] == 0 and full_counts[-1] == completed
    if endpoints_ok:
        endpoints_ok = all(n in full_counts for n in required)
        # Resume preserves old evaluation intervals and an old smoke-run endpoint.
        endpoints_ok = endpoints_ok and all(n in required or (0 < n <= resumed) for n in full_counts)
    check("full_evaluation_endpoints", endpoints_ok, recorded=full_counts, required=required,
          resumed_from_updates=resumed, current_interval=interval,
          retained_pre_resume_endpoints=[n for n in full_counts if isinstance(n, int) and n not in required])
    bad_full_rows = [i for i, row in enumerate(full) if row.get("update") != row.get("evaluated_updates")
                     or row.get("loss_scope") != "all_available_shots" or row.get("timing") != "postupdate_evaluation"
                     or not all(isinstance(row.get(key), (int, float)) and np.isfinite(row[key]) for key in (*METRIC_KEYS, *LOSS_KEYS))
                     or any(row.get(key, -1.) < 0 for key in LOSS_KEYS)]
    check("full_evaluation_timing_finite_values", not bad_full_rows, invalid_row_indices=bad_full_rows)
    check("final_metrics_full_history", bool(full) and metrics_agree(metrics["final"], full[-1], (*METRIC_KEYS, *LOSS_KEYS)))

    arrays = {name: np.load(out / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("true_velocity", "initial_velocity", "run_start_velocity", "final_velocity", "final_residual", "observed", "wavelet")}
    truth, background, start, final = (arrays[name] for name in ("true_velocity", "initial_velocity", "run_start_velocity", "final_velocity"))
    check("saved_velocity_array_shapes", all(arrays[name].shape == (94, 288) and arrays[name].dtype == np.float32
          for name in ("true_velocity", "initial_velocity", "run_start_velocity", "final_velocity", "final_residual")))
    check("saved_velocity_arrays_finite_positive", all(np.isfinite(arrays[name]).all()
          for name in ("true_velocity", "initial_velocity", "run_start_velocity", "final_velocity", "final_residual"))
          and all(float(arrays[name].min()) > 0 for name in ("true_velocity", "initial_velocity", "run_start_velocity", "final_velocity")))
    check("saved_residual_matches_fixed_background", bitwise_equal(arrays["final_residual"], final - background))
    observed = arrays["observed"]
    check("observed_shape_dtype", observed.shape == (1, 49, 2632, 288) and observed.dtype == np.float32,
          shape=list(observed.shape), dtype=str(observed.dtype))
    observed_finite, nonzero_tails = True, []
    for shot in range(observed.shape[1]):
        tail_nonzero = False
        for first in range(0, observed.shape[2], 128):
            block = observed[0, shot, first:min(first + 128, observed.shape[2])]
            observed_finite = observed_finite and bool(np.isfinite(block).all())
            if first + 128 > 1000:
                tail = observed[0, shot, max(first, 1000):min(first + 128, observed.shape[2])]
                tail_nonzero = tail_nonzero or bool(np.any(tail != 0))
        nonzero_tails.append(tail_nonzero)
    check("observed_finite_late_record_present", observed_finite and all(nonzero_tails),
          finite=observed_finite, each_shot_after_sample1000_nonzero=nonzero_tails)
    check("wavelet_saved_full_record", arrays["wavelet"].shape == (2632,) and np.isfinite(arrays["wavelet"]).all(),
          shape=list(arrays["wavelet"].shape))
    check("initial_checks_metadata", initial_checks["waveform_shape"] == [1, 49, 2632, 288]
          and initial_checks["velocity_units"] == "m/s" and initial_checks["observations_detached"] is True
          and initial_checks["fixed_initializer_requires_grad"] is False
          and initial_checks["coordinates_order"] == ["x", "z"] and initial_checks["coordinates_units"] == "km")
    check("fresh_zero_residual_initialization", resumed != 0 or (bitwise_equal(start, background)
          and initial_checks["max_residual_at_run_start_mps"] == 0.), applies_to_fresh_run=resumed == 0,
          max_run_start_residual_mps=float(np.max(np.abs(start - background))))
    for name, array, saved_metrics in (("initial", start, metrics["initial"]),
                                      ("fixed_background", background, metrics["fixed_background"]),
                                      ("final", final, metrics["final"])):
        recomputed = velocity_metrics(array, truth)
        check(f"{name}_velocity_metrics_cpu", metrics_agree(recomputed, saved_metrics, METRIC_KEYS), recomputed=recomputed)
    initial_entry = next((row for row in full if row.get("evaluated_updates") == resumed), {})
    check("run_start_metrics_full_history", metrics_agree(metrics["initial"], initial_entry, (*METRIC_KEYS, *LOSS_KEYS)))

    cls = cpu_model_class()
    model_config = config["model"]
    model = cls(np.array(background), config["data"]["grid_spacing_m"], model_config["neurons"],
                model_config["omega_0"], model_config["mean_kmps"], model_config["std_kmps"], config["parameterization"])
    zero_initial = model.net.linear[-1].weight.detach().count_nonzero().item() == 0 and model.net.linear[-1].bias.detach().count_nonzero().item() == 0
    check("parameterization_fresh_final_layer_zero", zero_initial)
    check("checkpoint_model_tensors_finite_cpu", all(tensor.device.type == "cpu" and torch.isfinite(tensor).all().item()
          for tensor in checkpoint["model"].values()))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    check("checkpoint_fixed_initializer_unchanged", not model.fixed_init_mps.requires_grad
          and bitwise_equal(model.fixed_init_mps.detach().numpy()[0], background))
    x, z = np.meshgrid(np.arange(288) * 15. / 1000., np.arange(94) * 15. / 1000.)
    expected_coords = np.stack((x, z), axis=-1).astype(np.float32)[None]
    check("checkpoint_coordinate_order_units", bitwise_equal(model.coords.detach().numpy(), expected_coords))
    with torch.no_grad():
        predicted = model().numpy()[0]
    difference = np.abs(predicted.astype(np.float64) - np.asarray(final, dtype=np.float64))
    check("checkpoint_final_model_cpu_matches_saved_array", np.allclose(predicted, final, atol=args.velocity_atol,
          rtol=args.velocity_rtol, equal_nan=False), exact_values=bitwise_equal(predicted, final),
          max_abs_difference_mps=float(difference.max()), atol_mps=args.velocity_atol, rtol=args.velocity_rtol)
    states = list(checkpoint["optimizer"]["state"].values())
    optimizer_ok = bool(states)
    for state in states:
        optimizer_ok = optimizer_ok and float(state["step"]) == completed
        for key in ("exp_avg", "exp_avg_sq"):
            optimizer_ok = optimizer_ok and bool(torch.isfinite(state[key]).all())
        optimizer_ok = optimizer_ok and bool((state["exp_avg_sq"] >= 0).all())
    check("adam_checkpoint_steps_finite_moments", optimizer_ok, state_count=len(states),
          expected_parameter_state_count=len(list(model.parameters())))
    check("adam_parameter_state_count", len(states) == len(list(model.parameters())))

    manifest = read_json(ROOT / "source_manifest.json")
    current_hashes = {name: sha256(ROOT / name) for name in manifest}
    check("current_source_manifest", current_hashes == manifest,
          mismatched_files=[name for name in manifest if current_hashes[name] != manifest[name]])
    original_hashes = {name: current_hashes[name] for name in SOURCE_FILES}
    run_hashes = read_json(out / "original_source_hashes.json")
    check("run_original_source_hashes", run_hashes == original_hashes == checkpoint["original_source_hashes"])
    check("checkpoint_initializer_provenance", checkpoint["provenance"] == read_json(out / "initial_model_provenance.json"))
    experiment_manifest = out / "experiment_source_hashes.json"
    if experiment_manifest.is_file():
        saved_experiment_hashes = read_json(experiment_manifest)
        experiment_hashes = {name: sha256(ROOT / name) for name in EXPERIMENT_FILES}
        check("optional_experiment_source_hashes", saved_experiment_hashes == experiment_hashes,
              mismatched_files=[name for name in EXPERIMENT_FILES
                                if saved_experiment_hashes.get(name) != experiment_hashes[name]])
        report["details"]["experiment_source_identity"] = "present and compared with current files"
    else:
        report["details"]["experiment_source_identity"] = "manifest absent; runner identity not verified"

    if args.reference_dir:
        ref = args.reference_dir.resolve()
        ref_config = read_json(ref / "config.json")
        ref_acquisition = read_json(ref / "acquisition.json")
        ref_hashes = read_json(ref / "original_source_hashes.json")
        ref_checkpoint = torch.load(ref / "checkpoint.pt", map_location="cpu", weights_only=False)
        ref_truth = np.load(ref / "true_velocity.npy", mmap_mode="r", allow_pickle=False)
        ref_background = np.load(ref / "initial_velocity.npy", mmap_mode="r", allow_pickle=False)
        ref_observed = np.load(ref / "observed.npy", mmap_mode="r", allow_pickle=False)
        check("reference_original_source_hashes", ref_hashes == original_hashes == ref_checkpoint["original_source_hashes"])
        check("reference_initializer_provenance", ref_checkpoint["provenance"] == read_json(ref / "initial_model_provenance.json")
              == checkpoint["provenance"])
        reference_experiment_manifest = ref / "experiment_source_hashes.json"
        if reference_experiment_manifest.is_file():
            reference_experiment_hashes = read_json(reference_experiment_manifest)
            reference_identity = {name: sha256(ROOT / name) for name in reference_experiment_hashes}
            report["details"]["reference_experiment_source_identity"] = {
                "matches_current": reference_experiment_hashes == reference_identity,
                "mismatched_files": [name for name in reference_experiment_hashes
                                     if reference_experiment_hashes[name] != reference_identity[name]],
                "note": "The historical runner may intentionally differ; original solver hash identity is checked above."}
        check("reference_is_completed_old400", ref_checkpoint["completed_updates"] == 400
              and read_json(ref / "metrics.json")["completed_updates"] == 400
              and ref_config["training"]["epochs"] == 400 and ref_config["acquisition"]["num_shots"] == 13
              and ref_config["forward"]["nt"] == 1000)
        check("reference_truth_bitwise_equal", bitwise_equal(truth, ref_truth))
        check("reference_fixed_initializer_bitwise_equal", bitwise_equal(background, ref_background))
        shared_settings = (config["data"] == ref_config["data"] and config["initialization"] == ref_config["initialization"]
                           and config["model"] == ref_config["model"]
                           and all(config["forward"][key] == ref_config["forward"][key] for key in config["forward"] if key != "nt")
                           and acquisition["receiver_x_indices"] == ref_acquisition["receiver_x_indices"]
                           and acquisition["source_z_index"] == ref_acquisition["source_z_index"]
                           and acquisition["receiver_z_index"] == ref_acquisition["receiver_z_index"])
        check("reference_common_acquisition_forward_settings", shared_settings)
        comparison = compare_observed(observed, acquisition, ref_observed, ref_acquisition,
                                      args.observation_atol, args.observation_rtol)
        check("reference_shared13_first1000_observations", comparison["allclose"] and comparison["dtype_equal"], **comparison)

    report["passed"] = all(item["passed"] for item in report["checks"])
    report["failed_checks"] = [item["name"] for item in report["checks"] if not item["passed"]]
    report["completed_updates"] = completed
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed stochastic run directory")
    parser.add_argument("--reference-dir", type=Path, help="Original completed 13-shot, 1000-sample, 400-update run")
    parser.add_argument("--output", type=Path, help="Create this JSON report exclusively; refuse an existing path")
    parser.add_argument("--velocity-atol", type=float, default=1e-2, help="CPU/saved velocity absolute tolerance in m/s")
    parser.add_argument("--velocity-rtol", type=float, default=1e-6)
    parser.add_argument("--observation-atol", type=float, default=1e-5)
    parser.add_argument("--observation-rtol", type=float, default=1e-5)
    args = parser.parse_args(argv)
    for name in ("velocity_atol", "velocity_rtol", "observation_atol", "observation_rtol"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("--output must be a new file in an existing directory")
    torch.set_num_threads(1)
    try:
        report = audit(args)
    except Exception as error:
        report = {"schema": "residual_ifwi_stochastic_cpu_audit_v1", "passed": False,
                  "run_dir": str(args.run_dir.resolve()), "device": "cpu",
                  "error": f"{type(error).__name__}: {error}"}
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    sys.stdout.write(serialized)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

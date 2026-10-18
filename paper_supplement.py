"""Specs, data preparation and screening selection for the paper's supplementary experiments.

Design: docs/paper_supplement.md (from the 2026-10-17 supplementary experiment design, items A-K).
Every run is one core_runner.py spec; queues list specs in run order; paper_queue.py runs them.

    python paper_supplement.py prepare-data                      # Overthrust CSV, hash-checked
    python paper_supplement.py specs                             # frozen specs + queues
    python paper_supplement.py select --name C_grid_lr --runs RUN_DIRS...   # after a screening queue
    python paper_supplement.py specs --selection experiments/paper_supplement/selection.json

Base configurations equal the paper's runs (E2/E3/E4 core and robustness batches, E2-MS);
tests check that a supplement spec differs from its base only in the intended setting.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "paper_supplement"
SEEDS = (0, 1, 3)
OVERTHRUST_CSV = "overthrust_94x401_mps.csv"
OVERTHRUST_CSV_SHA256 = "244891a5ebe4e40c4ed62f0a5574067b481c7e083dcec50f8481bdde4b4c3f49"

FOURIER = {"high_freq_num": 128, "low_freq_num": 128, "phi_num": 32, "alpha": 0.01, "lambda_init": "fr_inr"}
MULTISCALE = [{"max_hz": 3.0, "updates": 100}, {"max_hz": 5.0, "updates": 100}, {"max_hz": None}]
MARMOUSI = {
    "experiment_name": "E2", "parameterization": "residual", "seed": 3,
    "data": {"model_file": "vel_marmousi_376x1151.csv", "model_units": "m/s", "csv_header": "legacy",
             "downsample": 4, "crop_shape": None, "grid_spacing_m": 15.0},
    "initialization": {"sigma": 15.0, "gaussian_mode": "reflect", "gaussian_truncate": 4.0},
    "model": {"neurons": [2, 128, 128, 128, 128, 1], "omega_0": 30.0, "mean_kmps": 3.0, "std_kmps": 1.0},
    "acquisition": {"num_shots": 13, "source_x_indices": None, "source_start_index": 20, "source_spacing_index": 20,
                    "source_end_margin": 10, "source_depth_index": 1, "receiver_depth_index": 2, "receiver_stride": 1},
    "forward": {"dt_s": 0.0019, "nt": 1000, "frequency_hz": 8.0, "npad": 15, "order": 2, "free_surface": True,
                "pml_reflection": 1e-06},
    "training": {"epochs": 1000, "learning_rate": 0.0001, "checkpoint_interval": 50, "shot_batch_size": None},
}
OVERTHRUST = {
    **copy.deepcopy(MARMOUSI),
    "data": {"model_file": OVERTHRUST_CSV, "model_units": "m/s", "csv_header": "none", "downsample": 1,
             "crop_shape": None, "grid_spacing_m": 20.0},
    "acquisition": {"num_shots": 10, "source_x_indices": list(range(20, 381, 40)), "source_start_index": 20,
                    "source_spacing_index": 40, "source_end_margin": 10, "source_depth_index": 1,
                    "receiver_depth_index": 0, "receiver_stride": 1},
    "forward": {"dt_s": 0.002, "nt": 1500, "frequency_hz": 8.0, "npad": 15, "order": 2, "free_surface": True,
                "pml_reflection": 1e-06},
    "training": {"epochs": 1000, "learning_rate": 0.0001, "checkpoint_interval": 50, "shot_batch_size": 5},
}
CONDITIONS = {  # name -> (base config, robustness block); conditions share observations by name
    "clean": (MARMOUSI, {"condition": "marmousi_clean", "noise_sigma": 0.0, "noise_seed": 3}),
    "sigma2": (MARMOUSI, {"condition": "marmousi_noise_sigma2", "noise_sigma": 2.0, "noise_seed": 3}),
    "sigma4": (MARMOUSI, {"condition": "marmousi_noise_sigma4", "noise_sigma": 4.0, "noise_seed": 3}),
    "overthrust": (OVERTHRUST, {"condition": "overthrust", "noise_sigma": 0.0, "noise_seed": 3}),
}
ZERO, SMALL5 = {"kind": "zero_output"}, {"kind": "small_output", "rms_mps": 5.0}
PREFIT = {"kind": "prefit_background", "updates": 5000, "learning_rate": 0.0001, "selection": "best_background_fit"}
LR_SCHEDULE = {"warmup_updates": 20, "decay_start": 201, "decay_updates": 800, "final_factor": 0.1}
WARMUP_ONLY = {"warmup_updates": 20, "decay_start": 1, "decay_updates": 1, "final_factor": 1.0}
LINEAR_BACKGROUND = {"kind": "linear_depth", "top_mps": 1500.0, "gradient_per_s": 2.0}


def experiment(method, condition="clean", seed=3, name=None):
    """Config for E2 / E2-MS / E3 / E4 / direct / direct+Fourier under a condition."""
    base, _ = CONDITIONS[condition]
    cfg = copy.deepcopy(base)
    if method in ("E3", "E4", "directFR"):
        cfg["model"]["fourier"] = copy.deepcopy(FOURIER)
    if method == "E4":
        cfg["model"]["trainable_omega"] = {"learning_rate": 0.01}
    if method == "E2MS":
        cfg["training"]["frequency_schedule"] = copy.deepcopy(MULTISCALE)
    if method in ("direct", "directFR"):
        cfg["parameterization"] = "absolute"
    cfg["seed"] = seed
    cfg["experiment_name"] = name or f"{method}_{condition}_s{seed}"
    return cfg


def spec(name, cfg, control=ZERO, condition="clean", model=None, background=None, protocol="paper_supplement_v1",
         speedup=None):
    cfg = copy.deepcopy(cfg)
    cfg["experiment_name"] = name
    result = {"name": name, "experiment": cfg, "control": copy.deepcopy(control),
              "robustness": copy.deepcopy(CONDITIONS[condition][1]), "protocol": protocol}
    if model is not None:
        result["model"] = copy.deepcopy(model)
    if speedup is not None:
        result["speedup"] = copy.deepcopy(speedup)
    if background is not None:
        result["background"] = copy.deepcopy(background)
        result["robustness"]["condition"] += "_lineardepth"   # observations are shared per background
    return result


def short(cfg, updates=200):
    cfg = copy.deepcopy(cfg)
    cfg["training"]["epochs"] = updates
    return cfg


def fmt(value):
    """File-safe value: 3e-05 -> 3em05, 1e-04 -> 1em04, 10 -> 10, 2.5 -> 2p5."""
    text = f"{value:.0e}" if value < .01 else f"{value:g}"
    return text.replace(".", "p").replace("-", "m").replace("+", "")


def build_specs(selection=None):
    """{queue name: [spec, ...]} in run order (docs/paper_supplement.md, section 3)."""
    selection = selection or {}
    q = {}
    # F: zero-output E2, first five updates (layer changes and dv_k are logged for updates 1-5).
    first5 = experiment("E2", name="F_E2_first5_s3")
    first5["training"].update(epochs=5, checkpoint_interval=5)
    q["q1_quick_F"] = [spec("F_E2_first5_s3", first5)]
    # Screening (200 updates, waveform MSE decides): C grid lr, D E2/E3 lr (+ E2 residual scale), H scale.
    screen = []
    for lr in (3.0, 10.0, 30.0):
        cfg = short(experiment("E2"))
        cfg["training"]["learning_rate"] = lr
        screen.append(spec(f"C_screen_grid_lr{fmt(lr)}", cfg, model={"kind": "grid"}))
    for method in ("E2", "E3"):
        for lr in (3e-5, 1e-4, 3e-4):
            cfg = short(experiment(method))
            cfg["training"]["learning_rate"] = lr
            screen.append(spec(f"D_screen_{method}_lr{fmt(lr)}", cfg))
    cfg = short(experiment("E2"))
    cfg["model"]["std_kmps"] = 0.3                       # residual scale s_v = 1000 * std_kmps = 300 m/s
    screen.append(spec("D_screen_E2_sv300_lr1em04", cfg))
    for scale in (1.0, 3.0, 10.0):
        screen.append(spec(f"H_screen_ff_scale{fmt(scale)}", short(experiment("E2")),
                           model={"kind": "fourier_features", "scale_cpkm": scale}))
    q["q2_screen_CDH"] = screen
    # B: noisy E2-MS (sets the noise targets); E: direct + Fourier with background pretraining.
    q["q3_B_E"] = [spec(f"B_E2MS_{c}_s3", experiment("E2MS", c), condition=c) for c in ("sigma2", "sigma4")]
    q["q3_B_E"].append(spec("E_directFR_prefit_s3", experiment("directFR"), control=PREFIT))
    # A: same-queue seeds. Minimal: E2-MS, E3 (0 out), E4 (5 m/s); full adds E4 (0 out), E3 (5 m/s), E2.
    minimal, extra = [], []
    for seed in SEEDS:
        minimal += [spec(f"A_E2MS_s{seed}", experiment("E2MS", seed=seed)),
                    spec(f"A_E3_s{seed}", experiment("E3", seed=seed)),
                    spec(f"A_E4small5_s{seed}", experiment("E4", seed=seed), control=SMALL5)]
        extra += [spec(f"A_E4_s{seed}", experiment("E4", seed=seed)),
                  spec(f"A_E3small5_s{seed}", experiment("E3", seed=seed), control=SMALL5),
                  spec(f"A_E2_s{seed}", experiment("E2", seed=seed))]
    direct_seeds = [spec(f"A_directFR_prefit_s{seed}", experiment("directFR", seed=seed), control=PREFIT) for seed in (0, 1)]
    q["q4_A_minimal"] = minimal + direct_seeds
    q["q4_A_full_extra"] = extra
    # G: what drives the 114.96 m/s first step of the pretrained direct network.
    g1 = experiment("direct")
    g1["training"]["lr_schedule"] = copy.deepcopy(WARMUP_ONLY)
    g3 = experiment("direct")
    g3["training"]["learning_rate"] = 1e-5
    g1fr = experiment("directFR")
    g1fr["training"]["lr_schedule"] = copy.deepcopy(WARMUP_ONLY)
    q["q6_G"] = [spec("G1_direct_warmup20_s3", g1, control=PREFIT),
                 spec("G2_direct_keepadam_s3", experiment("direct"), control={**PREFIT, "keep_optimizer_state": True}),
                 spec("G3_direct_lr1em05_s3", g3, control=PREFIT),
                 spec("G1F_directFR_warmup20_s3", g1fr, control=PREFIT)]
    # J: late loss spike of sigma 4 E4 under a learning-rate schedule or beta2 = 0.99.
    j1 = experiment("E4", "sigma4")
    j1["training"]["lr_schedule"] = copy.deepcopy(LR_SCHEDULE)
    j2 = experiment("E4", "sigma4")
    j2["training"]["optimizer"] = {"name": "adam", "betas": [0.9, 0.99]}
    q["q6_J"] = [spec("J_E4_sigma4_lrsched_s3", j1, condition="sigma4"),
                 spec("J_E4_sigma4_beta2_s3", j2, condition="sigma4")]
    # K (optional): depth-only linear background, no truth involved.
    q["q7_K"] = [spec(f"K_{m}_lineardepth_s3", experiment(m), background=LINEAR_BACKGROUND) for m in ("E2MS", "E3", "E4")]
    # Speed: source encoding (one simultaneous-source simulation per update) and coarse low-pass stages.
    encoding = {"supershots": 1, "seed": 3}
    coarse = {"factor": 2, "source_depth_index": 1, "receiver_depth_index": 1}
    x1 = experiment("E3")
    x1["training"]["epochs"] = 3000
    long_ms = experiment("E2MS")
    long_ms["training"]["epochs"] = 3000            # same 100/100 low-pass stages; without lr decay the
    q["q8_speed"] = [                                # budget only sets where the trajectory stops
        spec("A_E3_s3", experiment("E3")),            # same-queue baselines for timing
        spec("A_E2MS_s3", experiment("E2MS")),
        spec("S1_E3_encoded_s3", x1, speedup={"source_encoding": encoding}),
        spec("S2_E2MS_coarse_s3", experiment("E2MS"), speedup={"coarse_lowpass": coarse}),
        spec("S4_E2MS_encoded_s3", long_ms, speedup={"source_encoding": encoding}),
        spec("S3_E2MS_coarse_encoded_s3", long_ms, speedup={"coarse_lowpass": coarse, "source_encoding": encoding})]
    # Formal runs that depend on a screening selection.
    if "C_grid_lr" in selection:
        lr, formal = float(selection["C_grid_lr"]), []
        for condition in ("clean", "sigma2", "sigma4", "overthrust"):
            for method in ("E2", "E2MS"):
                cfg = experiment(method, condition)
                cfg["training"]["learning_rate"] = lr
                tag = "MS" if method == "E2MS" else "full"
                formal.append(spec(f"C_grid{tag}_{condition}_lr{fmt(lr)}_s3", cfg, condition=condition, model={"kind": "grid"}))
        q["q5_C_formal"] = formal
    formal = []
    for method in ("E2", "E3"):
        key = f"D_{method}_lr"
        if key in selection:
            cfg = experiment(method)
            cfg["training"]["learning_rate"] = float(selection[key])
            formal.append(spec(f"D_{method}_lr{fmt(float(selection[key]))}_s3", cfg))
    if "H_scale" in selection:
        scale = float(selection["H_scale"])
        formal.append(spec(f"H_ff_scale{fmt(scale)}_s3", experiment("E2"), model={"kind": "fourier_features", "scale_cpkm": scale}))
    if formal:
        q["q5_DH_formal"] = formal
    return q


def write_specs(selection=None, out=OUT):
    queues = build_specs(selection)
    (out / "specs").mkdir(parents=True, exist_ok=True)
    (out / "queues").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    for queue, specs in queues.items():
        paths = []
        for item in specs:
            path = out / "specs" / f"{item['name']}.json"
            path.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            paths.append(f"specs/{item['name']}.json")
        (out / "queues" / f"{queue}.json").write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    # I: configs for the gradient report at v0 (Marmousi as the reference, Overthrust as the question).
    for name, cfg in (("I_marmousi", experiment("E2", name="I_marmousi")), ("I_overthrust", experiment("E2", "overthrust", name="I_overthrust"))):
        (out / "configs" / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return queues


def prepare_data(data_dir=ROOT / "data"):
    """data/overthrust_94x401_mps.csv exactly as the robustness batch wrote it (vp km/s x 1000, float32)."""
    vp = np.load(ROOT / "data" / "overthrust2d.npz", allow_pickle=False)["vp"]
    values = (vp.astype(np.float32) * np.float32(1000)).astype(np.float32)
    path = Path(data_dir) / OVERTHRUST_CSV
    with path.open("w", encoding="utf-8", newline="") as stream:
        np.savetxt(stream, values, delimiter=",", fmt="%.9g", newline="\r\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OVERTHRUST_CSV_SHA256:
        raise ValueError(f"{path} sha256 {digest} differs from the robustness batch ({OVERTHRUST_CSV_SHA256})")
    back = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if not np.array_equal(back, values):
        raise ValueError("Overthrust CSV does not round-trip float32")
    return path, digest


def select(run_dirs, name, selection_file):
    """Lowest all-shot full-band waveform MSE at the last evaluation (no truth); records every candidate."""
    candidates = []
    for directory in run_dirs:
        directory = Path(directory)
        run_spec = json.loads((directory / "run_spec.json").read_text(encoding="utf-8"))
        final = json.loads((directory / "evaluation_history.json").read_text(encoding="utf-8"))[-1]
        if final["evaluated_updates"] != run_spec["experiment"]["training"]["epochs"]:
            raise ValueError(f"{directory} has not finished its screening budget")
        cfg, model = run_spec["experiment"], run_spec.get("model", {})
        value = model["scale_cpkm"] if model.get("kind") == "fourier_features" else cfg["training"]["learning_rate"]
        candidates.append({"run": str(directory), "name": run_spec["name"], "value": value,
                           "data_mse": final["data_mse"], "evaluated_updates": final["evaluated_updates"]})
    if len({c["value"] for c in candidates}) != len(candidates):
        raise ValueError("Screening runs must differ in the selected setting")
    best = min(candidates, key=lambda c: c["data_mse"])
    path = Path(selection_file)
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record[name] = best["value"]
    record.setdefault("_candidates", {})[name] = {"rule": "lowest all-shot full-band waveform MSE at the screening budget",
                                                  "candidates": candidates}
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data")
    specs = sub.add_parser("specs")
    specs.add_argument("--selection", type=Path)
    chooser = sub.add_parser("select")
    chooser.add_argument("--name", required=True, choices=("C_grid_lr", "D_E2_lr", "D_E3_lr", "H_scale"))
    chooser.add_argument("--runs", nargs="+", required=True)
    chooser.add_argument("--file", type=Path, default=OUT / "selection.json")
    args = parser.parse_args()
    if args.command == "prepare-data":
        path, digest = prepare_data()
        print(f"Wrote {path} (sha256 {digest})")
    elif args.command == "specs":
        selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else None
        for queue, items in write_specs(selection).items():
            print(f"{queue}: {len(items)} runs")
    else:
        best = select(args.runs, args.name, args.file)
        print(f"{args.name} = {best['value']} ({best['name']}, waveform MSE {best['data_mse']:.7g}); wrote {args.file}")


if __name__ == "__main__":
    main()

"""P0 candidate backgrounds (design doc 7.1): controlled smoothings and perturbations of the truth plus
one truth-free linear background. They test whether a data-only score predicts IFWI success; they are
not a method. Each candidate is a float32 [nz, nx] .npy with its array hash in manifest.json.

    python -m genbg.candidates --template experiments/paper_supplement/specs/B_E2_smooth400_s3.json \
        --out experiments/genbg/candidates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from core_models import linear_depth_background, smooth400_background
from robust_observations import array_hash
from genbg.common import MANIFEST, ROOT, load_template, truth_and_default

GAUSS = (10, 15, 22, 30)                 # velocity smoothing sigma, grid points
SLOW = (6, 9, 14)                        # slowness smoothing sigma, grid points
PERTURB_LENGTHS_M = (600, 1500)
PERTURB_LEVELS = (0.03, 0.06)
PERTURB_SEEDS = (1, 2)
SCALES = (-0.05, 0.05)
LINEAR = {"top_mps": 1500., "gradient_per_s": 2.0}   # the q7_K linear background


def smooth(values, sigma):
    return gaussian_filter(np.asarray(values, dtype=np.float64), sigma=sigma, mode="reflect", truncate=4.)


FIELD_CLIP = 2.5


def random_field(shape, sigma_points, seed):
    """Zero-mean, unit-RMS Gaussian random field (smoothed white noise, reflect boundaries), clipped at
    +-FIELD_CLIP so a short correlation length cannot leave isolated velocity spikes."""
    field = smooth(np.random.RandomState(seed).randn(*shape), sigma_points)
    field -= field.mean()
    return np.clip(field / np.sqrt(np.mean(field ** 2)), -FIELD_CLIP, FIELD_CLIP)


def build(spec):
    cfg = spec["experiment"]
    truth, default, _ = truth_and_default(spec)
    dx = cfg["data"]["grid_spacing_m"]
    if cfg["initialization"]["sigma"] != 15. or cfg["initialization"].get("gaussian_mode", "reflect") != "reflect":
        raise ValueError("template must use the paper's sigma-15 reflect background")
    out = []

    def add(name, family, values, truth_free, **params):
        out.append({"id": name, "family": family, "params": params, "truth_free": truth_free,
                    "array": np.asarray(values, dtype=np.float32)})

    for sigma in GAUSS:
        # The loader's exact recipe (float32 input), so G15 is bit-identical to the runner's background.
        add(f"G{sigma}", "G", gaussian_filter(truth, sigma=sigma, mode="reflect", truncate=4.), False,
            velocity_sigma_points=sigma)
    g15 = out[GAUSS.index(15)]["array"]
    if array_hash(g15) != array_hash(default):
        raise ValueError("G15 differs from the runner's default background")
    for sigma in SLOW:
        add(f"S{sigma}", "S", 1. / smooth(1. / truth.astype(np.float64), sigma), False, slowness_sigma_points=sigma)
    add("SM400", "SM", smooth400_background(ROOT / "data", truth.shape), False,
        note="smooth400 background; recipe unknown")
    add("LIN", "L", linear_depth_background(truth.shape, dx, **LINEAR), True, **LINEAR)
    base_slowness = 1. / g15.astype(np.float64)
    for length in PERTURB_LENGTHS_M:
        for level in PERTURB_LEVELS:
            for seed in PERTURB_SEEDS:
                field = random_field(truth.shape, length / dx, 1000 * length + seed)
                add(f"P{length}_{round(level * 100)}_{seed}", "P", 1. / (base_slowness * (1. + level * field)), False,
                    correlation_length_m=length, slowness_rms_fraction=level, seed=seed, field_clip=FIELD_CLIP, base="G15")
    for scale in SCALES:
        add(("Kp" if scale > 0 else "Km") + str(round(abs(scale) * 100)), "K", g15.astype(np.float64) * (1. + scale),
            False, velocity_scale=1. + scale, base="G15")
    vmax_cfl = (1 / np.sqrt(2)) * dx / cfg["forward"]["dt_s"]
    for entry in out:
        values = entry["array"]
        if not np.isfinite(values).all() or values.min() <= 0 or values.max() >= vmax_cfl:
            raise ValueError(f"candidate {entry['id']} is non-physical or violates CFL")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = load_template(args.template)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for entry in build(spec):
        path = args.out / f"{entry['id']}.npy"
        np.save(path, entry["array"])
        manifest.append({"id": entry["id"], "family": entry["family"], "params": entry["params"],
                         "truth_free": entry["truth_free"], "file": path.resolve().relative_to(ROOT).as_posix(),
                         "array_hash": array_hash(entry["array"]), "shape": list(entry["array"].shape),
                         "min_mps": float(entry["array"].min()), "max_mps": float(entry["array"].max())})
        print(f"{entry['id']:>12}  {entry['array'].min():7.1f}-{entry['array'].max():7.1f} m/s")
    (args.out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(manifest)} candidates -> {args.out / MANIFEST}")


if __name__ == "__main__":
    main()

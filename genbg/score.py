"""Score every candidate background with one forward simulation of all shots (no adjoint).

Truth enters only through the observations (as in core_runner) and the oracle descriptors, which are
written separately and never used by a score.

    python -m genbg.score --candidates experiments/genbg/candidates \
        --template experiments/paper_supplement/specs/B_E2_smooth400_s3.json --device cuda:0 --out outputs/genbg/p0_scores
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments import residual_ifwi_experiment as e
from experiments.residual_ifwi_report import velocity_metrics
from genbg.common import load_template, read_manifest, truth_and_default, twt_bottom_ms
from genbg.picking import first_breaks, offsets_m, pick_scores, waveform_scores

TRUTH_FREE = ("fb_rms_ms", "cycle_skip_frac", "lp3_nmse", "lp5_nmse", "full_nmse")
ORACLE = ("rmse_mps", "ssim", "twt_rms_ms", "twt_max_ms")


def sync(device):
    if str(device).startswith("cuda"): torch.cuda.synchronize(device)


@torch.no_grad()
def simulate(problem, velocity, device):
    v = torch.as_tensor(velocity, dtype=torch.float32, device=device)[None]
    e.check_velocity(v, problem["config"])
    return torch.cat([e._simulate(item["solver"], v, problem["wavelet"]) for item in problem["propagators"]], dim=1)


def score_all(spec, entries, device, alpha=.05, min_offset_m=150., max_offset_m=None):
    cfg = spec["experiment"]
    truth, _, _ = truth_and_default(spec)
    problem = e.build_forward_problem(truth, cfg, device)
    observed = problem["observed"]
    dt, dx = cfg["forward"]["dt_s"], cfg["data"]["grid_spacing_m"]
    offsets = offsets_m(problem["geometry"], dx)
    obs_picks = first_breaks(observed[0].cpu().numpy(), dt, alpha)
    twt_true = twt_bottom_ms(truth, dx)
    rows, picks = [], {"observed": obs_picks}
    for entry in entries:
        sync(device); started = time.perf_counter()
        predicted = simulate(problem, entry["array"], device)
        sync(device); seconds = time.perf_counter() - started
        pred_picks = first_breaks(predicted[0].cpu().numpy(), dt, alpha)
        picks[entry["id"]] = pred_picks
        twt_error = twt_bottom_ms(entry["array"], dx) - twt_true
        metrics = velocity_metrics(entry["array"], truth)
        row = {"id": entry["id"], "family": entry["family"], "truth_free_candidate": entry["truth_free"],
               **pick_scores(obs_picks, pred_picks, offsets, cfg["forward"]["frequency_hz"], min_offset_m, max_offset_m),
               **waveform_scores(predicted, observed, dt),
               "rmse_mps": metrics["rmse_mps"], "ssim": metrics["ssim"],
               "twt_rms_ms": float(np.sqrt(np.mean(twt_error ** 2))), "twt_max_ms": float(np.abs(twt_error).max()),
               "forward_seconds": seconds}
        rows.append(row)
        print(f"{row['id']:>12} fb={row['fb_rms_ms']:7.2f} ms skip={row['cycle_skip_frac']:.3f} "
              f"lp3={row['lp3_nmse']:.4f} lp5={row['lp5_nmse']:.4f} full={row['full_nmse']:.4f} | "
              f"RMSE={row['rmse_mps']:.1f} TWT={row['twt_rms_ms']:.1f} ms ({seconds:.2f} s)", flush=True)
    settings = {"alpha": alpha, "min_offset_m": min_offset_m, "max_offset_m": max_offset_m,
                "truth_free_scores": list(TRUTH_FREE), "oracle_descriptors": list(ORACLE),
                "num_shots": problem["geometry"]["num_shots"], "num_receivers": problem["geometry"]["num_receivers"],
                "device": str(device)}
    return rows, picks, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=.05, help="first-break threshold, fraction of the trace maximum")
    parser.add_argument("--min-offset-m", type=float, default=150.)
    parser.add_argument("--max-offset-m", type=float)
    parser.add_argument("--only", nargs="*", help="score only these candidate ids")
    args = parser.parse_args()
    spec = load_template(args.template)
    entries = read_manifest(args.candidates)
    if args.only:
        entries = [x for x in entries if x["id"] in set(args.only)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows, picks, settings = score_all(spec, entries, args.device, args.alpha, args.min_offset_m, args.max_offset_m)
    (args.out / "scores.json").write_text(json.dumps({"settings": settings, "template": str(args.template),
                                                       "rows": rows}, indent=2), encoding="utf-8")
    with (args.out / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.out / "first_breaks.npz", **picks)
    print(f"scores -> {args.out / 'scores.json'}")


if __name__ == "__main__":
    main()

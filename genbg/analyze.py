"""P0 analysis: do the truth-free scores predict how E2 does from each background? (design doc 5, H1)

Outcome per candidate (truth, evaluation only): RMSE at the last update relative to the start,
rel = (RMSE_K - RMSE_0) / RMSE_0; success means RMSE_K <= 0.85 RMSE_0. Lower scores mean better
backgrounds, so a useful score has a positive Spearman rho with rel. Pre-registered pass rule: some
truth-free score has rho >= 0.7 and the candidate it ranks first succeeds.

    python -m genbg.analyze --scores outputs/genbg/p0_scores/scores.json --runs outputs/genbg/p0_runs \
        --out outputs/genbg/p0_report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from genbg.score import ORACLE, TRUTH_FREE

SUCCESS_FRACTION = .85
RHO_PASS = .7


def outcome(run_dir):
    path = Path(run_dir) / "evaluation_history.json"
    if not path.exists():
        return None
    evaluations = json.loads(path.read_text(encoding="utf-8"))
    status = json.loads((Path(run_dir) / "status.json").read_text(encoding="utf-8"))
    first, last = evaluations[0], evaluations[-1]
    rmse = [x["rmse_mps"] for x in evaluations]
    return {"state": status.get("state"), "updates": last["update"], "rmse_start": first["rmse_mps"],
            "rmse_end": last["rmse_mps"], "rmse_min": min(rmse),
            "rel_change": (last["rmse_mps"] - first["rmse_mps"]) / first["rmse_mps"],
            "success": bool(last["rmse_mps"] <= SUCCESS_FRACTION * first["rmse_mps"]),
            "data_mse_ratio": last["data_mse"] / first["data_mse"]}


def correlations(rows, keys):
    out = {}
    for key in keys:
        x = np.array([r[key] for r in rows], dtype=float); y = np.array([r["rel_change"] for r in rows], dtype=float)
        rho, p = spearmanr(x, y) if len(rows) >= 3 else (float("nan"), float("nan"))
        best = rows[int(np.argmin(x))]
        out[key] = {"spearman_rho": float(rho), "p_value": float(p), "ranked_first": best["id"],
                    "ranked_first_success": best["success"], "ranked_first_rmse_end": best["rmse_end"]}
    return out


def figure(rows, keys, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for r in rows:
            ax.scatter(r[key], 100 * r["rel_change"], c="tab:blue" if r["success"] else "tab:red", s=18)
            ax.annotate(r["id"], (r[key], 100 * r["rel_change"]), fontsize=6)
        ax.axhline(100 * (SUCCESS_FRACTION - 1), color="grey", lw=.8, ls="--")
        ax.set_xlabel(key); ax.set_ylabel("RMSE change at last update (%)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="paper_queue --out-root of the P0 queue")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    scores = json.loads(args.scores.read_text(encoding="utf-8"))["rows"]
    rows, missing = [], []
    for row in scores:
        result = outcome(args.runs / f"P0_E2_{row['id']}_s{args.seed}" / "attempt_01")
        if result is None or result["state"] != "completed":
            missing.append(row["id"]); continue
        rows.append({**row, **result})
    if len(rows) < 3:
        raise SystemExit(f"only {len(rows)} completed runs; missing {missing}")
    args.out.mkdir(parents=True, exist_ok=True)
    free, oracle = correlations(rows, TRUTH_FREE), correlations(rows, ORACLE)
    passed = [k for k, v in free.items() if v["spearman_rho"] >= RHO_PASS and v["ranked_first_success"]]
    report = {"completed": len(rows), "missing": missing, "success_rule": f"RMSE_end <= {SUCCESS_FRACTION} RMSE_start",
              "pass_rule": f"truth-free rho >= {RHO_PASS} and the first-ranked candidate succeeds",
              "p0_pass": bool(passed), "passing_scores": passed, "truth_free": free, "oracle": oracle}
    (args.out / "p0_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "p0_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figure(rows, list(TRUTH_FREE) + ["twt_rms_ms"], args.out / "p0_scatter.png")
    print(f"{len(rows)} runs, missing {missing}")
    for name, table in (("truth-free", free), ("oracle", oracle)):
        for key, v in table.items():
            print(f"{name:>10} {key:>16}: rho={v['spearman_rho']:+.3f} p={v['p_value']:.3g} "
                  f"first={v['ranked_first']} ({'success' if v['ranked_first_success'] else 'failure'})")
    print("P0", "PASS" if passed else "NOT PASSED", passed)


if __name__ == "__main__":
    main()

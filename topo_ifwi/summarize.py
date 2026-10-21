"""Markdown tables from run_minimal outputs (for proposal section 8.2).

Usage: .venv/bin/python -m topo_ifwi.summarize outputs/topo_ifwi/dumbbell_base outputs/topo_ifwi/dumbbell_review -- outputs/topo_ifwi/twobody
(directories before "--" share one warm start and are merged into one table)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def load(runs):
    """Merge summaries of runs that share one warm start (e.g. base groups + review groups)."""
    runs = [Path(r) for r in runs]
    s = json.loads((runs[0] / "summary.json").read_text())
    s["review_dirs"] = {g: runs[0] for g in s["results"]}
    for r in runs[1:]:
        t = json.loads((r / "summary.json").read_text())
        if abs(t["phase0"]["B_e"] - s["phase0"]["B_e"]) > 1e-6:
            raise SystemExit(f"{r} does not share the warm start of {runs[0]}")
        s["results"].update(t["results"])
        s["review_dirs"].update({g: r for g in t["results"]})
    return s


def summarize(*runs):
    s = load(runs)
    p0 = s["phase0"]
    out = [f"### {s['model']}", "",
           f"建模误差下限 E_A(v_true) = {s['floor']:.4f}；热启动 {p0['warm_seconds']:.0f} s，"
           f"每个训练步 {p0['sec_per_fb_step']:.2f} s（{s['device']}）。",
           f"阶段 0：located = {p0['located']}，topology_error = {p0['topology_error']}，"
           f"in_regime = {p0['in_regime']}；B_e = {p0['B_e']:.3f}，带内条形 {p0['bars_in_band']} 个，"
           f"热启动 IoU = {p0['metrics']['iou']:.3f}，β@0.5 = {tuple(p0['metrics']['betti_0p5'])}，"
           f"E_train = {p0['metrics']['e_train']:.4f}，E_eval = {p0['metrics']['e_eval']:.4f}。", "",
           "| 组 | IoU | HD95 (m) | 异常区 RMSE | 全域 RMSE | 带内 Betti 误差 | β@0.5 | E_train | E_eval | 训练步当量 | 仅前向 | 墙钟 (s) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|",
           f"| 热启动 | {fmt(p0['metrics']['iou'])} | {fmt(p0['metrics']['hd95_m'], 0)} | "
           f"{fmt(p0['metrics']['rmse_anomaly'], 0)} | {fmt(p0['metrics']['rmse'], 0)} | "
           f"{fmt(p0['metrics']['betti_band_err'], 2)} | {tuple(p0['metrics']['betti_0p5'])} | "
           f"{fmt(p0['metrics']['e_train'], 4)} | {fmt(p0['metrics']['e_eval'], 4)} | — | — | — |"]
    for g, r in s["results"].items():
        out.append(f"| {g} | {fmt(r['iou'])} | {fmt(r['hd95_m'], 0)} | {fmt(r['rmse_anomaly'], 0)} | "
                   f"{fmt(r['rmse'], 0)} | {fmt(r['betti_band_err'], 2)} | {tuple(r['betti_0p5'])} | "
                   f"{fmt(r['e_train'], 4)} | {fmt(r['e_eval'], 4)} | {r['fb_steps']} | {r['fwd_evals']} | "
                   f"{r['seconds']:.0f} |")
    out += ["", "| 组 | 审查次数 | 决定（类别:候选 → 次数） | 锚点 | 事后更接近真值的比例 | 结构候选 χ 中位数 |",
            "|---|---|---|---|---|---|"]
    for g, r in s["results"].items():
        f = s["review_dirs"][g] / f"reviews_{g}.json"
        if not f.exists() or not r["reviews"]:
            continue
        reviews = json.loads(f.read_text())
        chis = [v for rv in reviews for k, v in rv["chi"].items() if "waveform|" not in k or g == "B4"]
        closer = [x for x in r["posthoc_closer"] if x is not None]
        dec = "，".join(f"{k} → {v}" for k, v in sorted(r["decisions"].items()))
        out.append(f"| {g} | {r['reviews']} | {dec} | {r['anchors']} | "
                   f"{(sum(closer) / len(closer)) if closer else float('nan'):.2f} | "
                   f"{np.median(chis) if chis else float('nan'):.2f} |")
    return "\n".join(out)


def review_detail(run, group):
    reviews = json.loads((Path(run) / f"reviews_{group}.json").read_text())
    out = [f"#### {Path(run).name} / {group} 审查明细", "",
           "| 步 | 条形 | B / D | 候选: E（可行, 达标, β@0.5） | χ | 选择 | 类别 |", "|---|---|---|---|---|---|---|"]
    for r in reviews:
        bar = r["bar"]
        cands = "；".join(f"{c['name']}: {c['E']:.4f} ({'Y' if c['feasible'] else 'N'}, "
                         f"{'Y' if c['resolves'] else 'N'}, {tuple(c['betti_0p5'])})" for c in r["candidates"])
        chi = "；".join(f"{k}: {v:.1f}" for k, v in r["chi"].items())
        out.append(f"| {r['step']} | {bar['kind'] if bar else '—'} | "
                   f"{'%.2f / %.2f' % (bar['B'], bar['D']) if bar else '—'} | {cands} | {chi} | {r['chosen']} | {r['decision']} |")
    return "\n".join(out)


if __name__ == "__main__":
    # Arguments: groups of run directories separated by "+", e.g.  base + review  twobody
    groups, cur = [], []
    for a in sys.argv[1:]:
        if a == "--":
            groups.append(cur)
            cur = []
        else:
            cur.append(a)
    groups.append(cur)
    for runs in groups:
        print(summarize(*runs))
        print()
        for run in runs:
            for g in ("B3a", "B3b"):
                if (Path(run) / f"reviews_{g}.json").exists():
                    print(review_detail(run, g))
                    print()

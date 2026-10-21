"""Proposal figures: the threshold-band loss schematic (v0.2 change 13) and pilot result panels.

Usage: .venv/bin/python -m topo_ifwi.figures --out docs/figures/topology_ifwi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from topo_ifwi.persistence import band_loss

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
A, B, EPS = 0.4, 0.6, 0.05


def schematic_field(shape=(48, 80)):
    z, x = np.mgrid[:shape[0], :shape[1]]
    body = (((z - 24) / 9) ** 2 + ((x - 24) / 13) ** 2 <= 1) | (((z - 26) / 8) ** 2 + ((x - 56) / 11) ** 2 <= 1)
    hole = ((z - 24) / 3) ** 2 + ((x - 24) / 4) ** 2 <= 1
    q = np.where(body & ~hole, 0.92, 0.04)
    q[23:27, 37:44] = 0.5                                   # weak neck: the contested saddle
    q[6:9, 70:73] = 0.7                                     # small false island
    return gaussian_filter(q, 1.2)


def band_schematic(path):
    q = schematic_field()
    terms, d = band_loss(torch.tensor(q), A, B, EPS, kappa=1.0)
    pairs, kinds = terms["pairs"], terms["kinds"]
    flat = q.ravel()
    bars = [(flat[b], flat[dd], k) for (b, dd), k in zip(pairs, kinds)]
    h = terms["overlap"].numpy()
    keep = [i for i, (bb, dd, _) in enumerate(bars) if bb - dd > 0.02]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    im = ax[0].imshow(q, cmap="gray_r", vmin=0, vmax=1)
    ax[0].contour(q, levels=[A, B], colors=["tab:blue", "tab:red"], linewidths=0.8)
    ax[0].set_title("软异常分数 q（蓝: a=0.4，红: b=0.6）", fontsize=9)
    fig.colorbar(im, ax=ax[0], shrink=0.7)

    t = np.linspace(0, 1, 2)
    lo, hi = A - EPS, B + EPS
    ax[1].add_patch(plt.Polygon([(0, lo), (lo, lo), (hi, hi), (hi, 1), (0, 1)], color="tab:red", alpha=0.12,
                                label="受罚区: B > a-ε 且 D < b+ε"))
    ax[1].add_patch(plt.Polygon([(0, 0), (lo, lo), (0, lo)], color="tab:green", alpha=0.18, label="允许: 整条位于带下方或上方"))
    ax[1].add_patch(plt.Polygon([(hi, hi), (1, 1), (hi, 1)], color="tab:green", alpha=0.18))
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].plot(t, t, "k-", lw=0.6)
    for i in keep:
        bb, dd, k = bars[i]
        ax[1].plot(dd, bb, "o" if k == 0 else "s", ms=6, mfc="tab:red" if h[i] > 0 else "white",
                   mec="tab:red" if h[i] > 0 else "k")
    ax[1].plot(0, flat[d.essential], "k*", ms=10)
    ax[1].set_xlabel("死亡值 D", fontsize=9)
    ax[1].set_ylabel("出生值 B", fontsize=9)
    ax[1].set_title("超水平集持续图（圆 H0，方 H1，星 本质分支）\n实心红点: 与扩展带重叠、受罚", fontsize=9)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)

    ax[2].axvspan(A - EPS, B + EPS, color="tab:orange", alpha=0.15, label="扩展带 [a-ε, b+ε]")
    ax[2].axvspan(A, B, color="tab:orange", alpha=0.3, label="目标带 [a, b]")
    rows = [("本质 H0（要求 B_e ≥ b+ε）", 0.0, flat[d.essential], 0, 0.0)] + [
        (("H0" if bars[i][2] == 0 else "H1"), bars[i][1], bars[i][0], bars[i][2], h[i]) for i in keep]
    for j, (lab, lo, hi, k, hh) in enumerate(rows):
        ax[2].plot([lo, hi], [j, j], lw=4, color="tab:red" if hh > 0 else "k", solid_capstyle="butt")
        ax[2].text(1.01, j, lab if j == 0 else f"{lab}  h={hh:.2f}", va="center", fontsize=7)
    ax[2].set_xlim(0, 1)
    ax[2].set_yticks([])
    ax[2].set_xlabel("阈值 τ", fontsize=9)
    ax[2].set_title(f"条形码；R_PH = {float(terms['total']):.3f}", fontsize=9)
    ax[2].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/figures/topology_ifwi"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    band_schematic(args.out / "band_loss_schematic.png")
    print(json.dumps({"written": str(args.out / "band_loss_schematic.png")}))


if __name__ == "__main__":
    main()

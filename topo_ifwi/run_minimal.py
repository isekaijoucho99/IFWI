"""Minimal feasibility experiment for the topology-prior IFWI proposal v0.2.

Pipeline (proposal 3.3, v0.2 phases 0/A/B on a small scale):

1. Observed data on a 2x finer grid (no inverse crime), band-limited misfit E_A on 13 training
   shots, 4 held-out evaluation shots.
2. Warm start: plain residual-SIREN IFWI with low-pass stages 5 -> 8 -> 10 Hz.
3. Phase 0 report: is the body located (B_e >= b+eps, IoU) and does the band contain topology errors?
4. Structural stage from the same warm-start state, one run per group:
     B0  waveform only                     B1  + TV
     B2  + lambda_PH R_PH (no review)      B3a candidates + review, prior breaks data ties
     B3b candidates + review, ties keep the waveform update
     B4  waveform-only candidates with the same candidate/evaluation budget as B3
5. Metrics, candidate logs (with post-hoc truth checks that never feed back), timing, figures.

The truth is used only for metrics and post-hoc labels. Every weight is set by a fixed rule
(gradient-norm ratios at stage start), never by the truth.

Usage:  .venv/bin/python -m topo_ifwi.run_minimal --model dumbbell --out outputs/topo_ifwi/dumbbell
        .venv/bin/python -m topo_ifwi.run_minimal --smoke   (tiny budgets, checks the pipeline)
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from topo_ifwi.models import MODELS, ResidualSiren, build, score_parameters
from topo_ifwi.persistence import (anomaly_score, band_terms, betti_by_labels, superlevel_diagram)
from topo_ifwi.wave import Acoustic2D, bandpass

GROUPS = ("B0", "B1", "B2", "B3a", "B3b", "B4")

CONFIG = {
    "grid": {"nz": 96, "nx": 128, "dx": 20.0},
    "time": {"dt": 0.002, "nt": 1500, "f0": 5.0},
    "npml": 20,
    "acquisition": {"train_shots": 13, "eval_shots": 4, "src_iz": 1, "rec_iz": 1, "edge": 4},
    "observed": {"refine": 2, "snr_db": None, "lowcut_hz": 0.0},
    "v_anomaly": 4300.0,
    "network": {"hidden": 128, "depth": 4, "omega": 30.0, "scale": 1000.0, "lr": 2e-4, "seed": 0,
                "taper_rows": [4, 8]},
    "warm": [[5.0, 200], [8.0, 200], [10.0, 200]],
    "structural": {"high_hz": 10.0, "steps": 300, "review_every": 20, "n_inner": 10, "cand_lr_scale": 1.0,
                   "lr_factor": 0.5},
    "score": {"rho": 0.5, "temp_frac": 0.1, "a": 0.4, "b": 0.6, "eps": 0.05, "kappa": 1.0, "fg": 4},
    "weights": {"ph_ratio": 0.5, "tv_ratio": 0.5, "soft_ratio": 2.0},
    "review": {"tau_rel": 0.02, "r_max": 1000.0, "vmin": 1400.0, "vmax": 4800.0, "e_max_factor": 1.5,
               "min_improve_frac": 0.5, "saddle_radius": 2, "b4_lr_factors": [0.5, 1.0, 2.0]},
    "phase0": {"iou_min": 0.3},
}

SMOKE = {"time": {"nt": 600}, "warm": [[5.0, 4], [10.0, 4]],
         "structural": {"steps": 12, "review_every": 6, "n_inner": 2}}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ---------------------------------------------------------------- problem setup

class Problem:
    def __init__(self, cfg, model_name, device):
        g, t, acq = cfg["grid"], cfg["time"], cfg["acquisition"]
        self.cfg, self.device = cfg, torch.device(device)
        nz, nx, dx = g["nz"], g["nx"], g["dx"]
        self.m = build(model_name, nz, nx, dx, cfg["v_anomaly"])
        sc = cfg["score"]
        self.dv, self.temp, self.contrast = score_parameters(self.m["v_ref"], self.m["mask"], cfg["v_anomaly"],
                                                             sc["rho"], sc["temp_frac"])
        e = acq["edge"]
        train_x = np.linspace(e, nx - 1 - e, acq["train_shots"]).round().astype(int)
        mids = (train_x[:-1] + train_x[1:]) // 2
        eval_x = mids[np.linspace(0, len(mids) - 1, acq["eval_shots"]).round().astype(int)]
        src = [(acq["src_iz"], x) for x in np.concatenate([train_x, eval_x])]
        rec = [(acq["rec_iz"], x) for x in range(e, nx - e)]
        self.train = np.arange(len(train_x))
        self.eval = np.arange(len(train_x), len(src))
        self.solver = Acoustic2D(nz, nx, dx, t["dt"], t["nt"], src, rec, t["f0"], cfg["npml"], device)
        self.v_ref = torch.tensor(self.m["v_ref"], device=self.device)
        self.v_true = torch.tensor(self.m["v_true"], device=self.device)
        self.q_true = self.score(self.v_true).cpu().numpy()
        self.obs = self._observed(src, rec)
        self._filtered = {}

    def _observed(self, src, rec):
        cfg, r = self.cfg, self.cfg["observed"]["refine"]
        g, t = cfg["grid"], cfg["time"]
        v_fine = np.repeat(np.repeat(self.m["v_true"], r, 0), r, 1)
        fine = Acoustic2D(g["nz"] * r, g["nx"] * r, g["dx"] / r, t["dt"] / r, t["nt"] * r,
                          np.asarray(src) * r, np.asarray(rec) * r, t["f0"], cfg["npml"] * r, self.device)
        with torch.no_grad():
            d = fine.forward(torch.tensor(v_fine, device=self.device))[:, ::r]
        snr = cfg["observed"]["snr_db"]
        if snr is not None:
            gen = torch.Generator().manual_seed(1234)
            noise = torch.randn(d.shape, generator=gen).to(self.device)
            d = d + noise * d.square().mean().sqrt() * 10 ** (-snr / 20)
        return d

    def score(self, v):
        return anomaly_score(v, self.v_ref, self.dv, self.temp)

    def _obs_band(self, shots_key, high):
        key = (shots_key, high)
        if key not in self._filtered:
            idx = self.train if shots_key == "train" else self.eval
            d = bandpass(self.obs[idx], self.cfg["time"]["dt"], self.cfg["observed"]["lowcut_hz"], high)
            self._filtered[key] = (d, d.square().sum())
        return self._filtered[key]

    def predict(self, v, high, shots_key="train"):
        idx = self.train if shots_key == "train" else self.eval
        return bandpass(self.solver.forward(v, idx), self.cfg["time"]["dt"], self.cfg["observed"]["lowcut_hz"], high)

    def misfit(self, v, high, shots_key="train", pred=None):
        """E_A: band-limited residual energy normalised by the observed energy (proposal 4.1)."""
        d, norm = self._obs_band(shots_key, high)
        pred = self.predict(v, high, shots_key) if pred is None else pred
        return (pred - d).square().sum() / norm


# ---------------------------------------------------------------- training state

@dataclass
class State:
    model: ResidualSiren
    opt: torch.optim.Adam
    lr: float
    steps: int = 0
    fb: int = 0                       # forward+backward passes on the training shots
    fwd: int = 0                      # forward-only evaluations on the training shots
    history: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    reviews: list = field(default_factory=list)
    seconds: float = 0.0

    def clone(self, lr=None):
        model = copy.deepcopy(self.model)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        opt.load_state_dict(copy.deepcopy(self.opt.state_dict()))
        for group in opt.param_groups:
            group["lr"] = self.lr if lr is None else lr
        return State(model, opt, self.lr, self.steps, self.fb, self.fwd, list(self.history),
                     list(self.anchors), list(self.reviews), self.seconds)


def soft_target(q, mask, direction, target):
    """One-sided weighted quadratic on a region (proposal 5.2)."""
    sel = q[mask]
    gap = torch.relu(target - sel) if direction == "raise" else torch.relu(sel - target)
    return gap.square().mean()


def anchor_loss(P, v, anchors):
    if not anchors:
        return v.new_zeros(())
    q = P.score(v)
    return sum(w * soft_target(q, m, d, t) for m, d, t, w in anchors)


def train(P, st, n, high, extra=None):
    for _ in range(n):
        t0 = time.time()
        st.opt.zero_grad(set_to_none=True)
        v = st.model()
        e = P.misfit(v, high)
        loss = e + anchor_loss(P, v, st.anchors)
        if extra is not None:
            loss = loss + extra(v)
        loss.backward()
        st.opt.step()
        st.steps += 1
        st.fb += 1
        st.history.append(float(e.detach()))
        st.seconds += time.time() - t0


def grad_norm(model, loss):
    grads = torch.autograd.grad(loss, [p for p in model.parameters()], allow_unused=True, retain_graph=True)
    return float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))


def ratio_weight(P, st, high, extra_fn, ratio):
    """lambda = ratio * |grad E_A| / |grad extra| at the current state (fixed rule, no truth)."""
    v = st.model()
    g_e = grad_norm(st.model, P.misfit(v, high))
    x = extra_fn(v)
    g_x = grad_norm(st.model, x) if x.requires_grad else 0.0
    st.fb += 1
    return ratio * g_e / g_x if g_x > 0 else 0.0


def total_variation(v):
    km = v / 1000.
    dz, dx = km[1:, :-1] - km[:-1, :-1], km[:-1, 1:] - km[:-1, :-1]
    return torch.sqrt(dz ** 2 + dx ** 2 + 1e-6).mean()


def ph_value(P, v):
    sc = P.cfg["score"]
    q = P.score(v)
    d = superlevel_diagram(q.detach().cpu().double().numpy(), fg=sc["fg"])
    return band_terms(q, d, sc["a"], sc["b"], sc["eps"], sc["kappa"]), d, q


# ---------------------------------------------------------------- metrics (truth used only here)

def metrics(P, v, high):
    sc = P.cfg["score"]
    with torch.no_grad():
        q = P.score(v).cpu().numpy()
        e_train = float(P.misfit(v, high))
        e_eval = float(P.misfit(v, high, "eval"))
    vt, vn = P.m["v_true"], v.detach().cpu().numpy()
    region = ndimage.binary_dilation(P.m["mask"], iterations=3)
    pred, true = q >= 0.5, P.q_true >= 0.5
    union = (pred | true).sum()
    taus_band = np.linspace(sc["a"], sc["b"], 9)
    taus_wide = np.linspace(0.2, 0.8, 13)

    def betti_err(taus):
        errs = [np.abs(np.subtract(betti_by_labels(q, t, sc["fg"]), betti_by_labels(P.q_true, t, sc["fg"]))).sum()
                for t in taus]
        return float(np.mean(errs))

    return {"rmse": float(np.sqrt(np.mean((vn - vt) ** 2))),
            "rmse_anomaly": float(np.sqrt(np.mean((vn - vt)[region] ** 2))),
            "iou": float((pred & true).sum() / union) if union else 1.0,
            "hd95_m": hd95(pred, true, P.cfg["grid"]["dx"]),
            "betti_band_err": betti_err(taus_band), "betti_wide_err": betti_err(taus_wide),
            "betti_0p5": list(betti_by_labels(q, 0.5, sc["fg"])),
            "q_max": float(q.max()), "e_train": e_train, "e_eval": e_eval}


def hd95(pred, true, dx):
    if not pred.any() or not true.any():
        return None
    edge = lambda m: m & ~ndimage.binary_erosion(m)
    bp, bt = edge(pred), edge(true)
    d_to_t = ndimage.distance_transform_edt(~bt)[bp]
    d_to_p = ndimage.distance_transform_edt(~bp)[bt]
    return float(np.percentile(np.concatenate([d_to_t, d_to_p]), 95) * dx)


# ---------------------------------------------------------------- candidate review (proposal 4.5, 5.2-5.3)

def contested_bar(P, q, diagram, terms):
    """The finite bar with the largest band overlap h_i, with its two candidate regions."""
    sc = P.cfg["score"]
    lo, hi = sc["a"] - sc["eps"], sc["b"] + sc["eps"]
    h = terms["overlap"].detach().cpu().numpy()
    if len(h) == 0 or h.max() <= 0:
        return None
    i = int(np.argmax(h))
    birth, death = (int(x) for x in terms["pairs"][i])
    kind = int(terms["kinds"][i])
    qn = q.detach().cpu().numpy()
    shape = qn.shape
    B, D = qn.flat[birth], qn.flat[death]
    s4, s8 = ndimage.generate_binary_structure(2, 1), ndimage.generate_binary_structure(2, 2)
    fg_s, bg_s = (s4, s8) if sc["fg"] == 4 else (s8, s4)
    r = P.cfg["review"]["saddle_radius"]
    zz, xx = np.mgrid[:shape[0], :shape[1]]

    def disk(p):
        pz, px = np.unravel_index(p, shape)
        return (zz - pz) ** 2 + (xx - px) ** 2 <= r ** 2

    def component(level_mask, p, structure):
        lab, _ = ndimage.label(level_mask, structure=structure)
        return lab == lab.flat[p]

    if kind == 0:   # extra component: delete the island (lower its peak region) or connect at the saddle
        island = component(qn > D if D >= lo else qn >= lo, birth, fg_s)
        regions = {"lower": ("delete", island), "raise": ("connect", disk(death))}
    else:           # hole: fill the basin or open the ring at the closing pixel
        basin = component(qn < B if B <= hi else qn < hi, death, bg_s)
        regions = {"raise": ("fill", basin), "lower": ("open", disk(birth))}
    return {"index": i, "kind": "H0" if kind == 0 else "H1", "B": float(B), "D": float(D),
            "h": float(h[i]), "value": float(terms["bars"][i]), "regions": regions,
            "birth_zx": [int(x) for x in np.unravel_index(birth, shape)],
            "death_zx": [int(x) for x in np.unravel_index(death, shape)]}


def finite_ph(P, v):
    terms, _, _ = ph_value(P, v)
    return float(terms["bars"].sum()) if len(terms["bars"]) else 0.0


def review(P, st, rule, high, e_max):
    cfg, rc, sc = P.cfg, P.cfg["review"], P.cfg["score"]
    n_inner = cfg["structural"]["n_inner"]
    cand_lr = st.lr * cfg["structural"]["cand_lr_scale"]
    with torch.no_grad():
        v_t = st.model()
    terms, diagram, q_t = ph_value(P, v_t)
    e_t = float(P.misfit(v_t, high).detach())
    st.fwd += 1
    tau = rc["tau_rel"] * e_t
    bar = None
    if rule != "B4" and float(terms["essential"]) == 0:
        bar = contested_bar(P, q_t, diagram, terms)
    if rule != "B4" and bar is None:
        return None                                # nothing contested: the caller keeps training
    r_fin_t = float(terms["bars"].sum()) if len(terms["bars"]) else 0.0

    specs = []                                     # (name, kind, lr, soft)
    if rule == "B4":
        for f in rc["b4_lr_factors"]:
            specs.append((f"waveform_x{f:g}", "waveform", cand_lr * f, None))
    else:
        for direction, (name, region) in bar["regions"].items():
            target = sc["b"] + sc["eps"] + 0.05 if direction == "raise" else sc["a"] - sc["eps"] - 0.05
            mask = torch.tensor(region, device=P.device)
            specs.append((name, direction, cand_lr, (mask, direction, target)))
        specs.append(("waveform", "waveform", cand_lr, None))

    cands = [{"name": "keep", "kind": "keep", "state": st, "E": e_t}]
    for name, kind, lr, soft in specs:
        c = st.clone(lr=lr)
        extra = None
        if soft is not None:
            mask, direction, target = soft
            fn = lambda v, m=mask, d=direction, t=target: soft_target(P.score(v), m, d, t)
            w = ratio_weight(P, c, high, fn, cfg["weights"]["soft_ratio"])
            extra = lambda v, fn=fn, w=w: w * fn(v)
            soft = (mask, direction, target, w)
        train(P, c, n_inner, high, extra)
        for group in c.opt.param_groups:
            group["lr"] = st.lr
        with torch.no_grad():
            v_c = c.model()
            pred = P.predict(v_c, high)
            e_c = float(P.misfit(v_c, high, pred=pred))
        c.fwd += 1
        cands.append({"name": name, "kind": kind, "state": c, "E": e_c, "soft": soft, "v": v_c, "pred": pred})

    # Shared bookkeeping: every candidate's compute is charged to the run that made it.
    extra_fb = sum(c["state"].fb - st.fb for c in cands[1:])
    extra_fwd = sum(c["state"].fwd - st.fwd for c in cands[1:]) + 1
    extra_sec = sum(c["state"].seconds - st.seconds for c in cands[1:])

    v_t_np = v_t.cpu().numpy()
    vt_true = P.m["v_true"]
    for c in cands:
        v = c.get("v", v_t)
        vn = v.detach().cpu().numpy()
        c["dv_inf"] = float(np.abs(vn - v_t_np).max())
        c["dv_l2"] = float(np.sqrt(((vn - v_t_np) ** 2).sum()))
        c["feasible"] = bool(vn.min() >= rc["vmin"] and vn.max() <= rc["vmax"] and c["dv_inf"] <= rc["r_max"]
                             and c["E"] <= min(e_t + tau, e_max))
        c["R_fin"] = finite_ph(P, v) if c["kind"] != "keep" else r_fin_t
        need = rc["min_improve_frac"] * (bar["value"] if bar else 0.0)
        c["resolves"] = bool(bar is not None and r_fin_t - c["R_fin"] >= need)
        c["err_truth"] = float(np.sqrt(((vn - vt_true) ** 2).mean()))           # post-hoc only
        c["betti_0p5"] = list(betti_by_labels(P.score(v).detach().cpu().numpy(), 0.5, sc["fg"]))  # post-hoc only

    # Data distinguishability (v0.2, 4.5): chi = |W(F(v_1) - F(v_2))|^2 / |W d|^2 / tau.
    # chi < 1 means the two candidates' predicted data differ by less than the tie tolerance.
    _, norm = P._obs_band("train", high)
    preds = {c["name"]: c["pred"] for c in cands[1:]}
    chi = {}
    names = list(preds)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            chi[f"{n1}|{n2}"] = float((preds[n1] - preds[n2]).square().sum() / norm / tau)
    for c in cands[1:]:
        del c["pred"]

    feas = [c for c in cands if c["feasible"]]
    e_min = min(c["E"] for c in feas)
    tied = [c for c in feas if c["E"] <= e_min + tau]
    wave = next((c for c in feas if c["kind"] == "waveform"), None)
    structural = [c for c in tied if c["kind"] in ("raise", "lower") and c["resolves"]]
    if rule == "B3a":
        if structural:
            chosen = min(structural, key=lambda c: c["dv_l2"])
            decided = "prior" if len(tied) > 1 else "data"
        elif wave is not None and wave in tied:
            chosen, decided = wave, "fallback"
        else:
            chosen, decided = min(feas, key=lambda c: c["E"]), "data"
    elif rule == "B3b":
        best = min(feas, key=lambda c: c["E"])
        if wave is not None and wave["E"] <= best["E"] + tau:
            chosen = wave
            decided = "uncertain" if any(c["kind"] in ("raise", "lower") for c in tied) else "data"
        else:
            chosen, decided = best, "data"
    else:
        chosen, decided = min(feas, key=lambda c: c["E"]), "data"

    new = chosen["state"] if chosen["kind"] != "keep" else st.clone()
    new.fb, new.fwd = st.fb + extra_fb, st.fwd + extra_fwd
    new.seconds = st.seconds + extra_sec
    new.steps = st.steps + (n_inner if chosen["kind"] != "keep" else 0)
    new.history = list(st.history) + (chosen["state"].history[len(st.history):] if chosen["kind"] != "keep" else [])
    if chosen["kind"] in ("raise", "lower"):
        new.anchors = list(st.anchors) + [chosen["soft"]]
    wave_err = wave["err_truth"] if wave else None
    log = {"step": st.steps, "E_t": e_t, "tau": tau, "decision": decided, "chosen": chosen["name"],
           "bar": None if bar is None else {k: bar[k] for k in ("kind", "B", "D", "h", "birth_zx", "death_zx")},
           "candidates": [{k: c[k] for k in ("name", "E", "feasible", "resolves", "R_fin", "dv_inf", "err_truth",
                                             "betti_0p5")} for c in cands],
           "chi": chi, "posthoc_closer_than_waveform": None if wave_err is None else bool(chosen["err_truth"] < wave_err)}
    new.reviews = list(st.reviews) + [log]
    return new


# ---------------------------------------------------------------- groups

def run_group(P, warm, group, log):
    cfg = P.cfg
    s = cfg["structural"]
    high, K, n_inner = s["high_hz"], s["review_every"], s["n_inner"]
    st = warm.clone(lr=warm.lr * s["lr_factor"])
    st.lr = warm.lr * s["lr_factor"]
    st.fb = st.fwd = 0
    st.seconds = 0.0
    with torch.no_grad():
        e_max = cfg["review"]["e_max_factor"] * float(P.misfit(st.model(), high))
    extra, weight = None, None
    if group == "B1":
        weight = ratio_weight(P, st, high, total_variation, cfg["weights"]["tv_ratio"])
        extra = lambda v: weight * total_variation(v)
    if group == "B2":
        fn = lambda v: ph_value(P, v)[0]["total"]
        weight = ratio_weight(P, st, high, fn, cfg["weights"]["ph_ratio"])
        extra = lambda v: weight * fn(v)
    done = 0
    while done < s["steps"]:
        seg = min(K, s["steps"] - done)
        if group in ("B3a", "B3b", "B4") and seg > n_inner:
            new = review(P, st, group, high, e_max)
            if new is not None:
                committed = new.steps - st.steps
                st = new
                train(P, st, seg - committed, high)
            else:
                train(P, st, seg, high)
        else:
            train(P, st, seg, high, extra)
        done += seg
        if done % 60 == 0 or done == s["steps"]:
            m = metrics(P, st.model(), high)
            log(f"  {group} step {done:4d}  E={st.history[-1]:.4f}  IoU={m['iou']:.3f}  "
                f"betti_band={m['betti_band_err']:.2f}  rmse_anom={m['rmse_anomaly']:.0f}  "
                f"reviews={len(st.reviews)}  {st.seconds:.0f}s")
    return st, weight


# ---------------------------------------------------------------- main

def save_figure(P, panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, ax = plt.subplots(2, n, figsize=(2.3 * n, 4.2), constrained_layout=True)
    extent = [0, P.cfg["grid"]["nx"] * P.cfg["grid"]["dx"] / 1000, P.cfg["grid"]["nz"] * P.cfg["grid"]["dx"] / 1000, 0]
    for j, (title, v) in enumerate(panels):
        q = P.score(torch.tensor(v, device=P.device)).cpu().numpy()
        ax[0, j].imshow(v, cmap="jet", vmin=1500, vmax=4500, extent=extent)
        ax[1, j].imshow(q, cmap="gray_r", vmin=0, vmax=1, extent=extent)
        ax[1, j].contour(np.linspace(extent[0], extent[1], q.shape[1]), np.linspace(0, extent[2], q.shape[0]),
                         P.q_true, levels=[0.5], colors="r", linewidths=0.6)
        ax[0, j].set_title(title, fontsize=8)
        for a in ax[:, j]:
            a.tick_params(labelsize=6)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, default="dumbbell")
    ap.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    ap.add_argument("--out", type=Path, default=Path("outputs/topo_ifwi/run"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--snr-db", type=float)
    ap.add_argument("--lowcut-hz", type=float)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--warm-from", type=Path, help="warm_state.pt of an earlier run with the same config: skip the warm start")
    ap.add_argument("--struct-steps", type=int)
    ap.add_argument("--n-inner", type=int)
    ap.add_argument("--cand-lr-scale", type=float)
    ap.add_argument("--soft-ratio", type=float)
    ap.add_argument("--tau-rel", type=float)
    args = ap.parse_args()
    cfg = merge(CONFIG, SMOKE) if args.smoke else copy.deepcopy(CONFIG)
    for arg, (sec, key) in {"struct_steps": ("structural", "steps"), "n_inner": ("structural", "n_inner"),
                            "cand_lr_scale": ("structural", "cand_lr_scale"), "soft_ratio": ("weights", "soft_ratio"),
                            "tau_rel": ("review", "tau_rel")}.items():
        if getattr(args, arg) is not None:
            cfg[sec][key] = getattr(args, arg)
    if args.seed is not None:
        cfg["network"]["seed"] = args.seed
    if args.snr_db is not None:
        cfg["observed"]["snr_db"] = args.snr_db
    if args.lowcut_hz is not None:
        cfg["observed"]["lowcut_hz"] = args.lowcut_hz
    args.out.mkdir(parents=True, exist_ok=True)
    logf = open(args.out / "log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    P = Problem(cfg, args.model, args.device)
    log(f"model={args.model} device={args.device} contrast={P.contrast:.0f} dv={P.dv:.0f} T={P.temp:.0f} "
        f"CFL={P.solver.cfl(cfg['v_anomaly']):.3f}  obs {tuple(P.obs.shape)} in {time.time() - t0:.1f}s")
    high_final = cfg["structural"]["high_hz"]
    with torch.no_grad():
        floor = float(P.misfit(P.v_true, high_final))
    log(f"modelling-error floor E_A(v_true) at {high_final} Hz = {floor:.4f} (coarse solver vs fine-grid data)")

    torch.manual_seed(cfg["network"]["seed"])
    net = cfg["network"]
    model = ResidualSiren(P.m["v0"], net["hidden"], net["depth"], net["omega"], net["scale"],
                          net["taper_rows"]).to(P.device)
    st = State(model, torch.optim.Adam(model.parameters(), lr=net["lr"]), net["lr"])
    m0 = metrics(P, st.model(), high_final)
    log(f"initial: {json.dumps({k: m0[k] for k in ('rmse', 'iou', 'betti_band_err', 'e_train')})}")
    if args.warm_from:
        ck = torch.load(args.warm_from, map_location=P.device)
        if ck["model"] != args.model or ck["warm"] != cfg["warm"] or ck["network"] != cfg["network"]:
            raise SystemExit("--warm-from checkpoint was made with a different model, warm schedule or network")
        st.model.load_state_dict(ck["model_state"])
        st.opt.load_state_dict(ck["opt_state"])
        st.steps, st.fb, st.seconds, st.history = ck["steps"], ck["fb"], ck["seconds"], ck["history"]
        log(f"warm start loaded from {args.warm_from} ({st.steps} steps)")
    else:
        for high, n in cfg["warm"]:
            train(P, st, n, high)
            m = metrics(P, st.model(), high_final)
            log(f"warm {high:4.1f} Hz x{n}: E={st.history[-1]:.4f} E_eval={m['e_eval']:.4f} IoU={m['iou']:.3f} "
                f"q_max={m['q_max']:.3f} betti_band={m['betti_band_err']:.2f} betti@0.5={m['betti_0p5']} ({st.seconds:.0f}s)")
        torch.save({"model": args.model, "warm": cfg["warm"], "network": cfg["network"],
                    "model_state": st.model.state_dict(), "opt_state": st.opt.state_dict(), "steps": st.steps,
                    "fb": st.fb, "seconds": st.seconds, "history": st.history}, args.out / "warm_state.pt")
    sec_per_step = st.seconds / max(st.fb, 1)

    # Phase 0 (v0.2 change 1): is this the regime the method is meant for?
    with torch.no_grad():
        v_w = st.model()
    terms, diag, _ = ph_value(P, v_w)
    mw = metrics(P, v_w, high_final)
    sc = cfg["score"]
    located = float(terms["essential"]) == 0 and mw["iou"] >= cfg["phase0"]["iou_min"]
    phase0 = {"located": bool(located), "topology_error": mw["betti_band_err"] > 0,
              "in_regime": bool(located and mw["betti_band_err"] > 0),
              "B_e": float(P.score(v_w).max()), "bars_in_band": int((terms["overlap"] > 0).sum()) if len(terms["overlap"]) else 0,
              "R_PH": float(terms["total"]), "metrics": mw, "warm_seconds": st.seconds, "sec_per_fb_step": sec_per_step}
    log(f"PHASE 0: {json.dumps({k: phase0[k] for k in ('located', 'topology_error', 'in_regime', 'B_e', 'bars_in_band', 'R_PH')})}")
    (args.out / "phase0.json").write_text(json.dumps(phase0, indent=2))

    results, panels = {}, [("true", P.m["v_true"]), ("initial", P.m["v0"]), ("warm start", v_w.cpu().numpy())]
    for group in args.groups:
        log(f"group {group}")
        g, weight = run_group(P, st, group, log)
        with torch.no_grad():
            v = g.model()
        res = metrics(P, v, high_final)
        decisions = {}
        for r in g.reviews:
            decisions[f"{r['decision']}:{r['chosen']}"] = decisions.get(f"{r['decision']}:{r['chosen']}", 0) + 1
        res.update({"weight": weight, "fb_steps": g.fb, "fwd_evals": g.fwd, "seconds": g.seconds,
                    "committed_steps": g.steps - st.steps, "reviews": len(g.reviews), "decisions": decisions,
                    "anchors": len(g.anchors),
                    "posthoc_closer": [r["posthoc_closer_than_waveform"] for r in g.reviews]})
        results[group] = res
        log(f"  -> {json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in res.items() if k != 'posthoc_closer'})}")
        with open(args.out / f"reviews_{group}.json", "w") as f:
            json.dump(g.reviews, f, indent=1)
        np.save(args.out / f"v_{group}.npy", v.cpu().numpy())
        panels.append((group, v.cpu().numpy()))
    np.save(args.out / "v_warm.npy", v_w.cpu().numpy())
    summary = {"model": args.model, "config": cfg, "device": args.device, "floor": floor, "phase0": phase0,
               "results": results, "wall_seconds": time.time() - t0}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    save_figure(P, panels, args.out / "panels.png")
    log(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()

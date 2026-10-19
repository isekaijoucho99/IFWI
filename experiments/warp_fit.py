"""Wave-free check of the traveltime coordinate warp (core_models.traveltime_coordinates).

Fits the residual network v = v0 + 1000 f(q) directly to a target velocity with the paper's
Adam settings, once with (x, z) and once with warped coordinates, and records per depth band
how the fitted perturbation dv = v - v0 splits into wavenumbers the data can and cannot
resolve. No wave simulation: this isolates the representation bias that the warp changes.

Local resolution limit: k_res(q) = 2 f_max / v0(q) (cycles per unit length), the largest
wavenumber of a reflection-transmission gradient at f_max. Two targets:
* full:     the true model. Shows how fast resolvable and unresolvable parts are fitted.
* resolved: dv_true smoothed with a depth-dependent Gaussian whose half-amplitude wavenumber is
            k_res(q), i.e. roughly what band-limited data can constrain. Energy that the fit puts
            above k_res is then texture that no data asked for.
Truth is used only as the regression target, never by the warp.

python -m experiments.warp_fit --case marmousi --method E3 --target full resolved --axes none z xz --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

ROOT = Path(__file__).resolve().parents[1]


def depth_bands(nz, count=3):
    edges = np.linspace(0, nz, count + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def resolved_target(truth, background, dx_m, f_max):
    """dv_true smoothed per pixel with sigma = sqrt(2 ln 2) / (2 pi k_res), k_res = 2 f_max / v0."""
    dv = (truth - background).astype(np.float64)
    sigma_m = math.sqrt(2 * math.log(2)) / (2 * math.pi) * background.astype(np.float64) / (2 * f_max)
    sigma_px = sigma_m / dx_m
    levels = np.linspace(sigma_px.min(), sigma_px.max(), 12)
    stack = np.stack([gaussian_filter(dv, s, mode="reflect") for s in levels])
    position = np.interp(sigma_px, levels, np.arange(len(levels)))
    low = np.floor(position).astype(int).clip(0, len(levels) - 2)
    weight = position - low
    rows, cols = np.indices(dv.shape)
    smooth = (1 - weight) * stack[low, rows, cols] + weight * stack[low + 1, rows, cols]
    return (background + smooth).astype(np.float32)


def band_spectrum(field, background, band, dx_m, f_max):
    """Energy of a band's field above and below k_res = 2 f_max / mean v0 (Hann window, cycles/km)."""
    a, b = band
    block = field[a:b].astype(np.float64)
    block = block - block.mean()
    window = np.outer(np.hanning(block.shape[0]), np.hanning(block.shape[1]))
    power = np.abs(np.fft.fft2(block * window)) ** 2
    kz = np.fft.fftfreq(block.shape[0], dx_m / 1000)
    kx = np.fft.fftfreq(block.shape[1], dx_m / 1000)
    k = np.hypot(*np.meshgrid(kz, kx, indexing="ij"))
    k_res = 2 * f_max / float(background[a:b].mean()) * 1000
    total = float(power.sum()) or 1.
    return {"k_res_cpkm": k_res, "above_fraction": float(power[k > k_res].sum() / total),
            "energy_above": float(power[k > k_res].sum()), "energy_below": float(power[k <= k_res].sum())}


def diagnostics(v, target, truth, background, dx_m, f_max, bands):
    dv = v - background
    out = {"rmse_target_mps": float(np.sqrt(np.mean((v - target).astype(np.float64) ** 2))),
           "rmse_truth_mps": float(np.sqrt(np.mean((v - truth).astype(np.float64) ** 2))), "bands": []}
    for band in bands:
        a, b = band
        out["bands"].append({
            "rows": list(band),
            "rmse_truth_mps": float(np.sqrt(np.mean((v[a:b] - truth[a:b]).astype(np.float64) ** 2))),
            "dv": band_spectrum(dv, background, band, dx_m, f_max),
            "error": band_spectrum(v - target, background, band, dx_m, f_max)})
    return out


def fit(cfg, background, target, truth, axes, steps, every, device, f_max):
    from core_runner import build_model
    from experiments import residual_ifwi_experiment as e
    e._seed(cfg["seed"])
    warp = None if axes == "none" else {"kind": "traveltime", "axes": axes}
    model = build_model({"model": {"kind": "siren", **({"coordinate_warp": warp} if warp else {})}}, cfg, background).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    goal = torch.as_tensor(target, device=device)[None]
    dx = cfg["data"]["grid_spacing_m"]
    bands = depth_bands(background.shape[0])
    history, start = [], time.perf_counter()
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            with torch.no_grad():
                v = model()[0].cpu().numpy()
            history.append({"step": step, "seconds": time.perf_counter() - start,
                            **diagnostics(v, target, truth, background, dx, f_max, bands)})
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss = (((model() - goal) / 1000.) ** 2).mean()
        loss.backward()
        opt.step()
    report = getattr(model, "coordinate_warp_report", None)
    return history, report, v


def main(argv=None):
    import paper_supplement as ps
    from experiments.residual_ifwi_data import load_velocity_data
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=("marmousi", "overthrust"), default="marmousi")
    parser.add_argument("--method", choices=("E2", "E3"), default="E3")
    parser.add_argument("--axes", nargs="+", choices=("none", "z", "xz"), default=["none", "z", "xz"])
    parser.add_argument("--target", nargs="+", choices=("full", "resolved"), default=["full", "resolved"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--f-max", type=float, default=20.0, help="highest data frequency for k_res (8 Hz Ricker: ~20 Hz)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "warp_fit")
    args = parser.parse_args(argv)
    cfg = ps.experiment(args.method, "clean" if args.case == "marmousi" else "overthrust")
    truth, background, _ = load_velocity_data(cfg, ROOT)
    truth, background = truth.astype(np.float32), background.astype(np.float32)
    dx = cfg["data"]["grid_spacing_m"]
    targets = {"full": truth, "resolved": resolved_target(truth, background, dx, args.f_max)}
    out = args.output_dir / f"{args.case}_{args.method}"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.target:
        for axes in args.axes:
            history, report, v = fit(cfg, background, targets[name], truth, axes, args.steps, args.every, args.device, args.f_max)
            key = f"{name}_{axes}"
            (out / f"{key}.json").write_text(json.dumps({"case": args.case, "method": args.method, "target": name,
                "axes": axes, "f_max_hz": args.f_max, "warp": report, "history": history}, indent=2))
            np.save(out / f"{key}_velocity.npy", v)
            last = history[-1]
            summary[key] = {"rmse_target": last["rmse_target_mps"], "rmse_truth": last["rmse_truth_mps"],
                            "dv_above_fraction_by_band": [b["dv"]["above_fraction"] for b in last["bands"]],
                            "rmse_truth_by_band": [b["rmse_truth_mps"] for b in last["bands"]]}
            print(key, json.dumps(summary[key]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

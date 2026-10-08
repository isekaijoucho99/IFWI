"""IFWI vs G2IFWI on the 2D Overthrust (Sec. 4.2, 5.1-5.3) or Marmousi (--model marmousi) model.

python -m g2ifwi.invert --prior weights/prior_small.pt --out outputs/overthrust
Both methods share the first `--warmup` IFWI epochs and branch there (as in Fig. 4).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .common import metrics, pick_device
from .diffusion import DDPM, UNet, red_denoise
from .forward import AcousticModeler, ricker
from .inr import HashSirenVelocity

ROOT = Path(__file__).resolve().parents[1]


# Per-model acquisition. Overthrust follows Sec. 4.2; the paper has no Marmousi experiment, so the Marmousi
# values are those of the earlier IFWI code in this repo (94x288 grid, 15 m, 8 Hz, 1.9 ms x 1000 steps).
MODEL_DEFAULTS = {
    "overthrust": dict(dz=25.0, dt=0.002, nt=3000, shots=40, src_z=1, rec_z=1),
    "marmousi": dict(dz=15.0, dt=0.0019, nt=1000, shots=14, src_z=1, rec_z=2),
}


def apply_model_defaults(a):
    for k, v in MODEL_DEFAULTS[a.model].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def load_truth(a):
    if a.model == "marmousi":
        v = np.loadtxt(ROOT / "data" / "vel_marmousi_376x1151.csv", delimiter=",", dtype=np.float32)
        return v[::4, ::4]
    with np.load(ROOT / "data" / "overthrust2d.npz") as d:
        return d["vp"].astype(np.float32) * 1000


def build_problem(a, dev):
    truth = load_truth(a)[::a.stride, ::a.stride]
    nz, nx = truth.shape
    dz = a.dz * a.stride
    dt, nt = a.dt * a.stride, int(round(a.nt / a.stride))
    if a.model == "marmousi" and a.stride == 1:
        src_x = np.arange(20, nx - 10, 20)[:a.shots]       # as in the earlier Marmousi setup
    else:
        src_x = np.linspace(a.src_margin, nx - 1 - a.src_margin, a.shots).round().astype(int)
    mod = AcousticModeler(nz, nx, dz, dt, ricker(a.freq, dt, nt), src_x, a.src_z, np.arange(0, nx, a.rec_stride),
                          a.rec_z, npad=a.npad, chunk=a.chunk, shot_batch=a.shot_batch, device=dev)
    mod.check_cfl(float(truth.max()))
    return truth, mod


def initial_model(truth, kind, sigma):
    if kind == "1d":
        return np.tile(truth.mean(1, keepdims=True), (1, truth.shape[1]))
    return gaussian_filter(truth, sigma, mode="nearest")


def load_prior(path, dev):
    ck = torch.load(path, map_location=dev)
    net = UNet(**ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["model"])
    return net, DDPM(device=dev)


def run(a):
    apply_model_defaults(a)
    dev = pick_device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    truth, mod = build_problem(a, dev)
    nz, nx = truth.shape
    v_true = torch.from_numpy(truth).to(dev)
    t0 = time.time()
    d_obs = mod.simulate_all(v_true)
    if a.snr is not None:
        sigma = (d_obs.pow(2).mean() / 10 ** (a.snr / 10)).sqrt()
        d_obs = d_obs + sigma * torch.randn_like(d_obs)
    norm = float(d_obs.pow(2).sum())
    v0 = initial_model(truth, a.init, a.sigma)
    print(f"device={dev} grid={nz}x{nx} shots={mod.n_shots} nt={mod.nt} obs-data {time.time()-t0:.0f}s "
          f"init rmse={metrics(v0, truth)['rmse']:.1f}", flush=True)

    enc = dict(n_levels=a.levels, n_features=a.features, base_res=a.base_res, finest_res=a.finest_res,
               log2_hashmap_size=a.log2_T, init_scale=a.hash_init)
    net = HashSirenVelocity(nz, nx, v0, enc, dict(hidden=a.hidden, layers=a.layers, omega=a.omega),
                            dv_scale=a.dv_scale).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    prior = load_prior(a.prior, dev) if "g2ifwi" in a.methods else None

    def epoch(net, opt, e, use_prior):
        v = net()
        loss, g = mod.loss_and_grad(v, d_obs, norm)                      # g_data  (Alg. 1 line 6)
        info = dict(loss=loss, alpha=0.0)
        if use_prior and (e - a.warmup) % a.every == 0:
            with torch.no_grad():
                gen = torch.Generator(device=dev).manual_seed(a.seed * 100003 + e)
                v_d = red_denoise(prior[0], prior[1], v.detach(), a.t_start, a.n_steps, generator=gen)
                g_ddpm = v.detach() - v_d                                # Eq. (35)
                lam = a.lam if a.lam is not None else a.alpha * g.norm() / g_ddpm.norm().clamp_min(1e-30)
                info["alpha"] = float(lam * g_ddpm.norm() / g.norm().clamp_min(1e-30))
                info["lam"] = float(lam)
                g = g + lam * g_ddpm                                     # Eq. (37)
        opt.zero_grad(); v.backward(g); opt.step()                       # Eq. (39)
        info["rmse"] = float(((v.detach() - v_true) ** 2).mean().sqrt())
        return info

    hist = {}
    def run_stage(name, net, opt, lo, hi, use_prior):
        h = hist.setdefault(name, [])
        for e in range(lo, hi):
            info = epoch(net, opt, e, use_prior); h.append(info)
            if e % a.log == 0 or e == hi - 1:
                print(f"[{name}] epoch {e:4d} loss {info['loss']:.4e} rmse {info['rmse']:.1f} "
                      f"alpha {info['alpha']*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)

    warm = min(a.warmup, a.epochs)
    run_stage("warmup", net, opt, 0, warm, False)
    results = {}
    for name in a.methods:
        n = copy.deepcopy(net)
        o = torch.optim.Adam(n.parameters(), lr=a.lr); o.load_state_dict(opt.state_dict())
        run_stage(name, n, o, warm, a.epochs, name == "g2ifwi")
        with torch.no_grad():
            v = n().cpu().numpy()
        results[name] = v
    summary = {k: metrics(v, truth) for k, v in results.items()}
    summary["initial"] = metrics(v0, truth)
    print(json.dumps(summary, indent=2), flush=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    json.dump({k: v for k, v in hist.items()}, open(out / "history.json", "w"))
    np.savez(out / "models.npz", truth=truth, init=v0, **results)
    try:
        plot(out, truth, v0, results, hist, summary, a)
    except Exception as ex:  # plotting must never lose results
        print("plot failed:", ex)
    return summary


def plot(out, truth, v0, results, hist, summary, a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dz = a.dz * a.stride
    ext = [0, truth.shape[1] * dz / 1000, truth.shape[0] * dz / 1000, 0]
    panels = [("True", truth), ("Initial", v0)] + [(k.upper(), v) for k, v in results.items()]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    for axx, (t, v) in zip(ax.flat, panels):
        im = axx.imshow(v, extent=ext, aspect="auto", cmap="seismic", vmin=truth.min(), vmax=truth.max())
        s = summary.get(t.lower())
        axx.set_title(t + (f" RMSE={s['rmse']:.1f} SSIM={s['ssim']:.2f} PSNR={s['psnr']:.2f}" if s else ""))
    fig.colorbar(im, ax=ax.ravel().tolist()[:len(panels)], label="m/s")
    for axx in ax.flat[len(panels):]:
        axx.axis("off")
    fig.savefig(out / "models.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    w = hist["warmup"]
    for k in results:
        h = w + hist[k]
        ax[0].semilogy([i["loss"] for i in h], label=k); ax[1].plot([i["rmse"] for i in h], label=k)
    ax[0].set(title="data misfit", xlabel="epoch"); ax[1].set(title="model RMSE (m/s)", xlabel="epoch")
    for x in ax:
        x.axvline(len(w), ls=":", c="k"); x.legend()
    fig.savefig(out / "curves.png", dpi=130); plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_argument
    g("--out", default="outputs/overthrust"); g("--device"); g("--seed", type=int, default=0)
    g("--methods", nargs="+", default=["ifwi", "g2ifwi"], choices=["ifwi", "g2ifwi"])
    # acquisition / modelling (Sec. 4.2)
    g("--model", choices=list(MODEL_DEFAULTS), default="overthrust")
    g("--dz", type=float); g("--dt", type=float); g("--nt", type=int)
    g("--src-z", type=int); g("--rec-z", type=int)
    g("--freq", type=float, default=8.0); g("--shots", type=int); g("--src-margin", type=int, default=0)
    g("--rec-stride", type=int, default=1); g("--stride", type=int, default=1, help="spatial decimation for quick tests")
    g("--npad", type=int, default=30); g("--chunk", type=int, default=50); g("--shot-batch", type=int, default=8)
    g("--snr", type=float, help="add Gaussian noise at this SNR (dB), Sec. 5.2")
    g("--init", choices=["gauss", "1d"], default="gauss"); g("--sigma", type=float, default=15.0)
    # network (Table 1, Overthrust column)
    g("--levels", type=int, default=4); g("--features", type=int, default=1); g("--base-res", type=int, default=64)
    g("--finest-res", type=int, default=256); g("--log2-T", type=int, default=18)
    g("--hidden", type=int, default=128); g("--layers", type=int, default=2); g("--omega", type=float, default=30.0)
    g("--hash-init", type=float, default=1e-4); g("--dv-scale", type=float, default=1000.0)
    # optimisation / prior (Sec. 4.2)
    g("--epochs", type=int, default=200); g("--warmup", type=int, default=100); g("--lr", type=float, default=1e-4)
    g("--prior", default="weights/prior_small.pt")
    g("--lam", type=float, help="fixed diffusion weight lambda (units of this implementation's gradients)")
    g("--alpha", type=float, default=0.025, help="target ||lam g_ddpm||/||g_data|| when --lam is not given")
    g("--t-start", type=int, default=1); g("--n-steps", type=int, default=1); g("--every", type=int, default=1)
    g("--log", type=int, default=10)
    return p


if __name__ == "__main__":
    run(parser().parse_args())

"""Cheaper wave simulations per IFWI update: source encoding and coarse grids for low-pass stages.

Both build problem dictionaries in the format train_update expects, so the update itself is
unchanged; only what one update simulates differs.

Source encoding: the shots of one update are summed into ``supershots`` simultaneous-source
records with random +-1 weights, redrawn every update (deterministic in seed and update).
The solver is linear in the source, so the encoded prediction equals the weighted sum of the
single-shot predictions, and the encoded observation is the same weighted sum of the observed
records. One simulation replaces all shots; the price is crosstalk between shots, which acts as
gradient noise. The original solver file stays unchanged: a subclass injects every source with
its weight, after calling the original time step with a zero source.

Coarse low-pass stages: during a frequency-continuation stage with cutoff max_hz the network is
sampled on every factor-th grid point (an implicit representation can be sampled anywhere), and
the wave equation is solved on that grid with factor-times larger grid spacing and time step
(the CFL ratio is unchanged). Observations are the recorded traces of every factor-th receiver
and time sample, so no new data are modeled. Sources and receivers move to the nearest coarse
depth (1 -> 30 m for a 15 m grid); the source-depth change mainly alters the low-frequency
amplitude through the free-surface ghost, so the coarse wavelet is scaled once by the least-
squares factor between coarse predictions at the background and the low-passed observations.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from generator import wGenerator
from rnn_fd import Propagator2D, rnn2D
from experiments.frequency_continuation import energy_scales, lowpass

SPEEDUP_KEYS = {"source_encoding", "coarse_lowpass"}


def validate_speedup(setting, config):
    if setting is None:
        return
    if not isinstance(setting, dict) or not setting or set(setting) - SPEEDUP_KEYS:
        raise ValueError(f"spec.speedup takes {sorted(SPEEDUP_KEYS)}")
    encoding = setting.get("source_encoding")
    if encoding is not None and (set(encoding) != {"supershots", "seed"} or int(encoding["supershots"]) < 1):
        raise ValueError("source_encoding needs supershots >= 1 and seed")
    coarse = setting.get("coarse_lowpass")
    if coarse is not None:
        if set(coarse) != {"factor", "source_depth_index", "receiver_depth_index"} or int(coarse["factor"]) < 2:
            raise ValueError("coarse_lowpass needs factor >= 2, source_depth_index, receiver_depth_index")
        if config["training"].get("frequency_schedule") is None:
            raise ValueError("coarse_lowpass needs training.frequency_schedule")


class EncodedPropagator2D(Propagator2D):
    """Original time step with a zero source, then every source added with its weight."""

    def _Propagator2D___step_rnncell(self, source, u_prev, u_, vx_pre, vz_pre):
        zero = torch.zeros_like(source[0])
        u_, u_next, vx, vz = Propagator2D._Propagator2D___step_rnncell(self, [zero, source[1], source[2]], u_prev, u_, vx_pre, vz_pre)
        rows = self.source_z if self.freeSurface else self.source_z + self.npad
        add = torch.zeros_like(u_next[0, 0])
        add.index_put_((rows, self.source_x + self.npad), source[0][0] * self.weights, accumulate=True)
        return u_, u_next + add[None, None], vx, vz


class SourceEncoder:
    """Turns an all-shot problem into one problem of ``supershots`` encoded simulations."""

    def __init__(self, problem, supershots, seed):
        self.base, self.supershots, self.seed = problem, int(supershots), int(seed)
        cfg, geometry = problem["config"], problem["geometry_tensors"]
        fd = cfg["forward"]
        device = problem["observed"].device
        vmax = problem["propagators"][0]["solver"].fd.vmax
        self.propagators = []
        for index in range(self.supershots):
            solver = rnn2D(nz=geometry["nz"], nx=geometry["nx"], xs=geometry["xs"][:, :1], zs=geometry["zs"][:, :1],
                           xr=geometry["xr"][:, :1], zr=geometry["zr"][:, :1], dz=cfg["data"]["grid_spacing_m"],
                           dt=fd["dt_s"], npad=fd["npad"], order=2, vmax=vmax, log_para=fd["pml_reflection"],
                           freeSurface=fd["free_surface"], dtype=torch.float32, device=device)
            encoded = EncodedPropagator2D(geometry["nz"], geometry["nx"], cfg["data"]["grid_spacing_m"], fd["dt_s"],
                                          fd["npad"], 2, fd["free_surface"], fd["pml_reflection"], vmax, torch.float32,
                                          device).to(device)
            encoded.source_z, encoded.source_x = geometry["zs"][0].to(device), geometry["xs"][0].to(device)
            solver.fd = encoded
            self.propagators.append({"solver": solver, "first": index, "last": index + 1})

    def weights(self, update, index):
        rng = np.random.RandomState((self.seed * 1000003 + update * 101 + index) % 2**32)
        return rng.choice([-1., 1.], size=self.base["observed"].shape[1])

    def problem(self, update):
        """Encoded problem for this update: supershot k uses its own +-1 weights."""
        observed, encoded = self.base["observed"], []
        for index, item in enumerate(self.propagators):
            w = torch.as_tensor(self.weights(update, index), dtype=observed.dtype, device=observed.device)
            item["solver"].fd.weights = w.to(torch.float32)
            encoded.append(torch.einsum("s,bstr->btr", w, observed))
        return {**self.base, "observed": torch.stack(encoded, dim=1), "propagators": self.propagators}


class CoarseView(torch.nn.Module):
    """The network sampled on every factor-th grid point (same parameters)."""

    def __init__(self, model, factor):
        super().__init__()
        self.model, self.factor = model, int(factor)

    def forward(self):
        return self.model()[:, ::self.factor, ::self.factor]


def coarse_problem(problem, background, factor, source_depth_index, receiver_depth_index, schedule):
    """Problem on a factor-times coarser grid using decimated observations (no new data)."""
    f = int(factor)
    cfg = copy.deepcopy(problem["config"])
    fd, acq = cfg["forward"], cfg["acquisition"]
    nz, nx = background.shape
    nz_c, nx_c = len(range(0, nz, f)), len(range(0, nx, f))
    nt_c = len(range(0, fd["nt"], f))
    sources = [x // f for x in problem["geometry"]["source_x_indices"]]
    if any(x * f != s for x, s in zip(sources, problem["geometry"]["source_x_indices"])):
        raise ValueError("Source x positions must lie on the coarse grid")
    fd.update(dt_s=fd["dt_s"] * f, nt=nt_c)
    cfg["data"]["grid_spacing_m"] *= f
    acq.update(source_x_indices=sources, source_depth_index=source_depth_index,
               receiver_depth_index=receiver_depth_index, receiver_stride=1)
    device = problem["observed"].device
    xs = torch.as_tensor(sources, dtype=torch.long)[None]
    zs = torch.full_like(xs, source_depth_index)
    xr = torch.arange(nx_c, dtype=torch.long)[None, None].repeat(1, len(sources), 1)
    zr = torch.full_like(xr, receiver_depth_index)
    vmax = problem["propagators"][0]["solver"].fd.vmax
    batch = cfg["training"]["shot_batch_size"] or len(sources)
    propagators = []
    for first in range(0, len(sources), batch):
        last = min(first + batch, len(sources))
        solver = rnn2D(nz=nz_c, nx=nx_c, xs=xs[:, first:last], zs=zs[:, first:last], xr=xr[:, first:last],
                       zr=zr[:, first:last], dz=cfg["data"]["grid_spacing_m"], dt=fd["dt_s"], npad=fd["npad"], order=2,
                       vmax=vmax, log_para=fd["pml_reflection"], freeSurface=fd["free_surface"],
                       dtype=torch.float32, device=device).to(device)
        propagators.append({"solver": solver, "first": first, "last": last})
    receivers = problem["geometry"]["receiver_x_indices"]
    if receivers != list(range(len(receivers))):
        raise ValueError("Coarse stages assume one receiver per fine grid column")
    observed = problem["observed"][:, :, ::f, ::f][:, :, :nt_c, :nx_c].contiguous()
    wavelet = wGenerator(fd["dt_s"] * torch.arange(nt_c, dtype=torch.float32), fd["frequency_hz"]).ricker().to(device)
    result = {"propagators": propagators, "observed": observed, "wavelet": wavelet, "config": cfg,
              "geometry": {**problem["geometry"], "nz": nz_c, "nx": nx_c, "source_x_indices": sources,
                           "receiver_x_indices": list(range(nx_c)), "nt": nt_c},
              "geometry_tensors": {"nz": nz_c, "nx": nx_c, "xs": xs, "zs": zs, "xr": xr, "zr": zr},
              "frequency_scales": energy_scales(observed, schedule, fd["dt_s"])}
    # One amplitude factor for the moved source depth: least squares at the background, first cutoff.
    from experiments.residual_ifwi_experiment import _simulate
    cutoff = float(schedule[0]["max_hz"])
    v0 = torch.as_tensor(background[::f, ::f], dtype=torch.float32, device=device)[None]
    num = den = 0.
    with torch.no_grad():
        for item in propagators:
            p = lowpass(_simulate(item["solver"], v0, wavelet), cutoff, fd["dt_s"]).double()
            d = lowpass(observed[:, item["first"]:item["last"]], cutoff, fd["dt_s"]).double()
            num += float((p * d).sum())
            den += float(p.square().sum())
    scale = num / den if den > 0 else 1.
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Coarse source scaling failed")
    result["wavelet"] = wavelet * scale
    result["source_scale"] = scale
    return result

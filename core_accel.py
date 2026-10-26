"""Acceleration options for the core runner (spec.accel), all off unless the spec asks for them.

* function_space: function-space (Gauss-Newton / natural-gradient) update. One forward and adjoint
  simulation gives the velocity gradient g_v; an Adam step is taken in velocity space (step in m/s),
  dv = -s * m_hat / (sqrt(v_hat) + eps), and the network step solves the damped least-squares fit
      min_d ||J d - dv||^2 + lambda ||d||^2,    J = d v_theta / d theta,
  by conjugate gradients on (J^T J + lambda I) d = J^T dv. J^T is one backward pass through the
  network and J one double-backward of that linear map (network only, no wave simulation); the network costs a small fraction of a wave simulation, so
  tens of products add little time. lambda = damping * tr(J^T J) / P with one Hutchinson probe per
  update; a large damping leaves the network geometry in charge (close to a scaled gradient step),
  a small one follows the velocity-space step (close to grid FWI).
  (Energy natural gradient for PINNs: Mueller and Zeinhofer, ICML 2023; KFAC-PINN: Dangel et al.,
  NeurIPS 2024; Gauss-Newton natural gradient for implicit shapes, 2026.)
* supershot_schedule: number of simultaneous-source super-shots per update, growing over updates;
  "supershots": null means exact sequential shots (hybrid deterministic-stochastic optimization,
  Friedlander and Schmidt, SIAM J. Sci. Comput. 2012).
* time_window: record-length curriculum; early updates use only the first fraction of each record,
  so each update simulates fewer time steps (cost is proportional to the number of time steps).
* coordinate_anneal: coarse-to-fine input frequency; the network input coordinates are scaled by
  c(k) rising from "start" to 1 over "updates" updates, which lowers the first-layer frequency early
  (BARF, ICCV 2021; SAPE, NeurIPS 2021). The output layer starts at zero, so update 0 is still v0.
"""
from __future__ import annotations

import copy
import math

import torch

from core_speed import SourceEncoder
from experiments import residual_ifwi_experiment as e

ACCEL_KEYS = {"function_space", "supershot_schedule", "time_window", "coordinate_anneal", "fr_collapse"}
FS_KEYS = {"velocity_step_mps", "damping", "cg_iters", "betas", "eps"}


def validate_accel(setting):
    if setting is None:
        return
    if not isinstance(setting, dict) or not setting or set(setting) - ACCEL_KEYS:
        raise ValueError(f"spec.accel takes {sorted(ACCEL_KEYS)}")
    fs = setting.get("function_space")
    if fs is not None:
        if set(fs) - FS_KEYS or not {"velocity_step_mps", "damping"} <= set(fs):
            raise ValueError(f"accel.function_space needs velocity_step_mps and damping; optional {sorted(FS_KEYS)}")
        if not fs["velocity_step_mps"] > 0 or not fs["damping"] > 0:
            raise ValueError("velocity_step_mps and damping must be positive")
    for key in ("supershot_schedule", "time_window"):
        stages = setting.get(key)
        if stages is not None:
            if not isinstance(stages, list) or not stages or stages[0].get("from") != 1:
                raise ValueError(f"accel.{key} is a list of stages starting with from=1")
            if any(b["from"] <= a["from"] for a, b in zip(stages, stages[1:])):
                raise ValueError(f"accel.{key} stages must start at increasing updates")
    for stage in setting.get("time_window") or []:
        if set(stage) != {"from", "fraction"} or not 0 < stage["fraction"] <= 1:
            raise ValueError("time_window stages are {from, fraction in (0, 1]}")
    for stage in setting.get("supershot_schedule") or []:
        k = stage.get("supershots")
        if set(stage) != {"from", "supershots"} or (k is not None and int(k) < 1):
            raise ValueError("supershot_schedule stages are {from, supershots >= 1 or null}")
    collapse = setting.get("fr_collapse")
    if collapse is not None and (set(collapse) != {"at"} or int(collapse["at"]) < 1):
        raise ValueError("fr_collapse is {at >= 1}: merge the Fourier layers before that update")
    anneal = setting.get("coordinate_anneal")
    if anneal is not None and (set(anneal) != {"start", "updates"} or not 0 < anneal["start"] <= 1 or anneal["updates"] < 1):
        raise ValueError("coordinate_anneal is {start in (0, 1], updates >= 1}")


def stage_value(stages, update, key):
    value = None
    for stage in stages:
        if update >= stage["from"]:
            value = stage[key]
    return value


def coordinate_scale(anneal, update):
    """Cosine ramp from start to 1 over the first `updates` updates."""
    if anneal is None or update > anneal["updates"]:
        return 1.
    t = (update - 1) / anneal["updates"]
    return anneal["start"] + (1 - anneal["start"]) * 0.5 * (1 - math.cos(math.pi * t))


def time_window_problem(problem, fraction):
    """The same problem with only the first fraction of every record (fewer time steps)."""
    nt = problem["observed"].shape[2]
    keep = max(2, int(round(fraction * nt)))
    if keep >= nt:
        return problem
    cfg = copy.deepcopy(problem["config"])
    cfg["forward"]["nt"] = keep
    return {**problem, "config": cfg, "wavelet": problem["wavelet"][:keep],
            "observed": problem["observed"][:, :, :keep].contiguous()}


class StepProblems:
    """Problem for one update: time window first, then simultaneous-source encoding."""

    def __init__(self, problem, accel, encoding_seed):
        self.base, self.accel, self.seed = problem, accel or {}, encoding_seed
        self.windows, self.encoders = {}, {}

    def __call__(self, update, fixed_encoder=None):
        fraction = stage_value(self.accel.get("time_window") or [{"from": 1, "fraction": 1.}], update, "fraction")
        if fraction not in self.windows:
            self.windows[fraction] = time_window_problem(self.base, fraction)
        problem = self.windows[fraction]
        schedule = self.accel.get("supershot_schedule")
        if schedule is None:
            if fixed_encoder is None:
                return problem, {"time_fraction": fraction}
            supershots = fixed_encoder
        else:
            supershots = stage_value(schedule, update, "supershots")
            if supershots is None:
                return problem, {"time_fraction": fraction, "supershots": None}
        key = (fraction, int(supershots))
        if key not in self.encoders:
            self.encoders[key] = SourceEncoder(problem, int(supershots), self.seed)
        return self.encoders[key].problem(update), {"time_fraction": fraction, "supershots": int(supershots)}


class FunctionSpaceOptimizer:
    """Velocity-space Adam step projected onto the network tangent space (damped Gauss-Newton)."""

    def __init__(self, model, velocity_step_mps, damping, cg_iters=20, betas=(0.9, 0.999), eps=1e-8):
        self.model, self.damping, self.cg_iters = model, float(damping), int(cg_iters)
        self.betas, self.eps = tuple(betas), float(eps)
        self.param_groups = [{"lr": float(velocity_step_mps)}]     # lr_schedule scales the m/s step
        self.state = {"step": 0, "m": None, "v": None}

    def state_dict(self):
        return {"param_groups": copy.deepcopy(self.param_groups), "state": {k: (v.clone() if torch.is_tensor(v) else v)
                                                                           for k, v in self.state.items()}}

    def load_state_dict(self, saved):
        self.param_groups = copy.deepcopy(saved["param_groups"])
        self.state = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in saved["state"].items()}

    def zero_grad(self, set_to_none=True):
        pass

    def velocity_step(self, g_v):
        b1, b2 = self.betas
        if self.state["m"] is None:
            self.state["m"], self.state["v"] = torch.zeros_like(g_v), torch.zeros_like(g_v)
        self.state["step"] += 1
        k = self.state["step"]
        self.state["m"].mul_(b1).add_(g_v, alpha=1 - b1)
        self.state["v"].mul_(b2).addcmul_(g_v, g_v, value=1 - b2)
        m_hat = self.state["m"] / (1 - b1 ** k)
        v_hat = self.state["v"] / (1 - b2 ** k)
        return -self.param_groups[0]["lr"] * m_hat / (v_hat.sqrt() + self.eps)

    def project(self, dv):
        """Conjugate gradients on (J^T J + lambda I) d = J^T dv; returns d and diagnostics."""
        model = self.model
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        names, plist = [n for n, _ in named], [p for _, p in named]
        params = dict(named)
        with torch.enable_grad():
            field = model().reshape(-1)
            probe_out = torch.zeros_like(field, requires_grad=True)
            linear = torch.autograd.grad(field, plist, probe_out, create_graph=True)     # J^T w, linear in w

        def jt(u):
            return dict(zip(names, torch.autograd.grad(field, plist, u, retain_graph=True)))

        def j(t):
            return torch.autograd.grad(linear, probe_out, [t[n] for n in names], retain_graph=True)[0]

        def dot(a, b):
            return sum((a[n] * b[n]).sum() for n in names)

        probe = {n: torch.randn_like(params[n]) for n in names}
        count = sum(params[n].numel() for n in names)
        lam = self.damping * float(j(probe).square().sum()) / count
        rhs = jt(dv.reshape(-1))
        x = {n: torch.zeros_like(params[n]) for n in names}
        r = {n: rhs[n].clone() for n in names}
        p = {n: r[n].clone() for n in names}
        rr = dot(r, r)
        rhs_norm = float(rr.sqrt())
        iterations = 0
        for iterations in range(1, self.cg_iters + 1):
            jp = jt(j(p))
            ap = {n: jp[n] + lam * p[n] for n in names}
            alpha = rr / dot(p, ap)
            for n in names:
                x[n].add_(alpha * p[n]); r[n].sub_(alpha * ap[n])
            rr_new = dot(r, r)
            if float(rr_new.sqrt()) <= 1e-4 * rhs_norm:
                break
            for n in names:
                p[n] = r[n] + (rr_new / rr) * p[n]
            rr = rr_new
        fitted = j(x).reshape(dv.shape)
        fit = float((fitted - dv).norm() / (dv.norm() + 1e-30))
        return x, {"cg_iterations": iterations, "cg_relative_residual": float(rr_new.sqrt()) / (rhs_norm + 1e-30),
                   "damping_lambda": lam, "velocity_fit_relative_error": fit,
                   "velocity_step_rms_mps": float(dv.square().mean().sqrt())}


def function_space_update(model, problem, optimizer, max_hz=None):
    """One function-space update: g_v from one simulation pass, velocity Adam step, CG projection."""
    with torch.no_grad():
        start = model()
    e.check_velocity(start, problem["config"])
    velocity = start.detach().clone().requires_grad_(True)
    loss_value, objective_value = 0., 0.
    total = problem["observed"].shape[1]
    for item in problem["propagators"]:
        predicted = e._simulate(item["solver"], velocity, problem["wavelet"])
        observations = problem["observed"][:, item["first"]:item["last"]]
        with torch.no_grad():
            loss_value += float(e.waveform_mse(predicted, observations, total))
        objective = e.training_objective(predicted, observations, total, max_hz, problem)
        e.require_finite(objective, "function-space objective")
        objective.backward()
        objective_value += float(objective.detach())
        del predicted, objective
    g_v = velocity.grad.detach()
    e.require_finite(g_v, "velocity gradient")
    dv = optimizer.velocity_step(g_v)
    step, diagnostics = optimizer.project(dv)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    try:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in step:
                    p.add_(step[n])
                    e.require_finite(p, "updated parameter")
            e.check_velocity(model(), problem["config"])
    except BaseException:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(before[n])
        raise
    change = float(torch.stack([s.square().sum() for s in step.values()]).sum().sqrt())
    report = {"data_mse": loss_value, "gradient_norm": float(g_v.norm()), "parameter_update_norm": change,
              "velocity_before_mps": start.detach().cpu().numpy()[0].copy(), "function_space": diagnostics}
    if max_hz is not None:
        report.update(max_hz=float(max_hz), objective=objective_value)
    return report

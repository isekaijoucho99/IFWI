"""Stage-1 residual IFWI using unchanged original IRN and reference FD.

No velocity labels enter train_update: observations are its only target.
All physical velocities are m/s. IRN inputs remain the author's (x,z) km.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import uuid

import numpy as np
import torch

from generator import wGenerator
from ifwi_modules import IRN
from rnn_fd import rnn2D
from experiments.fourier_modules import validate_fourier
from experiments.frequency_continuation import (energy_scales, lowpass, stage_boundaries, stage_cutoff,
    validate_frequency_schedule)
from experiments.low_rank_modules import validate_low_rank
from experiments.lr_schedule import lr_factor, validate_lr_schedule
from experiments.training_options import (IterateEstimates, gain_scale, time_gain, time_power, validate_report_estimates,
    validate_time_gain)
from experiments.trainable_omega import omega_values, parameter_groups, validate_trainable_omega
from experiments.muon_optimizer import (build_optimizer, effective_hidden_weights, optimizer_name, stable_rank,
    validate_optimizer)
from experiments.residual_ifwi_data import load_velocity_data
from experiments.residual_ifwi_report import save_plots, save_sampling_plot, velocity_metrics

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "residual_ifwi_stage1_v1"


def default_config():
    return {
        "experiment_name": "residual_ifwi_stage1", "parameterization": "residual", "seed": 3,
        "data": {"model_file": "vel_marmousi_376x1151.csv", "model_units": "m/s",
                 "csv_header": "legacy", "downsample": 4, "crop_shape": None, "grid_spacing_m": 15.},
        "initialization": {"sigma": 15., "gaussian_mode": "reflect", "gaussian_truncate": 4.},
        "model": {"neurons": [2, 128, 128, 128, 128, 1], "omega_0": 30.,
                  "mean_kmps": 3., "std_kmps": 1.},
        "acquisition": {"num_shots": 13, "source_x_indices": None, "source_start_index": 20,
                        "source_spacing_index": 20, "source_end_margin": 10,
                        "source_depth_index": 1, "receiver_depth_index": 2, "receiver_stride": 1},
        "forward": {"dt_s": .0019, "nt": 1000, "frequency_hz": 8., "npad": 15,
                    "order": 2, "free_surface": True, "pml_reflection": 1e-6},
        "training": {"epochs": 400, "learning_rate": 1e-4,
                     "checkpoint_interval": 50, "shot_batch_size": None},
    }


def _positive(value, label, integer=False, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} must be {'nonnegative' if allow_zero else 'positive'}")


def validate_config(config):
    template = default_config()
    if not isinstance(config, dict) or set(config) != set(template):
        raise ValueError("Configuration must contain exactly the stage-1 sections")
    for name in ("data", "initialization", "model", "acquisition", "forward", "training"):
        allowed_optional = {"training": {"shots_per_update", "full_eval_interval", "optimizer", "frequency_schedule",
                                         "lr_schedule", "time_gain", "report_estimates"},
                            "model": {"fourier", "trainable_omega", "low_rank"},
                            "initialization": {"pretrain"}}.get(name, set())
        if (not isinstance(config[name], dict) or not set(template[name]).issubset(config[name])
                or set(config[name]) - set(template[name]) - allowed_optional):
            raise ValueError(f"Unknown/missing stage-1 settings in {name}")
    if config["parameterization"] not in ("residual", "absolute"):
        raise ValueError("parameterization must be residual or absolute")
    name = config["experiment_name"]
    if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("experiment_name must be a nonempty filename-safe label")
    _positive(config["seed"], "seed", integer=True, allow_zero=True)
    data, init, model, acq, fd, training = (config[x] for x in ("data", "initialization", "model", "acquisition", "forward", "training"))
    if not isinstance(data["model_file"], str) or not data["model_file"]:
        raise ValueError("model_file must be a nonempty path")
    if data["model_units"] not in ("m/s", "km/s") or data["csv_header"] not in ("legacy", "none"):
        raise ValueError("Declare model_units and csv_header explicitly")
    _positive(data["downsample"], "downsample", integer=True)
    _positive(data["grid_spacing_m"], "grid_spacing_m")
    if data["crop_shape"] is not None:
        if not isinstance(data["crop_shape"], list) or len(data["crop_shape"]) != 2:
            raise ValueError("crop_shape must be [nz,nx] or null")
        for length in data["crop_shape"]:
            _positive(length, "crop_shape", integer=True)
            if length < 3:
                raise ValueError("Each crop axis must have at least three cells")
    _positive(init["sigma"], "sigma", allow_zero=True)
    _positive(init["gaussian_truncate"], "gaussian_truncate")
    if init["gaussian_mode"] not in ("reflect", "nearest", "mirror", "wrap", "constant"):
        raise ValueError("Unsupported Gaussian boundary mode")
    pretrain = init.get("pretrain")
    if pretrain is not None:
        if not isinstance(pretrain, dict) or set(pretrain) != {"updates", "learning_rate"}:
            raise ValueError("initialization.pretrain must be null or contain exactly updates and learning_rate")
        _positive(pretrain["updates"], "pretrain.updates", integer=True)
        _positive(pretrain["learning_rate"], "pretrain.learning_rate")
    neurons, low_rank = model["neurons"], model.get("low_rank")
    validate_low_rank(low_rank)
    if low_rank is not None:
        # The factor networks replace the 2-D SIREN; a leftover neurons list would be silently unused.
        if neurons is not None:
            raise ValueError("model.neurons must be null when model.low_rank is set")
        if (config["parameterization"] == "residual" and low_rank.get("core_init", "zero") != "zero"
                and init.get("pretrain") is None):
            raise ValueError("A residual low-rank model needs core_init zero so that step 0 equals the background "
                             "(a random core is allowed with pretraining, which zeroes it afterwards)")
        hidden_layers = len(low_rank["hidden"])
    else:
        if not isinstance(neurons, list) or len(neurons) < 3 or neurons[0] != 2 or neurons[-1] != 1:
            raise ValueError("neurons must describe an original SIREN with 2 inputs and 1 output")
        for n in neurons:
            _positive(n, "neurons", integer=True)
        hidden_layers = len(neurons) - 2
    for key in ("omega_0", "mean_kmps", "std_kmps"):
        _positive(model[key], key)
    validate_fourier(model.get("fourier"))
    if model.get("fourier") is not None and hidden_layers < 2:
        raise ValueError("Fourier reparameterization needs at least one hidden-to-hidden layer")
    validate_trainable_omega(model.get("trainable_omega"))
    validate_optimizer(training.get("optimizer"))
    if optimizer_name(config) == "muon" and hidden_layers < 2:
        raise ValueError("Muon needs at least one hidden-to-hidden layer")
    for key in ("num_shots", "source_spacing_index", "receiver_stride"):
        _positive(acq[key], key, integer=True)
    for key in ("source_start_index", "source_end_margin", "source_depth_index", "receiver_depth_index"):
        _positive(acq[key], key, integer=True, allow_zero=True)
    if acq["source_x_indices"] is not None:
        if not isinstance(acq["source_x_indices"], list) or len(acq["source_x_indices"]) != acq["num_shots"]:
            raise ValueError("source_x_indices must match num_shots")
        for x in acq["source_x_indices"]:
            _positive(x, "source_x_indices", integer=True, allow_zero=True)
        if len(set(acq["source_x_indices"])) != len(acq["source_x_indices"]):
            raise ValueError("Source locations must be unique")
    for key in ("dt_s", "frequency_hz", "pml_reflection"):
        _positive(fd[key], key)
    for key in ("nt", "npad"):
        _positive(fd[key], key, integer=True)
    if fd["order"] != 2 or type(fd["order"]) is not int:
        raise ValueError("Stage 1 preserves the original second-order FD")
    if type(fd["free_surface"]) is not bool or fd["pml_reflection"] >= 1:
        raise ValueError("Invalid free_surface or PML reflection setting")
    if fd["frequency_hz"] >= .5 / fd["dt_s"]:
        raise ValueError("Source frequency must be below the temporal Nyquist frequency")
    validate_frequency_schedule(training.get("frequency_schedule"), fd["dt_s"])
    validate_lr_schedule(training.get("lr_schedule"))
    validate_time_gain(training.get("time_gain"))
    validate_report_estimates(training.get("report_estimates"))
    for key in ("epochs", "checkpoint_interval"):
        _positive(training[key], key, integer=True)
    _positive(training["learning_rate"], "learning_rate")
    if training["shot_batch_size"] is not None:
        _positive(training["shot_batch_size"], "shot_batch_size", integer=True)
    if training.get("shots_per_update") is not None:
        _positive(training["shots_per_update"], "shots_per_update", integer=True)
        if training["shots_per_update"] > acq["num_shots"]:
            raise ValueError("shots_per_update cannot exceed num_shots")
    if "full_eval_interval" in training:
        _positive(training["full_eval_interval"], "full_eval_interval", integer=True)
    return config


class VelocityParameterization(torch.nn.Module):
    """Original SIREN (or a low-rank factorized residual) with fixed background; only the network is trainable."""

    def __init__(self, initial_mps, dx_m, neurons, omega_0=30., mean_kmps=3.,
                 std_kmps=1., parameterization="residual", fourier=None, trainable_omega=False, low_rank=None,
                 zero_residual_head=True):
        super().__init__()
        initial = torch.as_tensor(initial_mps, dtype=torch.float32).detach().clone()
        if initial.ndim != 2 or not torch.isfinite(initial).all() or initial.min() <= 0:
            raise ValueError("initial_mps must be a finite positive [nz,nx] model")
        if parameterization not in ("residual", "absolute"):
            raise ValueError("Unknown velocity parameterization")
        _positive(dx_m, "dx_m")
        _positive(std_kmps, "std_kmps")
        self.register_buffer("fixed_init_mps", initial[None])
        self.std_kmps, self.mean_kmps = std_kmps, mean_kmps
        self.parameterization = parameterization
        if low_rank is not None:
            # Residual: zero core C, so step 0 equals the background, as the zero last layer does below.
            # Absolute (LR-IFWI paper): m = mean + std * output_gain * U C V^T in km/s.
            from experiments.low_rank_modules import LowRankResidual
            self.net = None
            self.low_rank = LowRankResidual(*initial.shape, dx_m, low_rank, omega_0,
                                            fourier=fourier, trainable_omega=trainable_omega)
            self.fourier_layers, self.omega_layers = self.low_rank.fourier_layers, self.low_rank.omega_layers
            return
        # Preserve the original NumPy-float64 construction before its float32 cast.
        x, z = np.meshgrid(np.arange(initial.shape[1]) * dx_m / 1000,
                           np.arange(initial.shape[0]) * dx_m / 1000)
        self.register_buffer("coords", torch.from_numpy(np.stack((x, z), axis=-1).astype(np.float32))[None])
        self.net = IRN(neuron=list(neurons), omega_0=omega_0, bias=True,
                       activation="sine", dropout=False, outermost_linear=True)
        self.low_rank = None
        # Replace hidden layers after IRN init, so seed-matched first/last layers are unchanged.
        self.fourier_layers = []
        if fourier is not None:
            from experiments.fourier_modules import reparameterize_irn
            self.fourier_layers = reparameterize_irn(self.net, fourier)
        # Wrap after FR so FR init still sees omega_0; step 0 equals the fixed-omega network.
        self.omega_layers = []
        if trainable_omega:
            from experiments.trainable_omega import wrap_irn_omega
            self.omega_layers = wrap_irn_omega(self.net)
        # A pretrained residual model keeps the random head for pretraining and zeroes it afterwards.
        if parameterization == "residual" and zero_residual_head:
            self.zero_output_head()

    def zero_output_head(self):
        """Zero the last linear layer (2-D) or the core C (low-rank): the residual becomes exactly 0."""
        with torch.no_grad():
            if self.low_rank is not None:
                self.low_rank.core.zero_()
            else:
                self.net.linear[-1].weight.zero_()
                self.net.linear[-1].bias.zero_()

    def raw_output(self):
        if self.low_rank is not None:
            return self.low_rank()[None]
        raw, _ = self.net(self.coords)
        return raw.squeeze(-1)

    def absolute_velocity(self):
        """(raw * std + mean) * 1000: the absolute-parameterization velocity of the current network."""
        return (self.raw_output() * self.std_kmps + self.mean_kmps) * 1000

    def forward(self):
        raw = self.raw_output()
        if self.parameterization == "residual":
            return self.fixed_init_mps + raw * (1000 * self.std_kmps)
        return (raw * self.std_kmps + self.mean_kmps) * 1000


def model_from_config(initial_mps, config):
    """VelocityParameterization exactly as the runner builds it from a validated config."""
    model = config["model"]
    return VelocityParameterization(initial_mps, config["data"]["grid_spacing_m"], model["neurons"],
        model["omega_0"], model["mean_kmps"], model["std_kmps"], config["parameterization"],
        fourier=model.get("fourier"), trainable_omega=model.get("trainable_omega") is not None,
        low_rank=model.get("low_rank"), zero_residual_head=config["initialization"].get("pretrain") is None)


def require_finite(value, label):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {label}; original outputs were not masked")


def check_velocity(v_mps, config):
    require_finite(v_mps, "physical velocity")
    if v_mps.ndim != 3 or v_mps.shape[0] != 1 or v_mps.min() <= 0:
        raise ValueError("Velocity must be positive m/s in shape [1,nz,nx]")
    courant = float(v_mps.detach().max()) * config["forward"]["dt_s"] / config["data"]["grid_spacing_m"]
    if courant >= 1 / math.sqrt(2):
        raise ValueError(f"Live order-2 CFL violation: vmax*dt/dx={courant:.6g} >= 1/sqrt(2); no clipping was applied")


def build_forward_problem(truth_mps, config, device):
    """Truth is used only here, under no_grad, to create detached observations."""
    validate_config(config)
    vp = torch.as_tensor(truth_mps, dtype=torch.float32, device=device)[None]
    check_velocity(vp, config)
    nz, nx = truth_mps.shape
    acq, fd = config["acquisition"], config["forward"]
    if min(nz, nx) < 3 or max(acq["source_depth_index"], acq["receiver_depth_index"]) >= nz:
        raise ValueError("Source/receiver depths must lie inside the model")
    if acq["source_x_indices"] is None:
        candidates = np.arange(acq["source_start_index"], nx - acq["source_end_margin"], acq["source_spacing_index"], dtype=np.int64)
        if len(candidates) < acq["num_shots"]:
            raise ValueError("Too few source positions for num_shots; reduce shots or specify source_x_indices")
        selected = np.linspace(0, len(candidates) - 1, acq["num_shots"]).round().astype(int)
        sources = candidates[selected]
    else:
        sources = np.asarray(acq["source_x_indices"], dtype=np.int64)
    if sources.min() < 0 or sources.max() >= nx:
        raise ValueError("Source x positions are outside the model")
    receivers = np.arange(0, nx, acq["receiver_stride"], dtype=np.int64)
    xs = torch.as_tensor(sources, dtype=torch.long)[None]
    zs = torch.full_like(xs, acq["source_depth_index"])
    xr = torch.as_tensor(receivers, dtype=torch.long)[None, None].repeat(1, len(sources), 1)
    zr = torch.full_like(xr, acq["receiver_depth_index"])
    geom = {"nz": nz, "nx": nx, "xs": xs, "zs": zs, "xr": xr, "zr": zr}
    wavelet = wGenerator(fd["dt_s"] * torch.arange(fd["nt"], dtype=torch.float32), fd["frequency_hz"]).ricker().to(device)
    batch = config["training"]["shot_batch_size"] or len(sources)
    propagators, observed = [], []
    for first in range(0, len(sources), batch):
        last = min(first + batch, len(sources))
        batch_geom = {"nz": nz, "nx": nx, "xs": xs[:, first:last], "zs": zs[:, first:last],
                      "xr": xr[:, first:last], "zr": zr[:, first:last]}
        solver = rnn2D(**batch_geom, dz=config["data"]["grid_spacing_m"], dt=fd["dt_s"],
            npad=fd["npad"], order=2, vmax=float(vp.max()), log_para=fd["pml_reflection"],
            freeSurface=fd["free_surface"], dtype=torch.float32, device=device).to(device)
        with torch.no_grad():
            prediction = _simulate(solver, vp, wavelet)
        observed.append(prediction.detach())
        propagators.append({"solver": solver, "first": first, "last": last})
    return {"propagators": propagators, "observed": torch.cat(observed, dim=1), "wavelet": wavelet,
            "geometry_tensors": geom, "config": copy.deepcopy(config),
            "geometry": {"nz": nz, "nx": nx, "axis_order": ["z", "x"],
                         "source_x_indices": sources.tolist(), "source_z_index": acq["source_depth_index"],
                         "receiver_x_indices": receivers.tolist(), "receiver_z_index": acq["receiver_depth_index"],
                         "num_shots": len(sources), "num_receivers": len(receivers), "nt": fd["nt"]}}


def _simulate(solver, v_mps, wavelet):
    outputs = solver(vmodel=v_mps, segment_wavelet=wavelet, option=0)
    for value in outputs:
        require_finite(value, "reference FD wavefield/records")
    return outputs[2]


def waveform_mse(prediction, observations, total_shots=None):
    """The original sum/B/S/T/R MSE; batch contributions use global S."""
    if prediction.shape != observations.shape or prediction.ndim != 4:
        raise ValueError("Waveforms must have matching [B,S,T,R] shapes")
    b, s, t, r = observations.shape
    return (prediction - observations).square().sum() / b / (total_shots or s) / t / r


def training_objective(prediction, observations, total_shots, max_hz, problem, power=None):
    """Waveform MSE; for a frequency-continuation stage and/or a late-arrival time gain t^power,
    the energy-normalized MSE of the low-passed and gained records."""
    if max_hz is None and power is None:
        return waveform_mse(prediction, observations, total_shots)
    dt_s = problem["config"]["forward"]["dt_s"]
    prediction, observations = lowpass(prediction, max_hz, dt_s), lowpass(observations, max_hz, dt_s)
    if power is None:
        return problem["frequency_scales"][float(max_hz)] * waveform_mse(prediction, observations, total_shots)
    scale = problem["gain_scales"][(None if max_hz is None else float(max_hz), float(power))]
    return scale * waveform_mse(time_gain(prediction, power, dt_s), time_gain(observations, power, dt_s), total_shots)


def train_update(model, problem, optimizer, max_hz=None, power=None, *, prior=None, update=None):
    """One optimizer update, with all shots or a uniform subset and exact accumulation.

    All shot batches are accumulated before the single step, so Muon orthogonalizes the
    summed gradient (orthogonalizing per batch and adding would be a different method).
    In a low-pass stage (max_hz) and/or with a time gain (power) the step follows that
    objective; data_mse stays the plain full-band waveform MSE of the same forward pass,
    comparable with every other run.
    """
    if prior is not None and (type(update) is not int or update < 1):
        raise ValueError("Prior guidance requires a one-based update number")
    selected_indices = None
    if problem["config"]["training"].get("shots_per_update") is not None:
        from experiments.residual_ifwi_stochastic import sample_shot_indices, select_shot_problem
        selected_indices = sample_shot_indices(problem["observed"].shape[1], problem["config"]["training"]["shots_per_update"])
        problem = select_shot_problem(problem, selected_indices)
    optimizer.zero_grad(set_to_none=True)
    before = [p.detach().clone() for p in model.parameters()]
    loss_value, objective_value, velocity_before = 0., 0., None
    for item in problem["propagators"]:
        velocity = model()
        check_velocity(velocity, problem["config"])
        if velocity_before is None:
            velocity_before = velocity.detach().cpu().numpy()[0].copy()
        predicted = _simulate(item["solver"], velocity, problem["wavelet"])
        observations = problem["observed"][:, item["first"]:item["last"]]
        if max_hz is None and power is None:
            loss = waveform_mse(predicted, observations, problem["observed"].shape[1])
            require_finite(loss, "waveform MSE")
            loss.backward()
        else:
            with torch.no_grad():
                loss = waveform_mse(predicted, observations, problem["observed"].shape[1])
            objective = training_objective(predicted, observations, problem["observed"].shape[1], max_hz, problem, power)
            require_finite(objective, "low-pass/gained waveform objective")
            objective.backward()
            objective_value += float(objective.detach())
            del objective
        loss_value += float(loss.detach())
        # No field graph is retained between batches; every batch uses the same theta.
        del loss, predicted, velocity
    # One frozen-prior injection after every shot batch, never once per microbatch.
    prior_report = prior.apply(model, update) if prior is not None else None
    gradients = [p.grad for p in model.parameters()]
    if any(g is None for g in gradients):
        raise RuntimeError("A SIREN parameter is disconnected from waveform loss")
    for grad in gradients:
        require_finite(grad, "SIREN parameter gradient")
    grad_norm = float(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt())
    hidden_before = effective_hidden_weights(model)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    try:
        optimizer.step()
        for parameter in model.parameters():
            require_finite(parameter, "updated SIREN parameter")
        with torch.no_grad():
            check_velocity(model(), problem["config"])
            change = float(torch.stack([(p - old).square().sum() for p, old in zip(model.parameters(), before)]).sum().sqrt())
    except BaseException:
        # Abort an invalid/interrupted update without leaving a partly advanced checkpoint.
        with torch.no_grad():
            for parameter, old in zip(model.parameters(), before):
                parameter.copy_(old)
        optimizer.load_state_dict(optimizer_before)
        raise
    report = {"data_mse": loss_value, "gradient_norm": grad_norm,
              "parameter_update_norm": change, "velocity_before_mps": velocity_before}
    if prior_report is not None:
        report["diffusion_prior"] = prior_report
    if max_hz is not None:
        report.update(max_hz=float(max_hz), objective=objective_value)
    if power is not None:
        report.update(time_gain_power=float(power), objective=objective_value)
    # Stable ranks of the effective hidden weights after the step and of this step's update
    # (None for a zero matrix, e.g. hidden updates while a residual head is still zero).
    hidden_after = effective_hidden_weights(model)
    report["hidden_weight_srank"] = [stable_rank(w) for w in hidden_after]
    report["hidden_update_srank"] = [stable_rank(a - b) for a, b in zip(hidden_after, hidden_before)]
    report["hidden_update_fro"] = [float((a - b).double().norm()) for a, b in zip(hidden_after, hidden_before)]
    omegas = omega_values(model)
    if omegas:
        report["omega_after"] = omegas
    if selected_indices is not None:
        report.update(shot_indices=selected_indices, source_x_indices=problem["geometry"]["source_x_indices"],
                      loss_scope="sampled_shots_mean")
    return report


@torch.no_grad()
def evaluate_data_mse(model, problem):
    velocity = model()
    check_velocity(velocity, problem["config"])
    total = 0.
    for item in problem["propagators"]:
        prediction = _simulate(item["solver"], velocity, problem["wavelet"])
        loss = waveform_mse(prediction, problem["observed"][:, item["first"]:item["last"]], problem["observed"].shape[1])
        require_finite(loss, "evaluated waveform MSE")
        total += float(loss)
    return total


@torch.no_grad()
def evaluate_velocity_mse(velocity_mps, problem):
    """All-shot full-band waveform MSE of a [nz,nx] velocity array (m/s)."""
    velocity = torch.as_tensor(np.asarray(velocity_mps, dtype=np.float32), device=problem["observed"].device)[None]
    check_velocity(velocity, problem["config"])
    total = 0.
    for item in problem["propagators"]:
        prediction = _simulate(item["solver"], velocity, problem["wavelet"])
        total += float(waveform_mse(prediction, problem["observed"][:, item["first"]:item["last"]], problem["observed"].shape[1]))
    return total


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def original_source_hashes():
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    names = ("ifwi_modules.py", "rnn_fd.py", "generator.py", "plot_functions.py")
    result = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    for name, digest in result.items():
        if digest != manifest[name]:
            raise ValueError(f"Original core identity differs from source_manifest: {name}")
    return result


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _resume_configuration(config):
    result = copy.deepcopy(config)
    result["training"].pop("epochs")
    result["training"].pop("checkpoint_interval")
    result["training"].pop("full_eval_interval", None)
    result["training"].setdefault("shots_per_update", None)
    result["training"].setdefault("frequency_schedule", None)
    result["training"].setdefault("lr_schedule", None)
    result["training"].setdefault("time_gain", None)
    result["training"].setdefault("report_estimates", None)
    result["model"].setdefault("fourier", None)
    result["model"].setdefault("trainable_omega", None)
    result["model"].setdefault("low_rank", None)
    result["initialization"].setdefault("pretrain", None)
    result["training"]["optimizer"] = result["training"].get("optimizer") or {"name": "adam"}
    return result


def _save_checkpoint(path, model, optimizer, config, completed, history, provenance, sources, full_evaluation_history=None,
                     estimates=None):
    payload = {"schema": SCHEMA, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "config": config, "completed_updates": completed, "history": history,
               "provenance": provenance, "original_source_hashes": sources,
               "fixed_init_units": "m/s", "rng": _rng_state()}
    if full_evaluation_history is not None:
        payload["full_evaluation_history"] = full_evaluation_history
    if estimates is not None:
        payload["estimates"] = estimates.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def pretrain_to_initial(model, background_mps, setting, trainable_omega, mean_kmps, std_kmps, emit=print):
    """LR-IFWI Stage 1 (Chen et al. 2025, Sec. II-E): fit the network to the initial model.

    Loss mean((m_hat - (m_init/1000 - mean)/std)^2) on the standardized output, Adam with
    setting['learning_rate']; the paper gives neither updates nor lr, so configs follow the
    original IFWI pretrain script (pretrain_marmousi.py: Adam 1e-4, 5000 updates, keep the
    lowest-loss state). Loss is evaluated at 0..updates completed updates. Truth is not used.
    The loss always uses the absolute-parameterization output (raw * std + mean), so a residual
    model pretrains exactly as its absolute counterpart; it then zeroes its output head (last
    layer or core C), starting Stage 2 exactly at the background with pretrained features.
    Returns (history, selected_update) and leaves the selected state loaded.
    """
    target = (torch.as_tensor(background_mps, dtype=torch.float32, device=model.fixed_init_mps.device) / 1000
              - mean_kmps) / std_kmps
    optimizer = torch.optim.Adam(parameter_groups(model, setting["learning_rate"], trainable_omega))
    best_loss, best_update, best_state, history = math.inf, 0, None, []
    for update in range(setting["updates"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model.absolute_velocity()[0] / 1000 - mean_kmps) / std_kmps - target).square().mean()
        require_finite(loss, "pretraining loss")
        value = float(loss.detach())
        history.append({"completed_updates": update, "normalized_mse": value,
                        "rmse_vs_initial_mps": math.sqrt(value) * 1000 * std_kmps})
        if value < best_loss:
            best_loss, best_update, best_state = value, update, copy.deepcopy(model.state_dict())
        if update % 500 == 0 or update == setting["updates"]:
            emit(f"Pretraining {update}/{setting['updates']}: RMSE to initial model {history[-1]['rmse_vs_initial_mps']:.3f} m/s")
        if update < setting["updates"]:
            loss.backward()
            optimizer.step()
    model.load_state_dict(best_state)
    if model.parameterization == "residual":
        model.zero_output_head()
    return history, best_update


def run_experiment(config, output_parent, device="cpu", resume=None, save_figures=True):
    config = copy.deepcopy(config)
    validate_config(config)
    stochastic = config["training"].get("shots_per_update") is not None
    full_eval_interval = config["training"].get("full_eval_interval", config["training"]["checkpoint_interval"])
    sources = original_source_hashes()
    fourier = config["model"].get("fourier")
    trainable_omega = config["model"].get("trainable_omega")
    low_rank = config["model"].get("low_rank")
    pretrain = config["initialization"].get("pretrain")
    schedule = config["training"].get("frequency_schedule")
    lr_setting = config["training"].get("lr_schedule")
    gain_setting = config["training"].get("time_gain")
    estimates_setting = config["training"].get("report_estimates")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    truth, background, provenance = load_velocity_data(config, ROOT)
    saved = None
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("schema") != SCHEMA or _resume_configuration(saved["config"]) != _resume_configuration(config):
            raise ValueError("Resume checkpoint schema/configuration differs")
        if saved["original_source_hashes"] != sources or saved["provenance"] != provenance:
            raise ValueError("Resume source/model/initializer provenance differs")
        if (saved["completed_updates"] > config["training"]["epochs"]
                or (not stochastic and saved["completed_updates"] == config["training"]["epochs"])):
            raise ValueError("Requested total epochs must exceed completed updates")
    label = f"{config['parameterization']}_seed{config['seed']}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out = Path(output_parent).resolve() / label
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    _json(out / "config.json", config)
    _json(out / "initial_model_provenance.json", provenance)
    _json(out / "original_source_hashes.json", sources)
    _json(out / "status.json", {"state": "preparing", "pid": os.getpid(), "completed_updates": 0})
    _json(out / "comparison_notes.json", {
        "parameterization": config["parameterization"],
        "initialization": (("absolute " + ("SIREN" if low_rank is None else "low-rank U C V^T") + ", mean/std km/s, "
                            + ("random" if pretrain is None else
                               f"pretrained on the Gaussian initial model ({pretrain['updates']} Adam updates, lr {pretrain['learning_rate']}, lowest-loss state)"))
                           if config["parameterization"] != "residual"
                           else "fixed sampled-grid Gaussian + " + ("zero last layer" if low_rank is None else "zero low-rank core C")
                           + ("" if pretrain is None else
                              f" after pretraining the network on the Gaussian initial model ({pretrain['updates']} Adam updates, "
                              f"lr {pretrain['learning_rate']}, lowest-loss state)")),
        "controlled_components": ["original IRN architecture" if fourier is None and low_rank is None
                                  else "original IRN sine layers" + ("" if low_rank is None else " inside both 1-D factor networks"),
                                  "reference FD", "observations", "waveform MSE",
                                  "Adam" if optimizer_name(config) == "adam" else
                                  "Muon on hidden matrices (FR: coefficients lamb) + Adam on inputs, heads, biases, core and omega",
                                  "acquisition", "update budget"],
        "optimizer": config["training"].get("optimizer") or {"name": "adam"},
        "changes_vs_original_random": (["fixed spatial background", "zero final-layer initialization", "residual parameterization"] if config["parameterization"] == "residual" else [])
                                      + ([] if fourier is None else [f"Fourier reparameterized hidden layers (W = Lambda B, lambda_init={fourier['lambda_init']})"])
                                      + ([] if trainable_omega is None else [f"per-layer trainable sine omega (init omega_0, Adam lr {trainable_omega['learning_rate']})"])
                                      + ([] if low_rank is None else [
                                          f"low-rank {config['parameterization']} U C V^T (rank_z={low_rank['rank_z']}, rank_x={low_rank['rank_x']}, "
                                          f"1-D factor IRNs hidden={low_rank['hidden']}, coord_scale={low_rank['coord_scale']}, "
                                          f"coordinates={low_rank.get('coordinates', 'km')}, core_init={low_rank.get('core_init', 'zero')}, "
                                          f"output_gain={low_rank['output_gain']})"])
                                      + ([] if schedule is None else [
                                          "frequency continuation: energy-normalized zero-phase low-pass misfit stages "
                                          + ", ".join(f"<= {s['max_hz']:g} Hz x {s['updates']}" for s in schedule[:-1])
                                          + ", then the full band"])
                                      + ([] if lr_setting is None else [
                                          f"learning-rate factor per update: warmup {lr_setting['warmup_updates']} updates per stage, "
                                          f"cosine decay to x{lr_setting['final_factor']:g} "
                                          f"over updates {lr_setting['decay_start']}-{lr_setting['decay_start'] + lr_setting['decay_updates'] - 1}"])
                                      + ([] if gain_setting is None else [
                                          f"late-arrival time gain (t/t_end)^{gain_setting['power']:g} on the misfit from update {gain_setting['start_update']} "
                                          "(energy-normalized; data_mse stays the plain full-band MSE)"]),
        **({} if schedule is None else {"frequency_schedule": schedule}),
        **({} if lr_setting is None else {"lr_schedule": lr_setting}),
        **({} if gain_setting is None else {"time_gain": gain_setting}),
        **({} if estimates_setting is None else {"report_estimates": estimates_setting,
                                                 "report_estimates_note": "EMA and lowest-misfit velocities are extra truth-free estimates; training and the last iterate are unchanged"}),
        "attribution": "Residual vs random absolute changes initialization and parameterization jointly; it does not isolate residual parameterization.",
        "historical_baseline_budget": 4001, "this_run_budget": config["training"]["epochs"],
        "runner_differences": "No unused TV coordinate derivatives, no NaN/Inf masking; original clipping effectively inactive; explicit final postupdate waveform evaluation.",
        "training_target": "detached synthetic waveforms only; truth velocity is never a SIREN training label",
    })
    if stochastic:
        runner_names = ("residual_ifwi.py", "experiments/residual_ifwi_experiment.py",
                        "experiments/residual_ifwi_stochastic.py", "experiments/residual_ifwi_data.py",
                        "experiments/residual_ifwi_report.py", "experiments/fourier_modules.py",
                        "experiments/trainable_omega.py", "experiments/low_rank_modules.py",
                        "experiments/muon_optimizer.py", "experiments/frequency_continuation.py",
                        "experiments/lr_schedule.py", "experiments/training_options.py")
        _json(out / "experiment_source_hashes.json", {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in runner_names})
        _json(out / "sampling_protocol.json", {
            "sampling": "uniform without replacement within each update; redraw independently next update",
            "rng": "global NumPy MT19937, full state saved in checkpoint",
            "shots_per_update": config["training"]["shots_per_update"],
            "available_shots": config["acquisition"]["num_shots"],
            "shot_microbatch_size": config["training"]["shot_batch_size"],
            "full_eval_interval": full_eval_interval,
            "training_objective": "mean waveform MSE over selected shots, all time samples and receivers",
            "time_segmentation": False,
            "dt_s": config["forward"]["dt_s"], "nt": config["forward"]["nt"],
            "record_sample_span_s": (config["forward"]["nt"] - 1) * config["forward"]["dt_s"],
            "record_nt_times_dt_s": config["forward"]["nt"] * config["forward"]["dt_s"],
            "common_window_samples": min(1000, config["forward"]["nt"]),
            "common_window_nt_times_dt_s": min(1000, config["forward"]["nt"]) * config["forward"]["dt_s"],
            "comparison": "Acquisition, record length, sampling and update budget changed jointly. Shared 13-shot first-1000-sample evaluation supports the previous experiment; no isolated causal attribution.",
        })
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    _json(out / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device), "git_head": head,
        "torch_threads": torch.get_num_threads(), "deterministic_cudnn": True,
        "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None})
    started = time.perf_counter()
    history, full_history, completed = [], [], 0
    model = optimizer = problem = estimates = None
    log = (out / "progress.log").open("w", encoding="utf-8", buffering=1)

    def emit(message):
        print(message, flush=True)
        print(message, file=log, flush=True)

    try:
        _seed(config["seed"])
        emit(f"Preparing reference FD: grid={truth.shape}, shots={config['acquisition']['num_shots']}, nt={config['forward']['nt']}, output={out}")
        preparation_started = time.perf_counter()
        problem = build_forward_problem(truth, config, device)
        if gain_setting is not None:
            cutoffs = [None] + [float(s["max_hz"]) for s in (schedule or [])[:-1]]
            power = float(gain_setting["power"])
            problem["gain_scales"] = {(c, power): gain_scale(problem["observed"], c, power, config["forward"]["dt_s"])
                                      for c in cutoffs}
        if schedule is not None:
            problem["frequency_scales"] = energy_scales(problem["observed"], schedule, config["forward"]["dt_s"])
            _json(out / "frequency_schedule.json", {
                "stages": [{"first_update": first, "last_update": last, "max_hz": cutoff,
                            "energy_scale": problem["frequency_scales"][cutoff] if cutoff is not None else 1.}
                           for first, last, cutoff in stage_boundaries(schedule)],
                "filter": "zero-phase Butterworth amplitude response 1/(1+(f/max_hz)^12) on predicted and observed records, time axis zero-padded to 2 nt",
                "objective": "energy_scale * waveform MSE of the filtered records; energy_scale = observed energy / filtered observed energy (all shots)",
                "logged": "data_mse is the full-band waveform MSE of the same forward pass; objective is the stage objective"})
        cfg_model = config["model"]
        model = model_from_config(background, config).to(device)
        if pretrain is not None and not saved:
            # LR-IFWI Stage 1; a resumed run restores the trained weights instead.
            pretrain_history, best_update = pretrain_to_initial(model, background, pretrain, trainable_omega,
                                                                cfg_model["mean_kmps"], cfg_model["std_kmps"], emit)
            with (out / "pretraining_history.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(pretrain_history[0]))
                writer.writeheader()
                writer.writerows(pretrain_history)
            with torch.no_grad():
                pretrained = model().cpu().numpy()[0]
            _json(out / "pretraining.json", {"updates": pretrain["updates"], "learning_rate": pretrain["learning_rate"],
                "selected_update": best_update, "selection": "lowest pretraining loss (original IFWI pretrain script)",
                "target": "Gaussian initial model, (v_init/1000 - mean_kmps)/std_kmps", "loss": "mean squared error, normalized units",
                "fit_rmse_vs_initial_mps": pretrain_history[best_update]["rmse_vs_initial_mps"],
                "output_head_zeroed_after_pretraining": config["parameterization"] == "residual",
                "rmse_vs_initial_mps": float(np.sqrt(np.mean((pretrained - background) ** 2))),
                "logged_only_rmse_vs_truth_mps": float(np.sqrt(np.mean((pretrained - truth) ** 2)))})
        optimizer = build_optimizer(model, config)
        # Configured learning rates, read before any saved (already scaled) state is loaded.
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        estimates = IterateEstimates(estimates_setting) if estimates_setting is not None else None
        if saved:
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            if estimates is not None:
                estimates.load_state_dict(saved["estimates"])
            completed, history = saved["completed_updates"], saved["history"]
            full_history = saved.get("full_evaluation_history", [])
            _restore_rng(saved["rng"])
            del saved
        resumed_from_updates = completed
        with torch.no_grad():
            run_initial = model().cpu().numpy()[0].copy()
        if stochastic:
            from experiments.residual_ifwi_stochastic import evaluate_windows
            initial_losses = evaluate_windows(model, problem)
        else:
            initial_losses = {"data_mse": evaluate_data_mse(model, problem)}
        initial_mse = initial_losses["data_mse"]
        initial_metrics = {**velocity_metrics(run_initial, truth), **initial_losses}
        if stochastic:
            initial_full_entry = {"evaluated_updates": completed, "update": completed,
                                  **initial_metrics, "loss_scope": "all_available_shots", "timing": "postupdate_evaluation"}
            if not full_history or full_history[-1]["evaluated_updates"] != completed:
                full_history.append(initial_full_entry)
            _json(out / "full_evaluation_history.json", full_history)
            if estimates is not None:
                estimates.observe_evaluation(initial_full_entry["data_mse"], completed, run_initial)
        background_metrics = velocity_metrics(background, truth)
        for name, array in (("true_velocity", truth), ("initial_velocity", background), ("run_start_velocity", run_initial)):
            np.save(out / f"{name}.npy", array)
        np.save(out / "observed.npy", problem["observed"].cpu().numpy())
        np.save(out / "wavelet.npy", problem["wavelet"].cpu().numpy())
        _json(out / "acquisition.json", problem["geometry"])
        _json(out / "initial_checks.json", {"velocity_units": "m/s", "residual_unit_scale_mps": 1000 * cfg_model["std_kmps"],
            "mean_applied_to_residual": False, "velocity_shape": [1, *truth.shape],
            **({"coordinates_shape": list(model.coords.shape), "coordinates_order": ["x", "z"], "coordinates_units": "km"}
               if low_rank is None else
               {"coordinates_shape": {"z": list(model.low_rank.z_coords.shape), "x": list(model.low_rank.x_coords.shape)},
                "coordinates_order": ["z", "x"],
                "coordinates_units": f"{low_rank.get('coordinates', 'km')} x coord_scale {low_rank['coord_scale']}",
                "low_rank_core_zero_at_start": bool(model.low_rank.core.detach().count_nonzero().item() == 0)}),
            "waveform_shape": list(problem["observed"].shape), "observations_detached": not problem["observed"].requires_grad,
            "fixed_initializer_requires_grad": model.fixed_init_mps.requires_grad,
            "max_residual_at_run_start_mps": float(np.max(np.abs(run_initial - background))),
            "resumed": resume is not None, "cfl": float(run_initial.max()) * config["forward"]["dt_s"] / config["data"]["grid_spacing_m"],
            "trainable_parameters": sum(p.numel() for p in model.parameters()), "initial_metrics": initial_metrics})
        # A complete initial checkpoint also makes preparation/first-step failures resumable.
        if stochastic:
            _save_checkpoint(out / "checkpoint.pt", model, optimizer, config, completed, history, provenance, sources, full_history,
                             estimates)
        preparation_seconds = time.perf_counter() - preparation_started
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        training_started = time.perf_counter()
        history_file = (out / "history.jsonl").open("w", encoding="utf-8", buffering=1)
        for entry in history:
            history_file.write(json.dumps(entry, allow_nan=False) + "\n")
        try:
            for update in range(completed + 1, config["training"]["epochs"] + 1):
                step_started = time.perf_counter()
                model_before, optimizer_before = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
                rng_before = _rng_state()
                prior_completed, prior_history_length = completed, len(history)
                max_hz, power = stage_cutoff(schedule, update), time_power(gain_setting, update)
                if schedule is not None and (update == resumed_from_updates + 1 or max_hz != stage_cutoff(schedule, update - 1)):
                    emit(f"Frequency stage from update {update}: " + ("full band" if max_hz is None else f"low-pass <= {max_hz:g} Hz"))
                if power is not None and (update == resumed_from_updates + 1 or time_power(gain_setting, update - 1) is None):
                    emit(f"Time gain (t/t_end)^{power:g} on the misfit from update {update}")
                try:
                    if lr_setting is not None:
                        factor = lr_factor(lr_setting, schedule, update)
                        for group, base in zip(optimizer.param_groups, base_lrs):
                            group["lr"] = base * factor
                    diagnostics = train_update(model, problem, optimizer, max_hz, power)
                    velocity_before = diagnostics.pop("velocity_before_mps")
                    entry = {"update": update, "evaluated_updates": update - 1, **diagnostics,
                             **velocity_metrics(velocity_before, truth), "step_seconds": time.perf_counter() - step_started}
                    if lr_setting is not None:
                        entry["lr_factor"] = factor
                    history.append(entry)
                    completed = update
                except BaseException:
                    model.load_state_dict(model_before)
                    optimizer.load_state_dict(optimizer_before)
                    _restore_rng(rng_before)
                    completed = prior_completed
                    del history[prior_history_length:]
                    raise
                if estimates is not None:
                    if not stochastic:          # sampled-shot MSEs are not comparable across updates
                        estimates.observe_evaluation(entry["data_mse"], update - 1, velocity_before)
                    with torch.no_grad():
                        estimates.observe_update(update, model().cpu().numpy()[0])
                # Logging failures after this commit still save a consistent advanced state.
                history_file.write(json.dumps(entry, allow_nan=False) + "\n")
                _json(out / "status.json", {"state": "running", "pid": os.getpid(), "completed_updates": completed,
                    "total_updates": config["training"]["epochs"], "last_training_mse": entry["data_mse"],
                    "last_loss_evaluated_updates": completed - 1, "elapsed_training_seconds": time.perf_counter() - training_started})
                emit(f"Update {update}/{config['training']['epochs']}: MSE={entry['data_mse']:.7g} RMSE(before)={entry['rmse_mps']:.3f} m/s grad={entry['gradient_norm']:.4g} step={entry['step_seconds']:.2f}s"
                     + (f" objective(<={entry['max_hz']:g}Hz)={entry['objective']:.7g}" if "max_hz" in entry else "")
                     + (f" gain=t^{entry['time_gain_power']:g}" if "time_gain_power" in entry else "")
                     + (f" objective={entry['objective']:.7g}" if "objective" in entry and "max_hz" not in entry else "")
                     + (f" lr_factor={entry['lr_factor']:.4g}" if "lr_factor" in entry else "")
                     + (f" omega={[round(w, 3) for w in entry['omega_after']]}" if "omega_after" in entry else ""))
                if stochastic and (update % full_eval_interval == 0 or update == config["training"]["epochs"]):
                    full_losses = evaluate_windows(model, problem)
                    with torch.no_grad():
                        evaluated_velocity = model().cpu().numpy()[0]
                    full_entry = {"evaluated_updates": completed, "update": completed,
                                  **full_losses, **velocity_metrics(evaluated_velocity, truth),
                                  "loss_scope": "all_available_shots", "timing": "postupdate_evaluation"}
                    full_history.append(full_entry)
                    _json(out / "full_evaluation_history.json", full_history)
                    if estimates is not None:
                        estimates.observe_evaluation(full_entry["data_mse"], completed, evaluated_velocity)
                    emit(f"Full evaluation {completed}: all-shot MSE={full_entry['data_mse']:.7g} RMSE={full_entry['rmse_mps']:.3f} m/s SSIM={full_entry['ssim']:.5f}")
                if update % config["training"]["checkpoint_interval"] == 0 or update == config["training"]["epochs"]:
                    _save_checkpoint(out / "checkpoint.pt", model, optimizer, config, completed, history, provenance, sources, full_history if stochastic else None,
                                     estimates)
                    _save_checkpoint(out / "checkpoints" / f"update_{update:04d}.pt", model, optimizer, config, completed, history, provenance, sources, full_history if stochastic else None,
                                     estimates)
                    with torch.no_grad():
                        snapshot = model().cpu().numpy()[0]
                    np.save(out / "checkpoints" / f"velocity_{update:04d}.npy", snapshot)
        finally:
            history_file.close()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - training_started
        with torch.no_grad():
            final = model().cpu().numpy()[0].copy()
        final_losses = ({key: full_history[-1][key] for key in ("data_mse", "common_1p9s_data_mse", "baseline_13shots_1p9s_data_mse") if key in full_history[-1]}
                        if stochastic else {"data_mse": evaluate_data_mse(model, problem)})
        final_mse = final_losses["data_mse"]
        final_metrics = {**velocity_metrics(final, truth), **final_losses}
        final_entry = {"evaluated_updates": completed, "update": completed, **final_metrics, "timing": "postupdate_endpoint"}
        plotted_history = full_history if stochastic else [*history, final_entry]
        np.save(out / "final_velocity.npy", final)
        np.save(out / "final_residual.npy", final - background)
        _json(out / "history.json", plotted_history)
        if stochastic:
            _json(out / "sampled_training_history.json", history)
            _json(out / "full_evaluation_history.json", full_history)
        columns = ["update", "evaluated_updates", "data_mse", "rmse_mps", "mae_mps", "ssim", "gradient_norm", "parameter_update_norm", "step_seconds"]
        with (out / "loss.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(plotted_history)
        if stochastic:
            with (out / "sampled_loss.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns + ["shot_indices", "source_x_indices"], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(history)
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if str(device).startswith("cuda") else 0.
        estimates_metrics = None
        if estimates is not None:
            # Extra truth-free estimates; "final" (the last iterate) stays the reported result.
            estimates.observe_evaluation(final_mse, completed, final)
            estimates_metrics = {"setting": estimates_setting}
            if estimates.ema is not None:
                ema = estimates.ema.astype(np.float32)
                np.save(out / "final_velocity_ema.npy", ema)
                estimates_metrics["ema"] = {"averaged_updates": estimates.ema_updates, **velocity_metrics(ema, truth),
                                            "data_mse": evaluate_velocity_mse(ema, problem)}
            lowest = estimates.best_velocity.astype(np.float32)
            np.save(out / "lowest_misfit_velocity.npy", lowest)
            estimates_metrics["lowest_misfit"] = {
                "completed_updates": estimates.best_updates, "data_mse": estimates.best_mse, **velocity_metrics(lowest, truth),
                "scope": ("all-shot evaluations (sampled-shot training MSEs excluded)" if stochastic
                          else "every pre-update all-shot evaluation and the final post-update evaluation")}
            for name in ("ema", "lowest_misfit"):
                if name in estimates_metrics:
                    emit(f"Estimate {name}: RMSE={estimates_metrics[name]['rmse_mps']:.3f} m/s SSIM={estimates_metrics[name]['ssim']:.5f} "
                         f"MSE={estimates_metrics[name]['data_mse']:.7g}")
        metrics = {"initial": initial_metrics, "final": final_metrics, "completed_updates": completed,
            **({} if estimates_metrics is None else {"estimates": estimates_metrics}),
            "fixed_background": background_metrics, "initial_metric_field": "run_start_velocity.npy",
            "resumed_from_updates": resumed_from_updates,
            "metric_velocity_units": "m/s", "ssim_data_range_mps": max(float(truth.max() - truth.min()), 1. if truth.max() == truth.min() else 0.),
            "loss_timing": ("Sampled training rows evaluate before each update; full evaluation rows and final metrics evaluate the saved postupdate model."
                            if stochastic else "Training rows evaluate before each update; final data_mse re-evaluates the saved postupdate model."),
            "timing": {"preparation_seconds": preparation_seconds, "training_seconds": training_seconds,
                       "total_seconds_before_plots": time.perf_counter() - started, "peak_cuda_allocated_gib": peak},
            "selection": "last completed update, no truth-based selection"}
        _json(out / "metrics.json", metrics)
        if save_figures:
            save_plots(out, truth, background, final, plotted_history, config["data"]["grid_spacing_m"])
            if stochastic:
                save_sampling_plot(out, history, full_history, config["training"]["shots_per_update"], config["acquisition"]["num_shots"])
        _json(out / "status.json", {"state": "completed", "completed_updates": completed, "output": str(out)})
        emit(f"Completed: RMSE {initial_metrics['rmse_mps']:.3f} -> {final_metrics['rmse_mps']:.3f} m/s, MSE {initial_mse:.7g} -> {final_mse:.7g}; training={training_seconds:.1f}s")
        return out
    except BaseException as error:
        if model is not None and optimizer is not None:
            _save_checkpoint(out / "interrupted.pt", model, optimizer, config, completed, history, provenance, sources, full_history if stochastic else None,
                             estimates)
        _json(out / "status.json", {"state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "completed_updates": completed, "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        log.close()

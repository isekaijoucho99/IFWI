"""Calibrate low_rank.output_gain so the first Adam step moves velocity as far as the reference.

Both models start exactly at v_init and see the same first-step gradient G = dL/dv
at v_init. G is the all-shot mean waveform MSE gradient, i.e. the expectation of
the stochastic 8-shot update; computing it costs about num_shots/shots_per_update
training updates of FD. Each model then takes one Adam step on dL/dtheta = (dv/dtheta)^T G
from the runner's exact initialization (same seed and RNG sequence), and the
velocity change is measured in m/s.

Adam's first step is lr * g / (|g| + eps), i.e. about lr * sign(g). In the low-rank
model only C moves on step 1, so dv = gain * U dC V^T is linear in output_gain up to
eps. The recommendation rescales the configured gain by the RMS ratio and the script
re-measures with it. This matches the first step only; later steps are not matched.
In both models that first step is nearly a constant shift (sign steps on sine features
with large constant parts; ~99 % of the energy in a full-grid Marmousi check), so the
match is of overall step size, not of spatial structure.

    python -m experiments.low_rank_calibration --config experiments/configs/residual_lr_ifwi_baseline1000.yaml \\
        --reference-config experiments/configs/residual_ifwi_baseline1000.yaml --device cuda:0 \\
        --output outputs/low_rank_calibration.json

Runs no training and writes only the JSON report.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from experiments.residual_ifwi_data import load_velocity_data
from experiments.residual_ifwi_experiment import (ROOT, _restore_rng, _rng_state, _seed, _simulate,
    build_forward_problem, model_from_config, require_finite, validate_config, waveform_mse)
from experiments.trainable_omega import parameter_groups

SHARED_SECTIONS = ("parameterization", "seed", "data", "initialization", "acquisition", "forward", "training")


def check_comparable(config, reference):
    """Only experiment_name and the model section may differ."""
    validate_config(config)
    validate_config(reference)
    if config["model"].get("low_rank") is None:
        raise ValueError("--config must set model.low_rank")
    differing = [name for name in SHARED_SECTIONS if config[name] != reference[name]]
    if differing:
        raise ValueError(f"Calibration needs identical non-model settings; these differ: {differing}")


def velocity_gradient(background, problem):
    """dL/dv at the background for the all-shot mean waveform MSE; returns (G, loss)."""
    velocity = torch.as_tensor(background, dtype=torch.float32,
                               device=problem["observed"].device)[None].requires_grad_(True)
    total, shots = 0., problem["observed"].shape[1]
    for item in problem["propagators"]:
        predicted = _simulate(item["solver"], velocity, problem["wavelet"])
        loss = waveform_mse(predicted, problem["observed"][:, item["first"]:item["last"]], shots)
        require_finite(loss, "waveform MSE")
        loss.backward()
        total += float(loss.detach())
        del loss, predicted
    require_finite(velocity.grad, "velocity gradient")
    return velocity.grad.detach(), total


def first_step_change(config, background, gradient, rng_state, device):
    """Velocity change (m/s, [nz,nx]) after one Adam step of the runner-initialized model."""
    _restore_rng(rng_state)
    model = model_from_config(background, config).to(device)
    optimizer = torch.optim.Adam(parameter_groups(model, config["training"]["learning_rate"],
                                                  config["model"].get("trainable_omega")))
    velocity = model()
    start = velocity.detach().clone()
    if not torch.equal(start, model.fixed_init_mps):
        raise ValueError("Model does not start at the background")
    optimizer.zero_grad(set_to_none=True)
    velocity.backward(gradient)
    optimizer.step()
    with torch.no_grad():
        return (model() - start)[0].cpu().double().numpy()


def summarize(change):
    return {"rms_mps": float(np.sqrt(np.mean(change ** 2))), "max_abs_mps": float(np.abs(change).max())}


def calibrate(config, reference, device="cpu"):
    check_comparable(config, reference)
    truth, background, _ = load_velocity_data(config, ROOT)
    _seed(config["seed"])
    problem = build_forward_problem(truth, config, device)
    # The runner builds its model right after the forward problem; replay that RNG state for each model.
    rng_state = _rng_state()
    gradient, loss = velocity_gradient(background, problem)
    gain = config["model"]["low_rank"]["output_gain"]
    reference_step = summarize(first_step_change(reference, background, gradient, rng_state, device))
    low_rank_step = summarize(first_step_change(config, background, gradient, rng_state, device))
    if low_rank_step["rms_mps"] == 0:
        raise ValueError("Low-rank first step does not change the velocity; C received no gradient")
    recommended = gain * reference_step["rms_mps"] / low_rank_step["rms_mps"]
    check = copy.deepcopy(config)
    check["model"]["low_rank"]["output_gain"] = recommended
    checked_step = summarize(first_step_change(check, background, gradient, rng_state, device))
    return {"config_experiment_name": config["experiment_name"],
            "reference_experiment_name": reference["experiment_name"],
            "gradient": "all-shot mean waveform MSE at v_init", "initial_all_shot_mse": loss,
            "velocity_gradient_rms": float(gradient.square().mean().sqrt()),
            "learning_rate": config["training"]["learning_rate"],
            "reference_first_step": reference_step,
            "low_rank_first_step": {"output_gain": gain, **low_rank_step},
            "recommended_output_gain": recommended,
            "recommended_first_step": {"output_gain": recommended, **checked_step,
                                       "rms_ratio_to_reference": checked_step["rms_mps"] / reference_step["rms_mps"]}}


def _load(path):
    import yaml
    with Path(path).open(encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Low-rank config to calibrate")
    parser.add_argument("--reference-config", type=Path, required=True, help="2-D SIREN config to match")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--crop", nargs=2, type=int, metavar=("NZ", "NX"), help="Smoke test only; applied to both")
    parser.add_argument("--shots", type=int, help="Smoke test only; applied to both")
    parser.add_argument("--nt", type=int, help="Smoke test only; applied to both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, reference = _load(args.config), _load(args.reference_config)
    for cfg in (config, reference):
        if args.crop:
            cfg["data"]["crop_shape"] = list(args.crop)
        if args.shots:
            cfg["acquisition"].update(num_shots=args.shots, source_x_indices=None)
            if cfg["training"].get("shots_per_update") is not None:
                cfg["training"]["shots_per_update"] = min(cfg["training"]["shots_per_update"], args.shots)
        if args.nt:
            cfg["forward"]["nt"] = args.nt
    torch.set_num_threads(args.threads)
    result = calibrate(config, reference, args.device)
    result["smoke_overrides"] = {"crop": args.crop, "shots": args.shots, "nt": args.nt}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    ref, low, rec = result["reference_first_step"], result["low_rank_first_step"], result["recommended_first_step"]
    print(f"reference first step: RMS {ref['rms_mps']:.4g} m/s, max {ref['max_abs_mps']:.4g} m/s")
    print(f"low-rank gain {low['output_gain']:.4g}: RMS {low['rms_mps']:.4g} m/s, max {low['max_abs_mps']:.4g} m/s")
    print(f"recommended output_gain {rec['output_gain']:.6g}: RMS {rec['rms_mps']:.4g} m/s "
          f"(x{rec['rms_ratio_to_reference']:.4f} of reference), max {rec['max_abs_mps']:.4g} m/s")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

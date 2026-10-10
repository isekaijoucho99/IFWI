"""Uniform shot sampling around the unchanged original finite-difference solver.

Microbatches cover all selected shots at a fixed parameter value. Time records
are never segmented: the reference PML states must remain continuous.
"""
import copy

import numpy as np
import torch

from rnn_fd import rnn2D


def sample_shot_indices(num_shots, shots_per_update):
    for value in (num_shots, shots_per_update):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Shot counts must be positive integers")
    if shots_per_update > num_shots:
        raise ValueError("Cannot sample more distinct shots than are available")
    # Existing checkpoint RNG serialization includes this global NumPy state.
    return np.random.choice(num_shots, shots_per_update, replace=False).tolist()


def select_shot_problem(problem, indices):
    """Gather geometry and observations in the same supplied shot order."""
    count = problem["observed"].shape[1]
    if not indices or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= count for i in indices):
        raise ValueError("Selected shot indices must be nonempty and inside the dataset")
    if len(set(indices)) != len(indices):
        raise ValueError("Selected shots must be unique")
    config = copy.deepcopy(problem["config"])
    config["training"]["shots_per_update"] = None
    geometry = problem["geometry_tensors"]
    selected = {"nz": geometry["nz"], "nx": geometry["nx"]}
    selected.update({name: geometry[name][:, indices] for name in ("xs", "zs", "xr", "zr")})
    device = problem["observed"].device
    fd = config["forward"]
    vmax = problem["propagators"][0]["solver"].fd.vmax
    batch_size = config["training"]["shot_batch_size"] or len(indices)
    propagators = []
    for first in range(0, len(indices), batch_size):
        last = min(first + batch_size, len(indices))
        batch_geometry = {"nz": selected["nz"], "nx": selected["nx"]}
        batch_geometry.update({name: selected[name][:, first:last] for name in ("xs", "zs", "xr", "zr")})
        solver = rnn2D(**batch_geometry, dz=config["data"]["grid_spacing_m"], dt=fd["dt_s"],
            npad=fd["npad"], order=2, vmax=vmax, log_para=fd["pml_reflection"],
            freeSurface=fd["free_surface"], dtype=torch.float32, device=device).to(device)
        propagators.append({"solver": solver, "first": first, "last": last})
    metadata = copy.deepcopy(problem["geometry"])
    metadata.update(num_shots=len(indices), source_x_indices=selected["xs"][0].tolist())
    return {**problem, "config": config, "geometry_tensors": selected,
            "geometry": metadata, "propagators": propagators,
            "observed": problem["observed"][:, indices]}


@torch.no_grad()
def evaluate_windows(model, problem):
    """One complete forward pass; short/shared windows are evaluation only."""
    from experiments.residual_ifwi_experiment import _simulate, check_velocity, require_finite, waveform_mse

    velocity = model()
    check_velocity(velocity, problem["config"])
    total_shots = problem["observed"].shape[1]
    common_nt = min(1000, problem["observed"].shape[2])
    sources = problem["geometry"]["source_x_indices"]
    baseline_sources = list(range(20, 261, 20))
    has_baseline = all(source in sources for source in baseline_sources)
    baseline_indices = {sources.index(source) for source in baseline_sources} if has_baseline else set()
    result = {"data_mse": 0., "common_1p9s_data_mse": 0.}
    if has_baseline:
        result["baseline_13shots_1p9s_data_mse"] = 0.
    for item in problem["propagators"]:
        predicted = _simulate(item["solver"], velocity, problem["wavelet"])
        observed = problem["observed"][:, item["first"]:item["last"]]
        result["data_mse"] += float(waveform_mse(predicted, observed, total_shots))
        result["common_1p9s_data_mse"] += float(waveform_mse(predicted[:, :, :common_nt], observed[:, :, :common_nt], total_shots))
        shared = [i - item["first"] for i in sorted(baseline_indices) if item["first"] <= i < item["last"]]
        if shared:
            result["baseline_13shots_1p9s_data_mse"] += float(waveform_mse(
                predicted[:, shared, :common_nt], observed[:, shared, :common_nt], len(baseline_sources)))
    for name, value in result.items():
        require_finite(torch.tensor(value), name)
    return result

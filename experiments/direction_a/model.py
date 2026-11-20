"""Parameter-space operators for matched residual velocity networks.

Velocity outputs and cotangents are indexed [z, x]. Flat directions follow
the order of model.parameters(), with each tensor flattened in row-major order.
"""

import copy
import math

import torch
from torch.func import functional_call

from experiments.fourier_modules import FRLinear
from experiments.residual_ifwi_experiment import VelocityParameterization


def make_model(initial_mps, spacing_m, width, seed, representation, device="cpu"):
    """Build the residual network at a zero-head checkpoint in double precision."""
    if representation not in ("siren", "fr"):
        raise ValueError("representation must be 'siren' or 'fr'")
    if type(width) is not int or width < 1:
        raise ValueError("width must be a positive integer")
    fourier = None
    if representation == "fr":
        # Sample one full set of DFT modes on the width-1 distinct grid points.
        # The final harmonic supplies the constant mode of the periodic grid.
        fourier = {"low_freq_num": 0, "high_freq_num": max(1, width - 1),
                   "phi_num": 4, "alpha": 0.01, "lambda_init": "pinv"}

    initial = torch.as_tensor(initial_mps, dtype=torch.float64, device="cpu").detach()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = VelocityParameterization(initial, spacing_m, [2, width, width, 1], fourier=fourier)
    model = model.to(device=device, dtype=torch.float64)
    # The legacy constructor stores float32 fields. Keep the supplied physical
    # background exactly, including values from a double-precision checkpoint.
    model.fixed_init_mps.copy_(initial.unsqueeze(0))
    return model


@torch.no_grad()
def matched_copy(model, scale, representation=None):
    """Change head scaling and optionally collapse FR weights at the same velocity."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if representation not in (None, "siren", "fr"):
        raise ValueError("representation must be 'siren' or 'fr'")
    if representation == "fr" and not model.fourier_layers:
        raise ValueError("Start from an FR checkpoint to obtain an exactly matched pair")
    result = copy.deepcopy(model)
    if representation == "siren":
        for index in result.fourier_layers:
            layer = result.net.linear[index]
            if not isinstance(layer, FRLinear):
                raise TypeError("Expected an FR hidden layer for collapse")
            # The legacy collapse utility creates a float32 weight. Materialize
            # the exact effective double weight and avoid drawing new random weights.
            plain = torch.nn.Linear(layer.in_features, layer.out_features, device="meta",
                                    dtype=layer.lamb.dtype)
            plain.weight = torch.nn.Parameter(layer.weight.detach(), requires_grad=layer.lamb.requires_grad)
            plain.bias = layer.bias
            result.net.linear[index] = plain.train(layer.training)
        result.fourier_layers = []
    result.std_kmps *= scale
    head = result.net.linear[-1]
    head.weight.div_(scale)
    head.bias.div_(scale)
    return result


@torch.no_grad()
def velocity(model):
    """Current physical velocity in m/s, detached from the network."""
    return model()[0]


def _parameter_direction(model, delta):
    params = dict(model.named_parameters())
    ref = next(iter(params.values()))
    delta = torch.as_tensor(delta, device=ref.device, dtype=ref.dtype).detach()
    sizes = [p.numel() for p in params.values()]
    if delta.ndim != 1 or delta.numel() != sum(sizes):
        raise ValueError(f"Expected a flat direction of length {sum(sizes)}")
    chunks = delta.split(sizes)
    return {name: chunk.reshape(p.shape) for (name, p), chunk in zip(params.items(), chunks)}


@torch.enable_grad()
def pullback(model, grad_v):
    """Apply the adjoint velocity Jacobian to a physical image gradient."""
    field = model()[0]
    covector = torch.as_tensor(grad_v, device=field.device, dtype=field.dtype).detach()
    if covector.shape != field.shape:
        raise ValueError(f"Expected a physical gradient of shape {tuple(field.shape)}")
    grads = torch.autograd.grad(field, tuple(model.parameters()), grad_outputs=covector)
    return torch.cat([g.reshape(-1) for g in grads]).detach()


def jvp(model, delta):
    """Apply the physical velocity Jacobian without materializing its dense matrix."""
    direction = _parameter_direction(model, delta)

    def field(*parameters):
        return functional_call(model, dict(zip(direction, parameters)), ())[0]

    # The original IRN enables coordinate gradients internally, which
    # torch.func.jvp disallows. The reverse-over-reverse JVP supports this IRN.
    _, image_delta = torch.autograd.functional.jvp(
        field, tuple(model.parameters()), tuple(direction.values()))
    return image_delta.detach()


@torch.no_grad()
def trial_velocity(model, delta):
    """Evaluate the finite parameter step without changing persistent parameters."""
    direction = _parameter_direction(model, delta)
    candidate = {name: p + direction[name] for name, p in model.named_parameters()}
    return functional_call(model, candidate, ())[0]


@torch.no_grad()
def apply_delta(model, delta):
    """Commit the finite step evaluated by trial_velocity."""
    direction = _parameter_direction(model, delta)
    for name, p in model.named_parameters():
        p.add_(direction[name])

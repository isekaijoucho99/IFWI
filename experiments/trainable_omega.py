"""Per-layer trainable sine frequency for the original IRN.

FR-IFWI (Kang et al., 2025, GJI 244, ggaf404, Sec. 3.2) makes "the frequency
parameter of the activation function in different layers" trainable; the
paper gives no initial value, parameterization or learning rate.

ifwi_modules.IRN is hash-verified and applies sin(omega_0 * layer(x)) with one
fixed omega_0. Each activated layer is wrapped so that it returns
layer(x) * omega / omega_0, giving sin(omega * layer(x)) with omega a trainable
scalar per layer, initialized to omega_0 (so step 0 is bit-identical to the
fixed-omega network). The outermost linear layer has no activation and stays
unwrapped, keeping the residual zero initialization.
"""
from __future__ import annotations

import math

import torch

OMEGA_KEYS = ("learning_rate",)


def validate_trainable_omega(setting):
    if setting is None:
        return
    if not isinstance(setting, dict) or set(setting) != set(OMEGA_KEYS):
        raise ValueError(f"model.trainable_omega must be null or contain exactly {list(OMEGA_KEYS)}")
    lr = setting["learning_rate"]
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("trainable_omega.learning_rate must be positive")


class OmegaScaledLayer(torch.nn.Module):
    """layer(x) * omega / omega_0; inside IRN this is sin(omega * layer(x))."""

    def __init__(self, layer, omega_0):
        super().__init__()
        self.layer, self.omega_0 = layer, float(omega_0)
        self.omega = torch.nn.Parameter(torch.tensor(self.omega_0, device=layer.bias.device))

    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

    def forward(self, x):
        return self.layer(x) * (self.omega / self.omega_0)


def wrap_irn_omega(irn):
    """Wrap every activated IRN layer in place; return their indices."""
    if not irn.outermost_linear:
        raise ValueError("trainable omega expects an IRN with a linear outermost layer")
    for index in range(len(irn.linear) - 1):
        irn.linear[index] = OmegaScaledLayer(irn.linear[index], irn.omega_0)
    return list(range(len(irn.linear) - 1))


def omega_values(module):
    return [float(m.omega.detach()) for m in module.modules() if isinstance(m, OmegaScaledLayer)]


def parameter_groups(module, learning_rate, trainable_omega):
    """One Adam group, or network + omega groups with their own learning rates."""
    if trainable_omega is None:
        return [{"params": list(module.parameters()), "lr": learning_rate}]
    omega = [m.omega for m in module.modules() if isinstance(m, OmegaScaledLayer)]
    ids = {id(p) for p in omega}
    rest = [p for p in module.parameters() if id(p) not in ids]
    return [{"params": rest, "lr": learning_rate},
            {"params": omega, "lr": trainable_omega["learning_rate"]}]

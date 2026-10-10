"""Fourier reparameterized (FR) hidden layers for the original IRN.

W = Lambda @ B with fixed cosine bases B and trainable coefficients Lambda
(Kang et al., 2025, GJI 244, ggaf404, Eq. 5-7; Shi et al., 2024, CVPR).
B follows the authors' public FR-INR ``sin_fr_layer``: low frequencies
(i+1)/L, high frequencies 1..H, P phases 2*pi*p/P, in_features points on
[-T_max/2, T_max/2] with T_max = 2*pi/lowest frequency, all scaled by alpha.

The original ``ifwi_modules.IRN`` is not edited: its hidden ``torch.nn.Linear``
layers are replaced after construction, so IRN.forward still applies
sin(omega_0 * layer(x)), as in FR-INR's ``sin_fr_layer``.

Only hidden-to-hidden layers are reparameterized, as in FR-INR. FR-IFWI Eq. 5
writes W^(n) for every layer but publishes no code; with the FR-INR bases a
2-input first layer has points +-T_max/2 where every basis takes the same value,
so B has rank 1 and x/z become indistinguishable. The last layer stays plain so
the residual zero initialization (start exactly at v_init) is unchanged.

lambda_init="fr_inr" is the authors' initialization (paper-faithful default).
lambda_init="pinv" (ours, ablation) sets lamb @ B = the seed-matched original
IRN weights, so only the training dynamics differ from the baseline.
"""
from __future__ import annotations

import math

import numpy as np
import torch

FOURIER_KEYS = ("high_freq_num", "low_freq_num", "phi_num", "alpha", "lambda_init")
LAMBDA_INITS = ("pinv", "fr_inr")


def default_fourier():
    """FR-INR 2d_image_fitting.py values for sine networks (alpha 0.01 for sin)."""
    return {"high_freq_num": 128, "low_freq_num": 128, "phi_num": 32, "alpha": .01, "lambda_init": "fr_inr"}


def validate_fourier(fourier):
    if fourier is None:
        return
    if not isinstance(fourier, dict) or set(fourier) != set(FOURIER_KEYS):
        raise ValueError(f"model.fourier must be null or contain exactly {list(FOURIER_KEYS)}")
    for key in ("high_freq_num", "low_freq_num"):
        value = fourier[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"fourier.{key} must be a nonnegative integer")
    if fourier["high_freq_num"] + fourier["low_freq_num"] == 0:
        raise ValueError("fourier needs at least one frequency")
    if type(fourier["phi_num"]) is not int or fourier["phi_num"] < 1:
        raise ValueError("fourier.phi_num must be a positive integer")
    alpha = fourier["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("fourier.alpha must be positive")
    if fourier["lambda_init"] not in LAMBDA_INITS:
        raise ValueError(f"fourier.lambda_init must be one of {LAMBDA_INITS}")


def fourier_bases(in_features, high_freq_num, low_freq_num, phi_num, alpha):
    """Return B [(L+H)*P, in_features], ordered low then high frequency, phase fastest."""
    phases = 2 * np.pi * np.arange(phi_num) / phi_num
    low = (np.arange(low_freq_num) + 1) / low_freq_num if low_freq_num else np.zeros(0)
    high = np.arange(high_freq_num, dtype=np.float64) + 1
    frequencies = np.concatenate((low, high))
    t_max = 2 * np.pi / frequencies.min()
    points = np.linspace(-t_max / 2, t_max / 2, in_features)
    bases = np.cos(frequencies[:, None, None] * points[None, None, :] + phases[None, :, None])
    return torch.from_numpy(alpha * bases.reshape(-1, in_features)).float()


class FRLinear(torch.nn.Module):
    """Drop-in replacement for torch.nn.Linear with weight = lamb @ bases."""

    def __init__(self, in_features, out_features, high_freq_num, low_freq_num, phi_num, alpha):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        # Deterministic from the config, so not stored in checkpoints.
        self.register_buffer("bases", fourier_bases(in_features, high_freq_num, low_freq_num, phi_num, alpha),
                             persistent=False)
        self.lamb = torch.nn.Parameter(torch.zeros(out_features, self.bases.shape[0]))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    @property
    def weight(self):
        return self.lamb @ self.bases

    def init_from_linear(self, linear):
        """Minimum-norm lamb with lamb @ bases = W0; keeps the original bias."""
        with torch.no_grad():
            pinv = torch.linalg.pinv(self.bases.double())
            self.lamb.copy_((linear.weight.double() @ pinv).float())
            self.bias.copy_(linear.bias)

    def init_fr_inr(self, omega_0):
        """Authors' FR-INR init: per-basis uniform scaled by sqrt(6/M)/|b_i|/omega_0, zero bias."""
        m = self.bases.shape[0]
        with torch.no_grad():
            bound = np.sqrt(6 / m) / torch.linalg.vector_norm(self.bases, dim=1) / omega_0
            self.lamb.copy_((torch.rand_like(self.lamb) * 2 - 1) * bound[None])
            self.bias.zero_()

    def rank(self):
        return int(torch.linalg.matrix_rank(self.bases.double()))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


def reparameterize_irn(irn, fourier):
    """Replace IRN hidden layers 1..n-2 in place; return their indices."""
    validate_fourier(fourier)
    keys = ("high_freq_num", "low_freq_num", "phi_num", "alpha")
    replaced = []
    for index in range(1, len(irn.linear) - 1):
        linear = irn.linear[index]
        if linear.bias is None:
            raise ValueError("FR reparameterization expects IRN layers with bias")
        layer = FRLinear(linear.in_features, linear.out_features, *(fourier[k] for k in keys))
        layer = layer.to(device=linear.weight.device)
        if layer.rank() < layer.in_features:
            raise ValueError(f"Fourier bases of layer {index} have rank {layer.rank()} < {layer.in_features}")
        if fourier["lambda_init"] == "pinv":
            layer.init_from_linear(linear)
        else:
            layer.init_fr_inr(irn.omega_0)
        irn.linear[index] = layer
        replaced.append(index)
    if not replaced:
        raise ValueError("IRN has no hidden-to-hidden layer to reparameterize")
    return replaced


def merged_weights(irn):
    """Plain weight matrices of every IRN layer (FR layers merged), for export or audits."""
    return [layer.weight.detach().clone() for layer in irn.linear]

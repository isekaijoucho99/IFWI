"""Fourier weight reparameterization for the existing IFWI SIREN.

W = Lambda B; B_ij = alpha cos(omega_i z_j + phi_i).
Kang et al., GJI, doi:10.1093/gji/ggaf404, equations (5)--(7).
Frequency/phase sampling and coefficient initialization follow Shi et al.,
CVPR 2024, and the public FR-INR sin_fr_layer implementation. The GJI paper
alone does not specify all implementation hyperparameters. See docs/fr_ifwi.md.

This layer is LINEAR. The parent IRN applies Sine once, after this layer.
It is not an FFT, a spatial-coordinate encoder, or a trainable basis frequency.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ifwi_modules import IRN

RECIPE_VERSION = "fr_inr_cosine_endpoint_v1"
DEFAULT_FOURIER = {
    "low_freq_num": 64, "high_freq_num": 64, "phi_num": 4,
    "alpha": .01, "target_layers": "hidden_only", "learnable_omega": False,
}


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def validate_fourier_options(settings: dict | None) -> dict:
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("model.fourier must be a mapping")
    settings = {} if settings is None else settings
    unknown = set(settings) - set(DEFAULT_FOURIER)
    if unknown:
        raise ValueError(f"Unknown Fourier settings: {sorted(unknown)}")
    result = {**DEFAULT_FOURIER, **settings}
    for key in ("low_freq_num", "high_freq_num"):
        result[key] = _integer(result[key], key, 0)
    if result["low_freq_num"] + result["high_freq_num"] == 0:
        raise ValueError("At least one Fourier frequency is required")
    result["phi_num"] = _integer(result["phi_num"], "phi_num", 1)
    result["alpha"] = _positive(result["alpha"], "alpha")
    if result["target_layers"] != "hidden_only":
        raise ValueError("Only target_layers=hidden_only is implemented")
    if type(result["learnable_omega"]) is not bool:
        raise ValueError("learnable_omega must be a boolean")
    return result


def validate_architecture(architecture: str, fourier: dict | None) -> dict | None:
    if architecture == "siren":
        if fourier is not None:
            raise ValueError("Fourier options require architecture=fr_siren; they are never ignored")
        return None
    if architecture != "fr_siren":
        raise ValueError("architecture must be siren or fr_siren")
    return validate_fourier_options(fourier)


class FourierLinear(nn.Module):
    """A Linear-compatible weight factorization with a fixed registered buffer.

    B has shape [M, in_features], coeff has shape [out_features, M].
    z samples the largest basis period, including BOTH endpoints, as in FR-INR.
    This entails dependent rows/columns; M >= in_features does not imply full
    column rank. diagnostics() reports the rank instead of silently modifying B.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 low_freq_num: int = 64, high_freq_num: int = 64,
                 phi_num: int = 4, alpha: float = .01, omega0: float = 30.,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = _integer(in_features, "in_features", 2)
        self.out_features = _integer(out_features, "out_features", 1)
        cfg = validate_fourier_options(dict(low_freq_num=low_freq_num,
            high_freq_num=high_freq_num, phi_num=phi_num, alpha=alpha))
        self.omega0 = _positive(omega0, "omega0")
        self.low_freq_num, self.high_freq_num = cfg["low_freq_num"], cfg["high_freq_num"]
        self.phi_num, self.alpha = cfg["phi_num"], cfg["alpha"]
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Construct FourierLinear in float32 or float64")

        # Float64 trigonometry avoids large-angle errors from float32 basis setup.
        # Deterministic construction deliberately consumes NO NumPy RNG draws.
        low = [(i + 1) / self.low_freq_num for i in range(self.low_freq_num)]
        high = [float(i + 1) for i in range(self.high_freq_num)]
        frequencies = torch.tensor(low + high, dtype=torch.float64)
        phases = torch.tensor([2*math.pi*i/self.phi_num for i in range(self.phi_num)], dtype=torch.float64)
        half_period = math.pi / float(frequencies.min())
        points = torch.from_numpy(np.linspace(-half_period, half_period, self.in_features))
        bases = self.alpha * torch.cos(
            frequencies[:, None, None] * points[None, None, :] + phases[None, :, None]
        ).reshape(-1, self.in_features)
        bases = bases.to(device=device, dtype=dtype)
        row_norm = bases.double().norm(dim=1)
        near_zero = self.alpha * math.sqrt(self.in_features) * 1e-10
        if not torch.isfinite(bases).all() or (row_norm <= near_zero).any():
            raise ValueError("Fourier sampling produced degenerate basis rows; change the frequency/phase counts or width")
        self.register_buffer("B", bases)
        self.coeff = nn.Parameter(torch.empty(self.out_features, bases.shape[0], device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(self.out_features, device=device, dtype=dtype))
        # FR-INR Sine initialization: each coefficient column has its own bound.
        bound = (math.sqrt(6/bases.shape[0]) / row_norm / self.omega0).to(dtype=dtype)
        with torch.no_grad():
            # Column-major draws match the order of the reference's column loop.
            draws = torch.empty(bases.shape[0], self.out_features, device=device, dtype=dtype).uniform_(-1, 1)
            self.coeff.copy_(draws.T * bound[None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Recompute W for every call; caching a detached W would disconnect coeff.
        return F.linear(x, self.coeff @ self.B, self.bias)

    @torch.no_grad()
    def diagnostics(self) -> dict:
        b = self.B.detach().cpu()
        singular = torch.linalg.svdvals(b.double())
        tolerance = float(singular[0]) * max(b.shape) * torch.finfo(b.dtype).eps
        rank = int((singular > tolerance).sum())
        condition = (float(singular[0] / singular[-1])
                     if rank == self.in_features else None)
        return {
            "recipe": RECIPE_VERSION, "basis_shape": list(b.shape),
            "coefficient_shape": list(self.coeff.shape), "numerical_rank": rank,
            "rank_tolerance": tolerance, "condition_number": condition,
            "effective_condition_number": float(singular[0] / singular[rank-1]) if rank else None,
            "singular_values": singular.tolist(),
            "basis_sha256": hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest(),
            "basis_dtype": str(b.dtype), "row_norm_min": float(b.double().norm(dim=1).min()),
            "low_freq_num": self.low_freq_num, "high_freq_num": self.high_freq_num,
            "phi_num": self.phi_num, "alpha": self.alpha, "omega0_at_initialization": self.omega0,
            "basis_is_trainable": False,
        }


class FourierIRN(IRN):
    """Keep the original first/output layers; reparameterize intermediate weights.

    Default fixed omega isolates FR from changes to activation frequencies.
    The optional independent experiment learns one omega per activated layer,
    including the first layer, as described (without full implementation details)
    in GJI section 3.2. No extra Sine is applied inside FourierLinear.
    """

    def __init__(self, neuron, omega_0: float = 30., fourier: dict | None = None):
        cfg = validate_fourier_options(fourier)
        omega_0 = _positive(omega_0, "omega_0")
        if not isinstance(neuron, (list, tuple)) or len(neuron) < 4 or neuron[0] != 2 or neuron[-1] != 1:
            raise ValueError("FRIRN needs 2 inputs, >=2 hidden layers and 1 linear output")
        for n in neuron:
            _integer(n, "neuron", 1)
        super().__init__(neuron=list(neuron), omega_0=omega_0, bias=True,
                         activation="sine", dropout=False, outermost_linear=True)
        self.fourier_config = cfg
        layer_cfg = {key: cfg[key] for key in ("low_freq_num", "high_freq_num", "phi_num", "alpha")}
        for index in range(1, len(self.linear)-1):
            old = self.linear[index]
            self.linear[index] = FourierLinear(old.in_features, old.out_features,
                omega0=omega_0, device=old.weight.device, dtype=old.weight.dtype, **layer_cfg)
        if cfg["learnable_omega"]:
            self.activation_omega = nn.Parameter(torch.full((len(self.linear)-1,), omega_0))
        else:
            self.register_parameter("activation_omega", None)

    def forward(self, coords):
        if self.activation_omega is None:
            return super().forward(coords)
        coords = coords.clone().detach().requires_grad_(True)
        feature = coords
        for index, layer in enumerate(self.linear[:-1]):
            feature = torch.sin(self.activation_omega[index] * layer(feature))
        return self.linear[-1](feature), coords

    def diagnostics(self) -> dict:
        return {"recipe": RECIPE_VERSION, **self.fourier_config,
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "activated_layers": len(self.linear)-1,
                "layers": [{"linear_index": index, **layer.diagnostics()}
                           for index, layer in enumerate(self.linear) if isinstance(layer, FourierLinear)]}

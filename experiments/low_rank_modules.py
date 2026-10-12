"""Low-rank residual for residual IFWI (LR-IFWI; Chen et al., 2025, TGRS 63, 5918816, Eq. 4-7).

R(z, x) = output_gain * U C V^T with U = Psi_z(z) [nz, rank_z], V = Psi_x(x) [nx, rank_x]
and a trainable core C [rank_z, rank_x]. Each Psi is an original ``ifwi_modules.IRN``
with one input, ``hidden`` sine layers and a linear output (paper: 3 x 128, sine,
omega_0 = 30; ranks 50 (depth, n1 = 94) and 100 (lateral, n2 = 288)).

Differences from the paper, all for the residual setting:
* The paper's Stage 1 pretraining on m_init is replaced by the fixed background.
* C starts at zero instead of random, so step 0 equals v_init exactly. Only C gets
  a gradient on the first update (dL/dU = G V C^T = 0), as only the last IRN layer
  does in the zero-initialized residual SIREN; U and V start moving afterwards.
  Zeroing U or V instead would leave every gradient at zero forever.
* Inputs are the runner's km coordinates times ``coord_scale``, not integer indices
  (with omega_0 = 30, indices 1..n alias). IRN draws first-layer weights from
  U(-1/in, 1/in): +-1/2 per axis for the 2-input SIREN, +-1 for a 1-input factor.
  coord_scale = 0.5 gives each factor the per-axis first-layer frequency range of
  the 2-input baseline.
* output_gain scales the product; calibrate it with experiments/low_rank_calibration.py.

FR (fourier_modules) and trainable omega (trainable_omega) apply to both factor
networks exactly as to the 2-D IRN: hidden-to-hidden layers and activated layers.
The factor output layers stay plain and unwrapped, and C stays a plain matrix.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ifwi_modules import IRN

LOW_RANK_KEYS = ("rank_z", "rank_x", "hidden", "coord_scale", "output_gain")


def default_low_rank():
    """Paper ranks and factor depth; coord_scale matches the 2-input IRN, gain uncalibrated."""
    return {"rank_z": 50, "rank_x": 100, "hidden": [128, 128, 128], "coord_scale": .5, "output_gain": 1.}


def _positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"low_rank.{label} must be a positive finite number")


def validate_low_rank(setting):
    if setting is None:
        return
    if not isinstance(setting, dict) or set(setting) != set(LOW_RANK_KEYS):
        raise ValueError(f"model.low_rank must be null or contain exactly {list(LOW_RANK_KEYS)}")
    for key in ("rank_z", "rank_x"):
        if type(setting[key]) is not int or setting[key] < 1:
            raise ValueError(f"low_rank.{key} must be a positive integer")
    hidden = setting["hidden"]
    if not isinstance(hidden, list) or not hidden or any(type(n) is not int or n < 1 for n in hidden):
        raise ValueError("low_rank.hidden must be a nonempty list of positive integers")
    for key in ("coord_scale", "output_gain"):
        _positive_number(setting[key], key)


def _axis(n, dx_m, scale):
    # Same float64 km construction as the 2-D runner coordinates, then scaled and cast.
    return torch.from_numpy((np.arange(n) * dx_m / 1000 * scale).astype(np.float32))[:, None]


class LowRankResidual(torch.nn.Module):
    """output_gain * Psi_z(z) @ C @ Psi_x(x).T, shape [nz, nx]."""

    def __init__(self, nz, nx, dx_m, setting, omega_0, fourier=None, trainable_omega=False):
        super().__init__()
        validate_low_rank(setting)
        self.setting = dict(setting)
        self.register_buffer("z_coords", _axis(nz, dx_m, setting["coord_scale"]))
        self.register_buffer("x_coords", _axis(nx, dx_m, setting["coord_scale"]))
        self.net_z = IRN(neuron=[1, *setting["hidden"], setting["rank_z"]], omega_0=omega_0, bias=True,
                         activation="sine", dropout=False, outermost_linear=True)
        self.net_x = IRN(neuron=[1, *setting["hidden"], setting["rank_x"]], omega_0=omega_0, bias=True,
                         activation="sine", dropout=False, outermost_linear=True)
        self.core = torch.nn.Parameter(torch.zeros(setting["rank_z"], setting["rank_x"]))
        self.output_gain = float(setting["output_gain"])
        self.fourier_layers, self.omega_layers = {}, {}
        for name, net in (("z", self.net_z), ("x", self.net_x)):
            if fourier is not None:
                from experiments.fourier_modules import reparameterize_irn
                self.fourier_layers[name] = reparameterize_irn(net, fourier)
            if trainable_omega:
                from experiments.trainable_omega import wrap_irn_omega
                self.omega_layers[name] = wrap_irn_omega(net)

    def factors(self):
        u, _ = self.net_z(self.z_coords)
        v, _ = self.net_x(self.x_coords)
        return u, v

    def forward(self):
        u, v = self.factors()
        return self.output_gain * (u @ self.core @ v.T)

    def named_layers(self):
        """[(label, layer)] in a fixed order: z factor, x factor; the core is not a layer."""
        return ([(f"z{i}", layer) for i, layer in enumerate(self.net_z.linear)]
                + [(f"x{i}", layer) for i, layer in enumerate(self.net_x.linear)])

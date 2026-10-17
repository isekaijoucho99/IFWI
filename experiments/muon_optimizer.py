"""Muon for hidden INR weight matrices, PyTorch Adam for everything else.

McGinnis et al., "Optimizing Rank for High-Fidelity Implicit Neural Representations",
ICML 2026 (PMLR 306), Sec. 3.3 and App. A: hidden-layer weight matrices are updated by
Muon (Jordan et al., 2024), input/output matrices and biases by Adam with
beta = (0.9, 0.999) as in PyTorch; no weight decay; the same lr schedule for both.

Muon follows the reference implementation (github.com/KellerJordan/Muon, MIT License):
EMA momentum (default 0.95) with Nesterov blending, Frobenius normalization, five quintic
Newton-Schulz steps in bfloat16 with coefficients (3.4445, -4.7750, 2.0315), and a
max(1, rows/cols)^0.5 scale. Two deliberate differences: the gradient is not modified in
place (same arithmetic; p.grad stays available for diagnostics), and the auxiliary
optimizer is torch.optim.Adam with PyTorch defaults (the paper's setting) instead of the
reference's hand-written Adam with betas (0.9, 0.95) and eps 1e-10.

Hidden matrices are chosen by position in each original IRN (linear layers 1..n-2), not
by shape: the 1-output head of a 2-D SIREN is a 1 x 128 matrix and must stay in Adam.
FR layers contribute their coefficient matrix ``lamb`` (W = lamb @ bases), so Muon
orthogonalizes the coefficient update; the low-rank core C stays in Adam.
"""
from __future__ import annotations

import math

import torch

from experiments.trainable_omega import OmegaScaledLayer, parameter_groups

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_KEYS = ("name", "learning_rate", "momentum", "nesterov", "ns_steps")


def newton_schulz_orthogonalize(G, steps=5):
    """Approximate the polar factor P Q^T of G = P S Q^T (singular values land near 0.5-1.5)."""
    if G.ndim != 2:
        raise ValueError("Newton-Schulz orthogonalization expects a matrix")
    a, b, c = NS_COEFFICIENTS
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Single-device Muon for 2-D parameters (hidden weight or FR coefficient matrices)."""

    def __init__(self, params, lr, momentum=0.95, nesterov=True, ns_steps=5):
        if not lr > 0 or not 0 <= momentum < 1 or int(ns_steps) != ns_steps or ns_steps < 1:
            raise ValueError("Muon needs lr > 0, 0 <= momentum < 1 and ns_steps >= 1")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=bool(nesterov), ns_steps=int(ns_steps)))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError("Muon only updates matrices")

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buffer = state["momentum_buffer"]
                buffer.lerp_(p.grad, 1 - group["momentum"])
                update = p.grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = newton_schulz_orthogonalize(update, group["ns_steps"]).to(p.dtype)
                update = update * max(1., p.size(0) / p.size(1)) ** .5
                p.add_(update, alpha=-group["lr"])


class MuonWithAdam:
    """The optimizer interface the runner uses, over a Muon part and an Adam part."""

    def __init__(self, muon, adam):
        self.muon, self.adam = muon, adam

    @property
    def param_groups(self):
        return self.adam.param_groups + self.muon.param_groups

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.adam.step()
        self.muon.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state):
        if set(state) != {"muon", "adam"}:
            raise ValueError("Not a Muon + Adam optimizer state")
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])


def validate_optimizer(setting):
    if setting is None:
        return
    if not isinstance(setting, dict) or setting.get("name") not in ("adam", "muon"):
        raise ValueError("training.optimizer must be null, {name: adam} or a muon setting")
    if setting["name"] == "adam":
        # Optional Adam betas/eps (PyTorch defaults when absent); the learning rate stays training.learning_rate.
        if not set(setting) <= {"name", "betas", "eps"}:
            raise ValueError("training.optimizer for adam takes only optional betas and eps (lr is training.learning_rate)")
        betas = setting.get("betas", [.9, .999])
        if (not isinstance(betas, (list, tuple)) or len(betas) != 2
                or any(isinstance(b, bool) or not isinstance(b, (int, float)) or not 0 <= b < 1 for b in betas)):
            raise ValueError("adam betas must be two numbers in [0, 1)")
        eps = setting.get("eps", 1e-8)
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("adam eps must be positive")
        return
    if set(setting) != set(MUON_KEYS):
        raise ValueError(f"training.optimizer for muon must contain exactly {list(MUON_KEYS)}")
    lr, momentum, steps = setting["learning_rate"], setting["momentum"], setting["ns_steps"]
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("muon learning_rate must be positive")
    if isinstance(momentum, bool) or not isinstance(momentum, (int, float)) or not 0 <= momentum < 1:
        raise ValueError("muon momentum must be in [0, 1)")
    if type(setting["nesterov"]) is not bool or type(steps) is not int or steps < 1:
        raise ValueError("muon nesterov must be a bool and ns_steps a positive integer")


def optimizer_name(config):
    return (config["training"].get("optimizer") or {"name": "adam"})["name"]


def irn_networks(model):
    from ifwi_modules import IRN
    return [m for m in model.modules() if isinstance(m, IRN)]


def hidden_layers(model):
    """[(label, layer)] for every hidden-to-hidden layer (unwrapped from trainable omega)."""
    layers = []
    for n, irn in enumerate(irn_networks(model)):
        for index in range(1, len(irn.linear) - 1):
            layer = irn.linear[index]
            layers.append((f"net{n}.{index}", layer.layer if isinstance(layer, OmegaScaledLayer) else layer))
    return layers


def hidden_matrices(model):
    """Trainable matrices handed to Muon: Linear.weight, or FRLinear.lamb (its weight is not a leaf)."""
    from experiments.fourier_modules import FRLinear
    matrices = []
    for _, layer in hidden_layers(model):
        if isinstance(layer, FRLinear):
            matrices.append(layer.lamb)
        elif isinstance(layer, torch.nn.Linear):
            matrices.append(layer.weight)
        else:
            raise TypeError(f"Unsupported hidden layer {type(layer).__name__}")
    if not matrices:
        raise ValueError("Muon needs at least one hidden-to-hidden layer")
    return matrices


def build_optimizer(model, config):
    """Adam exactly as before, or Muon on hidden matrices plus Adam on all other parameters."""
    setting = config["training"].get("optimizer") or {"name": "adam"}
    groups = parameter_groups(model, config["training"]["learning_rate"], config["model"].get("trainable_omega"))
    if setting["name"] == "adam":
        options = {**({"betas": tuple(setting["betas"])} if "betas" in setting else {}),
                   **({"eps": setting["eps"]} if "eps" in setting else {})}
        return torch.optim.Adam(groups, **options)
    hidden = hidden_matrices(model)
    ids = {id(p) for p in hidden}
    auxiliary = [{**g, "params": [p for p in g["params"] if id(p) not in ids]} for g in groups]
    return MuonWithAdam(Muon(hidden, lr=setting["learning_rate"], momentum=setting["momentum"],
                             nesterov=setting["nesterov"], ns_steps=setting["ns_steps"]),
                        torch.optim.Adam([g for g in auxiliary if g["params"]]))


def stable_rank(matrix):
    """||M||_F^2 / ||M||_2^2, or None for a zero matrix (undefined, not 'collapsed')."""
    m = matrix.detach().double()
    spectral = torch.linalg.matrix_norm(m, ord=2)
    return float(m.square().sum() / spectral ** 2) if spectral > 0 else None


@torch.no_grad()
def effective_hidden_weights(model):
    """Effective hidden weights W (FR: lamb @ bases); omega scaling leaves stable rank unchanged."""
    return [layer.weight.detach().clone() for _, layer in hidden_layers(model)]

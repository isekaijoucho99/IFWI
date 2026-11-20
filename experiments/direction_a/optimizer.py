"""Flat parameter updates for the Direction A optimizer comparison."""
from dataclasses import dataclass, field
import math

import torch


@dataclass(frozen=True)
class _State:
    first: torch.Tensor | None = None
    second: torch.Tensor | None = None
    first_age: int = 0
    second_age: int = 0


@dataclass(frozen=True)
class UpdateProposal:
    """Parameter displacements and the candidate state computed from the clipped gradient."""

    delta: torch.Tensor
    current: torch.Tensor
    history: torch.Tensor
    instant: torch.Tensor
    raw_norm: float
    clip_factor: float
    _origin: _State = field(repr=False)
    _candidate: _State = field(repr=False)


class UpdateRule:
    """Stage SGD or Adam updates. Only accept() advances the moment estimates."""

    def __init__(self, method, lr, *, beta1=.9, beta2=.999, eps=1e-8, clip_norm=None):
        if method not in ("sgd", "adam_no_momentum", "adam"):
            raise ValueError(f"Unknown update method {method!r}")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive and finite")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam betas must lie in [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite")
        if clip_norm is not None and (not math.isfinite(clip_norm) or clip_norm <= 0):
            raise ValueError("clip_norm must be positive and finite")
        self.method = method
        self.lr = lr
        self.beta1 = beta1 if method == "adam" else 0.
        self.beta2 = beta2
        self.eps = eps
        self.clip_norm = clip_norm
        self._state = _State()

    @torch.no_grad()
    def propose(self, grad: torch.Tensor, *, restart=False) -> UpdateProposal:
        """Use the accepted state. A restart keeps the second moment and its age."""
        if grad.ndim != 1 or not grad.numel() or not grad.is_floating_point():
            raise ValueError("grad must be a nonempty flat real floating-point tensor")
        state = self._state
        previous_second = state.second
        if previous_second is not None and (
            grad.shape != previous_second.shape or grad.device != previous_second.device
            or grad.dtype != previous_second.dtype
        ):
            raise ValueError("Gradient shape, dtype and device must match the accepted moments")
        raw_norm = torch.linalg.vector_norm(grad).item()
        if not math.isfinite(raw_norm):
            raise ValueError("Gradient norm must be finite")
        # Match the coefficient used by torch.nn.utils.clip_grad_norm_.
        clip_factor = 1. if self.clip_norm is None else min(1., self.clip_norm / (raw_norm + 1e-6))
        gradient = grad * clip_factor if clip_factor < 1. else grad
        if self.method == "sgd":
            delta = gradient * -self.lr
            return UpdateProposal(delta, delta, torch.zeros_like(delta), delta,
                                  raw_norm, clip_factor, state, _State())

        if previous_second is None:
            previous_second = torch.zeros_like(gradient)
        second_age = state.second_age + 1
        second = previous_second.mul(self.beta2).addcmul_(gradient, gradient, value=1 - self.beta2)
        denom = second.sqrt().div_(math.sqrt(1 - self.beta2 ** second_age)).add_(self.eps)
        instant = gradient.div(denom).mul_(-self.lr)

        # Both terms use the same bias-corrected second moment. Only the first
        # moment's bias correction restarts after a momentum reset.
        previous_first = None if restart else state.first
        first_age = 1 if restart else state.first_age + 1
        correction = 1 - self.beta1 ** first_age
        current = instant * ((1 - self.beta1) / correction)
        if previous_first is None or self.beta1 == 0.:
            history = torch.zeros_like(gradient)
            first = gradient.mul(1 - self.beta1)
        else:
            history = previous_first.div(denom).mul_(-self.lr * self.beta1 / correction)
            first = previous_first.lerp(gradient, 1 - self.beta1)
        candidate = _State(first, second, first_age, second_age)
        return UpdateProposal(current + history, current, history, instant,
                              raw_norm, clip_factor, state, candidate)

    def accept(self, proposal: UpdateProposal) -> None:
        """Commit the gradient moments once, independently of the step length applied."""
        if proposal._origin is not self._state:
            raise ValueError("Proposal belongs to a different rule or a superseded state")
        self._state = proposal._candidate

"""Unmodified official DFM Sinkhorn/loss functions (MIT).

Copyright (c) 2026 Drift Flow Matching authors.
Source: merry7cherry/drift-flow-matching at
619663f0e7cbd17e58d98501daaf389d2810077b,
image_generation/src/driftfm/methods/drift_flow_matching.py.
Full license and provenance: third_party/drift_flow_matching/.
Only project imports, timestep/config helpers and method classes were omitted.
"""
from __future__ import annotations
import torch

UPSTREAM_COMMIT = "619663f0e7cbd17e58d98501daaf389d2810077b"


def _expand_marginals(
    marginals: torch.Tensor,
    *,
    expected_size: int,
    leading_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Broadcast Sinkhorn marginals to `leading_shape + [expected_size]`.

    Args:
        marginals: `[expected_size]` or `leading_shape + [expected_size]`.
        expected_size: Required final axis size.
        leading_shape: Broadcasted batch shape.
        device: Target device.
        dtype: Target dtype.
        name: Error-label for validation.

    Returns:
        Expanded marginals with shape `leading_shape + [expected_size]`.
    """
    marginals = marginals.to(device=device, dtype=dtype)
    if marginals.ndim == 1:
        if marginals.shape[0] != expected_size:
            raise ValueError(f"{name} must have shape [{expected_size}], got {tuple(marginals.shape)}")
        view_shape = (1,) * len(leading_shape) + (expected_size,)
        return marginals.reshape(view_shape).expand(*leading_shape, expected_size)
    if marginals.shape[:-1] != leading_shape or marginals.shape[-1] != expected_size:
        raise ValueError(
            f"{name} must have shape {leading_shape + (expected_size,)}, got {tuple(marginals.shape)}"
        )
    return marginals


def _sinkhorn_from_logits(
    logits: torch.Tensor,
    *,
    row_marginals: torch.Tensor,
    col_marginals: torch.Tensor,
    num_iters: int = 20,
    eps: float = 1e-12,
    return_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Solve Sinkhorn transport plans from pairwise logits.

    Args:
        logits: Pairwise scores with shape `[..., N_src, N_tgt]`.
        row_marginals: Source marginals with shape `[N_src]` or `[..., N_src]`.
        col_marginals: Target marginals with shape `[N_tgt]` or `[..., N_tgt]`.
        num_iters: Sinkhorn iterations.
        eps: Numerical clamp.
        return_dtype: Optional output dtype.

    Returns:
        Transport plan with shape `[..., N_src, N_tgt]`.
    """
    if logits.ndim < 2:
        raise ValueError(f"logits must have shape [..., N_src, N_tgt], got {tuple(logits.shape)}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be > 0, got {num_iters}")

    num_source_points, num_target_points = logits.shape[-2], logits.shape[-1]
    orig_dtype = logits.dtype
    work_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    if return_dtype is None:
        return_dtype = orig_dtype

    logits = logits.to(dtype=work_dtype)
    leading_shape = logits.shape[:-2]
    row_marginals = _expand_marginals(
        row_marginals,
        expected_size=num_source_points,
        leading_shape=leading_shape,
        device=logits.device,
        dtype=work_dtype,
        name="row_marginals",
    )
    col_marginals = _expand_marginals(
        col_marginals,
        expected_size=num_target_points,
        leading_shape=leading_shape,
        device=logits.device,
        dtype=work_dtype,
        name="col_marginals",
    )

    log_row_marginals = torch.log(row_marginals.clamp_min(eps))
    log_col_marginals = torch.log(col_marginals.clamp_min(eps))
    log_u = torch.zeros_like(logits[..., :, 0])
    log_v = torch.zeros_like(logits[..., 0, :])

    for _ in range(int(num_iters)):
        log_u = log_row_marginals - torch.logsumexp(logits + log_v.unsqueeze(-2), dim=-1)
        log_v = log_col_marginals - torch.logsumexp(logits + log_u.unsqueeze(-1), dim=-2)

    plan = torch.exp(logits + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).clamp_min(0.0)
    return plan.to(dtype=return_dtype)


def _row_normalize_plan(plan: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a transport plan row-wise.

    Args:
        plan: Transport plan with shape `[..., N_src, N_tgt]`.
        eps: Numerical clamp.

    Returns:
        Row-normalized weights with shape `[..., N_src, N_tgt]`.
    """
    return plan / plan.sum(dim=-1, keepdim=True).clamp_min(eps)


def _compute_sinkhorn_barycentric_projection(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    kernel_temp: float,
    num_iters: int,
    eps: float,
    row_marginals: torch.Tensor | None = None,
    col_marginals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project grouped source points onto Sinkhorn barycenters.

    Args:
        source_points: Source block with shape `[G, N_src, D]`.
        target_points: Target block with shape `[G, N_tgt, D]`.
        kernel_temp: Sinkhorn kernel temperature.
        num_iters: Sinkhorn iterations.
        eps: Numerical clamp.
        row_marginals: Optional source marginals with shape `[N_src]` or `[G, N_src]`.
        col_marginals: Optional target marginals with shape `[N_tgt]` or `[G, N_tgt]`.

    Returns:
        Barycentric projections with shape `[G, N_src, D]`.
    """
    if source_points.ndim != 3 or target_points.ndim != 3:
        raise ValueError("source_points and target_points must have shape [G, N, D]")
    if source_points.shape[0] != target_points.shape[0]:
        raise ValueError("source_points and target_points must share the same G axis")
    if target_points.shape[1] == 0:
        return torch.zeros_like(source_points)

    orig_dtype = source_points.dtype
    work_dtype = torch.float32 if source_points.dtype in (torch.float16, torch.bfloat16) else source_points.dtype
    source_points = source_points.to(dtype=work_dtype)
    target_points = target_points.to(dtype=work_dtype)
    if row_marginals is None:
        row_marginals = torch.full(
            (source_points.shape[1],),
            1.0 / float(source_points.shape[1]),
            device=source_points.device,
            dtype=work_dtype,
        )
    if col_marginals is None:
        col_marginals = torch.full(
            (target_points.shape[1],),
            1.0 / float(target_points.shape[1]),
            device=target_points.device,
            dtype=work_dtype,
        )
    distances = torch.cdist(source_points, target_points)
    logits = -distances / float(kernel_temp)
    plan = _sinkhorn_from_logits(
        logits,
        row_marginals=row_marginals,
        col_marginals=col_marginals,
        num_iters=num_iters,
        eps=eps,
    )
    weights = _row_normalize_plan(plan, eps=eps)
    return torch.matmul(weights, target_points).to(dtype=orig_dtype)


def compute_drift_grouped_sinkhorn(
    gen: torch.Tensor,
    pos: torch.Tensor,
    *,
    kernel_temp_pos: float = 0.05,
    kernel_temp_neg: float = 0.05,
    num_sinkhorn_iters: int = 20,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Same-class attraction minus generated-sample repulsion, shape [G, B, D]."""
    if gen.ndim != 3 or pos.shape != gen.shape:
        raise ValueError("gen and pos must share shape [G, B, D]")
    if kernel_temp_pos <= 0 or kernel_temp_neg <= 0:
        raise ValueError("Sinkhorn kernel temperatures must be positive")
    pos_proj = _compute_sinkhorn_barycentric_projection(
        gen, pos, kernel_temp=kernel_temp_pos, num_iters=num_sinkhorn_iters, eps=eps,
    )
    neg_proj = _compute_sinkhorn_barycentric_projection(
        gen, gen, kernel_temp=kernel_temp_neg, num_iters=num_sinkhorn_iters, eps=eps,
    )
    return pos_proj - neg_proj


def _adaptive_matching_loss(
    predicted_points: torch.Tensor,
    target_points: torch.Tensor,
    norm_eps: float,
    norm_p: float,
) -> torch.Tensor:
    """Compute adaptive squared-error matching loss.

    Args:
        predicted_points: Predictions with shape `[N, *shape]`.
        target_points: Targets with shape `[N, *shape]`.
        norm_eps: Stability constant.
        norm_p: Adaptive weighting exponent.

    Returns:
        Scalar loss tensor.
    """
    terms = (predicted_points - target_points) ** 2
    terms = terms.reshape(terms.shape[0], -1).sum(dim=1)
    adaptive_weight = (terms.detach() + norm_eps) ** norm_p
    return torch.mean(terms / adaptive_weight)

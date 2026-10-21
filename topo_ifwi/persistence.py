"""Superlevel-set persistence of a 2-D anomaly score and the threshold-band loss (proposal 4.2-4.3).

Conventions (fixed once, checked by topo_ifwi/tests/test_persistence.py against scipy labels):

* Filtration: superlevel sets Omega_tau = {q >= tau} on the pixel grid, tau decreasing.
* Foreground connectivity ``fg`` is 4 (pixels sharing an edge) or 8; holes are components of
  the complement {q < tau} under the dual connectivity (8 for fg=4, 4 for fg=8) that do not
  touch the image border. This pair makes Alexander duality hold on the grid.
* A bar (B, D) with B >= D is alive for D < tau <= B, so beta_k(tau) = #{bars: D < tau <= B}.
* H0: the component born at its maximum pixel dies at the merge pixel (elder rule). The global
  maximum is the essential branch: B_e = max q, D_e = 0 by convention (no gradient through D_e).
* H1: computed by duality as H0 of the sublevel filtration of q on the complement with a virtual
  border node that is older than every pixel. A hole is born at the pixel that closes it off from
  the border (B = q at that pixel) and dies at its minimum (D).

Each bar keeps the flat indices of its birth and death pixels, so a torch loss can gather q at
those pixels: the gradient reaches the scores through the critical pixels only (piecewise
differentiable, proposal proposition 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class Diagram:
    shape: tuple
    essential: int                 # flat index of the global maximum
    h0: np.ndarray                 # [n, 2] int (birth_idx, death_idx), finite H0 bars
    h1: np.ndarray                 # [m, 2] int (birth_idx, death_idx), H1 bars

    def values(self, q):
        q = np.asarray(q, dtype=np.float64).ravel()
        return {"essential": float(q[self.essential]),
                "h0": np.stack([q[self.h0[:, 0]], q[self.h0[:, 1]]], 1) if len(self.h0) else np.zeros((0, 2)),
                "h1": np.stack([q[self.h1[:, 0]], q[self.h1[:, 1]]], 1) if len(self.h1) else np.zeros((0, 2))}


def _neighbour_table(shape, offsets):
    nz, nx = shape
    iz, ix = np.divmod(np.arange(nz * nx), nx)
    table = np.full((nz * nx, len(offsets)), -1, dtype=np.int64)
    for k, (dz, dx) in enumerate(offsets):
        jz, jx = iz + dz, ix + dx
        ok = (jz >= 0) & (jz < nz) & (jx >= 0) & (jx < nx)
        table[ok, k] = jz[ok] * nx + jx[ok]
    border = (iz == 0) | (iz == nz - 1) | (ix == 0) | (ix == nx - 1)
    return table, border


def _find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def _sweep(order, table, border, use_border):
    """Union-find sweep in ``order``; returns (pairs, first) where each pair is
    (young_birth_pixel, merge_pixel) and ``first`` is the pixel that started the oldest component.
    With use_border, a virtual node (index n) is the oldest component and touches border pixels."""
    n = len(order)
    parent = np.full(n + 1, -1, dtype=np.int64)
    birth_pos = np.empty(n + 1, dtype=np.int64)    # sweep position of the component's birth pixel
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    if use_border:
        parent[n], birth_pos[n] = n, -1
    pairs = []
    for p in order:
        parent[p], birth_pos[p] = p, pos[p]
        neighbours = [j for j in table[p] if j >= 0 and parent[j] >= 0]
        if use_border and border[p]:
            neighbours.append(n)
        for j in neighbours:
            rp, rj = _find(parent, p), _find(parent, j)
            if rp == rj:
                continue
            old, young = (rp, rj) if birth_pos[rp] < birth_pos[rj] else (rj, rp)
            young_birth = order[birth_pos[young]]
            if young_birth != p:                     # zero-length bars carry no information
                pairs.append((young_birth, p))
            parent[young] = old
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2), order[0]


def superlevel_diagram(q, fg=4):
    """Persistence diagram of {q >= tau}; ties are broken by flat index (stable sort)."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all():
        raise ValueError("q must be a finite 2-D array")
    if fg not in (4, 8):
        raise ValueError("fg must be 4 or 8")
    flat = q.ravel()
    order = np.argsort(-flat, kind="stable")
    table_fg, border = _neighbour_table(q.shape, _N4 if fg == 4 else _N8)
    h0, essential = _sweep(order, table_fg, border, use_border=False)
    # H0 pairs from the descending sweep are (birth=peak, death=merge pixel): already (B, D).
    table_bg, _ = _neighbour_table(q.shape, _N8 if fg == 4 else _N4)
    basins, _ = _sweep(order[::-1], table_bg, border, use_border=True)
    # Ascending sweep pairs are (basin minimum, closing pixel) = (D, B) of the superlevel hole.
    h1 = basins[:, ::-1].copy()
    return Diagram(q.shape, int(essential), h0, h1)


def betti_from_diagram(diagram, q, tau):
    v = diagram.values(q)
    b0 = int(v["essential"] >= tau) + int(((v["h0"][:, 1] < tau) & (tau <= v["h0"][:, 0])).sum())
    b1 = int(((v["h1"][:, 1] < tau) & (tau <= v["h1"][:, 0])).sum())
    return b0, b1


def betti_by_labels(q, tau, fg=4):
    """Independent Betti numbers from connected-component labelling (used as ground truth)."""
    from scipy import ndimage
    s4 = ndimage.generate_binary_structure(2, 1)
    s8 = ndimage.generate_binary_structure(2, 2)
    fg_mask = np.asarray(q) >= tau
    _, b0 = ndimage.label(fg_mask, structure=s4 if fg == 4 else s8)
    labels, nbg = ndimage.label(~fg_mask, structure=s8 if fg == 4 else s4)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    b1 = nbg - len(set(edge.tolist()) - {0})
    return int(b0), int(b1)


def band_terms(q, diagram, a, b, eps, kappa):
    """Torch terms of R_PH (proposal 4.3); q is a 2-D tensor on any device.

    R_PH = [b+eps-B_e]_+^2 + sum_i h_i^2 (1 + kappa l_i^2),
    h_i = [min(B_i, b+eps) - max(D_i, a-eps)]_+,  l_i = B_i - D_i, over finite H0 and H1 bars.
    Returns a dict with the total, the essential term, per-bar overlaps and their kinds.
    """
    if not 0 < a - eps < a < b < b + eps < 1:
        raise ValueError("need 0 < a-eps < a < b < b+eps < 1")
    flat = q.reshape(-1)
    lo, hi = a - eps, b + eps
    essential = torch.relu(hi - flat[diagram.essential]) ** 2
    pairs = np.concatenate([diagram.h0, diagram.h1]) if len(diagram.h0) + len(diagram.h1) else np.zeros((0, 2), int)
    kinds = np.array([0] * len(diagram.h0) + [1] * len(diagram.h1), dtype=np.int64)
    if len(pairs):
        idx = torch.as_tensor(pairs, device=q.device)
        B, D = flat[idx[:, 0]], flat[idx[:, 1]]
        h = torch.relu(torch.minimum(B, torch.full_like(B, hi)) - torch.maximum(D, torch.full_like(D, lo)))
        bars = h ** 2 * (1 + kappa * (B - D) ** 2)
    else:
        h = bars = flat.new_zeros(0)
    return {"total": essential + bars.sum(), "essential": essential, "bars": bars, "overlap": h,
            "pairs": pairs, "kinds": kinds}


def band_loss(q, a, b, eps, kappa, fg=4):
    """R_PH(q) with the diagram recomputed from the current (detached) scores."""
    diagram = superlevel_diagram(q.detach().cpu().double().numpy(), fg=fg)
    return band_terms(q, diagram, a, b, eps, kappa), diagram


def anomaly_score(v, v_ref, dv, temperature):
    """q = sigmoid((v - v_ref - dv) / T), proposal 4.2 (all in m/s)."""
    return torch.sigmoid((v - v_ref - dv) / temperature)

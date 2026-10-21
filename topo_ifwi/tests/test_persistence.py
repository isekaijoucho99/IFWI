"""Phase-A unit tests for the persistence code and the band loss (proposal 3.1, 6.2).

Run: .venv/bin/python -m pytest topo_ifwi/tests -q
"""
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from topo_ifwi.persistence import (band_loss, band_terms, betti_by_labels, betti_from_diagram,
                                   superlevel_diagram)

A, B, EPS = 0.4, 0.6, 0.05


def disk(shape, cz, cx, r):
    z, x = np.mgrid[:shape[0], :shape[1]]
    return ((z - cz) ** 2 + (x - cx) ** 2) <= r ** 2


def field(mask, high=0.95, low=0.05, sigma=1.0):
    return gaussian_filter(np.where(mask, high, low).astype(float), sigma)


@pytest.mark.parametrize("fg", [4, 8])
@pytest.mark.parametrize("seed", range(6))
def test_betti_matches_labels_on_random_fields(fg, seed):
    rng = np.random.default_rng(seed)
    q = gaussian_filter(rng.random((30, 41)), 1.2)
    q = (q - q.min()) / (q.max() - q.min())
    d = superlevel_diagram(q, fg=fg)
    for tau in np.linspace(0.02, 0.98, 25):
        assert betti_from_diagram(d, q, tau) == betti_by_labels(q, tau, fg=fg), tau


SHAPE = (40, 60)
CASES = {
    "single":     (disk(SHAPE, 20, 30, 8), (1, 0)),
    "broken":     (disk(SHAPE, 20, 18, 6) | disk(SHAPE, 20, 42, 6), (2, 0)),
    "holed":      (disk(SHAPE, 20, 30, 12) & ~disk(SHAPE, 20, 30, 5), (1, 1)),
    "false_isle": (disk(SHAPE, 20, 25, 9) | disk(SHAPE, 6, 52, 2), (2, 0)),
    "flat":       (np.zeros(SHAPE, bool), (0, 0)),
}


@pytest.mark.parametrize("name", CASES)
def test_unit_shapes(name):
    mask, expected = CASES[name]
    q = field(mask)
    d = superlevel_diagram(q)
    assert betti_from_diagram(d, q, 0.5) == expected == betti_by_labels(q, 0.5)


def test_single_body_satisfies_band_and_zero_loss_gives_betti():
    """Proposition 1: R_PH = 0 => beta0 = 1 and beta1 = 0 for every tau in [a, b]."""
    q = field(CASES["single"][0])
    terms, d = band_loss(torch.tensor(q), A, B, EPS, kappa=1.0)
    assert float(terms["total"]) == 0.0
    for tau in np.linspace(A, B, 21):
        assert betti_by_labels(q, tau) == (1, 0)


@pytest.mark.parametrize("name", ["broken", "holed", "false_isle", "flat"])
def test_defects_are_penalised(name):
    q = field(CASES[name][0])
    terms, _ = band_loss(torch.tensor(q), A, B, EPS, kappa=1.0)
    assert float(terms["total"]) > 1e-3


def test_constant_above_band_is_feasible_but_shapeless():
    """The zero set is reachable (q > b+eps everywhere), which is why topology alone cannot fix shape."""
    terms, _ = band_loss(torch.full((10, 12), B + 2 * EPS, dtype=torch.float64), A, B, EPS, kappa=1.0)
    assert float(terms["total"]) == 0.0


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    q0 = gaussian_filter(rng.random((16, 20)), 1.0)
    q0 = 0.1 + 0.8 * (q0 - q0.min()) / (q0.max() - q0.min())
    d = superlevel_diagram(q0)                       # fixed pairing (proposition 3: fixed active pairs)
    q = torch.tensor(q0, requires_grad=True)
    band_terms(q, d, A, B, EPS, 1.0)["total"].backward()
    pixels = np.unique(np.concatenate([d.h0.ravel(), d.h1.ravel(), [d.essential]]))
    assert len(pixels) > 0
    for p in pixels[:12]:
        e = np.zeros(q0.size)
        e[p] = 1e-7
        f = lambda s: float(band_terms(torch.tensor(q0 + s * e.reshape(q0.shape)), d, A, B, EPS, 1.0)["total"])
        fd = (f(1) - f(-1)) / 2e-7
        assert q.grad.view(-1)[p].item() == pytest.approx(fd, rel=1e-4, abs=1e-8)
    # Non-critical pixels get no gradient.
    others = np.setdiff1d(np.arange(q0.size), pixels)
    assert torch.count_nonzero(q.grad.view(-1)[others]) == 0


def test_kappa_restores_gradient_on_bars_spanning_the_band():
    """A bar covering [a-eps, b+eps] has constant overlap; only kappa > 0 gives a descent direction,
    lowering the peak (delete) and raising the saddle (connect) at the same time."""
    mask = CASES["broken"][0]
    q0 = field(mask)
    d = superlevel_diagram(q0)
    vals = d.values(q0)["h0"]
    i = int(np.argmax(vals[:, 0] - vals[:, 1]))
    assert vals[i, 0] > B + EPS and vals[i, 1] < A - EPS
    birth, death = d.h0[i]
    for kappa, nonzero in ((0.0, False), (1.0, True)):
        q = torch.tensor(q0, requires_grad=True)
        band_terms(q, d, A, B, EPS, kappa)["total"].backward()
        gb, gd = q.grad.view(-1)[birth].item(), q.grad.view(-1)[death].item()
        if nonzero:
            assert gb > 0 and gd < 0
        else:
            assert gb == 0 and gd == 0

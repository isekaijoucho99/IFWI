import numpy as np
import torch

from g2ifwi.forward import AcousticModeler, ricker


def make(dtype=torch.float64):
    nz, nx, dz, dt, nt = 20, 30, 20.0, 0.002, 160
    return AcousticModeler(nz, nx, dz, dt, ricker(10, dt, nt, dtype=dtype), [8, 20], 1, np.arange(0, nx, 2), 1,
                           npad=8, chunk=37, shot_batch=1, dtype=dtype), nz, nx


def test_gradient_matches_finite_difference():
    m, nz, nx = make()
    torch.manual_seed(0)
    v_true = 2500 + 300 * torch.rand(nz, nx, dtype=torch.float64)
    v0 = torch.full((nz, nx), 2600.0, dtype=torch.float64)
    d = m.simulate_all(v_true)
    _, g = m.loss_and_grad(v0, d)
    for (i, j) in [(5, 10), (12, 20), (3, 4)]:
        e = 1.0
        vp, vm = v0.clone(), v0.clone()
        vp[i, j] += e; vm[i, j] -= e
        lp, _ = m.loss_and_grad(vp, d); lm, _ = m.loss_and_grad(vm, d)
        fd = (lp - lm) / (2 * e)
        assert abs(fd - g[i, j].item()) < 2e-3 * max(abs(fd), 1e-12) + 1e-14, (fd, g[i, j].item())


def test_chunking_invariant():
    m, nz, nx = make()
    v = 2500 + 300 * torch.rand(nz, nx, dtype=torch.float64)
    a = m.simulate_all(v)
    m.chunk = 160
    b = m.simulate_all(v)
    assert torch.allclose(a, b)
    assert a.abs().max() > 0 and torch.isfinite(a).all()

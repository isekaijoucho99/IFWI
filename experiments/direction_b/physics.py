"""Dimensionless five-point Helmholtz operator and instrumented complex GMRES."""

from dataclasses import dataclass
from numbers import Real
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator, aslinearoperator, gmres


class Helmholtz:
    """Use C-order indexing, zero exterior values and dimensionless loss.

    A = L - (2 pi f h)^2 diag(m) - i diag(loss), where m is squared
    slowness. At distance d cells from the nearest edge, including d=0
    on the edge, loss = damping + sponge * max(1 - d/3, 0)^2.
    For any unit vector v, |Im(v* A v)| >= damping, implying ||A v|| >=
    damping. The same inverse-norm bound holds for A*.
    """

    def __init__(self, m, spacing, frequency, damping, sponge=0.0):
        model = np.asarray(m)
        if model.ndim != 2 or not model.size or np.iscomplexobj(model):
            raise ValueError("m must be a nonempty real 2-D squared-slowness model")
        model = np.asarray(model, dtype=np.float64)
        if not np.isfinite(model).all() or np.any(model <= 0):
            raise ValueError("m must contain finite positive squared slownesses")
        for name, value in (("spacing", spacing), ("frequency", frequency),
                            ("damping", damping)):
            if not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive scalar")
        if not isinstance(sponge, Real) or not np.isfinite(sponge) or sponge < 0:
            raise ValueError("sponge must be a finite nonnegative scalar")

        nz, nx = self.shape = model.shape
        self.coefficient = (2 * np.pi * float(frequency) * float(spacing)) ** 2
        self.K = 1 / float(damping)

        def axis_laplacian(n):
            return sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))

        laplacian = sparse.kronsum(axis_laplacian(nx), axis_laplacian(nz), format="csr")
        z, x = np.ogrid[:nz, :nx]
        edge_distance = np.minimum(np.minimum(z, nz - 1 - z), np.minimum(x, nx - 1 - x))
        self.damping_profile = damping + sponge * np.maximum(1 - edge_distance / 3, 0) ** 2
        diagonal = -self.coefficient * model.ravel() - 1j * self.damping_profile.ravel()
        self.A = (laplacian + sparse.diags(diagonal)).tocsr()
        self.AH = self.A.getH().tocsr()


@dataclass(frozen=True)
class SolveResult:
    x: np.ndarray
    iterations: int
    matvecs: int
    seconds: float


class SolveFailure(RuntimeError):
    """Preserve the work performed by a failed solve for its caller's ledger."""

    def __init__(self, message, *, iterations, matvecs, seconds):
        super().__init__(message)
        self.iterations = iterations
        self.matvecs = matvecs
        self.seconds = seconds
        self.work = dict(iterations=iterations, matvecs=matvecs, solve_matvecs=matvecs)


def solve(matrix, rhs, rtol, x0=None):
    """Run restarted GMRES, counting every matrix application including verification."""
    start = perf_counter()
    operator = aslinearoperator(matrix)
    n, columns = operator.shape
    if columns != n:
        raise ValueError("GMRES requires a square operator")
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be positive and finite")

    def vector(values, name):
        values = np.asarray(values, dtype=np.complex128)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite flat vector of length {n}")
        return values

    b = vector(rhs, "rhs")
    initial = None if x0 is None else vector(x0, "x0")
    rhs_norm = norm(b)
    if rhs_norm == 0:
        return SolveResult(np.zeros_like(b), 0, 0, perf_counter() - start)

    matvecs = 0
    iterations = 0

    def multiply(v):
        nonlocal matvecs
        matvecs += 1
        return operator.matvec(v)

    def count_iteration(_):
        nonlocal iterations
        iterations += 1

    # GMRES forms squared norms internally. Normalize the source and initial
    # guess together to preserve the relative stopping test for weak sources.
    limits = np.finfo(np.float64)
    scale = np.clip(rhs_norm, limits.tiny, limits.max)
    scaled_rhs = b / scale
    scaled_initial = None if initial is None else initial / scale
    counted = LinearOperator(operator.shape, matvec=multiply, dtype=np.complex128)
    x, info = gmres(
        counted, scaled_rhs, x0=scaled_initial, rtol=rtol, atol=0.0,
        restart=min(30, n), maxiter=400,
        callback=count_iteration, callback_type="pr_norm",
    )
    x *= scale
    relative_residual = norm((b - multiply(x)) / scale, check_finite=False) / norm(scaled_rhs)
    if info != 0 or not np.isfinite(relative_residual) or relative_residual > rtol:
        raise SolveFailure(
            f"GMRES failed after {iterations} iterations (info={info}, "
            f"relative residual={relative_residual:.3e}, tolerance={rtol:.3e})",
            iterations=iterations, matvecs=matvecs, seconds=perf_counter() - start,
        )
    return SolveResult(x, iterations, matvecs, perf_counter() - start)

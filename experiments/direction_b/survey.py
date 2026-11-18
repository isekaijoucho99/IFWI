"""Small frequency-domain survey. All shot/frequency contributions are summed."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import splu

from .control import Case
from .physics import Helmholtz


def geometry(shape, shots):
    nz, nx = shape
    if min(shape) < 6 or not 1 <= shots <= nx - 4:
        raise ValueError("use a grid of at least 6 x 6 and 1 through nx-4 shots")
    xs = np.rint(np.linspace(2, nx - 3, shots)).astype(int) if shots > 1 else [nx // 2]
    receivers = nx + np.arange(1, nx - 1, 2)
    return 2 * nx + np.asarray(xs), receivers


@dataclass
class Observation:
    frequency: float
    source: np.ndarray
    receivers: np.ndarray
    data: np.ndarray


def observations(truth_velocity, spacing, frequencies, damping, acquisition, *, sponge=0.):
    sources, receivers = acquisition
    survey = []
    for frequency in frequencies:
        op = Helmholtz(np.asarray(truth_velocity, dtype=float) ** -2, spacing, frequency, damping, sponge=sponge)
        direct = splu(op.A.tocsc())
        for index in sources:
            source = np.zeros(truth_velocity.size, dtype=complex)
            source[index] = 1.
            wave = direct.solve(source)
            survey.append(Observation(frequency, source, receivers, wave[receivers]))
    return survey


def build_cases(m, spacing, damping, survey, *, sponge=0.):
    operators = {}
    cases = []
    for obs in survey:
        if obs.frequency not in operators:
            operators[obs.frequency] = Helmholtz(m, spacing, obs.frequency, damping, sponge=sponge)
        cases.append(Case(operators[obs.frequency], obs.source, obs.receivers, obs.data))
    return cases


def reference(cases):
    """Offline sparse LU with one factorization per frequency, including A^H solves."""
    start = perf_counter()
    gradient = np.zeros(cases[0].source.size)
    factors = {}
    misfit, worst = 0., 0.
    for case in cases:
        op = case.op
        if op not in factors:
            factors[op] = splu(op.A.tocsc())
        lu = factors[op]
        u = lu.solve(case.source)
        residual = u[case.receivers] - case.data
        misfit += .5 * np.vdot(residual, residual).real
        rhs = np.zeros_like(case.source)
        rhs[case.receivers] = residual
        adj = lu.solve(rhs, trans="H")
        gradient += op.coefficient * (adj.conj() * u).real
        for defect, source in [(case.source - op.A @ u, case.source), (rhs - op.AH @ adj, rhs)]:
            rhs_norm = norm(source)
            relative = norm(defect) / rhs_norm if rhs_norm else norm(defect)
            worst = max(worst, relative)
    return float(misfit), gradient, {"seconds": perf_counter() - start, "max_relative_residual": float(worst)}

"""Small 2-D acoustic FWI with a frozen geological patch denoiser.

Gathers have shape [frequency, shot, receiver]. The latent vector has one
real coordinate per grid point (depth-major). All methods use the same
nonlinear velocity transform, noise level and prior. W already includes the
sketch normalization. Its rows weight both source amplitudes and native
observations, with one forward and one Hermitian-adjoint RHS per row/frequency.
"""

from copy import deepcopy
from functools import lru_cache

import numpy as np
from scipy.special import expit, logit, softmax
from scipy.sparse.linalg import splu

from experiments.direction_b.model import DATA, load_patch
from experiments.direction_b.physics import Helmholtz
from experiments.direction_b.survey import geometry


V_MIN = 1000.
V_MAX = 6500.
PATCH_STD = 0.24


def _latent(velocity):
    p = (np.asarray(velocity, dtype=float) - V_MIN) / (V_MAX - V_MIN)
    if not np.all((p > 0.) & (p < 1.)):
        raise ValueError("velocity must lie strictly within the logit bounds")
    return logit(p)


class GeologicalPatchPrior:
    """Uniform Gaussian mixture around training patches in logit velocity.

    Denoising is exact for clean mixture samples corrupted by white Gaussian
    noise of standard deviation sigma. DAPS uses a scalar conditional variance
    from a moment-matched isotropic Gaussian. The true conditional covariance
    also depends on the noisy patch through its component probabilities.
    """

    def __init__(self, centers, component_std):
        self.centers = np.array(centers, dtype=float, copy=True)
        if self.centers.ndim != 2 or not self.centers.size or not np.isfinite(self.centers).all():
            raise ValueError("prior centers must be a nonempty finite matrix")
        if not np.isfinite(component_std) or component_std <= 0:
            raise ValueError("component standard deviation must be finite and positive")
        self.centers.setflags(write=False)
        self.dimension = self.centers.shape[1]
        self.component_std = float(component_std)
        self.marginal_variance = self.component_std**2 + np.mean(self.centers.var(axis=0))

    def sample(self, rng):
        center = self.centers[rng.integers(len(self.centers))]
        return center + self.component_std * rng.normal(size=self.dimension)

    def denoise(self, noisy, sigma):
        y = np.asarray(noisy, dtype=float)
        if y.shape != (self.dimension,) or not np.isfinite(y).all():
            raise ValueError("denoise expects a finite latent vector of the prior dimension")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("noise standard deviation must be finite and nonnegative")
        if sigma == 0:
            return y.copy()
        total = self.component_std**2 + sigma**2
        difference = self.centers - y
        logits = -0.5 * np.einsum("ij,ij->i", difference, difference) / total
        weights = softmax(logits)
        return y + (sigma**2 / total) * (weights @ difference)

    def conditional_variance(self, sigma):
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("noise standard deviation must be finite and nonnegative")
        return float(self.marginal_variance * sigma**2 / (self.marginal_variance + sigma**2))


@lru_cache(maxsize=4)
def _frozen_prior(shape):
    """Pool nonoverlapping tiles away from both evaluation strips, independent of seed."""
    with np.load(DATA / "overthrust2d.npz") as archive:
        overthrust = archive["vp"].astype(float) * 1000.
    marmousi = np.loadtxt(DATA / "vel_marmousi_376x1151.csv", delimiter=",", skiprows=1)[::4, ::4]
    nz, nx = shape
    rng = np.random.default_rng(1742)
    splits = {}
    centers = []
    for name, grid in (("marmousi", marmousi), ("overthrust", overthrust)):
        zlen, xlen = grid.shape
        test_start = (xlen - nx) // 2
        test_stop = test_start + nx
        # Exclude the entire central strip, including a patch-width buffer.
        lo, hi = max(0, test_start - nx), min(xlen, test_stop + nx)
        origins = [(z, x) for z in range(0, zlen - nz + 1, nz)
                   for x in range(0, xlen - nx + 1, nx) if x + nx <= lo or x >= hi]
        if len(origins) > 32:
            indices = sorted(rng.choice(len(origins), 32, replace=False))
            origins = [origins[i] for i in indices]
        splits[name] = dict(grid_shape=list(grid.shape), excluded_x=[lo, hi],
                            test_x=[test_start, test_stop], training_origins=origins)
        centers.extend(_latent(grid[z:z + nz, x:x + nx]).ravel() for z, x in origins)
    if len(centers) < 2:
        raise ValueError("shape leaves fewer than two independent tiles outside the held-out strips")
    prior = GeologicalPatchPrior(centers, PATCH_STD)
    metadata = dict(family="uniform Gaussian mixture of logit-velocity patches",
                    denoiser="exact posterior mean under additive Gaussian corruption",
                    conditional_variance="moment-matched isotropic Gaussian approximation",
                    components=len(centers), component_std=PATCH_STD, training_seed=1742,
                    training_splits=splits, test_and_guard_strips_used_for_training=False)
    return prior, metadata


class _FrequencyOperator:
    """Retain the fixed Branch B stencil and update its model-dependent diagonal."""

    def __init__(self, shape, spacing, frequency, damping, sponge):
        self.reference_m = 3500.**-2
        op = Helmholtz(np.full(shape, self.reference_m), spacing, frequency, damping, sponge)
        self.coefficient = op.coefficient
        self.template = op.A.tocsc()
        self.template.sort_indices()
        columns = np.repeat(np.arange(self.template.shape[1]), np.diff(self.template.indptr))
        self.diagonal = np.flatnonzero(self.template.indices == columns)

    def factor(self, m):
        matrix = self.template.copy()
        matrix.data[self.diagonal] += self.coefficient * (self.reference_m - m)
        return splu(matrix)


class SeismicProblem:
    def __init__(self, dataset, truth, spacing, shots, frequencies, seed, prior_metadata):
        self.shape = truth.shape
        self.dimension = truth.size
        self.truth = _latent(truth).ravel()
        self.sources, self.receivers = geometry(self.shape, shots)
        self.shots = len(self.sources)
        self.frequencies = tuple(float(f) for f in frequencies)
        if (not self.frequencies or any(not np.isfinite(f) or f <= 0 for f in self.frequencies)
                or len(set(self.frequencies)) != len(self.frequencies)):
            raise ValueError("acquisition requires distinct positive finite frequencies in Hz")
        self.frequencies_count = len(self.frequencies)
        self.heldout_frequency = 1.25 * max(self.frequencies)
        self.noise_std = 0.003
        damping, sponge = 0.08, 0.25
        self._operators = {f: _FrequencyOperator(self.shape, spacing, f, damping, sponge)
                           for f in (*self.frequencies, self.heldout_frequency)}
        self.work = {phase: dict(factorizations=0, forward_rhs=0, adjoint_rhs=0)
                     for phase in ("setup", "likelihood", "prediction", "heldout_prediction")}
        self._native_rhs = np.zeros((self.dimension, self.shots), dtype=complex)
        self._native_rhs[self.sources, np.arange(self.shots)] = 1.
        self.metadata = dict(
            dataset=dataset, shape=list(self.shape), spacing_m=spacing, damping=damping,
            sponge=sponge, frequencies_hz=list(self.frequencies), heldout_frequency_hz=self.heldout_frequency,
            source_indices=self.sources.tolist(), receiver_indices=self.receivers.tolist(),
            latent="one logit velocity coordinate per cell, flattened in C order",
            velocity_bounds_mps=[V_MIN, V_MAX], noise_std=self.noise_std,
            noise="independent zero-mean Gaussian real and imaginary parts per receiver, shot and frequency",
            data_seed=seed, prior=deepcopy(prior_metadata), solver="sparse LU, shared across RHS per frequency",
            setup_forward_solves=(self.frequencies_count + 1) * self.shots,
            prediction_forward_solves=self.frequencies_count * self.shots,
            heldout_forward_solves_per_prediction=self.shots,
            gradient_rhs_per_encoded_row=2 * self.frequencies_count,
            factorizations_per_gradient=self.frequencies_count)
        train_seed, heldout_seed = np.random.SeedSequence(seed).spawn(2)

        def observe(frequencies, noise_seed):
            signal = self._predict(self.truth, frequencies, "setup")
            rng = np.random.default_rng(noise_seed)
            noise = rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
            return signal + self.noise_std * noise

        self.data = observe(self.frequencies, train_seed)
        self.heldout_data = observe((self.heldout_frequency,), heldout_seed)

    def _fraction(self, x):
        x = np.asarray(x)
        if x.shape != (self.dimension,) or np.iscomplexobj(x) or not np.isfinite(x).all():
            raise ValueError("x must contain one finite real latent coordinate per grid cell")
        return expit(x)

    def velocity(self, x):
        return (V_MIN + (V_MAX - V_MIN) * self._fraction(x)).reshape(self.shape)

    def _predict(self, x, frequencies, phase):
        m = self.velocity(x).ravel() ** -2
        work = self.work[phase]
        gathers = []
        for frequency in frequencies:
            lu = self._operators[frequency].factor(m)
            work["factorizations"] += 1
            wave = lu.solve(self._native_rhs)
            work["forward_rhs"] += self.shots
            gathers.append(wave[self.receivers].T)
        return np.stack(gathers)

    def predict(self, x):
        """Noise-free full gather on the assimilated frequencies."""
        return self._predict(x, self.frequencies, "prediction")

    def predict_heldout(self, x):
        """Noise-free full gather at a new frequency with the same source/receiver array."""
        return self._predict(x, (self.heldout_frequency,), "heldout_prediction")

    def value_grad(self, x, W):
        """Return the sketched Gaussian log-likelihood cost and its real gradient.

        If E[W.T @ W] = I, both are unbiased estimators of the native sums.
        This cost sketches the native misfit, including its measurement noise.
        """
        W = np.asarray(W)
        if (W.ndim != 2 or W.shape[1] != self.shots or len(W) == 0
                or np.iscomplexobj(W) or not np.isfinite(W).all()):
            raise ValueError("W must be a finite real matrix of shape (encoded rows, shots)")
        fraction = self._fraction(x)
        velocity = V_MIN + (V_MAX - V_MIN) * fraction
        m = velocity ** -2
        dm_dx = -2. * (V_MAX - V_MIN) * fraction * (1. - fraction) / velocity**3
        source = np.zeros((self.dimension, len(W)), dtype=complex)
        source[self.sources, :] = W.T
        adjoint_rhs = np.zeros_like(source)
        work = self.work["likelihood"]
        loss, grad_m = 0., np.zeros(self.dimension)
        for frequency, data in zip(self.frequencies, self.data):
            op = self._operators[frequency]
            lu = op.factor(m)
            work["factorizations"] += 1
            wave = lu.solve(source)
            work["forward_rhs"] += len(W)
            residual = wave[self.receivers].T - W @ data
            loss += 0.5 * np.vdot(residual, residual).real
            adjoint_rhs[self.receivers, :] = residual.T
            adjoint = lu.solve(adjoint_rhs, trans="H")
            work["adjoint_rhs"] += len(W)
            grad_m += op.coefficient * (adjoint.conj() * wave).real.sum(axis=1)
        variance = self.noise_std**2
        return float(loss / variance), (grad_m * dm_dx) / variance


def seismic_problem(dataset="marmousi", shape=(8, 12), shots=4, frequencies=(5., 8.), seed=3):
    """Build a centered benchmark task and common frozen geological prior.

    Training uses the repo's two full geological grids, with the centered
    lateral strips and guard bands removed for both. The smooth background
    provided with each benchmark is unused. The task is a small damped
    constant-density experiment with zero exterior Dirichlet values.
    """
    truth, _, spacing = load_patch(dataset, shape)
    prior, metadata = _frozen_prior(tuple(shape))
    return SeismicProblem(dataset, truth, spacing, shots, frequencies, seed, metadata), prior

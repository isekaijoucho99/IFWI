"""Analytic and quadrature references for simultaneous-source sampling.

The model interface returns a likelihood-only value and gradient in latent
coordinates. Each row of W already includes the sketch normalization. Sensor
noise is independent in the individual real observations before encoding.
"""

from dataclasses import dataclass
from math import erfc, pi, sqrt

import numpy as np


@dataclass(frozen=True)
class NormalPrior:
    dimension: int

    def sample(self, rng):
        return rng.standard_normal(self.dimension)

    def denoise(self, noisy, sigma):
        """E[x | x + sigma*z] with independent standard normal x and z."""
        return np.asarray(noisy) / (1.0 + sigma**2)

    def conditional_variance(self, sigma):
        return sigma**2 / (1.0 + sigma**2)


def ula_audit(seed=3, *, precision=1.5, step_size=0.22, gradient_noise_variance=4.0,
              steps=18, replicas=40_000):
    """Audit x'=(1-ha)x - h*e + sqrt(2h)z from x=0 across independent replicas.

    The invariant variance is (2+h*v)/(a*(2-h*a)) when 0<h*a<2.
    Only the empirical standard error uses Gaussian e. The variance identity
    needs only centered IID e of variance v, independent of x and z.
    """
    a, h, v = precision, step_size, gradient_noise_variance
    if not (a > 0 and 0 < h*a < 2 and np.isfinite(v) and v >= 0):
        raise ValueError("require precision > 0, 0 < step_size*precision < 2, and finite variance >= 0")
    if steps < 1 or replicas < 2:
        raise ValueError("require at least one step and two independent replicas")
    rng = np.random.default_rng(seed)
    contraction = 1 - h*a
    full = np.zeros(replicas)
    stochastic = np.zeros(replicas)
    for _ in range(steps):
        brownian = sqrt(2*h) * rng.standard_normal(replicas)
        full = contraction * full + brownian
        stochastic = contraction * stochastic + brownian - h * sqrt(v) * rng.standard_normal(replicas)

    def summarize(samples, invariant):
        finite_horizon = invariant * (1 - contraction**(2*steps))
        return {
            "stationary_variance": invariant,
            "finite_horizon_variance": finite_horizon,
            "empirical_variance": float(samples.var(ddof=1)),
            "mc_standard_error": finite_horizon * sqrt(2 / (replicas - 1)),
        }

    full_stationary = 2 / (a * (2 - h*a))
    stochastic_excess = h*v / (a * (2 - h*a))
    return {
        "seed": seed, "steps": steps, "replicas": replicas, "precision": a,
        "step_size": h, "gradient_noise_variance": v, "target_variance": 1/a,
        "finite_step_excess": full_stationary - 1/a,
        "stochastic_gradient_excess": stochastic_excess,
        "full": summarize(full, full_stationary),
        "stochastic": summarize(stochastic, full_stationary + stochastic_excess),
    }


class GaussianProblem:
    """Four shot amplitudes with two linearly coupled latent coordinates."""

    dimension = 2
    shots = 4
    frequencies_count = 0  # Closed-form predictions and gradients use no PDE solves.
    noise_std = 0.75
    prior = NormalPrior(dimension)
    design = np.array([[1.0, 0.55], [0.90, 0.15], [0.10, 0.90], [-0.35, 0.95]])

    def __init__(self, rng, truth_scale):
        self.truth = truth_scale * self.prior.sample(rng)
        self.data = self.predict(self.truth) + self.noise_std * rng.normal(size=(self.shots, 1))

    def predict(self, x):
        return (np.asarray(x) @ self.design.T)[..., None]

    def value_grad(self, x, W):
        encoded_design = W @ self.design
        residual = encoded_design @ x - (W @ self.data)[:, 0]
        return (float(residual @ residual / (2 * self.noise_std**2)),
                (encoded_design.T @ residual) / self.noise_std**2)

    def posterior(self, W=None):
        """Exact normalized target from exp(-Phi_W) times the unit Gaussian prior."""
        if W is None:
            W = np.eye(self.shots)
        design = W @ self.design / self.noise_std
        data = (W @ self.data)[:, 0] / self.noise_std
        covariance = np.linalg.inv(np.eye(self.dimension) + design.T @ design)
        return covariance @ (design.T @ data), covariance


def gaussian_problem(seed=3, truth_scale=1.0):
    """New prior-predictive dataset per seed. truth_scale changes only the data generator."""
    if not np.isfinite(truth_scale) or truth_scale <= 0:
        raise ValueError("truth_scale must be positive and finite")
    return GaussianProblem(np.random.default_rng(seed), truth_scale)


class LayeredProblem:
    """Single narrowband reflection from a horizontal interface at depth 1 km.

    The reflected ray length is sqrt((2*depth)^2 + offset^2). Its travel time
    equals this length times a bounded layer slowness in seconds per kilometer.
    Shot amplitudes and source phases are known. Both quadratures are observed.
    """

    dimension = 1
    shots = 4
    frequencies_count = 0  # The reflected phase and its derivative are evaluated analytically.
    frequency_hz = 2.8
    noise_std = 0.78
    prior = NormalPrior(dimension)
    _angular_path = 2 * pi * frequency_hz * np.hypot(2.0, [0.0, 0.18, 0.32, 0.45])
    _amplitude = np.array([0.80, 0.80, 0.72, 0.70])
    _source_phase = np.array([0.0, 0.15, -0.30, 0.60])
    _slowness_center = 0.5
    _slowness_half_range = 0.22

    def __init__(self, rng):
        self.truth = np.array([-0.28])
        self.data = self.predict(self.truth) + self.noise_std * rng.normal(size=(self.shots, 2))

    def predict(self, x):
        slowness = self._slowness_center + self._slowness_half_range * np.tanh(np.asarray(x))
        phase = slowness * self._angular_path + self._source_phase
        return self._amplitude[:, None] * np.stack((np.cos(phase), np.sin(phase)), axis=-1)

    def value_grad(self, x, W):
        prediction = self.predict(x)
        residual = W @ (prediction - self.data)
        phase_derivative = self._angular_path * self._slowness_half_range * (1 - np.tanh(x[0])**2)
        derivative = np.column_stack((-prediction[:, 1], prediction[:, 0])) * phase_derivative[:, None]
        scaled_residual = residual / self.noise_std**2
        return (float(0.5 * np.sum(scaled_residual * residual)),
                np.array([np.sum(scaled_residual * (W @ derivative))]))

    def reference(self, W=None, *, bound=8.0, intervals=8192):
        """Nested trapezoidal rules in the unbounded latent coordinate.

        Finite separators are posterior-density minima between adjacent modes.
        Evidence uses the normalized prior and omits the constant likelihood
        prefactor, so each likelihood factor is bounded above by one.
        """
        if not np.isfinite(bound) or bound <= 0 or intervals < 8 or intervals % 2:
            raise ValueError("expected positive finite bound and even interval count >= 8")
        if W is None:
            W = np.eye(self.shots)
        x = np.linspace(-bound, bound, intervals + 1)
        residual = W @ (self.predict(x[:, None]) - self.data)
        log_density = -0.5 * (x**2 + np.sum(residual**2, axis=(1, 2)) / self.noise_std**2)
        log_density -= 0.5 * np.log(2*pi)
        peak = float(log_density.max())

        def quadrature(stride):
            weights = np.exp(log_density[::stride] - peak)
            weights[[0, -1]] *= 0.5
            total = weights.sum()
            log_evidence = peak + np.log(total * 2*bound*stride / intervals)
            return weights / total, float(log_evidence)

        weights, evidence = quadrature(1)
        coarse_weights, coarse_evidence = quadrature(2)
        peaks = np.flatnonzero((log_density[1:-1] > log_density[:-2]) &
                               (log_density[1:-1] >= log_density[2:])) + 1
        minima = [i + np.argmin(log_density[i:j+1]) for i, j in zip(peaks[:-1], peaks[1:])]
        edges = x[np.asarray(minima, dtype=int)]
        bins = np.searchsorted(edges, x, side="right")
        mass = np.bincount(bins, weights=weights, minlength=len(edges)+1)
        coarse_mass = np.bincount(bins[::2], weights=coarse_weights, minlength=len(edges)+1)

        # On a short interval around the best grid point, bound ||Wr(x)|| via
        # ||Wr(anchor)|| + radius * sup ||W f'(x)||. Integrating this constant
        # lower bound times a lower bound on the Gaussian prior bounds evidence
        # independently of the quadrature, including for rank-deficient W.
        best = int(log_density.argmax())
        radius = 0.02
        lipschitz = (np.linalg.norm(W, 2) * self._slowness_half_range
                     * np.linalg.norm(self._amplitude * self._angular_path))
        residual_bound = np.linalg.norm(residual[best]) + radius*lipschitz
        log_evidence_lower = (np.log(2*radius / sqrt(2*pi)) -
                              0.5 * ((abs(x[best]) + radius)**2 + (residual_bound/self.noise_std)**2))
        prior_tail = erfc(bound / sqrt(2))
        tail_bound = np.exp(min(0.0, np.log(prior_tail) - log_evidence_lower)) if prior_tail else 0.0
        mean = weights @ x
        return x[:, None], weights, {
            "domain": [-bound, bound], "fine_intervals": intervals, "coarse_intervals": intervals//2,
            "mode_locations": x[peaks].tolist(), "mode_edges": edges.tolist(),
            "mode_probabilities": mass.tolist(), "mean": float(mean),
            "variance": float(weights @ ((x-mean)**2)),
            "log_evidence_without_likelihood_constant": evidence,
            "log_evidence_refinement_difference": abs(coarse_evidence - evidence),
            "max_mode_mass_refinement_difference": float(np.max(abs(mass - coarse_mass))),
            "posterior_tail_mass_upper_bound": float(tail_bound),
        }


def layered_problem(seed=3):
    """A fixed layered earth with reproducible observational noise."""
    return LayeredProblem(np.random.default_rng(seed))

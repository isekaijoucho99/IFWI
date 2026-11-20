"""Clean-space DAPS-style annealing with explicit shot-gradient accounting."""

from math import ceil, sqrt
from numbers import Integral

import numpy as np


def _integer(name, number, minimum=1):
    if isinstance(number, bool) or not isinstance(number, Integral) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _positive(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _vector(value, dimension, description):
    value = np.asarray(value)
    if np.iscomplexobj(value) or value.shape != (dimension,):
        raise ValueError(f"{description} must be a real vector with shape ({dimension},)")
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise FloatingPointError(f"nonfinite {description}")
    return value


def _rng(seed, *key):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def _signs(rng, rows, shots):
    signs = 2 * rng.integers(2, size=(rows, shots)) - 1
    return signs / sqrt(rows)


def _gradient(problem, x, weights, record, role):
    rows = len(weights)
    if record["used"] + rows > record["budget"]:
        raise RuntimeError("gradient evaluation exceeds the stage budget")
    record["used"] += rows
    record["costs"][role] += rows
    record["calls"][role] += 1
    value, grad = problem.value_grad(x, weights)
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite likelihood")
    return _vector(grad, problem.dimension, "likelihood gradient")


def _adaptive_choice(problem, center, record, pilots, width, noise_ratio, probes, requested):
    """Estimate one-row covariance at the center and lease one choice for this stage."""
    grads = np.array([_gradient(problem, center, _signs(pilots, 1, problem.shots),
                                record, "pilot") for _ in range(probes)])
    deviations = grads - grads.mean(axis=0)
    # The small probe Gram matrix has the same nonzero eigenvalues as the
    # parameter-space sample covariance, including in a large image domain.
    gram = deviations @ deviations.T / (probes - 1)
    largest = float(np.linalg.eigvalsh(gram)[-1])
    record.update(pilot_lambda_max=largest, pilot_trace=float(np.trace(gram)))

    # A k-row gradient contributes eta^2 Cov(g_1)/k to the step noise.
    # Full-shot evaluation has zero encoding variance and costs S rows.
    eta = record["eta"]
    available = (record["budget"] - record["used"]) // requested
    needed = max(width, ceil(eta * largest / (2 * noise_ratio)))
    rows = min(needed, available, problem.shots)
    if rows == problem.shots:
        record.update(operator="identity", rows=rows, encoding_noise_ratio=0.0)
    else:
        if largest > 0:
            eta = min(eta, 2 * noise_ratio * rows / largest)
        record.update(rows=rows, eta=eta, encoding_noise_ratio=eta * largest / (2 * rows))


def _steps(method, budget, shots, rows, inner_steps):
    if inner_steps is not None:
        return inner_steps
    if method == "svrg":
        # The first transition uses the full anchor gradient directly at x=center.
        return 1 + (budget - shots) // (2 * rows)
    return budget // rows


def _advance(problem, center, record, identity, fixed, sketches, brownian, inner_steps):
    method, rows = record["method"], record["rows"]
    tau2, eta = record["tau2"], record["eta"]
    steps = _steps(method, record["budget"] - record["used"], problem.shots, rows, inner_steps)
    x = center
    anchor = None
    if method == "svrg":
        anchor = _gradient(problem, center, identity, record, "anchor")
    for step in range(steps):
        if method == "svrg" and step == 0:
            gradient = anchor
        else:
            if record["operator"] == "identity":
                weights = identity
            elif fixed is not None:
                weights = fixed
            else:
                weights = _signs(sketches, rows, problem.shots)
            gradient = _gradient(problem, x, weights, record, "current")
            if method == "svrg":
                control = _gradient(problem, center, weights, record, "control")
                gradient = gradient - control + anchor
        drift = gradient + (x - center) / tau2
        x = x - eta * drift + sqrt(2 * eta) * brownian.standard_normal(problem.dimension)
        if not np.isfinite(x).all():
            raise FloatingPointError("nonfinite Langevin transition")
        record["steps"] += 1
    record["displacement_norm"] = float(np.linalg.norm(x - center))
    if not np.isfinite(record["displacement_norm"]):
        raise FloatingPointError("nonfinite terminal displacement")
    return x


def _finish_record(record, frequencies):
    record["unused"] = record["budget"] - record["used"]
    record["forward_solves"] = record["adjoint_solves"] = record["used"] * frequencies


def _work_totals(records, budget):
    totals = {key: sum(r[key] for r in records) for key in
              ("used", "steps", "rejected_steps", "forward_solves", "adjoint_solves")}
    return dict(totals, budget=budget, unused=budget - totals["used"], stages=records)


def sample(problem, prior, *, method, particles, sigmas, budget, encoded_rows=1,
           step_fraction=.12, noise_ratio=.2, probes=4, seed=3, encoding_seed=19, inner_steps=None):
    """Return terminal clean particles and a JSON-safe computational ledger.

    Each positive, descending sigma denotes a denoise/conditional-ULA stage.
    After that stage we re-noise its terminal state at the next sigma. The
    final next sigma is zero. ULA uses eta = step_fraction * conditional_variance
    and Gaussian noise with covariance 2 * eta * I. Adaptive stages may reduce eta.

    The per-particle, per-stage budget counts the rows of every evaluated W.
    One row requires one forward and one adjoint RHS per modeled frequency.
    inner_steps=None fills the cap with complete transitions. Otherwise each
    stage must fit exactly that many transitions. All fixed stages and particles
    share one W. Refresh, adaptive sketches and SVRG correction pairs resample
    W at every transition that consumes a sketch.
    """
    if method not in ("full", "fixed", "refresh", "svrg", "adaptive"):
        raise ValueError("unknown sampling method")
    shots = _integer("problem.shots", problem.shots)
    dimension = _integer("problem.dimension", problem.dimension)
    frequencies = _integer("problem.frequencies_count", problem.frequencies_count, 0)
    particles = _integer("particles", particles)
    budget = _integer("budget", budget)
    encoded_rows = _integer("encoded_rows", encoded_rows)
    seed = _integer("seed", seed, 0)
    encoding_seed = _integer("encoding_seed", encoding_seed, 0)
    step_fraction = _positive("step_fraction", step_fraction)
    noise_ratio = _positive("noise_ratio", noise_ratio)
    probes = _integer("probes", probes, 2 if method == "adaptive" else 0)
    sigmas = np.asarray(sigmas, dtype=float)
    if (sigmas.ndim != 1 or not sigmas.size or not np.isfinite(sigmas).all()
            or np.any(sigmas <= 0) or np.any(np.diff(sigmas) >= 0)):
        raise ValueError("sigmas must be a strictly descending sequence of finite positive levels")
    if inner_steps is not None:
        inner_steps = _integer("inner_steps", inner_steps)
    requested = inner_steps or 1
    rows = shots if method == "full" else encoded_rows
    minimum = rows * requested
    if method == "svrg":
        minimum = shots + 2 * rows * (requested - 1)
    elif method == "adaptive":
        minimum = probes + min(rows, shots) * requested
    if budget < minimum:
        raise ValueError(f"stage budget {budget} is too small for {method}, need at least {minimum}")

    identity = np.eye(shots) if method in ("full", "adaptive", "svrg") else None
    fixed = _signs(_rng(encoding_seed, 1, 0), rows, shots) if method == "fixed" else None
    all_costs = dict(anchor=0, pilot=0, current=0, control=0)
    records = []
    total_budget = particles * len(sigmas) * budget
    samples = np.empty((particles, dimension))
    for particle in range(particles):
        initial = _rng(seed, 0, particle, 0)
        clean = _vector(prior.sample(initial), dimension, "initial prior sample")
        noisy = clean + sigmas[0] * initial.standard_normal(dimension)
        for stage, sigma in enumerate(sigmas):
            next_sigma = float(sigmas[stage+1]) if stage+1 < len(sigmas) else 0.0
            tau2 = _positive("conditional variance", prior.conditional_variance(float(sigma)))
            eta = _positive("Langevin step size", step_fraction * tau2)
            record = dict(method=method, particle=particle, stage=stage, sigma=float(sigma),
                          sigma_next=next_sigma, tau2=tau2, eta=eta, rows=rows,
                          operator="identity" if method == "full" else "rademacher",
                          pilot_lambda_max=None, pilot_trace=None, encoding_noise_ratio=None,
                          budget=budget, used=0, costs=all_costs.copy(), calls=all_costs.copy(),
                          steps=0, rejected_steps=0)
            # Stage-specific streams keep later Brownian and re-noising draws
            # coupled across methods even when earlier step counts differ.
            sketches = _rng(encoding_seed, 1, 1, particle, stage)
            brownian = _rng(seed, 0, particle, 1, stage)
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    noisy = _vector(noisy, dimension, "noisy stage input")
                    center = _vector(prior.denoise(noisy, float(sigma)), dimension, "denoised center")
                    if method == "adaptive":
                        pilots = _rng(encoding_seed, 1, 2, particle, stage)
                        _adaptive_choice(problem, center, record, pilots, rows, noise_ratio, probes, requested)
                    try:
                        clean = _advance(problem, center, record, identity, fixed, sketches, brownian, inner_steps)
                    except FloatingPointError:
                        record["rejected_steps"] += 1
                        raise
                    if next_sigma:
                        renoise = _rng(seed, 0, particle, 2, stage)
                        noisy = clean + next_sigma * renoise.standard_normal(dimension)
            except FloatingPointError as exc:
                _finish_record(record, frequencies)
                error = FloatingPointError(f"nonfinite computation at particle {particle}, stage {stage}: {exc}")
                error.stage = record
                error.stats = _work_totals(records + [record], total_budget)
                raise error from exc
            _finish_record(record, frequencies)
            records.append(record)
        samples[particle] = clean
    stats = dict(method=method, particles=particles, sigmas=sigmas.tolist(), encoded_rows=encoded_rows,
                 step_fraction=step_fraction, noise_ratio=noise_ratio, probes=probes, seed=seed,
                 encoding_seed=encoding_seed, inner_steps=inner_steps,
                 **_work_totals(records, total_budget))
    if fixed is not None:
        stats["fixed_weights"] = fixed.tolist()
    return samples, stats

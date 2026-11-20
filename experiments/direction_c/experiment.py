"""Run one posterior comparison and persist its samples, references and solve ledger."""

import csv
import gzip
import json
from collections import Counter
from time import perf_counter

import numpy as np

from .moments import gaussian_laws
from .sampler import sample
from .scores import gaussian_distance, interval, mode_score, predictive, sample_distance
from .toys import gaussian_problem, layered_problem, ula_audit


METHODS = ("full", "fixed", "refresh", "svrg", "adaptive", "refresh_wide", "refresh_half")
STEP_FRACTION = {"gaussian": .06, "layered": .0015, "seismic": .035}


def write_json(path, result):
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sampler_options(args, tier):
    return dict(sigmas=args.sigmas, budget=args.budget, encoded_rows=args.encoded_rows,
                step_fraction=args.step_fraction or STEP_FRACTION[tier], probes=args.probes,
                noise_ratio=args.noise_ratio, inner_steps=args.inner_steps)


def _stages(records):
    summary = []
    for stage in sorted({r["stage"] for r in records}):
        part = [r for r in records if r["stage"] == stage]
        row = dict(stage=stage, sigma=part[0]["sigma"],
                   row_count_histogram=dict(Counter(r["rows"] for r in part)),
                   operator_counts=dict(Counter(r["operator"] for r in part)),
                   mean_langevin_time=float(np.mean([r["eta"] * r["steps"] for r in part])))
        for key in ("eta", "steps", "pilot_lambda_max", "pilot_trace", "encoding_noise_ratio"):
            values = [r[key] for r in part if r[key] is not None]
            if values:
                row["mean_" + key] = float(np.mean(values))
        summary.append(row)
    return summary


def _write_records(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, allow_nan=False) + "\n")


def run_case(root, tier, dataset, args):
    physical = tier == "seismic"
    case = dataset if physical else tier
    folder = root / case
    folder.mkdir()
    particles = args.seismic_particles if physical else args.particles
    config = sampler_options(args, tier)
    reference = {}
    if physical:
        from .conditional import audit_conditional
        from .importance import importance_reference
        from .seismic import seismic_problem
        problem, prior = seismic_problem(dataset, shape=tuple(args.shape), seed=args.seed)
        grid, weights, ref_info = importance_reference(
            problem, prior, samples_per_component=args.importance_per_component, seed=args.seed+902)
        reference.update(kind="stratified full-likelihood prior importance sampling", **ref_info)
        reference["conditional_approximation"] = audit_conditional(
            prior, args.sigmas, draws=512, seed=args.seed+903)
        np.savez_compressed(folder / "reference.npz", latent=grid, weights=weights)
    elif tier == "gaussian":
        problem = gaussian_problem(args.seed)
        prior = problem.prior
        mu, cov = problem.posterior()
        reference.update(kind="analytic full-shot posterior", mean=mu.tolist(), covariance=cov.tolist(),
                         independent_noise_ula=ula_audit(seed=args.seed))
    else:
        problem = layered_problem(args.seed)
        prior = problem.prior
        grid, weights, ref_info = problem.reference()
        reference.update(kind="full-shot quadrature", **ref_info)
        np.savez_compressed(folder / "reference.npz", latent=grid, weights=weights)
    acquisition = dict(truth=problem.truth.tolist(), observed_real=problem.data.real.tolist(),
                       observed_imag=problem.data.imag.tolist() if np.iscomplexobj(problem.data) else None,
                       noise_std=problem.noise_std, shots=problem.shots)
    if physical:
        acquisition.update(problem.metadata)
    results = dict(case=case, particles=particles, acquisition=acquisition, reference=reference, methods={})
    rows = []
    for label in args.methods:
        method = "refresh" if label.startswith("refresh_") else label
        options = dict(config)
        if label == "refresh_wide":
            options["encoded_rows"] *= 2
        elif label == "refresh_half":
            options["step_fraction"] *= .5
        start_work = dict(problem.work["likelihood"]) if physical else {}
        start = perf_counter()
        try:
            draws, ledger = sample(problem, prior, method=method, particles=particles,
                                   seed=args.seed+101, encoding_seed=args.encoding_seed, **options)
        except FloatingPointError as exc:
            if hasattr(exc, "stats"):
                write_json(folder / (label + "_failure.json"), dict(error=str(exc), **exc.stats))
            raise
        seconds = perf_counter() - start
        records = ledger.pop("stages")
        _write_records(folder / (label + "_stages.jsonl.gz"), records)
        coordinates = np.stack([problem.velocity(x) for x in draws]) if physical else draws
        truth = problem.velocity(problem.truth) if physical else problem.truth
        archive = dict(latent=draws, mean=coordinates.mean(axis=0),
                       quantiles_05_50_95=np.quantile(coordinates, [.05, .50, .95], axis=0))
        np.savez_compressed(folder / (label + ".npz"), **archive)
        post = interval(coordinates, truth)
        if tier == "gaussian":
            dist = gaussian_distance(draws, mu, cov)
        else:
            dist = sample_distance(draws, grid, weights=weights)
        pred = predictive(problem, draws, seed=args.seed+201)
        metrics = dict(interval=post, interval_units="m/s" if physical else "latent",
                       latent_distribution=dist, posterior_predictive=pred)
        if tier == "gaussian" and method in ("full", "fixed", "refresh"):
            metrics["moment_laws"] = gaussian_laws(
                problem, [r for r in records if r["particle"] == 0], method, ledger.get("fixed_weights"))
            if method == "fixed":
                sketch_mu, sketch_cov = problem.posterior(np.array(ledger["fixed_weights"]))
                metrics["fixed_target"] = dict(mean=sketch_mu.tolist(), covariance=sketch_cov.tolist(),
                                               distance_from_draws=gaussian_distance(draws, sketch_mu, sketch_cov))
        if tier == "layered":
            metrics["modes"] = mode_score(draws, grid, weights, reference["mode_edges"])
            if method == "fixed":
                fixed_grid, fixed_w, fixed_info = problem.reference(np.array(ledger["fixed_weights"]))
                metrics["fixed_target"] = dict(reference=fixed_info, mass_in_full_target_basins=
                    np.histogram(fixed_grid.ravel(), np.r_[-np.inf, reference["mode_edges"], np.inf],
                                 weights=fixed_w)[0].tolist())
        physical_work = {k: v-start_work[k] for k, v in problem.work["likelihood"].items()} if physical else {
            "forward_rhs": 0, "adjoint_rhs": 0, "factorizations": 0}
        if physical and (ledger["forward_solves"], ledger["adjoint_solves"]) != (
                physical_work["forward_rhs"], physical_work["adjoint_rhs"]):
            raise RuntimeError("sampler work disagrees with the Helmholtz operator counters")
        results["methods"][label] = dict(ledger=ledger, stage_summary=_stages(records), metrics=metrics,
                                         pde_work=physical_work, sampler_wall_seconds=seconds)
        rows.append(dict(case=case, method=label, particles=particles, budget=ledger["budget"],
                         used=ledger["used"], unused=ledger["unused"], steps=ledger["steps"],
                         **physical_work, **post, interval_units=metrics["interval_units"], **dist,
                         mode_tv=metrics.get("modes", {}).get("total_variation"),
                         predictive_coverage90=pred["coverage90"],
                         predictive_rmse_sigma=pred["mean_rmse_sigma"],
                         predictive_width90_sigma=pred["interval_width90_sigma"],
                         sampler_wall_seconds=seconds))
        print(f"{case:12} {label:14} {ledger['used']:6}/{ledger['budget']} shot gradients; "
              f"W1={dist['sliced_w1']:.4g}", flush=True)
    if physical:
        results["pde_work_all_phases"] = problem.work
    write_json(folder / "report.json", results)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

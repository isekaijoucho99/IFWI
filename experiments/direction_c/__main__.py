"""Command line entry point for the simultaneous-source posterior study."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .calibration import calibrate
from .experiment import METHODS, STEP_FRACTION, run_case, sampler_options, write_csv, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tiers", nargs="+", choices=STEP_FRACTION, default=list(STEP_FRACTION))
    parser.add_argument("--datasets", nargs="+", choices=["marmousi", "overthrust"], default=["marmousi", "overthrust"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--particles", type=int, default=96, help="independent particles per toy method")
    parser.add_argument("--seismic-particles", type=int, default=24)
    parser.add_argument("--shape", type=int, nargs=2, metavar=("NZ", "NX"), default=[8, 12])
    parser.add_argument("--budget", type=int, default=28, help="shot-gradient cap per particle and stage")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[2., 1., .45, .2])
    parser.add_argument("--encoded-rows", type=int, default=1)
    parser.add_argument("--step-fraction", type=float, help="override tier-specific eta / conditional variance")
    parser.add_argument("--inner-steps", type=int, help="match transition counts within the budget cap")
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--noise-ratio", type=float, default=.2)
    parser.add_argument("--importance-per-component", type=int, default=256,
                        help="number of full-likelihood reference draws per geological prior component")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--encoding-seed", type=int, default=19)
    parser.add_argument("--sbc-replicates", type=int, default=0,
                        help="paired Gaussian datasets per condition for exact and selected core methods")
    parser.add_argument("--sbc-particles", type=int, default=32)
    parser.add_argument("--mismatch-scale", type=float, default=1.8, help="truth prior SD in the misspecification test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/direction_c"))
    args = parser.parse_args(argv)
    if min(args.particles, args.seismic_particles, args.sbc_particles, args.importance_per_component) < 2:
        parser.error("ensemble and reference sizes must be at least two")
    if min(args.seed, args.encoding_seed, args.sbc_replicates) < 0:
        parser.error("seeds and sbc-replicates must be nonnegative")
    if args.step_fraction is not None and args.step_fraction <= 0:
        parser.error("--step-fraction must be positive")
    for values in (args.tiers, args.datasets, args.methods):
        if len(set(values)) != len(values):
            parser.error("tiers, datasets and methods must be distinct within each list")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_dir / (stamp + "_" + uuid4().hex[:8])
    root.mkdir(parents=True)
    write_json(root / "config.json", dict(vars(args), output_dir=str(args.output_dir),
                                          default_step_fractions=STEP_FRACTION))
    print(root.resolve(), flush=True)
    rows = []
    for tier in args.tiers:
        for dataset in (args.datasets if tier == "seismic" else [None]):
            rows.extend(run_case(root, tier, dataset, args))
            write_csv(root / "summary.csv", rows)
    if args.sbc_replicates:
        cal = calibrate(replicates=args.sbc_replicates, particles=args.sbc_particles, seed=args.seed+301,
                        methods=[m for m in args.methods if m in METHODS[:5]],
                        sampler_kwargs=sampler_options(args, "gaussian"), mismatch_scale=args.mismatch_scale)
        write_json(root / "calibration.json", cal)
    return root


if __name__ == "__main__":
    main()

"""Independent stage-1 Residual IFWI CLI; original experiment.py stays unchanged."""
import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from experiments.residual_ifwi_experiment import ROOT, default_config, run_experiment, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/residual_ifwi_stage1.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sigma", type=float, help="Gaussian standard deviation in inversion-grid cells")
    parser.add_argument("--parameterization", choices=("residual", "absolute"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--crop", nargs=2, type=int, metavar=("NZ", "NX"))
    parser.add_argument("--shot-batch", type=int, help="Fixed shot accumulation size; 0 means all shots")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/residual_ifwi")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and display configuration without creating outputs")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    overrides = (("epochs", "training", "epochs"), ("sigma", "initialization", "sigma"),
        ("seed", None, "seed"), ("parameterization", None, "parameterization"),
        ("shots", "acquisition", "num_shots"), ("nt", "forward", "nt"),
        ("crop", "data", "crop_shape"), ("learning_rate", "training", "learning_rate"),
        ("checkpoint_interval", "training", "checkpoint_interval"))
    for argument, section, key in overrides:
        value = getattr(args, argument)
        if value is not None:
            (config[section] if section else config)[key] = value
    if args.shot_batch is not None:
        config["training"]["shot_batch_size"] = None if args.shot_batch == 0 else args.shot_batch
    if args.threads < 1:
        parser.error("--threads must be positive")
    validate_config(config)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    torch.set_num_threads(args.threads)
    out = run_experiment(config, args.output_dir, args.device, resume=args.resume)
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

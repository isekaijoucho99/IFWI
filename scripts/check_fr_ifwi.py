"""Check the full Fourier configuration and initialization WITHOUT wave simulation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(config_path):
    import numpy as np
    import torch
    import yaml
    from experiments.residual_ifwi_experiment import (
        VelocityParameterization, check_velocity, load_velocity_data,
        original_source_hashes, validate_config,
    )
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    validate_config(config)
    if config["model"].get("architecture") != "fr_siren" or config["parameterization"] != "residual":
        raise ValueError("This preflight expects a residual FR-SIREN configuration")
    source_hashes = original_source_hashes()
    truth, background, provenance = load_velocity_data(config, ROOT)
    # This script constructs CPU tensors only; do not seed or initialize CUDA.
    np.random.seed(config["seed"])
    torch.random.default_generator.manual_seed(config["seed"])
    model = VelocityParameterization(background, config["data"]["grid_spacing_m"],
                                     parameterization=config["parameterization"], **config["model"])
    with torch.no_grad():
        velocity = model()
        check_velocity(velocity, config)
        exact = torch.equal(velocity, torch.from_numpy(background)[None])
    if not exact:
        raise RuntimeError("Initial Fourier residual is nonzero")
    return {"passed": True, "finite_difference_executed": False, "device": "cpu",
            "velocity_shape": list(velocity.shape), "initial_residual_exactly_zero": exact,
            "original_source_hashes": source_hashes, "model": model.net.diagnostics(),
            "background_sha256": provenance["initial"]["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"experiments/configs/residual_ifwi_fr1000.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    result = check(args.config)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    summary = {"passed": result["passed"], "velocity_shape": result["velocity_shape"],
               "initial_residual_exactly_zero": result["initial_residual_exactly_zero"],
               "trainable_parameters": result["model"]["trainable_parameters"],
               "basis_ranks": [x["numerical_rank"] for x in result["model"]["layers"]]}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

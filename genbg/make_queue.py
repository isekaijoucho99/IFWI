"""Write one E2 spec per candidate background and the P0 queue for paper_queue.py.

Each spec is the template with spec.background = {kind: file, path, array_hash}, its own observation
cache (the cache identity contains the background hash) and a fixed update budget without a target,
so every run reports the same horizon. G15 and SM400 run first as the known failure/success references.

    python -m genbg.make_queue --candidates experiments/genbg/candidates \
        --template experiments/paper_supplement/specs/B_E2_smooth400_s3.json --epochs 400 --out experiments/genbg
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from genbg.common import load_template, read_manifest

FIRST = ("G15", "SM400")


def make_spec(template, entry, epochs, seed):
    spec = copy.deepcopy(template)
    name = f"P0_E2_{entry['id']}_s{seed}"
    spec["name"] = name
    spec["experiment"]["experiment_name"] = name
    spec["experiment"]["seed"] = seed
    spec["experiment"]["training"]["epochs"] = epochs
    spec["experiment"]["training"]["checkpoint_interval"] = min(50, epochs)
    spec["background"] = {"kind": "file", "path": entry["file"], "array_hash": entry["array_hash"]}
    spec["robustness"] = {"condition": f"marmousi_clean_genbg_{entry['id']}", "noise_sigma": 0.0, "noise_seed": seed}
    spec.pop("target", None)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="folder that receives specs/ and queues/")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--queue-name", default="p0_rank")
    args = parser.parse_args()
    template = load_template(args.template)
    entries = read_manifest(args.candidates)
    order = [x for i in FIRST for x in entries if x["id"] == i] + [x for x in entries if x["id"] not in FIRST]
    (args.out / "specs").mkdir(parents=True, exist_ok=True); (args.out / "queues").mkdir(parents=True, exist_ok=True)
    queue = []
    for entry in order:
        spec = make_spec(template, entry, args.epochs, args.seed)
        path = args.out / "specs" / f"{spec['name']}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        queue.append(f"specs/{path.name}")
    queue_path = args.out / "queues" / f"{args.queue_name}.json"
    queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    print(f"{len(queue)} specs -> {queue_path}")


if __name__ == "__main__":
    main()

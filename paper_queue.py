"""Run a queue of core_runner specs one after another on one GPU (docs/paper_supplement.md).

Each spec runs in its own process into OUT_ROOT/<name>/attempt_01. Finished runs (status
"completed") are skipped, so a queue can be restarted; a failed or interrupted run stops the
queue for inspection (resume it explicitly with core_runner.py --resume into a new attempt).
With --wait-idle the queue waits before every run until nvidia-smi reports no other GPU work,
so runs of one comparison are timed under the same conditions.

    python paper_queue.py experiments/paper_supplement/queues/q4_A_minimal.json \\
        --out-root outputs/paper_supplement --shared-dir F:/project/IFWI/outputs/robustness_overthrust_noise_20261016/shared
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def gpu_busy(max_utilization=10, max_memory_mib=1024):
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None                                   # no nvidia-smi: cannot check
    rows = [[float(v) for v in line.split(",")] for line in output.strip().splitlines() if line.strip()]
    return any(u > max_utilization or m > max_memory_mib for u, m in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queue", type=Path, help="JSON list of spec paths (relative to the queue's parent folder)")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--shared-dir", type=Path, help="shared observations root passed to core_runner")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait-idle", action="store_true", help="wait for an idle GPU before every run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = [(args.queue.parent.parent / p).resolve() for p in json.loads(args.queue.read_text(encoding="utf-8"))]
    args.out_root.mkdir(parents=True, exist_ok=True)
    log = args.out_root / f"queue_{args.queue.stem}.jsonl"
    for path in specs:
        name = json.loads(path.read_text(encoding="utf-8"))["name"]
        out = args.out_root / name / "attempt_01"
        status = out / "status.json"
        if status.exists():
            state = json.loads(status.read_text(encoding="utf-8")).get("state")
            if state == "completed":
                print(f"skip {name}: completed", flush=True)
                continue
            raise SystemExit(f"{name}: previous attempt is '{state}' at {out}; inspect it and resume explicitly")
        command = [sys.executable, str(ROOT / "core_runner.py"), "--spec", str(path), "--out", str(out),
                   "--device", args.device, "--no-plots"]
        if args.shared_dir:
            command += ["--shared-dir", str(args.shared_dir)]
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        while args.wait_idle and gpu_busy():
            print("GPU busy; waiting 60 s", flush=True)
            time.sleep(60)
        started = time.time()
        code = subprocess.run(command, cwd=ROOT).returncode
        record = {"name": name, "spec": str(path), "out": str(out), "returncode": code,
                  "wall_seconds": time.time() - started, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if code != 0:
            raise SystemExit(f"{name} failed (exit {code}); queue stopped, see {out}/progress.log")
    print(f"Queue {args.queue.stem} done", flush=True)


if __name__ == "__main__":
    main()

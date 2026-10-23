"""Run the time-to-target plan: exact velocities at k* and continuation of runs that missed the target.

Each entry continues an existing checkpoint with core_runner.py --extend (or residual_ifwi.py --resume
for the residual-runner run) and stops at the first update whose velocity RMSE reaches the target.
Continuation is exact (tests: 3+2 updates equal 5; resuming from an intermediate checkpoint gives the
same velocity at the target update), so an entry that starts from the checkpoint just before k*
reproduces the model at k* without re-running the whole inversion.

    python target_plan.py experiments/paper_supplement/target_plan.json --roots SPEED=F:/... Q12=F:/... \
        --out-root outputs/paper_supplement/targets --shared-dir SHARED [--dry-run]

Roots are the batch output directories named in the plan ({SPEED}, {Q12}, ...). Finished entries
(status.json state completed) are skipped, so the plan can be restarted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def expand(text, roots):
    for key, value in roots.items():
        text = text.replace('{' + key + '}', value)
    if '{' in text:
        raise SystemExit(f'Unresolved root in {text!r}; pass it with --roots')
    return text


def command(entry, roots, out, shared):
    checkpoint = expand(entry['checkpoint'], roots)
    if entry.get('runner', 'core') == 'core':
        cmd = [sys.executable, str(ROOT / 'core_runner.py'), '--extend', checkpoint, '--epochs', str(entry['epochs']),
               '--out', str(out), '--no-plots']
        if entry.get('stop_rmse') is not None:
            cmd += ['--stop-rmse', str(entry['stop_rmse'])]
        if entry.get('stop_plateau') is not None:
            cmd += ['--stop-plateau', str(entry['stop_plateau']['window']), str(entry['stop_plateau']['rel'])]
        if shared:
            cmd += ['--shared-dir', shared]
        return cmd
    # residual_ifwi.py has no stop option: it runs to the budget and k* is read from its history.
    return [sys.executable, str(ROOT / 'residual_ifwi.py'), '--config', expand(entry['config'], roots),
            '--epochs', str(entry['epochs']), '--resume', checkpoint, '--output-dir', str(out)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('plan', type=Path)
    parser.add_argument('--roots', nargs='*', default=[], help='KEY=path pairs for the plan placeholders')
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--shared-dir')
    parser.add_argument('--only', nargs='*', help='run only these entry names')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    roots = dict(pair.split('=', 1) for pair in args.roots)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    for entry in plan:
        if args.only and entry['name'] not in args.only:
            continue
        out = args.out_root / entry['name']
        status = out / 'status.json'
        if status.exists() and json.loads(status.read_text(encoding='utf-8')).get('state') == 'completed':
            print(f"skip {entry['name']} (completed)")
            continue
        cmd = command(entry, roots, out, args.shared_dir)
        print(' '.join(cmd), flush=True)
        if args.dry_run:
            continue
        args.out_root.mkdir(parents=True, exist_ok=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise SystemExit(f"{entry['name']} failed; plan stopped")


if __name__ == '__main__':
    main()

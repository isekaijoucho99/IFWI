"""Run OFAT cases through the original random IFWI baseline, with numbered snapshots."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml
from experiment_runtime import write_json
from parameter_sweep import make_cases, validate_case
from baseline_protocol import (ORIGINAL_PROTOCOL, baseline_config,
                               validate_baseline_protocol, validate_original_sources)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, help='Update budget; defaults to the baseline config')
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shot-batch-size', type=int, help='Rejected by the original full-shot protocol')
    parser.add_argument('--reuse-baseline', type=Path, help='Continue an original numbered baseline checkpoint')
    parser.add_argument('--completed-baseline', type=Path, help='Import the completed legacy random run; baseline training is skipped')
    parser.add_argument('--baseline-config', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preliminary', action='store_true', help='Explicitly permit a short validation budget; never a formal result')
    args = parser.parse_args()
    if args.reuse_baseline and args.completed_baseline:
        parser.error('Choose either a completed baseline or a checkpoint continuation')
    legacy_metadata = None
    if args.completed_baseline:
        legacy_metadata = json.loads((args.completed_baseline/'config.json').read_text(encoding='utf-8-sig'))
    config_path = args.baseline_config or ROOT/'experiments/configs/legacy_random_baseline.yaml'
    base = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    args.seeds = args.seeds or ([legacy_metadata['seed']] if legacy_metadata else [base.get('seed', 3)])
    base.setdefault('execution', {'protocol': ORIGINAL_PROTOCOL})
    base['seed'] = args.seeds[0]
    args.iterations = args.iterations if args.iterations is not None else base['training']['max_iterations']
    if legacy_metadata and args.seeds != [legacy_metadata['seed']]:
        parser.error('The imported legacy baseline must use its original single seed')
    if args.reuse_baseline and (len(args.seeds) != 1 or not args.reuse_baseline.is_file()):
        parser.error('--reuse-baseline requires one seed and an existing checkpoint')
    args.log_interval = args.log_interval if args.log_interval is not None else base['training']['log_interval']
    if min(args.iterations, args.log_interval) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive budgets and unique seeds are required')
    if args.shot_batch_size is not None:
        parser.error('Original baseline uses full shot input; omit --shot-batch-size')
    base['training'].update(max_iterations=args.iterations, log_interval=args.log_interval)
    validate_baseline_protocol(base, preliminary=args.preliminary)
    if args.seeds != [3]:
        parser.error('This baseline protocol is anchored to seed=3')
    validate_original_sources(ROOT)
    cases = make_cases(base, preliminary=args.preliminary)
    if legacy_metadata:
        from import_completed_baseline import validate_legacy_config
        validate_legacy_config(legacy_metadata, cases[0]['config'], args.iterations)
    if args.reuse_baseline:
        from baseline_experiment import validate_resume
        validate_resume(cases[0]['config'], args.reuse_baseline)
    if args.dry_run:
        print(json.dumps(cases, indent=2))
        return 0
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new output directory; existing runs are never overwritten')
    (out/'logs').mkdir(parents=True, exist_ok=True)
    snapshot = out/'source'
    paths = list(ROOT.glob('*.py')) + list((ROOT/'experiments').rglob('*.py'))
    paths += list((ROOT/'experiments/configs').glob('*.yaml'))
    paths += [ROOT/'data'/base['data']['model_file']]
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        target = snapshot/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[rel.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(out/'source_hashes.json', hashes)
    write_json(out/'baseline_protocol.json', dict(protocol=ORIGINAL_PROTOCOL,
        baseline=baseline_config(), original_source_hashes=validate_original_sources(snapshot),
        training_entry='IFWI2D.train', shot_input='full',
        best_selection='preupdate total loss at zero-based logged epochs; postupdate model saved',
        checkpoints='checkpoints/MarmousiI_random-checkpoint-{completed_updates}.pth',
        completed_update_cadence='1 + 100*j, j=0..40; final update',
        stdout='original EpochLog, IFWI2D.train and IFWI2D.predict'))
    legacy_bundle = None
    if args.completed_baseline:
        legacy_bundle = out/'completed_legacy_baseline'
        (legacy_bundle/'checkpoints').mkdir(parents=True)
        (legacy_bundle/'author_sources').mkdir()
        for name in ('config.json','initial_velocity.npy','true_velocity.npy','best_velocity.npy','loss.csv'):
            shutil.copy2(args.completed_baseline/name, legacy_bundle/name)
        for name in ('metrics.json', 'progress.log'):
            if (args.completed_baseline/name).is_file():
                shutil.copy2(args.completed_baseline/name, legacy_bundle/name)
        checkpoint_name = f"MarmousiI_random-checkpoint-{legacy_metadata['epochs']}.pth"
        shutil.copy2(args.completed_baseline/'checkpoints'/checkpoint_name,legacy_bundle/'checkpoints'/checkpoint_name)
        author_sources = args.completed_baseline.resolve()/'author_sources'
        if not author_sources.exists():
            author_sources = args.completed_baseline.resolve().parents[2]
        for name in ('ifwi_modules.py','rnn_fd.py','generator.py'):
            shutil.copy2(author_sources/name,legacy_bundle/'author_sources'/name)
        # Retain every original model snapshot, not only the final baseline.
        for path in (args.completed_baseline/'checkpoints').glob('MarmousiI_random-checkpoint-*.pth'):
            shutil.copy2(path, legacy_bundle/'checkpoints'/path.name)
        write_json(out/'completed_baseline_reference.json', dict(original_run=str(args.completed_baseline.resolve()),
            checkpoint_sha256=hashlib.sha256((legacy_bundle/'checkpoints'/checkpoint_name).read_bytes()).hexdigest(),
            original_epoch_label=legacy_metadata['epochs']-1,completed_updates=legacy_metadata['epochs'],
            input_hashes={p.relative_to(legacy_bundle).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in legacy_bundle.rglob('*') if p.is_file()}))
    reused_checkpoint = None
    if args.reuse_baseline:
        (out/'reused_baseline/checkpoints').mkdir(parents=True)
        reused_checkpoint = out/'reused_baseline/checkpoints'/args.reuse_baseline.name
        shutil.copy2(args.reuse_baseline, reused_checkpoint)
        write_json(out/'reused_baseline/config.json', cases[0]['config'])
        write_json(out/'reuse_manifest.json', dict(original_checkpoint=str(args.reuse_baseline.resolve()),
            checkpoint_sha256=hashlib.sha256(reused_checkpoint.read_bytes()).hexdigest()))
    (out/'configs').mkdir()
    for case in cases:
        (out/'configs'/f"{case['name']}.yaml").write_text(
            yaml.safe_dump(case['config'], sort_keys=False), encoding='utf-8')
    jobs = []
    for seed in args.seeds:
        for case in cases:
            jobs.append(dict(name=case['name'], factor=case['factor'], value=case['value'],
                seed=seed, config=case['config'], directory=f"runs/{case['name']}_seed{seed}"))
    write_json(out/'manifest.json', dict(iterations=args.iterations, seeds=args.seeds, jobs=jobs,
        protocol=ORIGINAL_PROTOCOL, preliminary=args.preliminary,
        completed_baseline_bundle='completed_legacy_baseline' if legacy_bundle else None))
    env = os.environ.copy()
    env.update(PYTHONUTF8='1')
    records = []
    if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for job in jobs:
            for name, digest in hashes.items():
                if hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Frozen source changed: ' + name)
            registered_path = out/'configs'/f"{job['name']}.yaml"
            registered = yaml.safe_load(registered_path.read_text(encoding='utf8'))
            if registered != job['config']:
                raise ValueError('Registered config changed: ' + job['name'])
            validate_case(cases[0]['config'], dict(job, config=registered), preliminary=args.preliminary)
            log = out/'logs'/f"{job['name']}_seed{job['seed']}.log"
            write_json(out/'active.json', dict(state='running', name=job['name'], seed=job['seed'],
                completed_runs=len(records), total_runs=len(jobs), log=str(log)))
            print(f"START {job['name']} seed={job['seed']}", flush=True)
            start = time.perf_counter()
            command = [sys.executable, '-X', 'utf8', '-u', str(snapshot/'experiments/run_experiment.py'),
                '--config', str(out/'configs'/f"{job['name']}.yaml"),
                '--output-dir', str(out/job['directory']), '--device', args.device, '--seed', str(job['seed'])]
            if args.preliminary:
                command += ['--preliminary']
            if job['name'] == 'baseline' and reused_checkpoint:
                command += ['--resume', str(reused_checkpoint)]
            if job['name'] == 'baseline' and legacy_bundle:
                command = [sys.executable, '-X', 'utf8', '-u', str(snapshot/'experiments/import_completed_baseline.py'),
                    '--legacy-run', str(legacy_bundle), '--config', str(out/'configs/baseline.yaml'),
                    '--output-dir', str(out/job['directory']), '--device', args.device, '--seed', str(job['seed'])]
            with log.open('wb') as stream:
                code = subprocess.run(command, cwd=snapshot, env=env,
                    stdout=stream, stderr=subprocess.STDOUT).returncode
            records.append(dict(name=job['name'], seed=job['seed'], exit_code=code,
                                seconds=time.perf_counter()-start))
            write_json(out/'runs.json', records)
            print(f"END {job['name']} exit={code}", flush=True)
            if code and job['name']=='baseline' and legacy_bundle:
                raise RuntimeError('Completed legacy baseline import failed; do not launch unmatched treatments')
            # Preserve failures and continue the other registered cases at their fixed budgets.
            subprocess.run([sys.executable, '-X', 'utf8',
                str(snapshot/'experiments/summarize_parameter_sweep.py'), '--suite', str(out)],
                cwd=snapshot, env=env, check=True)
        failures = sum(r['exit_code'] != 0 for r in records)
        write_json(out/'active.json', dict(state='completed' if not failures else 'completed_with_failures',
            completed_runs=len(records), failed_runs=failures, total_runs=len(jobs)))
        return int(bool(failures))
    except BaseException as exc:
        write_json(out/'active.json', dict(state='failed', completed_runs=len(records), error=str(exc)))
        raise
    finally:
        if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    raise SystemExit(main())

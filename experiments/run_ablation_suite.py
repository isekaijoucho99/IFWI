"""Run the registered ten-way IFWI ablation, sequentially on one GPU."""
import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment_runtime import write_json

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIGS = ['baseline', 'depth_weighted_loss', 'attention', 'attention_residual',
           'attention_film', 'prior_only', 'prior_normalized_tv',
           'prior_normalized_charbonnier', 'data_huber', 'layerwise_conservative']
CONFIGS = LEGACY_CONFIGS + ['st_multiscale', 'st_time', 'st_space', 'st_joint']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--configs', nargs='+', choices=CONFIGS, default=LEGACY_CONFIGS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1 or args.log_interval < 1:
        parser.error('iterations and log-interval must be positive')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.configs)) != len(args.configs):
        parser.error('Duplicate seeds/configs are not independent replicates')
    out = args.output_dir.resolve()
    commands = []
    for seed in args.seeds:
        for name in args.configs:
            config_name = 'feature_baseline' if name == 'baseline' else name
            command = [sys.executable, '-X', 'utf8', '-u', str(ROOT/'experiments/run_experiment.py'),
                       '--config', str(ROOT/'experiments/configs'/f'{config_name}.yaml'),
                       '--output-dir', str(out/'runs'), '--device', args.device,
                       '--iterations', str(args.iterations), '--log-interval', str(args.log_interval),
                       '--seed', str(seed)]
            commands.append({'name': name, 'seed': seed, 'command': command})
    if args.dry_run:
        import json
        print(json.dumps(commands, indent=2))
        return 0
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a new output directory; completed and interrupted suites are preserved')
    (out/'logs').mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT/'experiments').rglob('*.py')) + sorted((ROOT/'experiments/configs').glob('*.yaml'))
    paths += [ROOT/n for n in ['ifwi_modules.py', 'rnn_fd.py', 'generator.py', 'plot_functions.py']]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def verify_frozen_sources():
        for name, digest in hashes.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Experiment source/config changed during suite: '+name)
    write_json(out/'source_hashes.json', hashes)
    write_json(out/'manifest.json', commands)
    records = []
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for job in commands:
            verify_frozen_sources()
            log = out/'logs'/f"{job['name']}_seed{job['seed']}.log"
            write_json(out/'active.json', dict(job, state='running', log=str(log),
                completed_runs=len(records), total_runs=len(commands)))
            print(f"START {job['name']} seed={job['seed']}", flush=True)
            start = time.perf_counter()
            with log.open('wb') as stream:
                code = subprocess.run(job['command'], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT).returncode
            records.append(dict(job, exit_code=code, seconds=time.perf_counter()-start))
            write_json(out/'runs.json', records)
            print('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-2:]), flush=True)
            if code: raise RuntimeError(f"Failed {job['name']} seed={job['seed']}; see {log}")
            verify_frozen_sources()
        with (out/'logs/comparison.log').open('wb') as stream:
            subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'experiments/compare_results.py'),
                '--results-dir', str(out/'runs')], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        verify_frozen_sources()
        write_json(out/'active.json', {'state': 'completed', 'run_count': len(records)})
    except BaseException as exc:
        write_json(out/'active.json', {'state': 'failed', 'completed_runs': len(records), 'error': str(exc)})
        raise
    finally:
        if os.name == 'nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

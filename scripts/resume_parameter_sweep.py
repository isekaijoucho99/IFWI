"""Continue a frozen sweep from saved state, without changing training sources."""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from parameter_sweep import validate_case
from baseline_protocol import ORIGINAL_PROTOCOL, validate_baseline_protocol, validate_original_sources


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def inside(suite, path):
    suite, path = Path(suite).resolve(), Path(path).resolve()
    if path == suite or not path.is_relative_to(suite):
        raise ValueError('Recovery path must stay inside the suite: '+str(path))
    return path


def job_root(suite, job):
    if job['directory'] != f"runs/{job['name']}_seed{job['seed']}":
        raise ValueError('Unexpected registered job directory')
    return inside(suite, Path(suite)/job['directory'])


def verify_suite(suite):
    import yaml
    suite = Path(suite).resolve()
    manifest = read(suite/'manifest.json')
    original = manifest.get('protocol') == ORIGINAL_PROTOCOL
    if any(job.get('config', {}).get('execution', {}).get('protocol') == ORIGINAL_PROTOCOL
           for job in manifest['jobs']) and not original:
        raise ValueError('Original jobs require the original suite protocol marker')
    preliminary = manifest.get('preliminary', False)
    if original:
        if not isinstance(preliminary, bool) or manifest.get('seeds') != [3]:
            raise ValueError('Original suite requires seed=3 and an explicit boolean preliminary flag')
        # A rewritten caller manifest cannot redefine the protected author files.
        validate_original_sources(suite/'source')
    for name, digest in read(suite/'source_hashes.json').items():
        path = inside(suite/'source', suite/'source'/name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source/data/config changed: '+name)
    baselines = {j['seed']: j['config'] for j in manifest['jobs'] if j['factor']=='baseline'}
    if len(baselines) != sum(j['factor']=='baseline' for j in manifest['jobs']):
        raise ValueError('Duplicate registered baseline for a seed')
    for job in manifest['jobs']:
        job_root(suite, job)
        if original:
            if job['seed'] != 3 or job['config'].get('seed') != job['seed']:
                raise ValueError('Original job/config seed differs: '+job['name'])
            validate_baseline_protocol(job['config'], allow_treatment=job['factor']!='baseline',
                                       preliminary=preliminary)
        if job['seed'] not in baselines:
            raise ValueError('Missing registered baseline for seed: '+str(job['seed']))
        validate_case(baselines[job['seed']], job, preliminary=preliminary)
        if job['config']['training']['max_iterations'] != manifest['iterations']:
            raise ValueError('Registered budget mismatch')
        path = inside(suite, suite/'configs'/f"{job['name']}.yaml")
        if yaml.safe_load(path.read_text(encoding='utf-8-sig')) != job['config']:
            raise ValueError('Executed YAML differs from registered config: '+job['name'])
    return manifest


def original_job(job):
    return job.get('config', {}).get('execution', {}).get('protocol') == ORIGINAL_PROTOCOL


def completed_baseline_bundle(suite, manifest=None):
    """Locate an explicitly saved import bundle, including historical suites."""
    suite = Path(suite).resolve()
    if manifest is None:
        manifest = read(suite/'manifest.json') if (suite/'manifest.json').is_file() else {}
    name = manifest.get('completed_baseline_bundle')
    if name is None and (suite/'completed_baseline_reference.json').is_file():
        name = 'completed_legacy_baseline'
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError('Completed baseline bundle must be a relative suite path')
    path = inside(suite, suite/name)
    if not path.is_dir():
        raise ValueError('Completed baseline bundle is missing: '+str(path))
    return path


def plan_job(suite, job):
    directory = job_root(suite, job)
    candidates = list(directory.glob('*/status.json'))
    if len(candidates) > 1:
        raise ValueError('Ambiguous run directory: '+job['name'])
    if not candidates:
        return {'action': 'start'}
    run = inside(suite, candidates[0].parent)
    status = read(run/'status.json')
    config = read(run/'config.json')
    registered = copy.deepcopy(job['config'])
    registered_seed = registered.pop('seed', job['seed'])
    if config.pop('seed', None) != job['seed'] or registered_seed != job['seed'] or config != registered:
        raise ValueError('Existing run config/seed differs: '+job['name'])
    original = original_job(job)
    if original and (run/'last.pth').exists():
        raise ValueError('Original protocol cannot resume a schema2 last.pth checkpoint')
    if status['state']=='completed':
        if status['completed_updates'] != job['config']['training']['max_iterations']:
            raise ValueError('Completed run has a different update budget')
        return {'action': 'skip', 'run': run}
    saved_updates = status.get('completed_updates')
    if original:
        numbered = []
        for path in (run/'checkpoints').glob('MarmousiI_random-checkpoint-*.pth'):
            match = re.fullmatch(r'MarmousiI_random-checkpoint-([0-9]+)\.pth', path.name)
            if match:
                update = int(match.group(1))
                if update < 1:
                    raise ValueError('Original checkpoint update must be positive: '+path.name)
                numbered.append((update, path))
        if not numbered:
            raise ValueError('Interrupted original run has no numbered checkpoint: '+str(run))
        saved_updates, checkpoint = max(numbered, key=lambda entry: entry[0])
        if saved_updates >= job['config']['training']['max_iterations']:
            raise ValueError('Original checkpoint already reaches the budget; completion needs review')
    else:
        checkpoint = run/'last.pth'
    if not checkpoint.is_file():
        raise ValueError('Interrupted run has no checkpoint; refusing to replay silently: '+str(run))
    return {'action': 'resume', 'run': run, 'checkpoint': checkpoint,
            'saved_updates': saved_updates}


def command_for_job(suite, job, plan, python, device):
    suite = Path(suite).resolve()
    manifest = read(suite/'manifest.json') if (suite/'manifest.json').is_file() else {}
    bundle = completed_baseline_bundle(suite, manifest) if job['name']=='baseline' else None
    importer = bundle is not None and plan['action']=='start'
    entry = 'import_completed_baseline.py' if importer else 'run_experiment.py'
    command = [str(python), '-X', 'utf8', '-u', str(suite/'source/experiments'/entry),
               '--config', str(suite/'configs'/f"{job['name']}.yaml"),
               '--output-dir', str(job_root(suite, job)), '--device', device,
               '--seed', str(job['seed'])]
    if importer:
        command += ['--legacy-run', str(bundle)]
    elif original_job(job) and manifest.get('preliminary', False):
        command += ['--preliminary']
    if plan['action']=='resume':
        command += ['--resume', str(plan['checkpoint'])]
    return command


def archive_interrupted_run(suite, job, plan):
    run = inside(suite, plan['run'])
    checkpoint = inside(suite, plan['checkpoint'])
    if not checkpoint.is_relative_to(run):
        raise ValueError('Recovery checkpoint must belong to the interrupted run')
    relative_checkpoint = checkpoint.relative_to(run)
    destination = inside(suite, Path(suite)/'archive'/f"{job['name']}_seed{job['seed']}"/run.name)
    if destination.exists():
        raise ValueError('Recovery archive already exists')
    # Both resolved targets have been checked before moving a directory.
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = read(run/'status.json')
    status.update(state='interrupted', recovery_reason='Continue from saved checkpoint in a new run directory')
    write(run/'status.json', status)
    run.rename(destination)
    return destination/relative_checkpoint


@contextlib.contextmanager
def exclusive_lock(suite):
    with (Path(suite)/'recovery.lock').open('a+b') as stream:
        if stream.tell()==0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name=='nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def continue_suite(suite, device, dry_run=False):
    manifest = verify_suite(suite)
    plans = [(job, plan_job(suite, job)) for job in manifest['jobs']]
    if dry_run:
        for job, plan in plans:
            print(json.dumps(dict(name=job['name'], action=plan['action'],
                                 saved_updates=plan.get('saved_updates'),
                                 checkpoint=str(plan.get('checkpoint', '')))))
        return
    completed = sum(plan['action']=='skip' for _, plan in plans)
    env = dict(os.environ, PYTHONUTF8='1')
    if manifest.get('protocol') != ORIGINAL_PROTOCOL:
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sleep_guard = False
    if os.name=='nt':
        import ctypes
        sleep_guard = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    events_path = suite/'recovery_events.json'
    events = read(events_path) if events_path.exists() else []
    try:
        for job, plan in plans:
            if plan['action']=='skip':
                print('SKIP completed '+job['name'], flush=True)
                continue
            verify_suite(suite)
            if plan['action']=='resume':
                plan['checkpoint'] = archive_interrupted_run(suite, job, plan)
            command = command_for_job(suite, job, plan, sys.executable, device)
            event = dict(name=job['name'], action=plan['action'], saved_updates=plan.get('saved_updates'),
                         checkpoint=str(plan.get('checkpoint', '')), command=command,
                         started_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                         orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            if plan['action']=='resume':
                event['checkpoint_sha256'] = hashlib.sha256(plan['checkpoint'].read_bytes()).hexdigest()
            events.append(event); write(events_path, events)
            log = suite/'logs'/f"{job['name']}_seed{job['seed']}.log"
            write(suite/'active.json', dict(state='running', name=job['name'], seed=job['seed'],
                  completed_runs=completed, total_runs=len(plans), log=str(log), launcher='Windows Task Scheduler'))
            print(f"START {job['name']}: {plan['action']}; saved updates={plan.get('saved_updates')}", flush=True)
            with log.open('ab') as stream:
                stream.write(f"\nRECOVERY: {plan['action']}; saved updates={plan.get('saved_updates')}; unsaved updates replayed\n".encode())
                stream.flush()
                code = subprocess.run(command, cwd=suite/'source', env=env,
                                      stdout=stream, stderr=subprocess.STDOUT).returncode
            event.update(exit_code=code, finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            write(events_path, events)
            if code:
                raise RuntimeError(f'{job["name"]} exited with code {code}; inspect {log}')
            if plan_job(suite, job)['action'] != 'skip':
                raise RuntimeError('Child exited without a completed run')
            completed += 1
            subprocess.run([sys.executable, '-X', 'utf8',
                            str(suite/'source/experiments/summarize_parameter_sweep.py'), '--suite', str(suite)],
                           cwd=suite/'source', env=env, check=True)
        write(suite/'active.json', dict(state='completed', completed_runs=completed, total_runs=len(plans)))
    except BaseException as error:
        current = read(suite/'active.json') if (suite/'active.json').exists() else {}
        current.update(state='failed', error=str(error))
        write(suite/'active.json', current)
        raise
    finally:
        if sleep_guard:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    suite = args.suite.resolve()
    if args.dry_run:
        continue_suite(suite, args.device, True)
        return
    with (suite/'recovery.console.log').open('a', encoding='utf-8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                with exclusive_lock(suite):
                    continue_suite(suite, args.device)
            except BaseException:
                traceback.print_exc()
                raise


if __name__=='__main__':
    main()

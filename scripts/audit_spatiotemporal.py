"""Archive frozen inputs and audit completed spatiotemporal experiments on CPU.

--snapshot-only is safe while training is running. A normal audit requires all
five registered 100-update runs. No wave propagation or GPU operation is used.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'baseline', 'st_multiscale', 'st_time', 'st_space', 'st_joint'}


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON value in ' + str(path) + ': ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def source_path(root, name):
    relative = Path(name)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()) or '..' in relative.parts:
        raise ValueError('Unsafe manifest path: ' + name)
    return path


def frozen_inputs(suite, root):
    frozen = read_json(suite / 'source_hashes.json')
    original = read_json(root / 'source_manifest.json')
    if not frozen or len([name for name in original if name.lower().endswith('.csv')]) != 2:
        raise ValueError('Require frozen sources and two original CSV files in source_manifest.json')
    expected = dict(frozen)
    for name, value in original.items():
        if name in expected and expected[name] != value:
            raise ValueError('Conflicting original/source digest: ' + name)
        expected[name] = value
    content = {}
    for name, value in expected.items():
        data = source_path(root, name).read_bytes()
        if digest(data) != value:
            raise ValueError('Frozen/original file changed: ' + name)
        content[name] = data
    return frozen, original, content


def snapshot_suite(suite, root=ROOT):
    suite, root = Path(suite).resolve(), Path(root).resolve()
    frozen, original, content = frozen_inputs(suite, root)
    archive = suite / 'source_snapshot.zip'
    index = suite / 'snapshot_manifest.json'
    if archive.exists() or index.exists():
        if not archive.is_file() or not index.is_file():
            raise ValueError('Incomplete existing snapshot; refusing to overwrite it')
        result = read_json(index)
        if digest(archive.read_bytes()) != result['archive_sha256']:
            raise ValueError('Snapshot ZIP digest mismatch')
        with zipfile.ZipFile(archive) as saved:
            if len(saved.namelist()) != len(set(saved.namelist())) or set(saved.namelist()) != set(result['files']):
                raise ValueError('Snapshot ZIP member mismatch')
            for name, value in result['files'].items():
                if digest(saved.read(name)) != value:
                    raise ValueError('Snapshot member digest mismatch: ' + name)
            for name, data in content.items():
                if result['files'].get(name) != digest(data):
                    raise ValueError('Snapshot differs from frozen/original manifest: ' + name)
            for name, path in [('source_manifest.json', root/'source_manifest.json'),
                               ('audit_metadata/source_hashes.json', suite/'source_hashes.json')]:
                if saved.read(name) != path.read_bytes():
                    raise ValueError('Snapshot manifest differs: ' + name)
        return result
    extras = list((root/'tests').rglob('*.py'))
    extras += list(root.glob('*requirements*.txt')) + list((root/'experiments').glob('*requirements*.txt'))
    extras += [root/'source_manifest.json']
    for optional in (root/'test_framework.py', root/'scripts/audit_spatiotemporal.py'):
        if optional.is_file():
            extras.append(optional)
    for path in extras:
        content[path.relative_to(root).as_posix()] = path.read_bytes()
    content['audit_metadata/source_hashes.json'] = (suite/'source_hashes.json').read_bytes()
    if (suite/'manifest.json').is_file():
        content['audit_metadata/manifest.json'] = (suite/'manifest.json').read_bytes()
    temporary = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as saved:
        for name, data in sorted(content.items()):
            saved.writestr(name, data)
    result = {'schema': 1, 'archive': archive.name, 'archive_sha256': digest(temporary.read_bytes()),
              'files': {name: digest(data) for name, data in sorted(content.items())},
              'frozen_source_count': len(frozen), 'original_file_count': len(original),
              'note': 'Tests and requirements captured at snapshot creation; frozen training inputs verified against manifests.'}
    os.replace(temporary, archive)
    write_json(index, result)
    return result


@lru_cache(maxsize=1)
def numerics():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    return np, torch


def finite_tree(value, label):
    np, torch = numerics()
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite tensor: ' + label)
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite array: ' + label)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, label + '/' + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            finite_tree(item, label + '/' + str(index))
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError('Nonfinite scalar: ' + label)


def close(actual, expected, label):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), rel_tol=2e-6, abs_tol=1e-10):
        raise ValueError(f'Mismatch {label}: {actual} versus {expected}')


def array_digest(value):
    np, _ = numerics()
    a = np.ascontiguousarray(value)
    return digest(str((a.shape, str(a.dtype))).encode() + a.tobytes())


def equal_states(left, right, label):
    _, torch = numerics()
    if set(left) != set(right):
        raise ValueError('State key mismatch: ' + label)
    for key in left:
        if left[key].dtype != right[key].dtype or not torch.equal(left[key], right[key]):
            raise ValueError('State tensor mismatch: ' + label + '/' + key)


def reconstruct_velocity(state, config, shape):
    """Evaluate the saved vanilla SIREN on CPU; never construct the wave solver.

    The verified run_experiment.build_model protocol fixes mean=3, std=1 km/s.
    Unsupported network protocols fail rather than silently using these formulas.
    """
    np, torch = numerics()
    mc = config['model']
    if (mc.get('network_type', 'vanilla') != 'vanilla' or mc.get('activation', 'sine') != 'sine'
            or mc.get('dropout', False) or len(mc['neuron']) < 3):
        raise ValueError('CPU stage reconstruction requires the vanilla, dropout-free SIREN protocol')
    finite_tree(state, 'reconstruction state')
    dz = config['data']['dz']
    x, z = np.meshgrid(np.arange(shape[1]) * dz / 1000, np.arange(shape[0]) * dz / 1000)
    feature = torch.stack([torch.from_numpy(x).float(), torch.from_numpy(z).float()], dim=-1)[None]
    expected_keys = set()
    with torch.no_grad():
        count = len(mc['neuron']) - 1
        for index in range(count):
            key = f'linear.{index}.'
            expected_keys.add(key + 'weight')
            weight = state[key + 'weight']
            if weight.shape != (mc['neuron'][index+1], mc['neuron'][index]) or weight.dtype != torch.float32:
                raise ValueError('Unsupported SIREN weight shape/dtype: ' + key)
            bias = state.get(key + 'bias')
            if mc.get('bias', True):
                expected_keys.add(key + 'bias')
                if bias is None or bias.shape != (mc['neuron'][index+1],):
                    raise ValueError('Invalid SIREN bias: ' + key)
            elif bias is not None:
                raise ValueError('Bias contradicts network config')
            feature = torch.nn.functional.linear(feature, weight, bias)
            if index == 0 or not mc.get('outermost_linear', True) or index < count - 1:
                feature = torch.sin(mc.get('omega_0', 30) * feature)
        if set(state) != expected_keys or feature.shape != (1, *shape, 1):
            raise ValueError('Unsupported SIREN state or output shape')
        velocity = ((feature[0, ..., 0] + 3.) * 1000).numpy()
    finite_tree(velocity, 'CPU velocity')
    return velocity


def stage_metrics(truth, prediction, config):
    np, _ = numerics()
    from scipy.ndimage import gaussian_filter
    sys.path.insert(0, str(ROOT / 'experiments'))
    from improved_modules.evaluate_deep import evaluate_deep_layers
    settings = config['evaluation']
    depth = settings.get('depth_threshold', .5)
    dz = config['data']['dz']
    result = evaluate_deep_layers(truth, prediction, depth, settings.get('corner_size', .25), dz)
    result.pop('depth_profile', None)
    # Smooth the complete model first; cropping first changes the boundary condition.
    a, b = truth.astype(np.float64), prediction.astype(np.float64)
    smooth_a = gaussian_filter(a, sigma=10., mode='reflect')
    smooth_b = gaussian_filter(b, sigma=10., mode='reflect')
    start = int(a.shape[0] * depth)
    result.update(deep_background_rmse_mps=float(np.sqrt(np.mean((smooth_b[start:] - smooth_a[start:])**2))),
                  deep_detail_rmse_mps=float(np.sqrt(np.mean(((b-smooth_b)[start:] - (a-smooth_a)[start:])**2))),
                  sigma_cells=10., dz_m=float(dz), sigma_m=10.*float(dz),
                  scale_diagnostic='Operational spatial smoothing scale; not fault-location truth or waveform temporal frequency')
    return result


def audit_run(path, frozen_sources, budget=100):
    np, torch = numerics()
    path = Path(path)
    config = read_json(path/'config.json')
    status = read_json(path/'status.json')
    summary = read_json(path/'training_summary.json')
    comparison = read_json(path/'comparison_contract.json')
    if status.get('state') != 'completed':
        raise ValueError('Run not completed: ' + path.name)
    last = torch.load(path/'last.pth', map_location='cpu', weights_only=False)
    best = torch.load(path/'best.pth', map_location='cpu', weights_only=False)
    if last.get('schema') != 2 or best.get('schema') != 2:
        raise ValueError('Unsupported checkpoint schema: ' + path.name)
    finite_tree(last, 'last checkpoint'); finite_tree(best, 'best checkpoint')
    if summary['parameter_count'] != sum(value.numel() for value in last['model'].values()):
        raise ValueError('Parameter count differs from checkpoint')
    for value in (last['completed_updates'], status['completed_updates'], summary['completed_updates'],
                  comparison['budget'], config['training']['max_iterations']):
        if value != budget:
            raise ValueError('Expected complete update budget: ' + path.name)
    if last['config'] != config or comparison['seed'] != config['seed']:
        raise ValueError('Checkpoint/config identity mismatch: ' + path.name)
    contracted = dict(config)
    for name in ('experiment_name', 'description', 'evaluation'):
        contracted.pop(name, None)
    contracted['training'] = {k:v for k,v in config['training'].items() if k not in ('max_iterations','log_interval')}
    if last['contract']['config'] != contracted:
        raise ValueError('Resume contract config mismatch: ' + path.name)
    if last['contract']['sources'] != frozen_sources or comparison['sources'] != frozen_sources:
        raise ValueError('Checkpoint source mismatch: ' + path.name)
    selection = 'data_mse' if config.get('spatiotemporal', {}).get('enabled') else 'total_loss'
    for saved in (last, best, summary):
        if saved.get('selection_metric') != selection:
            raise ValueError('Selection metric mismatch: ' + path.name)
    with (path/'loss_history.csv').open(newline='', encoding='utf-8') as stream:
        history = list(csv.DictReader(stream))
    if len(history) != budget or len(last['history']) != budget:
        raise ValueError('Incomplete checkpoint or CSV history: ' + path.name)
    for update, (row, saved) in enumerate(zip(history, last['history']), 1):
        if int(row['completed_updates']) != update or saved['completed_updates'] != update:
            raise ValueError('Noncontiguous update history: ' + path.name)
        if set(row) != set(saved):
            raise ValueError('History fields differ from checkpoint: ' + path.name)
        for field, value in saved.items():
            if value is None:
                if row[field] != '':
                    raise ValueError('CSV null mismatch: ' + field)
            elif isinstance(value, (int, float)):
                if float(row[field]) != value or not math.isfinite(value):
                    raise ValueError('CSV/checkpoint numeric mismatch: ' + field)
            elif row[field] != str(value):
                raise ValueError('CSV/checkpoint text mismatch: ' + field)
        for field in ('loss_before_update','data_loss_before_update','prior_loss_before_update',
                      'data_mse_before_update','loss_after_update','data_loss_after_update',
                      'data_mse_after_update','gradient_norm'):
            if saved[field] is not None and saved[field] < 0:
                raise ValueError('Negative loss/gradient diagnostic: ' + field)
        rates = json.loads(row['learning_rates'])
        if not rates or any(not math.isfinite(v) or v <= 0 for v in rates):
            raise ValueError('Invalid learning-rate history')
    metric_field = 'data_mse_after_update' if selection == 'data_mse' else 'loss_after_update'
    evaluated = [row for row in last['history'] if row[metric_field] is not None]
    if not evaluated or last['history'][-1][metric_field] is None:
        raise ValueError('Missing saved final objective')
    chosen = min(evaluated, key=lambda row: row[metric_field])
    update, score = chosen['completed_updates'], chosen[metric_field]
    if update != last['best_update'] or update != best['completed_updates'] or update != summary['best_update']:
        raise ValueError('Incorrect best update: ' + path.name)
    for label, value in [('last best',last['best_loss']),('last selection',last['selection_score']),
                         ('best selection',best['selection_score']),('summary best',summary['best_loss']),
                         ('summary selection',summary['selection_score'])]:
        close(value, score, path.name+'/'+label)
    close(best['loss_after_update'], chosen['loss_after_update'], 'best objective')
    if best['objective_step'] != update-1 or last['objective_step'] != budget-1:
        raise ValueError('Objective step mismatch')
    equal_states(last['best_state'], best['model'], 'best snapshot')
    arrays = {}
    for name in ('v_true','initial_velocity','last_velocity','best_velocity','v_pred'):
        arrays[name] = np.load(path/(name+'.npy'), allow_pickle=False)
        if arrays[name].ndim != 2 or not np.isfinite(arrays[name]).all():
            raise ValueError('Invalid velocity artifact: ' + name)
    truth = arrays['v_true']
    if any(a.shape != truth.shape for a in arrays.values()):
        raise ValueError('Velocity grid mismatch')
    if not np.array_equal(arrays['best_velocity'], arrays['v_pred']):
        raise ValueError('best_velocity and v_pred differ')
    observed = np.load(path/'observed.npy', allow_pickle=False)
    if not np.isfinite(observed).all():
        raise ValueError('Nonfinite observations')
    for key, value in [('truth',truth[None]),('observed',observed)]:
        if last['contract'][key] != array_digest(value):
            raise ValueError('Data contract hash mismatch: ' + key)
    for name, state in [('last_velocity',last['model']),('best_velocity',best['model'])]:
        predicted = reconstruct_velocity(state, config, truth.shape)
        # CUDA/CPU sin/GEMM roundoff is expected; this is not a bitwise resume test.
        if not np.allclose(predicted, arrays[name], rtol=2e-5, atol=.02):
            raise ValueError('Saved velocity disagrees with CPU checkpoint prediction: ' + name)
    final_metrics = read_json(path/'final_metrics.json')
    best_metrics = read_json(path/'metrics.json')
    finite_tree(final_metrics, 'final metrics'); finite_tree(best_metrics, 'best metrics')
    if final_metrics['completed_updates'] != budget or best_metrics['best_update'] != update:
        raise ValueError('Metrics snapshot index mismatch')
    close(final_metrics['data_mse'], last['history'][-1]['data_mse_after_update'], 'final raw MSE')
    close(best_metrics['data_mse'], chosen['data_mse_after_update'], 'best raw MSE')
    close(final_metrics['data_objective'],last['history'][-1]['data_loss_after_update'],'final data objective')
    close(best_metrics['data_objective'],chosen['data_loss_after_update'],'best data objective')
    for label, metrics, pred in [('final',final_metrics,arrays['last_velocity']),('best',best_metrics,arrays['best_velocity'])]:
        actual = stage_metrics(truth, pred, config)
        for key in ('full_rmse','deep_rmse','deep_ssim'):
            if key in metrics and actual[key] is not None:
                close(metrics[key], actual[key], label+'/'+key)
    return {'record': {'experiment':config['experiment_name'],'seed':config['seed'],'run_id':path.name,
                      'completed_updates':budget,'best_update':update,'selection_metric':selection,
                      'selection_score':score,'last_checkpoint_sha256':digest((path/'last.pth').read_bytes()),
                      'best_checkpoint_sha256':digest((path/'best.pth').read_bytes())},
            'path':path,'config':config,'last':last,'truth':truth,'initial':arrays['initial_velocity'],
            'comparison':comparison}


def audit_suite(suite, root=ROOT):
    np, torch = numerics()
    suite, root = Path(suite).resolve(), Path(root).resolve()
    snapshot = snapshot_suite(suite, root)
    frozen = read_json(suite/'source_hashes.json')
    sources = {k:v for k,v in frozen.items() if k.endswith('.py')}
    jobs = read_json(suite/'manifest.json')
    if len(jobs) != 5 or {job['name'] for job in jobs} != GROUPS or len({job['seed'] for job in jobs}) != 1:
        raise ValueError('Require the five registered groups with one common seed')
    expected = {(job['name'],job['seed']) for job in jobs}
    for job in jobs:
        cmd = job['command']
        if int(cmd[cmd.index('--iterations')+1]) != 100:
            raise ValueError('This audit requires 100-update experiments')
    results = []
    for path in sorted((suite/'runs').iterdir()):
        if not path.is_dir() or path.name == 'comparison':
            continue
        cfg = read_json(path/'config.json')
        key = (cfg['experiment_name'],cfg['seed'])
        if key not in expected:
            raise ValueError('Unexpected or duplicate run: ' + path.name)
        expected.remove(key)
        results.append(audit_run(path, sources))
    if expected:
        raise ValueError('Incomplete suite: ' + str(sorted(expected)))
    for result in results[1:]:
        if result['comparison'] != results[0]['comparison']:
            raise ValueError('Different comparison protocols')
        if not np.array_equal(result['initial'],results[0]['initial']):
            raise ValueError('Initial velocity arrays differ')
        if not np.array_equal(result['truth'],results[0]['truth']):
            raise ValueError('Truth arrays differ')
    stage_rows, missing_stages = [], []
    stage_root = suite/'stage_snapshots'
    for result in results:
        candidates = list(dict.fromkeys([stage_root/result['path'].name,stage_root/result['config']['experiment_name']]))
        existing = [path for path in candidates if path.is_dir()]
        if len(existing) > 1:
            raise ValueError('Ambiguous stage snapshot directories: ' + result['path'].name)
        directory = existing[0] if existing else candidates[0]
        for update in (30,70,100):
            paths = [directory/f'update{update:06d}.pth', directory/f'update{update}.pth']
            present = [path for path in paths if path.is_file()]
            if len(present) > 1:
                raise ValueError('Ambiguous stage snapshot names: ' + str(directory))
            if not present:
                missing_stages.append(str(paths[0].relative_to(suite)))
                continue
            path = present[0]
            ck = torch.load(path,map_location='cpu',weights_only=False)
            finite_tree(ck, 'stage checkpoint')
            if (ck.get('schema') != 2 or ck['completed_updates'] != update or ck['objective_step'] != update-1
                    or ck['config'] != result['config'] or ck['contract'] != result['last']['contract']
                    or ck['history'] != result['last']['history'][:update]):
                raise ValueError('Stage snapshot metadata/history mismatch: ' + str(path))
            row = ck['history'][update-1]
            if row['data_mse_after_update'] is None:
                raise ValueError('Stage snapshot lacks matching raw MSE')
            if update == 100:
                equal_states(ck['model'],result['last']['model'],'final stage')
            prediction = reconstruct_velocity(ck['model'],result['config'],result['truth'].shape)
            stage_rows.append(dict(experiment=result['config']['experiment_name'],seed=result['config']['seed'],
                run_id=result['path'].name,completed_updates=update,objective_step=update-1,
                cutoff_hz=row.get('cutoff_hz'),data_mse=row['data_mse_after_update'],
                training_objective=row['loss_after_update'],checkpoint_sha256=digest(path.read_bytes()),
                **stage_metrics(result['truth'],prediction,result['config'])))
    if stage_rows:
        target = suite/'stage_metrics.csv'
        temporary = target.with_suffix('.csv.tmp')
        with temporary.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(stage_rows[0])); writer.writeheader(); writer.writerows(stage_rows)
        os.replace(temporary,target)
    report = {'state':'passed','run_count':len(results),'completed_updates':100,
              'frozen_and_original_hashes_verified':True,'initial_models_exactly_equal':True,
              'snapshot_archive_sha256':snapshot['archive_sha256'],'runs':[r['record'] for r in results],
              'stage_metrics_count':len(stage_rows),'missing_optional_stage_snapshots':missing_stages,
              'stage_metric_note':'Gaussian sigma=10 cells on full grids before deep crop, reflect boundary; operational spatial scale only. Raw MSE read from matching checkpoint history; no wave propagation.',
              'cpu_velocity_tolerance':{'rtol':2e-5,'atol_mps':.02},
              'audit_script_sha256':digest(Path(__file__).read_bytes())}
    write_json(suite/'checkpoint_audit.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    try:
        result = snapshot_suite(args.suite) if args.snapshot_only else audit_suite(args.suite)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        if not args.snapshot_only and args.suite.is_dir():
            write_json(args.suite/'checkpoint_audit.json',{'state':'failed','error':str(exc)})
        parser.error(str(exc))
    display = ({'archive':result['archive'],'archive_sha256':result['archive_sha256'],'file_count':len(result['files'])}
               if args.snapshot_only else result)
    print(json.dumps(display, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

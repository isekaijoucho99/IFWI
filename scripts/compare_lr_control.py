"""Read-only audit for the registered modern13 constant-versus-cosine LR pair.

Does not train, load pickled checkpoints, choose a model using velocity truth,
or write inside either source run. All existing training code stays unchanged.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = '22fa517bc9c9217e9c08d47c585a56a916702a91'
SOURCE_MANIFEST = ROOT / 'experiments/configs/lr_control_sources.json'
SCHEDULE = {'warmup_epochs': 0, 'max_epochs': 4001, 'eta_min': 1e-5}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_pair(constant, cosine):
    """Reject extra factors and reject two identically but incorrectly changed arms."""
    sys.path.insert(0, str(ROOT / 'experiments'))
    from baseline_protocol import baseline_config
    reference = baseline_config()
    for key in ('experiment_name', 'description', 'execution'):
        reference.pop(key)
    reference['optimizer']['scheduler_params'] = SCHEDULE
    for supplied, enabled in ((constant, False), (cosine, True)):
        normalized = copy.deepcopy(supplied)
        for key in ('experiment_name', 'description'):
            normalized.pop(key, None)
        reference['optimizer']['use_scheduler'] = enabled
        # Canonical JSON preserves bool/int distinction that plain dict equality loses.
        if json_hash(normalized) != json_hash(reference):
            raise ValueError('LR configuration is not the registered modern13 pair')
    return {'upstream_commit': UPSTREAM_COMMIT, 'only_treatment': 'optimizer.use_scheduler',
            'constant_config_sha256': json_hash(constant), 'cosine_config_sha256': json_hash(cosine),
            'historical_frozen_runtime_verified': False}


def load_config(path):
    path = Path(path)
    with path.open(encoding='utf-8') as stream:
        return json.load(stream) if path.suffix == '.json' else yaml.safe_load(stream)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, str(value.dtype))).encode() + value.tobytes()).hexdigest()


def _number(value, label, nonnegative=True):
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError('Invalid ' + label)
    return value


def _close(actual, expected, label):
    if not math.isclose(_number(actual, label, False), expected, rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError('Mismatch: ' + label)


def _read_run(path, enabled):
    config = _json(path / 'config.json')
    status = _json(path / 'status.json')
    summary = _json(path / 'training_summary.json')
    contract = _json(path / 'comparison_contract.json')
    environment = _json(path / 'environment.json')
    if status.get('state') != 'completed' or any(item.get(key) != 4001 for item, key in
            ((status, 'completed_updates'), (summary, 'completed_updates'), (contract, 'budget'))):
        raise ValueError('Both runs must be completed at exactly 4001 updates')
    if summary.get('resumed_from_updates') != 0 or summary.get('executed_updates') != 4001:
        raise ValueError('This first LR comparison accepts fresh runs only; resumed initial metadata is ambiguous')
    expected_sources = _json(SOURCE_MANIFEST)
    if expected_sources.get('upstream_commit') != UPSTREAM_COMMIT:
        raise ValueError('Unexpected source manifest version')
    for sources in (contract.get('sources'), environment.get('source_hashes')):
        if not sources or sources != expected_sources['sources']:
            raise ValueError('Run source hashes differ from the pinned modern runtime')
    for key, expected in [('seed', 3), ('clip_grad', None), ('data', config['data']),
                          ('params', {'dz': 15, 'dt': .0019, 'nt': 1000}),
                          ('evaluation', {'depth_threshold': .5, 'corner_size': .25}),
                          ('backbone', {k: config['model'][k] for k in
                           ('neuron', 'omega_0', 'activation', 'outermost_linear', 'dropout')})]:
        if contract.get(key) != expected:
            raise ValueError('Run comparison contract differs from configuration: ' + key)
    arrays = {}
    for name in ('v_true', 'initial_velocity', 'last_velocity', 'best_velocity', 'observed'):
        value = np.load(path / (name + '.npy'), mmap_mode='r', allow_pickle=False)
        shape = (1, 13, 1000, 288) if name == 'observed' else (94, 288)
        if value.shape != shape or value.dtype != np.dtype('float32') or not np.isfinite(value).all():
            raise ValueError('Unexpected shape/dtype/nonfinite array: ' + name)
        arrays[name] = value
    if (arrays['v_true'] <= 0).any():
        raise ValueError('Nonpositive truth')
    if contract.get('truth') != _array_hash(arrays['v_true'][None]) or contract.get('observed') != _array_hash(arrays['observed']):
        raise ValueError('Saved data arrays differ from run contract')
    wavelet_hash = contract.get('wavelet', '')
    if len(wavelet_hash) != 64 or any(c not in '0123456789abcdef' for c in wavelet_hash):
        raise ValueError('Invalid wavelet hash')
    with (path / 'loss_history.csv').open(newline='', encoding='utf-8') as stream:
        history = list(csv.DictReader(stream))
    if [int(row['completed_updates']) for row in history] != list(range(1, 4002)):
        raise ValueError('Incomplete or duplicate completed updates in loss history')
    evaluated = []
    for update, row in enumerate(history, 1):
        expected_lr = (1e-5 + 9e-5 * (1 + math.cos(math.pi * (update - 1) / 4000)) / 2) if enabled else 1e-4
        rates = json.loads(row['learning_rates'])
        if len(rates) != 1 or not math.isclose(float(rates[0]), expected_lr, rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError('Unexpected learning rate at update ' + str(update))
        row['lr'] = rates[0]
        _number(row['gradient_norm'], 'gradient norm')
        before = _number(row['data_mse_before_update'], 'before-update MSE')
        _close(row['loss_before_update'], before, 'plain MSE objective')
        _close(row['data_loss_before_update'], before, 'plain MSE data loss')
        _close(row['prior_loss_before_update'], 0., 'disabled prior')
        if row.get('cutoff_hz'):
            raise ValueError('Unexpected frequency schedule')
        save = update % 100 == 0 or update == 4001
        after_fields = ('data_mse_after_update', 'loss_after_update', 'data_loss_after_update')
        if any(bool(row.get(key)) != save for key in after_fields):
            raise ValueError('Unexpected post-update evaluation cadence')
        if save:
            score = _number(row['data_mse_after_update'], 'after-update MSE')
            for key in after_fields: _close(row[key], score, 'plain MSE evaluation')
            evaluated.append((score, update))
    best_score, best_update = min(evaluated)
    if summary.get('selection_metric') != 'total_loss' or summary.get('best_update') != best_update:
        raise ValueError('Best selection differs from minimum logged waveform MSE')
    _close(summary['best_loss'], best_score, 'best loss')
    _close(summary['selection_score'], best_score, 'selection score')
    rows = {}
    for label, filename, array_name, update, mse in [
            ('final', 'final_metrics.json', 'last_velocity', 4001, evaluated[-1][0]),
            ('best', 'metrics.json', 'best_velocity', best_update, best_score)]:
        metrics = _json(path / filename)
        key = 'completed_updates' if label == 'final' else 'best_update'
        if metrics.get(key) != update or metrics.get('selection_metric') != 'total_loss':
            raise ValueError('Wrong snapshot update/selection metric')
        _close(metrics['data_mse'], mse, 'snapshot MSE')
        prediction = arrays[array_name].astype(np.float64)
        error = prediction - arrays['v_true'].astype(np.float64)
        full_rmse = float(np.sqrt(np.mean(error ** 2)))
        deep_rmse = float(np.sqrt(np.mean(error[47:] ** 2)))
        _close(metrics['full_rmse'], full_rmse, 'full RMSE')
        _close(metrics['deep_rmse'], deep_rmse, 'deep RMSE')
        rows[label] = dict(arm='cosine' if enabled else 'constant', completed_updates=update,
                          waveform_mse=mse, full_rmse_mps=full_rmse, deep_rmse_mps=deep_rmse,
                          deep_velocity_std_mps=float(prediction[47:].std()),
                          deep_horizontal_tv_mps=float(np.abs(np.diff(prediction[47:], axis=1)).mean()),
                          velocity_min_mps=float(prediction.min()), velocity_max_mps=float(prediction.max()))
    inputs = ['config.json', 'status.json', 'training_summary.json', 'comparison_contract.json',
              'environment.json', 'loss_history.csv', 'final_metrics.json', 'metrics.json']
    inputs += [name + '.npy' for name in arrays]
    return dict(config=config, contract=contract, environment=environment, history=history,
                arrays=arrays, rows=rows, inputs={name: _file_hash(path / name) for name in inputs})


def compare_runs(constant_path, cosine_path, output_path):
    """Validate before writing; truth is used only for fixed final/best diagnostics."""
    constant_path, cosine_path, output_path = [Path(p).resolve() for p in (constant_path, cosine_path, output_path)]
    if constant_path == cosine_path:
        raise ValueError('Provide two distinct runs')
    if output_path.exists() or any(run == output_path or run in output_path.parents for run in (constant_path, cosine_path)):
        raise ValueError('Use a new output directory outside both input runs')
    report = validate_pair(_json(constant_path / 'config.json'), _json(cosine_path / 'config.json'))
    a, b = _read_run(constant_path, False), _read_run(cosine_path, True)
    if a['contract'] != b['contract']:
        raise ValueError('Run comparison contracts differ')
    for key in ('python', 'device', 'packages'):
        if key not in a['environment'] or not a['environment'][key] or a['environment'][key] != b['environment'].get(key):
            raise ValueError('Run environment differs or is missing: ' + key)
    if not np.array_equal(a['arrays']['initial_velocity'], b['arrays']['initial_velocity']):
        raise ValueError('Run initial velocities differ')
    report.update(best_selection='minimum post-update waveform MSE among logged points',
                  final_update=4001, independent_velocity_rmse_check=True,
                  initial_velocity_sha256=_array_hash(a['arrays']['initial_velocity']),
                  analysis_script_sha256=_file_hash(__file__),
                  source_manifest_sha256=_file_hash(SOURCE_MANIFEST),
                  input_file_sha256={'constant': a['inputs'], 'cosine': b['inputs']},
                  environment={k: a['environment'][k] for k in ('python', 'device', 'packages')},
                  limitations=['Single seed; no claim of convergence or velocity improvement',
                               'Historical frozen runtime has not been matched',
                               'Checkpoint losses are verified from exported metrics/logs, not re-propagated here',
                               'Wavelet hash is checked across contracts; the runner does not export wavelet.npy'])
    output_path.mkdir(parents=True)
    for name, label in [('fixed_final.csv', 'final'), ('waveform_best.csv', 'best')]:
        rows = [run['rows'][label] for run in (a, b)]
        with (output_path / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout='constrained')
    for run, label in ((a, 'constant'), (b, 'cosine')):
        h = run['history']; x = [int(row['completed_updates']) for row in h]
        axes[0].plot(np.asarray(x)-1, [float(r['data_mse_before_update']) for r in h], label=label, linewidth=.8)
        saved = [r for r in h if r['data_mse_after_update']]
        axes[0].scatter([int(r['completed_updates']) for r in saved], [float(r['data_mse_after_update']) for r in saved], s=10)
        axes[1].plot(x, [float(r['gradient_norm']) for r in h], label=label, linewidth=.8)
        axes[2].plot(x, [r['lr'] for r in h], label=label)
    for ax, ylabel in zip(axes, ['Waveform MSE', 'Parameter gradient norm', 'Learning rate']):
        ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend()
    axes[2].set_xlabel('Completed update (MSE line: before update; dots: after update)')
    fig.savefig(output_path / 'loss_grad_lr.png', dpi=150); plt.close(fig)
    (output_path / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check-configs', help='Validate the exact LR-only configurations without training')
    check.add_argument('--constant', type=Path, default=ROOT / 'experiments/configs/modern13_constant_lr.yaml')
    check.add_argument('--cosine', type=Path, default=ROOT / 'experiments/configs/modern13_cosine_lr.yaml')
    compare = sub.add_parser('compare', help='Compare completed fresh paired runs; never train')
    compare.add_argument('--constant-run', type=Path, required=True)
    compare.add_argument('--cosine-run', type=Path, required=True)
    compare.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'check-configs':
            report = validate_pair(load_config(args.constant), load_config(args.cosine))
        else:
            report = compare_runs(args.constant_run, args.cosine_run, args.output)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Validate and summarize a manifest-backed IFWI ablation suite on CPU.

Usage: python scripts/summarize_ablations.py --suite results/.../ablation100
Use --partial only for an explicitly marked preview of completed runs.
Training artifacts are read only; generated files live in <suite>/report.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

# Limit numerical workers before importing NumPy/Matplotlib. No Torch/GPU import.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np


LABELS = {
    'baseline': 'Baseline', 'depth_weighted_loss': 'Depth preconditioner',
    'attention': 'Legacy attention', 'attention_residual': 'Residual depth gate',
    'attention_film': 'Feature FiLM', 'prior_only': 'Legacy prior',
    'prior_normalized_tv': 'Normalized TV prior',
    'prior_normalized_charbonnier': 'Normalized Charbonnier',
    'data_huber': 'Huber data objective', 'layerwise_conservative': 'Conservative layerwise LR',
}
METRICS = ('data_mse', 'data_objective', 'full_rmse', 'deep_rmse', 'deep_ssim',
           'deep_gradient_fidelity', 'deep_velocity_std_mps', 'deep_horizontal_tv_mps',
           'velocity_min_mps', 'velocity_max_mps', 'bottom_left_rmse', 'bottom_right_rmse')


def read_json(path):
    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON value {value}: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant)


def command_value(command, flag):
    if flag not in command or command.index(flag) + 1 >= len(command):
        raise ValueError(f'Manifest command requires {flag}')
    return command[command.index(flag) + 1]


def array(path):
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f'Expected finite two-dimensional velocity: {path}')
    return value


def require_close(actual, expected, label):
    if not math.isfinite(float(actual)) or not np.isclose(actual, expected, rtol=2e-5, atol=1e-6):
        raise ValueError(f'Artifact mismatch for {label}: {actual} versus {expected}')


def validate_metrics(metrics, prediction, truth, depth, label):
    for key in METRICS:
        if key not in metrics:
            raise ValueError(f'Missing {key} in {label}')
        if metrics[key] is None:
            if key not in ('deep_ssim', 'deep_gradient_fidelity'):
                raise ValueError(f'Unexpected null {key} in {label}')
        elif not isinstance(metrics[key], (int, float)) or not math.isfinite(metrics[key]):
            raise ValueError(f'Invalid {key} in {label}')
    if metrics['data_mse'] < 0 or metrics['data_objective'] < 0:
        raise ValueError(f'Negative data loss in {label}')
    index = int(truth.shape[0] * depth)
    if not 0 <= index < truth.shape[0]:
        raise ValueError(f'Invalid deep region in {label}')
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    expected = {'full_rmse': np.sqrt(np.mean(error ** 2)),
                'deep_rmse': np.sqrt(np.mean(error[index:] ** 2)),
                'deep_velocity_std_mps': prediction[index:].std(),
                'deep_horizontal_tv_mps': np.abs(np.diff(prediction[index:], axis=1)).mean(),
                'velocity_min_mps': prediction.min(), 'velocity_max_mps': prediction.max()}
    for key, value in expected.items():
        require_close(metrics[key], value, label + '/' + key)


def load_suite(suite, partial=False):
    suite = Path(suite)
    manifest_path = suite / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('manifest.json is required, including for partial previews')
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError('Manifest must contain an explicit nonempty job list')
    expected = {}
    for job in manifest:
        key = (job['name'], int(job['seed']))
        if key in expected:
            raise ValueError(f'Duplicate job identity in manifest: {key}')
        budget = int(command_value(job['command'], '--iterations'))
        if budget < 1 or int(command_value(job['command'], '--seed')) != key[1]:
            raise ValueError(f'Invalid manifest budget or seed: {key}')
        expected[key] = budget
    sources = read_json(suite / 'source_hashes.json')
    runs, skipped, seen, contracts = [], [], set(), {}
    directories = sorted(p for p in (suite / 'runs').iterdir() if p.is_dir()) if (suite / 'runs').exists() else []
    for path in directories:
        if not (path / 'config.json').is_file():
            # compare_results.py writes this known non-run directory after the
            # suite finishes. Do not treat arbitrary configless folders as safe.
            if path.name == 'comparison' and all((path / name).is_file() for name in
                                                 ('runs.csv', 'by_seed.csv', 'summary.csv')):
                continue
            raise ValueError(f'Unexpected directory without run identity: {path}')
        config = read_json(path / 'config.json')
        key = (config['experiment_name'], int(config['seed']))
        if key not in expected or key in seen:
            raise ValueError(f'Unexpected or duplicate run identity: {key} at {path.name}')
        seen.add(key)
        status = read_json(path / 'status.json') if (path / 'status.json').is_file() else {'state': 'missing status'}
        if status.get('state') != 'completed':
            skipped.append(f'{key[0]} / seed={key[1]}: {status.get("state", "unknown state")}')
            continue
        budget = expected[key]
        contract = read_json(path / 'comparison_contract.json')
        summary = read_json(path / 'training_summary.json')
        if any(value != budget for value in (status['completed_updates'], summary['completed_updates'],
                                            config['training']['max_iterations'], contract['budget'])):
            raise ValueError(f'Completed update counts differ from manifest: {path.name}')
        if contract['seed'] != key[1]:
            raise ValueError(f'Contract seed differs from manifest: {path.name}')
        expected_sources = {name: digest for name, digest in sources.items() if name.endswith('.py')}
        if not expected_sources or contract['sources'] != expected_sources:
            raise ValueError(f'Run source hashes differ from suite snapshot: {path.name}')
        if key[1] in contracts and contracts[key[1]] != contract:
            raise ValueError(f'Comparison contract mismatch: {path.name}')
        contracts[key[1]] = contract
        with (path / 'loss_history.csv').open(newline='', encoding='utf-8') as stream:
            history = list(csv.DictReader(stream))
        if [int(row['completed_updates']) for row in history] != list(range(1, budget + 1)):
            raise ValueError(f'Incomplete or repeated loss-history updates: {path.name}')
        for row in history:
            for field in ('data_mse_before_update', 'data_mse_after_update'):
                if field not in row or (field == 'data_mse_before_update' and not row[field]):
                    raise ValueError(f'Missing raw MSE history: {path.name}')
                if row[field] and (not math.isfinite(float(row[field])) or float(row[field]) < 0):
                    raise ValueError(f'Invalid raw MSE history: {path.name}')
        truth, initial, final, best = (array(path / name) for name in
            ('v_true.npy', 'initial_velocity.npy', 'last_velocity.npy', 'best_velocity.npy'))
        if any(value.shape != truth.shape for value in (initial, final, best)):
            raise ValueError(f'Velocity array shapes differ: {path.name}')
        depth = contract['evaluation']['depth_threshold']
        final_metrics = read_json(path / 'final_metrics.json')
        best_metrics = read_json(path / 'metrics.json')
        if final_metrics['completed_updates'] != budget:
            raise ValueError(f'Fixed-final metrics have wrong update number: {path.name}')
        best_update = int(best_metrics['best_update'])
        if best_update != summary['best_update'] or not 1 <= best_update <= budget:
            raise ValueError(f'Invalid best update: {path.name}')
        validate_metrics(final_metrics, final, truth, depth, path.name + '/final')
        validate_metrics(best_metrics, best, truth, depth, path.name + '/best')
        if not history[-1]['data_mse_after_update'] or not history[best_update - 1]['data_mse_after_update']:
            raise ValueError(f'Missing final or best raw-MSE evaluation: {path.name}')
        require_close(final_metrics['data_mse'], float(history[-1]['data_mse_after_update']), path.name + '/final MSE')
        require_close(best_metrics['data_mse'], float(history[best_update - 1]['data_mse_after_update']), path.name + '/best MSE')
        if (int(summary['parameter_count']) < 1 or not math.isfinite(float(summary['total_seconds']))
                or float(summary['total_seconds']) < 0
                or summary['executed_updates'] + summary['resumed_from_updates'] != budget):
            raise ValueError(f'Invalid timing or parameter count: {path.name}')
        runs.append(dict(name=key[0], seed=key[1], path=path, config=config, contract=contract,
                         summary=summary, history=history, truth=truth, initial=initial,
                         final=final, best=best, final_metrics=final_metrics, best_metrics=best_metrics))
    skipped.extend(f'{name} / seed={seed}: not started or missing run' for name, seed in expected if (name, seed) not in seen)
    if skipped and not partial:
        raise ValueError('Suite is incomplete; use --partial for a marked preview: ' + '; '.join(skipped))
    if not runs:
        raise ValueError('No completed valid runs to summarize')
    protocols = [{k: v for k, v in contract.items() if k not in ('seed', 'observed')} for contract in contracts.values()]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError('Comparison protocol differs across seeds')
    if any(not np.array_equal(run['truth'], runs[0]['truth']) for run in runs[1:]):
        raise ValueError('Saved truth arrays differ across runs')
    order = {key: position for position, key in enumerate(expected)}
    runs.sort(key=lambda run: order[(run['name'], run['seed'])])
    return runs, skipped, len(expected)


def initial_rows(runs):
    """Evaluate saved starting velocities; waveform MSE comes from existing logs."""
    rows = []
    for run in runs:
        prediction = run['initial'].astype(np.float64)
        truth = run['truth'].astype(np.float64)
        index = int(truth.shape[0] * run['contract']['evaluation']['depth_threshold'])
        error = prediction - truth
        # A resumed invocation saves the resumed model as initial_velocity.npy.
        # Its matching before-update MSE is at resumed_from, not history row 0.
        update = int(run['summary']['resumed_from_updates'])
        rows.append(dict(experiment=run['name'], seed=run['seed'], run_id=run['path'].name,
                         updates=update, data_mse=float(run['history'][update]['data_mse_before_update']),
                         full_rmse=float(np.sqrt(np.mean(error ** 2))),
                         deep_rmse=float(np.sqrt(np.mean(error[index:] ** 2))),
                         deep_velocity_std_mps=float(prediction[index:].std())))
    return rows


def table_rows(runs, snapshot):
    rows = []
    for run, initial in zip(runs, initial_rows(runs)):
        metrics, summary = run[snapshot + '_metrics'], run['summary']
        row = dict(experiment=run['name'], seed=run['seed'], run_id=run['path'].name,
                   updates=metrics['completed_updates'] if snapshot == 'final' else metrics['best_update'],
                   objective=run['config']['loss'].get('data_objective', 'mse'))
        row.update({key: metrics[key] for key in METRICS})
        row.update(parameter_count=summary['parameter_count'], seconds=summary['total_seconds'],
                   timing_scope=summary.get('timing_scope', 'unspecified'),
                   executed_updates=summary['executed_updates'], resumed_from_updates=summary['resumed_from_updates'])
        row.update({'initial_' + key: initial[key] for key in
                    ('updates', 'data_mse', 'full_rmse', 'deep_rmse', 'deep_velocity_std_mps')})
        rows.append(row)
    return rows


def write_table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def markdown_table(rows):
    lines = ['| Group / seed | Step | Raw MSE | Full RMSE | Deep RMSE | Deep SSIM | Deep std | Parameters | Time (s) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        values = [f'{row["experiment"]} / {row["seed"]}', str(row['updates']), format_number(row['data_mse'], 7),
                  format_number(row['full_rmse']), format_number(row['deep_rmse']), format_number(row['deep_ssim'], 5),
                  format_number(row['deep_velocity_std_mps']), str(row['parameter_count']), format_number(row['seconds'], 1)]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def initial_markdown_table(rows):
    lines = ['| Group / seed | Initial step | Raw MSE | Full RMSE | Deep RMSE | Deep std |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        values = [f'{row["experiment"]} / {row["seed"]}', str(row['updates']), format_number(row['data_mse'], 7),
                  format_number(row['full_rmse']), format_number(row['deep_rmse']), format_number(row['deep_velocity_std_mps'])]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def figures(runs, output, preview):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    groups = [('attention', ['baseline', 'attention', 'attention_residual', 'attention_film']),
              ('losses', ['baseline', 'prior_only', 'prior_normalized_tv', 'prior_normalized_charbonnier', 'data_huber']),
              ('optimization', ['baseline', 'depth_weighted_loss', 'layerwise_conservative'])]
    links = []
    for seed in sorted({run['seed'] for run in runs}):
        seed_runs = [run for run in runs if run['seed'] == seed]
        first = seed_runs[0]
        dz = first['contract']['params']['dz'] / 1000
        nz, nx = first['truth'].shape
        extent = [-dz / 2, (nx - .5) * dz, (nz - .5) * dz, -dz / 2]
        # Match the original fwi.py velocity plots, including fixed limits.
        vmin = 1.0 if first['contract']['params']['dz'] == 15 else float(first['truth'].min()) / 1000
        vmax = 4.7 if first['contract']['params']['dz'] == 15 else float(first['truth'].max()) / 1000
        for category, names in groups:
            chosen = [run for run in seed_runs if run['name'] in names]
            if not chosen:
                continue
            arrays = [first['truth']] + [run['final'] for run in chosen]
            titles = ['Truth'] + [f'{LABELS.get(run["name"], run["name"])}\nMSE {run["final_metrics"]["data_mse"]:.5f}; deep RMSE {run["final_metrics"]["deep_rmse"]:.0f} m/s' for run in chosen]
            ncols = 3 if len(arrays) > 4 else 2
            nrows = math.ceil(len(arrays) / ncols)
            figure, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.7 * nrows), squeeze=False, layout='constrained')
            for axis, value, title in zip(axes.flat, arrays, titles):
                im = axis.imshow(value / 1000, origin='upper', extent=extent, aspect='equal', interpolation='nearest',
                                 cmap='RdBu_r', vmin=vmin, vmax=vmax)
                axis.set(title=title, xlabel='Distance (km)', ylabel='Depth (km)')
            for axis in list(axes.flat)[len(arrays):]:
                axis.set_visible(False)
            figure.colorbar(im, ax=list(axes.flat)[:len(arrays)], label='Velocity (km/s)', shrink=.88, extend='both')
            tag = 'PARTIAL PREVIEW - ' if preview else ''
            figure.suptitle(f'{tag}Fixed final models: {category}, seed {seed}, {first["summary"]["completed_updates"]} updates')
            filename = f'velocity_{category}_seed{seed}.png'
            figure.savefig(output / filename, dpi=160)
            plt.close(figure)
            links.append(filename)
        figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
        for axis, (category, names) in zip(axes, groups):
            chosen = [run for run in seed_runs if run['name'] in names]
            colors = ['#2166ac', '#b2182b', '#4393c3', '#d6604d', '#053061']
            styles = ['-', '--', '-.', ':', '--']
            for position, run in enumerate(chosen):
                history = run['history']
                x = [int(row['completed_updates']) - 1 for row in history]
                y = [float(row['data_mse_before_update']) for row in history]
                line, = axis.plot(x, y, linewidth=1.3, color=colors[position], linestyle=styles[position],
                                  label=LABELS.get(run['name'], run['name']))
                evaluated = [row for row in history if row['data_mse_after_update']]
                axis.scatter([int(row['completed_updates']) for row in evaluated],
                             [float(row['data_mse_after_update']) for row in evaluated],
                             s=12, facecolors='none', edgecolors=line.get_color(), linewidths=.7)
            axis.set(title=category.capitalize(), xlabel='Completed updates at evaluation', ylabel='Raw waveform MSE')
            axis.grid(alpha=.2)
            if chosen:
                axis.legend(fontsize=8, loc='best')
        figure.suptitle(('PARTIAL PREVIEW - ' if preview else '') + f'Common data metric, seed {seed}; lines: before update, circles: after update')
        filename = f'raw_mse_seed{seed}.png'
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
        links.append(filename)
    return links


def summarize(suite, partial=False):
    suite = Path(suite).resolve()
    runs, skipped, expected_count = load_suite(suite, partial)
    output = suite / ('report_partial' if partial else 'report')
    output.mkdir(parents=True, exist_ok=True)
    final, best = table_rows(runs, 'final'), table_rows(runs, 'best')
    initial = initial_rows(runs)
    write_table(output / 'fixed_final.csv', final)
    write_table(output / 'best_snapshot.csv', best)
    write_table(output / 'initial_metrics.csv', initial)
    links = figures(runs, output, bool(skipped) or partial)
    title = 'IFWI 优化对照报告' if not partial else 'IFWI 优化对照：未完成预览'
    text = [f'# {title}', '', f'已验证 {len(runs)}/{expected_count} 组。单次运行逐组展示，不计算单种子置信区间。', '',
            '## 各组自身的初始模型参照', '',
            '从保存的 initial_velocity.npy 与真值计算全域/深层 RMSE 和深层标准差；初始 Raw MSE 直接读取对应更新前的训练记录，未重新正演。'
            '旧注意力改变初始输出，因此必须使用各组自己的初始数组，不能假定所有组初始模型相同。恢复训练的初始步数指本次调用开始时的更新数。', '',
            initial_markdown_table(initial), '',
            '## 固定末步结果', '',
            '所有组使用预定预算的最后一次更新；Raw MSE 始终是未加权的普通炮集均方误差。RMSE、深层标准差的单位为 m/s。', '',
            markdown_table(final), '', '## 按训练目标选择的最佳快照', '',
            '旧协议 Best 由各组自己的总训练目标选择；启用 spatiotemporal 的新协议按固定 Raw MSE 选择，具体见 training_summary.json 的 selection_metric。时变频带、先验或 Huber 总 loss 不可直接横向比较。以下仍报告独立 Raw MSE。', '',
            markdown_table(best), '', '## 解释边界', '',
            '- 数据拟合和地下速度结构必须同时评估；更低的 Raw MSE 不保证更低的速度误差。',
            '- 相对其他训练组的改善不等于相对自身初始模型的改善；近常量初始模型也可能取得较低 RMSE，不能据此认定其结构正确。',
            '- 深层标准差接近零提示近常量恢复，应结合深层 RMSE、SSIM 和速度图，不能仅据 RMSE 判断改善。',
            '- 本轮短预算合成对照不证明长期收敛、跨种子稳健性或真实数据效果。Huber 在无噪声数据上的结果保留为实测结果。',
            '- 时间来自训练调用的 total_seconds；包含该调用内部评估，不包含全部进程启动与图像生成。恢复训练仅记录本次调用耗时。',
            '- 图像展示固定末步、未经平滑的数组。速度图沿用 fwi.py 的 RdBu_r 蓝白红色图，dz=15 时统一为 1–4.7 km/s；其他网格取真值范围。色条两端三角标识范围外数值，显示饱和不改变数组或评估指标。相同坐标位置对应相同物理位置。', '',
            '## 相对自身初始模型的固定末步变化', '']
    for row, reference in zip(final, initial):
        changes = []
        for key, label in [('data_mse', 'Raw MSE'), ('full_rmse', '全域 RMSE'), ('deep_rmse', '深层 RMSE')]:
            change = 100 * (row[key] / reference[key] - 1) if reference[key] else None
            changes.append(f'{label} {change:+.2f}%' if change is not None else f'{label} 初始为零，未计算比例')
        text.append(f'- {row["experiment"]} / seed={row["seed"]}：' + '；'.join(changes) + '。')
    text += ['', '## 相对同种子训练基线的固定末步变化', '']
    baselines = {row['seed']: row for row in final if row['experiment'] == 'baseline'}
    for row in final:
        reference = baselines.get(row['seed'])
        if reference is None or row['experiment'] == 'baseline':
            continue
        changes = []
        for key, label in [('data_mse', 'Raw MSE'), ('deep_rmse', '深层 RMSE')]:
            change = 100 * (row[key] / reference[key] - 1) if reference[key] else None
            changes.append(f'{label} {change:+.2f}%' if change is not None else f'{label} 基线为零，未计算比例')
        text.append(f'- {row["experiment"]} / seed={row["seed"]}：' + '；'.join(changes) + '。')
    index = int(runs[0]['truth'].shape[0] * runs[0]['contract']['evaluation']['depth_threshold'])
    text += ['', f'真值深层标准差：{runs[0]["truth"][index:].std():.2f} m/s；此值仅用于解释结构，未用于训练或选择最佳快照。']
    if skipped:
        text += ['', '## 未完成或失败组（预览明确保留）', ''] + ['- ' + reason for reason in skipped]
    text += ['', '## 图与可复核产物', '']
    text += [f'![{name}]({name})' for name in links]
    text += ['', '- [各组初始参照](initial_metrics.csv)', '- [固定末步完整表](fixed_final.csv)', '- [最佳快照完整表](best_snapshot.csv)',
             '- [实验清单](../manifest.json)', '- [冻结源码记录](../source_hashes.json)', '']
    (output / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    audit = dict(expected=expected_count, included=len(runs), partial=partial, skipped=skipped,
                 manifest_sha256=hashlib.sha256((suite / 'manifest.json').read_bytes()).hexdigest(),
                 reporting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 run_ids=[run['path'].name for run in runs])
    (output / 'report_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--partial', action='store_true', help='Explicit preview of completed jobs; manifest still required')
    args = parser.parse_args()
    try:
        output = summarize(args.suite, args.partial)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    print(f'Saved verified report: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Validate a complete spatiotemporal IFWI suite and render a CPU-only report.

Usage: python scripts/summarize_spatiotemporal.py --suite results/.../suite
Only <suite>/report_st is written. Training artifacts remain read only.
There is no partial-report mode: missing runs or diagnostics are errors.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from summarize_ablations import (
    initial_markdown_table, initial_rows, load_suite,
    markdown_table, require_close, table_rows, write_table,
)
import numpy as np


GROUPS = ('baseline', 'st_multiscale', 'st_time', 'st_space', 'st_joint')
LABELS = {'baseline': 'Baseline', 'st_multiscale': 'Frequency continuation',
          'st_time': 'Continuation + temporal weights',
          'st_space': 'Continuation + spatial weights',
          'st_joint': 'Continuation + joint weights'}
COLORS = ('#555555', '#2166ac', '#4393c3', '#b2182b', '#67001f')
STYLES = ('-', '--', '-.', ':', '-')


def stage_ranges(config, budget):
    """Return inclusive update ranges; schedule entries count finished updates."""
    if not isinstance(budget, int) or budget < 1:
        raise ValueError('Training budget must be a positive integer')
    boundaries, cutoffs = config['schedule_steps'], config['cutoffs_hz']
    if (not isinstance(boundaries, list) or not isinstance(cutoffs, list)
            or len(cutoffs) != len(boundaries) + 1
            or any(type(step) is not int or step < 1 for step in boundaries)
            or any(b <= a for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError('Invalid frequency-continuation schedule')
    if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0) for value in cutoffs):
        raise ValueError('Cutoffs must be finite positive frequencies or null')
    ordered = [float('inf') if cutoff is None else cutoff for cutoff in cutoffs]
    if any(right < left for left, right in zip(ordered, ordered[1:])):
        raise ValueError('Frequency continuation cutoffs must not decrease')
    starts = [0] + boundaries
    stops = boundaries + [max(budget, boundaries[-1] + 1 if boundaries else budget)]
    return [(start + 1, min(stop, budget), cutoff)
            for start, stop, cutoff in zip(starts, stops, cutoffs) if start < budget]


def cutoff_at(stages, completed_updates):
    for first, last, cutoff in stages:
        if first <= completed_updates <= last:
            return cutoff
    raise ValueError(f'Update outside frequency schedule: {completed_updates}')


def validate_cutoff(value, expected, label):
    if expected is None:
        if value not in (None, ''):
            raise ValueError(f'Expected unfiltered objective in {label}: {value}')
    elif value in (None, ''):
        raise ValueError(f'Missing cutoff in {label}')
    else:
        require_close(float(value), expected, label)


def load_attention(run, stages):
    """Read actual saved weights and optimizer velocity changes, without Torch."""
    expected_updates = {int(row['completed_updates']) for row in run['history']
                        if row['data_mse_after_update']}
    observed = np.load(run['path'] / 'observed.npy', mmap_mode='r', allow_pickle=False)
    if observed.ndim != 4 or observed.shape[0] != 1:
        raise ValueError('Expected observed shots with shape [1, shot, time, receiver]')
    temporal_shape = observed.shape[1:3]
    nz, nx = run['truth'].shape
    required = {'spatial_weights', 'temporal_weights_shot_time', 'velocity_change',
                'completed_updates', 'objective_step'}
    snapshots = {}
    for path in sorted(run['path'].glob('attention_step*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            if not required.issubset(saved.files):
                raise ValueError(f'Missing diagnostic fields: {path.name}')
            update, objective_step = saved['completed_updates'], saved['objective_step']
            if (update.shape != () or objective_step.shape != ()
                    or not np.issubdtype(update.dtype, np.integer)
                    or not np.issubdtype(objective_step.dtype, np.integer)):
                raise ValueError(f'Noninteger diagnostic update metadata: {path.name}')
            update, objective_step = int(update), int(objective_step)
            if (path.name != f'attention_step{update:06d}.npz' or update in snapshots
                    or update not in expected_updates or objective_step != update - 1):
                raise ValueError(f'Diagnostic update does not match training history: {path.name}')
            spatial = saved['spatial_weights'].copy()
            temporal = saved['temporal_weights_shot_time'].copy()
            change = saved['velocity_change'].copy()
        if spatial.shape != (1, nz, nx) or temporal.shape != temporal_shape or change.shape != (nz, nx):
            raise ValueError(f'Diagnostic shape mismatch: {path.name}')
        if any(not np.isfinite(value).all() for value in (spatial, temporal, change)):
            raise ValueError(f'Nonfinite attention diagnostic: {path.name}')
        if np.any(spatial <= 0) or np.any(temporal <= 0):
            raise ValueError(f'Attention weights must be positive: {path.name}')
        config = run['config']['spatiotemporal']
        for values, enabled, strength in (
                (spatial, config.get('spatial_attention', False), config.get('spatial_strength', .25)),
                (temporal, config.get('time_attention', False), config.get('time_strength', .3))):
            if not enabled and not np.array_equal(values, np.ones_like(values)):
                raise ValueError(f'Disabled attention is not identity: {path.name}')
            if np.max(np.abs(values - 1)) > strength + 3e-6:
                raise ValueError(f'Attention exceeds declared bounds: {path.name}')
            require_close(float(values.mean()), 1., path.name + '/mean attention weight')
        snapshots[update] = dict(path=path, update=update, objective_step=objective_step,
                                 spatial=spatial[0], temporal=temporal, change=change)
    if set(snapshots) != expected_updates:
        raise ValueError(f'Missing attention diagnostics for {run["path"].name}: '
                         f'{sorted(expected_updates - set(snapshots))}')
    selected = []
    for first, last, cutoff in stages:
        candidates = [step for step in snapshots if first <= step <= last]
        if not candidates:
            raise ValueError(f'No saved attention diagnostic in stage {first}-{last}')
        selected.append(dict(snapshots[max(candidates)], cutoff=cutoff))
    return snapshots, selected


def validate_st_suite(runs):
    """Add schedule/selection/diagnostic checks to the existing artifact audit."""
    diagnostics, schedules = {}, {}
    for seed in sorted({run['seed'] for run in runs}):
        chosen = [run for run in runs if run['seed'] == seed]
        if {run['name'] for run in chosen} != set(GROUPS) or len(chosen) != len(GROUPS):
            raise ValueError(f'Seed {seed} requires exactly these groups: {", ".join(GROUPS)}')
        baseline = next(run for run in chosen if run['name'] == 'baseline')
        if baseline['config'].get('spatiotemporal', {}).get('enabled', False):
            raise ValueError('Baseline must not enable spatiotemporal weighting')
        for run in chosen:
            if not np.array_equal(run['initial'], baseline['initial']):
                raise ValueError(f'Initial velocities differ from baseline: {run["path"].name}')
            if run['name'] == 'baseline':
                continue
            config = run['config']['spatiotemporal']
            if not config.get('enabled', False):
                raise ValueError(f'Spatiotemporal controller disabled: {run["path"].name}')
            modes = {'st_multiscale': (False, False), 'st_time': (True, False),
                     'st_space': (False, True), 'st_joint': (True, True)}
            if (config.get('time_attention', False), config.get('spatial_attention', False)) != modes[run['name']]:
                raise ValueError(f'Attention switches do not match the group label: {run["path"].name}')
            stages = stage_ranges(config, int(run['summary']['completed_updates']))
            dt = run['contract']['params']['dt']
            if not math.isfinite(dt) or dt <= 0 or any(c is not None and c > .5 / dt for _, _, c in stages):
                raise ValueError('Sampling interval and frequency cutoff are inconsistent')
            if seed in schedules and schedules[seed] != stages:
                raise ValueError('Frequency schedules differ across comparison groups')
            schedules[seed] = stages
            summary = run['summary']
            if summary.get('selection_metric') != 'data_mse':
                raise ValueError(f'New groups must select snapshots by raw MSE: {run["path"].name}')
            evaluated = [row for row in run['history'] if row['data_mse_after_update']]
            best_row = min(evaluated, key=lambda row: float(row['data_mse_after_update']))
            if int(best_row['completed_updates']) != int(summary['best_update']):
                raise ValueError(f'Best snapshot is not the first minimum raw MSE: {run["path"].name}')
            for row in run['history']:
                validate_cutoff(row['cutoff_hz'], cutoff_at(stages, int(row['completed_updates'])),
                                run['path'].name + '/history cutoff')
            for snapshot in ('final', 'best'):
                metrics = run[snapshot + '_metrics']
                update = metrics['completed_updates'] if snapshot == 'final' else metrics['best_update']
                if (metrics.get('selection_metric') != 'data_mse'
                        or metrics.get('objective_step') != update - 1 or 'cutoff_hz' not in metrics):
                    raise ValueError(f'Invalid objective metadata: {run["path"].name}/{snapshot}')
                validate_cutoff(metrics.get('cutoff_hz'), cutoff_at(stages, update),
                                run['path'].name + '/' + snapshot + ' cutoff')
            diagnostics[(run['name'], seed)] = load_attention(run, stages)
    return schedules, diagnostics


def spatial_extent(run):
    spacing = run['contract']['params']['dz'] / 1000
    nz, nx = run['truth'].shape
    return [-spacing / 2, (nx - .5) * spacing, (nz - .5) * spacing, -spacing / 2]


def cutoff_label(cutoff):
    return 'Unfiltered' if cutoff is None else f'{cutoff:g} Hz cutoff'


def save_figure(figure, output, stem):
    figure.savefig(output / (stem + '.png'), dpi=300)
    figure.savefig(output / (stem + '.pdf'))
    figure.savefig(output / (stem + '.svg'))
    return stem + '.png'


def diagnostic_figure(joint, selected):
    """Show actual saved joint weights and delta-v at selected stage endpoints."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    gain_span = max(float(np.max(np.abs(item['spatial'] - 1))) for item in selected)
    change_span = max(float(np.max(np.abs(item['change']))) for item in selected)
    # An exactly constant diagnostic uses a narrow, explicitly stated display range.
    gain_span, change_span = max(gain_span, 1e-6), max(change_span, 1e-6)
    gain_norm = TwoSlopeNorm(vmin=1 - gain_span, vcenter=1, vmax=1 + gain_span)
    change_norm = TwoSlopeNorm(vmin=-change_span, vcenter=0, vmax=change_span)
    figure, axes = plt.subplots(len(selected), 3, figsize=(15, 3.2 * len(selected)),
                                squeeze=False, layout='constrained')
    dt, extent = joint['contract']['params']['dt'], spatial_extent(joint)
    for row, item in zip(axes, selected):
        spatial = row[0].imshow(item['spatial'], origin='upper', extent=extent,
                               cmap='RdBu_r', norm=gain_norm, aspect='equal', interpolation='nearest')
        change = row[1].imshow(item['change'], origin='upper', extent=extent,
                              cmap='RdBu_r', norm=change_norm, aspect='equal', interpolation='nearest')
        for axis in row[:2]:
            axis.set(xlabel='Distance (km)', ylabel='Depth (km)')
        row[0].set_title(f'Spatial gradient-response weights\nUpdate {item["update"]}; '
                         f'objective step {item["objective_step"]}')
        row[1].set_title(f'Actual optimizer velocity change (m/s)\n{cutoff_label(item["cutoff"])}')
        times = np.arange(item['temporal'].shape[1]) * dt
        row[2].plot(times, item['temporal'].mean(axis=0), color=COLORS[-1], linewidth=1.4)
        row[2].axhline(1., color='.6', linestyle=':', linewidth=.8)
        row[2].set(title='Temporal weights: receiver average, then shot mean',
                   xlabel='Recorded time (s)', ylabel='Mean temporal weight', xlim=(times[0], times[-1]))
        row[2].grid(alpha=.18)
    figure.colorbar(spatial, ax=list(axes[:, 0]), label='Spatial weight; neutral = 1', shrink=.82)
    figure.colorbar(change, ax=list(axes[:, 1]), label='Single-update velocity change (m/s)', shrink=.82)
    figure.suptitle(f'Joint-weight diagnostics, seed {joint["seed"]}: last saved update of each frequency stage')
    return figure


def make_figures(runs, schedules, diagnostics, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False})
    links = []
    for seed in sorted(schedules):
        lookup = {run['name']: run for run in runs if run['seed'] == seed}
        baseline, joint = lookup['baseline'], lookup['st_joint']
        extent, stages = spatial_extent(baseline), schedules[seed]
        budget = int(baseline['summary']['completed_updates'])
        figure, axes = plt.subplots(2, 3, figsize=(15, 6), layout='constrained')
        panels = [('Truth', baseline['truth'])] + [
            (f'{LABELS[name]}\nRaw MSE {lookup[name]["final_metrics"]["data_mse"]:.5g}; '
             f'deep RMSE {lookup[name]["final_metrics"]["deep_rmse"]:.0f} m/s', lookup[name]['final'])
            for name in GROUPS]
        for axis, (title, values) in zip(axes.flat, panels):
            im = axis.imshow(values / 1000, origin='upper', extent=extent, aspect='equal',
                             interpolation='nearest', cmap='RdBu_r', vmin=1., vmax=4.7)
            axis.set(title=title, xlabel='Distance (km)', ylabel='Depth (km)')
        figure.colorbar(im, ax=list(axes.flat), label='Velocity (km/s)', shrink=.86, extend='both')
        figure.suptitle(f'Fixed final models: seed {seed}, {budget} completed updates')
        links.append(save_figure(figure, output, f'velocity_seed{seed}'))
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(11, 5.2), layout='constrained')
        for name, color, style in zip(GROUPS, COLORS, STYLES):
            history = lookup[name]['history']
            axis.plot([int(row['completed_updates']) - 1 for row in history],
                      [float(row['data_mse_before_update']) for row in history],
                      color=color, linestyle=style, linewidth=1.3, label=LABELS[name])
            evaluated = [row for row in history if row['data_mse_after_update']]
            axis.scatter([int(row['completed_updates']) for row in evaluated],
                         [float(row['data_mse_after_update']) for row in evaluated],
                         s=13, facecolors='none', edgecolors=color, linewidths=.8)
        for first, last, cutoff in stages:
            if first > 1:
                axis.axvline(first - 1, color='.55', linestyle=':', linewidth=.8)
            axis.text((first - 1 + last) / 2, 1.02, cutoff_label(cutoff),
                      ha='center', va='bottom', transform=axis.get_xaxis_transform(), fontsize=8)
        axis.set(xlabel='Completed updates at evaluation', ylabel='Raw unweighted waveform MSE',
                 xlim=(0, budget))
        axis.grid(alpha=.18)
        axis.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
        figure.suptitle(f'Common evaluation metric, seed {seed}\nLines: before update; circles: after update; '
                       'stage labels apply to continuation groups')
        links.append(save_figure(figure, output, f'raw_mse_seed{seed}'))
        plt.close(figure)

        figure = diagnostic_figure(joint, diagnostics[('st_joint', seed)][1])
        links.append(save_figure(figure, output, f'attention_diagnostics_seed{seed}'))
        plt.close(figure)
    return links


def percent_change(value, reference):
    return '参照为零，未计算比例' if reference == 0 else f'{100 * (value / reference - 1):+.2f}%'


def stage_figures(suite, output):
    """Three measured checkpoints, not an interpolated convergence trajectory."""
    import matplotlib.pyplot as plt
    path = suite / 'stage_metrics.csv'
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    links = []
    metrics = [('deep_rmse', 'Deep RMSE (m/s)'),
               ('deep_background_rmse_mps', 'Deep smoothed-background RMSE (m/s)'),
               ('deep_detail_rmse_mps', 'Deep residual-detail RMSE (m/s)'),
               ('data_mse', 'Raw waveform MSE')]
    for seed in sorted({row['seed'] for row in rows}):
        scales = {(float(row['sigma_cells']), float(row['sigma_m'])) for row in rows if row['seed'] == seed}
        if len(scales) != 1:
            raise ValueError('Spatial smoothing scales differ across stage comparisons')
        sigma_cells, sigma_m = scales.pop()
        figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
        for axis, (metric, label) in zip(axes.flat, metrics):
            for name, color, style in zip(GROUPS, COLORS, STYLES):
                chosen = sorted([row for row in rows if row['seed'] == seed and row['experiment'] == name],
                                key=lambda row: int(row['completed_updates']))
                if [int(row['completed_updates']) for row in chosen] != [30, 70, 100]:
                    raise ValueError('Require all three stage snapshots to draw the stage comparison')
                axis.plot([30, 70, 100], [float(row[metric]) for row in chosen],
                          color=color, linestyle=style, marker='o', markersize=4, label=LABELS[name])
            axis.set(xlabel='Completed updates (saved snapshots only)', ylabel=label, xticks=[30, 70, 100])
            axis.grid(alpha=.18)
        axes[0, 0].legend(fontsize=7)
        figure.suptitle(f'Stage comparisons, seed {seed}; background: Gaussian sigma {sigma_cells:g} cells ({sigma_m:g} m)')
        links.append(save_figure(figure, output, f'stage_metrics_seed{seed}'))
        plt.close(figure)
    return links


def summarize(suite):
    suite = Path(suite).resolve()
    runs, skipped, expected_count = load_suite(suite, partial=False)
    schedules, diagnostics = validate_st_suite(runs)
    output = suite / 'report_st'
    output.mkdir(parents=True, exist_ok=True)
    final, best, initial = table_rows(runs, 'final'), table_rows(runs, 'best'), initial_rows(runs)
    for rows in (final, best):
        for row in rows:
            run = next(item for item in runs if item['path'].name == row['run_id'])
            row.update(training_objective=LABELS[row['experiment']],
                       selection_metric=run['summary'].get('selection_metric', 'total_loss'))
    write_table(output / 'final.csv', final)
    write_table(output / 'best.csv', best)
    write_table(output / 'initial.csv', initial)
    focus = []
    for run in runs:
        key = (run['name'], run['seed'])
        if key not in diagnostics:
            continue
        depth = int(run['truth'].shape[0] * run['contract']['evaluation']['depth_threshold'])
        for update, item in sorted(diagnostics[key][0].items()):
            weights, change = item['spatial'].astype(float), item['change'].astype(float)
            energy = float(np.sum(change**2))
            focus.append(dict(experiment=run['name'], seed=run['seed'], updates=update,
                shallow_mean_weight=float(weights[:depth].mean()), deep_mean_weight=float(weights[depth:].mean()),
                shallow_update_rms_mps=float(np.sqrt(np.mean(change[:depth]**2))),
                deep_update_rms_mps=float(np.sqrt(np.mean(change[depth:]**2))),
                deep_mean_change_mps=float(change[depth:].mean()),
                deep_update_energy_fraction=float(np.sum(change[depth:]**2)/energy) if energy else None))
    write_table(output / 'attention_focus.csv', focus)
    links = make_figures(runs, schedules, diagnostics, output)
    links += stage_figures(suite, output)
    text = ['# IFWI 时空权重与频率延拓对照报告', '',
            f'已验证完整清单中的 {len(runs)}/{expected_count} 次运行。各组按实际种子逐项展示；'
            '表中 RMSE、深层标准差的单位为 m/s，Raw MSE 为原始未加权炮集误差。', '',
            '## 方案与可比范围', '',
            '- baseline：原始 MSE；st_multiscale：仅频率延拓；st_time：延拓加时间权重；'
            'st_space：延拓加空间梯度权重；st_joint：延拓加二者。',
            '- 五组初始速度数组已逐点精确比较相同。网格、观测、波子、预算、种子、主干配置和冻结源码由 comparison contract 校验。',
            '- 各组训练目标不同：阶段滤波和时间权重会改变被优化的数据目标；空间权重改变回传的速度梯度。'
            '不可把这些训练 loss 当作统一误差横向排序。跨组比较使用独立计算的原始未加权 Raw MSE 与速度结构指标。',
            '- 新四组在已保存、已评估的步数中按 Raw MSE 选择最佳快照，并核对其确为首次最小值。'
            'baseline 仍按自身总目标选择；本组总目标就是 MSE。固定末步结果不经过最佳快照筛选。', '',
            '## 实际频率阶段', '']
    for seed, stages in sorted(schedules.items()):
        pieces = [f'更新 {first}–{last}：' + ('未滤波' if cutoff is None else f'{cutoff:g} Hz 截止频率')
                  for first, last, cutoff in stages]
        text.append(f'- seed={seed}：' + '；'.join(pieces) + '。阶段边界表示已完成的更新数。')
    text += ['', '## 固定末步结果', '', markdown_table(final), '',
             '## 相对基线及仅频率延拓的变化', '',
             '负值表示指标下降；同时提供两种参照，区分频率延拓与额外时空权重的贡献。']
    for seed in sorted(schedules):
        lookup = {row['experiment']: row for row in final if row['seed'] == seed}
        for name in GROUPS[1:]:
            row = lookup[name]
            comparisons = []
            for reference in ('baseline', 'st_multiscale'):
                if name == reference:
                    continue
                comparisons.append(f'相对 {reference}：Raw MSE '
                    f'{percent_change(row["data_mse"], lookup[reference]["data_mse"])}，深层 RMSE '
                    f'{percent_change(row["deep_rmse"], lookup[reference]["deep_rmse"])}')
            text.append(f'- {name} / seed={seed}：' + '；'.join(comparisons) + '。')
    text += ['', '## 初始模型参照', '', initial_markdown_table(initial), '',
             '训练后的组间优势不代表已恢复真实地下结构；还需与各自初始模型的误差、深层标准差和速度图比较。', '',
             '## 最佳已评估快照', '', markdown_table(best), '',
             '## 时空诊断解释', '',
             '- 空间图是梯度响应构造的权重代理，不是真实照明度、真实断层位置或地质真值。权重大不等于该位置已正确恢复。',
             '- Δv 图直接读取该次参数更新前后的实际速度差（m/s），不是负梯度图，也不是累计速度变化。'
             '空间权重在反向传播时作用于速度梯度，实际参数更新还经过网络 Jacobian 与优化器，二者不可等同。',
             '- 时间权重文件已沿接收道取均值，图中再沿炮号求均值，仅展示道均值的时间概况；'
             '它隐藏了不同炮与接收道间的差异，不能解释为全炮集都具有同一条时间权重曲线。',
             '- attention_focus.csv 同时记录浅/深部平均空间权重与实际更新 RMS；深部更新能量占比只反映更新位置，不代表方向正确或误差改善。',
             '- 诊断图对每个频率阶段选最后一个实际保存的更新；标题同时给出更新数与优化目标使用的步数。'
             '空间权重色标以 1 为中心，实际 Δv 色标以 0 为中心；各自跨阶段共享范围。全常量图使用 1e-6 的最小显示半宽。', '',
             '| Joint seed | Stage | Saved update | Objective step | Raw artifact |',
             '| --- | --- | ---: | ---: | --- |']
    for seed in sorted(schedules):
        for item in diagnostics[('st_joint', seed)][1]:
            rel = item['path'].relative_to(suite).as_posix()
            text.append(f'| {seed} | {cutoff_label(item["cutoff"])} | {item["update"]} | '
                        f'{item["objective_step"]} | [{item["path"].name}](../{rel}) |')
    seeds = sorted(schedules)
    budgets = sorted({int(run['summary']['completed_updates']) for run in runs})
    text += ['', '## 限制与图像口径', '',
             f'- 实际种子：{", ".join(map(str, seeds))}；实际预算：{", ".join(map(str, budgets))} 步。'
             '单种子、100 步设置只能用于短程诊断，不能证明长期收敛、多种子稳健性或真实数据效果；不生成单种子置信区间。',
             '- 速度图使用未经平滑或裁剪的固定末步数组，统一 RdBu_r 蓝白红及 1–4.7 km/s 色标；'
             '色条两端三角表示显示范围外的值，显示饱和不改变数值或评估指标。距离与深度均为 km。',
             '- Raw MSE 曲线使用全部逐步更新前记录与实际更新后评估点；竖线是频率阶段边界，基线不参与滤波。',
             '- 时间为训练调用内 total_seconds，含内部评估；不包括所有进程启动和绘图。恢复训练时间仅覆盖本次调用。', '',
             '- 阶段误差图仅连接第 30、70、100 步的实际检查点，不代表中间完整轨迹。背景是完整网格 sigma=10 cells 的高斯平滑，细节是原模型减去该背景；先计算再裁取深部。它们不是断层定位指标。', '',
             '## 图与完整数据', '']
    for link in links:
        text += [f'![{link}]({link})', '', f'[矢量 PDF]({Path(link).with_suffix(".pdf").name})', '']
    text += ['- [固定末步完整表](final.csv)', '- [最佳已评估快照](best.csv)', '- [初始模型参照](initial.csv)',
             '- [空间关注与实际更新分布](attention_focus.csv)',
             '- [阶段背景与细节指标](../stage_metrics.csv)',
             '- [实验清单](../manifest.json)', '- [冻结源码记录](../source_hashes.json)', '']
    (output / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    audit = dict(expected_runs=expected_count, included_runs=len(runs), skipped=skipped,
                 seeds=seeds, budgets=budgets, run_ids=[run['path'].name for run in runs],
                 manifest_sha256=hashlib.sha256((suite / 'manifest.json').read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 diagnostic_counts={f'{name}/seed{seed}': len(items[0])
                                    for (name, seed), items in diagnostics.items()})
    (output / 'report_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    args = parser.parse_args()
    try:
        output = summarize(args.suite)
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        parser.error(str(error))
    print(f'Saved verified spatiotemporal report: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

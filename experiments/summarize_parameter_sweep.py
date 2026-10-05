"""Audit and summarize fixed-update endpoints; never pool different acquisition losses."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from parameter_sweep import validate_case
from improved_modules.evaluate_deep import evaluate_deep_layers


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(out):
    manifest = read(out/'manifest.json')
    rows = []
    baselines = {}
    for job in manifest['jobs']:
        if job['factor'] == 'baseline': baselines[job['seed']] = job['config']
    for job in manifest['jobs']:
        validate_case(baselines[job['seed']], job, preliminary=manifest.get('preliminary', False))
        candidates = list((out/job['directory']).glob('*/status.json'))
        row = {k: job[k] for k in ('name', 'factor', 'value', 'seed')}
        row['state'] = 'pending'
        if len(candidates) > 1: raise ValueError('Ambiguous run directory')
        if candidates:
            run = candidates[0].parent
            status = read(run/'status.json')
            row.update(state=status['state'], completed_updates=status['completed_updates'])
            if status['state'] == 'completed':
                if status['completed_updates'] != manifest['iterations']:
                    raise ValueError('Unequal update budgets')
                actual = read(run/'config.json')
                if actual.get('seed') != job['seed']:
                    raise ValueError('Executed seed differs from registered seed')
                expected = dict(job['config'], seed=job['seed'])
                if actual != expected: raise ValueError('Executed config differs from registered config')
                metrics = read(run/'final_metrics.json')
                summary = read(run/'training_summary.json')
                geometry = read(run/'acquisition.json')
                if geometry['num_shots'] != actual['data']['num_shots']:
                    raise ValueError('Incorrect realized shot count')
                row.update(num_shots=geometry['num_shots'], hidden_layers=len(actual['model']['neuron'])-2,
                    width=actual['model']['neuron'][1], omega=actual['model']['omega_0'])
                for key in ('full_rmse', 'deep_rmse', 'deep_ssim', 'deep_gradient_fidelity',
                            'bottom_left_rmse', 'bottom_right_rmse', 'data_mse', 'deep_velocity_std_mps'):
                    row[key] = metrics[key]
                row.update(parameter_count=summary['parameter_count'], total_seconds=summary['total_seconds'],
                           peak_cuda_gib=summary['peak_cuda_bytes']/1024**3 if summary['peak_cuda_bytes'] is not None else None)
                truth = np.load(run/'v_true.npy')
                initial_file = run/'random_initial_velocity.npy'
                initial = np.load(initial_file if initial_file.is_file() else run/'initial_velocity.npy')
                init = evaluate_deep_layers(truth, initial,
                    depth_threshold=actual['evaluation'].get('depth_threshold', .5),
                    corner_size=actual['evaluation'].get('corner_size', .25), dz=actual['data']['dz'])
                row.update(initial_deep_rmse=init['deep_rmse'],
                           deep_rmse_change_from_init=row['deep_rmse']-init['deep_rmse'])
                baseline_dirs = list((out/f"runs/baseline_seed{job['seed']}").glob('*/final_metrics.json'))
                if baseline_dirs:
                    baseline_run = baseline_dirs[0].parent
                    np.testing.assert_array_equal(truth, np.load(baseline_run/'v_true.npy'))
                    if job['factor'] == 'shots':
                        baseline_initial = baseline_run/'random_initial_velocity.npy'
                        np.testing.assert_array_equal(initial, np.load(baseline_initial if baseline_initial.is_file()
                                                                       else baseline_run/'initial_velocity.npy'))
                        x = geometry['source_x_indices'][0]
                        bx = read(baseline_run/'acquisition.json')['source_x_indices'][0]
                        if not set(bx).issubset(x) or (min(x),max(x)) != (min(bx),max(bx)):
                            raise ValueError('Shot treatment changed aperture or removed baseline sources')
                    elif job['factor'] != 'baseline':
                        np.testing.assert_array_equal(np.load(run/'observed.npy'), np.load(baseline_run/'observed.npy'))
                    bm = read(baseline_dirs[0])
                    row['deep_rmse_vs_baseline_pct'] = 100*(row['deep_rmse']/bm['deep_rmse']-1)
        rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ['# 单变量参数实验', '', f"每组预算：{manifest['iterations']} 次完整数据更新；固定末步比较。",
        '这是流程验证，不用于选参。' if manifest.get('preliminary') else '这是正式预算结果；待全部完成后比较各组。',
        '炮数组使用不同训练数据，训练 MSE 不作为跨炮数的统一排名。',
        '深度/宽度/omega 会改变初始函数；同时记录初始误差。omega 包含原网络初始化规则的影响。',
        '单种子结果仅用于观察趋势；不能证明统计稳定性。负的 RMSE 变化表示改善。', '',
        '|实验|seed|状态|全域 RMSE|深层 RMSE|深层 SSIM|相对基线深层 RMSE %|',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        def fmt(key):
            value = row.get(key)
            return f'{value:.4f}' if value is not None else '—'
        lines.append(f"|{row['name']}|{row['seed']}|{row['state']}|{fmt('full_rmse')}|{fmt('deep_rmse')}|{fmt('deep_ssim')}|{fmt('deep_rmse_vs_baseline_pct')}|")
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    completed = [r for r in rows if r['state'] == 'completed']
    if completed:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
        fig.suptitle(f"{manifest['iterations']} updates; {len(completed)}/{len(rows)} completed"
                     + (' — smoke test only' if manifest.get('preliminary') else ''))
        baseline_config=next(j['config'] for j in manifest['jobs'] if j['factor']=='baseline')
        baseline_values=[baseline_config['data']['num_shots'],len(baseline_config['model']['neuron'])-2,
                         baseline_config['model']['neuron'][1],baseline_config['model']['omega_0']]
        for ax, factor, baseline_value in zip(axes.flat, ['shots', 'depth', 'width', 'omega'], baseline_values):
            for seed in manifest['seeds']:
                points = [(baseline_value if r['factor']=='baseline' else r['value'], r['deep_rmse'])
                          for r in completed if r['seed']==seed and r['factor'] in ('baseline',factor)]
                points.sort()
                if points: ax.plot(*zip(*points), 'o-', label=f'seed={seed}')
            ax.set(xlabel=factor, ylabel='Final deep RMSE (m/s)', title=f'{factor}: other factors fixed')
            ax.set_xticks(sorted({baseline_value} | {j['value'] for j in manifest['jobs'] if j['factor'] == factor}))
            ax.grid(alpha=.3)
            if ax.lines: ax.legend()
        fig.savefig(out/'parameter_effects.png', dpi=160); plt.close(fig)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    summarize(parser.parse_args().suite.resolve())

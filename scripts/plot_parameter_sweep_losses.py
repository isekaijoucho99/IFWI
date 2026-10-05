"""Export complete registered loss histories without importing the training code."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import traceback


FACTORS = ('shots', 'depth', 'width', 'omega')
LOSS_FIELDS = ('completed_updates', 'total_loss', 'data_loss', 'prior_loss',
               'loss_after_update', 'data_loss_after_update', 'gradient_norm',
               'regularization_statistic')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_loss_history(path):
    return _parse_history(Path(path).read_bytes(), path)


def collect_cases(suite, manifest):
    suite = Path(suite).resolve()
    cases = []
    for job in manifest['jobs']:
        directory = (suite / job['directory']).resolve()
        if not directory.is_relative_to(suite) or Path(job['name']).name != job['name']:
            raise ValueError('Invalid registered result path')
        candidates = list(directory.glob('*/status.json'))
        if len(candidates) > 1:
            raise ValueError('Ambiguous registered run: ' + job['name'])
        case = dict(job, state='pending', rows=[], source=None, source_sha256=None)
        if candidates:
            run = candidates[0].parent
            case['state'] = read_json(candidates[0])['state']
            histories = [p for p in (run / 'loss_history.csv', run / 'legacy_loss_history.csv',
                                     run / 'loss.csv') if p.is_file()]
            if len(histories) > 1:
                raise ValueError('Ambiguous loss history: ' + str(run))
            if histories:
                # Snapshot once: exported values and provenance refer to the same bytes.
                source = histories[0]
                raw = source.read_bytes()
                case.update(source=str(source), source_sha256=hashlib.sha256(raw).hexdigest())
                case['rows'] = _parse_history(raw, source)
        cases.append(case)
    return cases


def _parse_history(raw, source):
    # Ignore only a trailing write that has not reached its line terminator yet.
    complete = raw[:raw.rfind(b'\n') + 1].decode('utf-8-sig')
    reader = csv.DictReader(io.StringIO(complete))
    if not reader.fieldnames:
        return []
    legacy = 'total_loss' in reader.fieldnames
    rows = []
    for record in reader:
        update = float(record['completed_updates'])
        if not update.is_integer() or update < 1 or (rows and update <= rows[-1]['completed_updates']):
            raise ValueError('Loss update numbers must be positive and strictly increasing: ' + str(source))
        def optional(field):
            value = record.get(field)
            return float(value) if value not in (None, '') else None
        total = optional('total_loss' if legacy else 'loss_before_update')
        if total is None:
            raise ValueError('Missing training loss: ' + str(source))
        rows.append(dict(completed_updates=int(update), total_loss=total,
                         data_loss=optional('data_loss' if legacy else 'data_loss_before_update'),
                         prior_loss=optional('prior_loss_before_update'),
                         loss_after_update=optional('loss_after_update'),
                         data_loss_after_update=optional('data_loss_after_update'),
                         gradient_norm=optional('gradient_norm'),
                         regularization_statistic=optional('regularization_statistic')))
    return rows


def atomic_text(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, path)


def save_figure(fig, path):
    for extension in ('.png', '.pdf'):
        destination = path.with_suffix(extension)
        temporary = destination.with_name(destination.stem + '.tmp' + extension)
        fig.savefig(temporary, dpi=160, facecolor='white')
        os.replace(temporary, destination)


def label(case):
    if case['factor'] == 'baseline':
        return f"baseline (seed {case['seed']})"
    return f"{case['factor']}={case['value']} (seed {case['seed']})"


def draw_curve(ax, case, logarithmic, **style):
    rows = case['rows']
    x = [r['completed_updates'] for r in rows]
    y = [r['total_loss'] if math.isfinite(r['total_loss']) and
         (not logarithmic or r['total_loss'] > 0) else math.nan for r in rows]
    ax.plot(x, y, linewidth=1, label=label(case), **style)


def configure_axis(ax, budget, logarithmic):
    if logarithmic:
        ax.set_yscale('log')
    ax.set(xlim=(0, budget), xlabel='Completed Adam updates',
           ylabel='Loss (log scale)' if logarithmic else 'Loss (linear scale)')
    ax.grid(alpha=.2)


def export_losses(suite):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suite = Path(suite).resolve()
    manifest = read_json(suite / 'manifest.json')
    cases = collect_cases(suite, manifest)
    out = suite / 'loss_curves'
    out.mkdir(exist_ok=True)
    budget = manifest['iterations']
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'path.simplify': False})
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), layout='constrained')
    fig.suptitle(f'IFWI loss histories | budget: {budget} updates | {timestamp}', fontsize=14)
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
    for column, factor in enumerate(FACTORS):
        selected = [c for c in cases if c['factor'] in ('baseline', factor)]
        pending = [c['name'] for c in selected if not c['rows']]
        for row, logarithmic in enumerate((False, True)):
            ax = axes[row, column]
            configure_axis(ax, budget, logarithmic)
            ax.set_title(factor + ('\nAwaiting: ' + ', '.join(pending) if pending else ''), fontsize=10)
            index = 0
            for case in selected:
                if not case['rows']:
                    continue
                if case['factor'] == 'baseline':
                    style = dict(color='#333333', linestyle='--')
                else:
                    style = dict(color=colors[index % len(colors)], linestyle='-')
                    index += 1
                draw_curve(ax, case, logarithmic, **style)
            if ax.lines:
                ax.legend(fontsize=8, loc='upper right')
    fig.supxlabel('Raw pre-update loss; no smoothing. Different shot datasets: do not rank velocity quality by loss.',
                  fontsize=10)
    save_figure(fig, out / 'all_loss_curves')
    plt.close(fig)
    for case in cases:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
        last = case['rows'][-1]['completed_updates'] if case['rows'] else 0
        fig.suptitle(f"{label(case)} | {case['state']} | {last}/{budget} recorded updates")
        for ax, logarithmic in zip(axes, (False, True)):
            configure_axis(ax, budget, logarithmic)
            if case['rows']:
                draw_curve(ax, case, logarithmic, color='#0072B2')
            else:
                ax.text(.5, .5, 'Awaiting training data', ha='center', va='center', transform=ax.transAxes)
        fig.supxlabel('Raw pre-update total loss (waveform MSE in this suite); no smoothing.', fontsize=9)
        save_figure(fig, out / f"{case['name']}_seed{case['seed']}")
        plt.close(fig)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=('name', 'factor', 'value', 'seed', 'state') + LOSS_FIELDS)
    writer.writeheader()
    metadata = []
    for case in cases:
        for record in case['rows']:
            writer.writerow(dict({k: case[k] for k in ('name', 'factor', 'value', 'seed', 'state')}, **record))
        rows = case['rows']
        metadata.append(dict(name=case['name'], seed=case['seed'], state=case['state'],
                             recorded_rows=len(rows), last_recorded_update=rows[-1]['completed_updates'] if rows else None,
                             source=case['source'], source_sha256=case['source_sha256'],
                             missing_update_count=rows[-1]['completed_updates'] - len(rows) if rows else None,
                             nonfinite_loss_count=sum(not math.isfinite(r['total_loss']) for r in rows)))
    atomic_text(out / 'all_loss_history.csv', '\ufeff' + stream.getvalue())
    atomic_text(out / 'manifest.json', json.dumps(dict(exported_at=timestamp, budget=budget,
                objective='raw pre-update total loss', smoothed=False, cases=metadata), indent=2, ensure_ascii=False))
    lines = ['# 全部实验 loss 曲线', '', f'导出时间：{timestamp}；每组预算 {budget} 次 Adam 更新。', '',
             '[分组总览](all_loss_curves.png) · [PDF](all_loss_curves.pdf) · [全部原始数值](all_loss_history.csv)', '',
             '每组均保存线性和对数纵轴；逐步原始记录，无平滑或抽样。横轴沿用原记录的完成更新次数。',
             '仅读取当前注册的实验，不合并归档中被重放的尾段。待运行组没有虚构的零 loss。',
             '本轮 total loss 与 data MSE 相同（无先验项）；不同炮数的 loss 仅用于观察各自收敛，不能直接排名反演质量。',
             '旧基线的 regularization_statistic 是未进入目标的统计量，保留在 CSV 中，不画作训练 loss。',
             'CSV 同时保留新组已有的更新后 loss 和梯度范数；更新后 loss 仅在原保存点有值，不插值补齐。',
             '非有限 loss 在图中保留为缺口；对数图不显示非正值，原数值仍完整保留在 CSV。', '',
             '|实验|状态|记录数|最新更新|单独曲线|', '|---|---|---:|---:|---|']
    for case, item in zip(cases, metadata):
        filename = f"{case['name']}_seed{case['seed']}"
        lines.append(f"|{case['name']}|{case['state']}|{item['recorded_rows']}|{item['last_recorded_update'] or '—'}|"
                     f"[PNG]({filename}.png) · [PDF]({filename}.pdf)|")
    atomic_text(out / 'index.md', '\n'.join(lines) + '\n')
    return metadata


def input_signature(suite):
    manifest = read_json(suite / 'manifest.json')
    paths = [suite / 'manifest.json', suite / 'active.json']
    for job in manifest['jobs']:
        for status in (suite / job['directory']).glob('*/status.json'):
            paths.extend((status, status.parent / 'loss_history.csv',
                          status.parent / 'legacy_loss_history.csv', status.parent / 'loss.csv'))
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.is_file())


def watch(suite, interval):
    previous = None
    while True:
        try:
            signature = input_signature(suite)
            state = read_json(suite / 'active.json').get('state') if (suite / 'active.json').exists() else None
            if signature != previous:
                metadata = export_losses(suite)
                previous = signature
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} exported {sum(bool(c['recorded_rows']) for c in metadata)} histories", flush=True)
            if state in ('completed', 'failed'):
                print('Training suite state: ' + state + '; final available curves exported.', flush=True)
                return
        except (OSError, ValueError):
            # A plot viewer may briefly lock an output; retry without affecting training.
            traceback.print_exc()
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least one second')
    suite = args.suite.resolve()
    if args.watch:
        with (suite / 'loss-export.log').open('a', encoding='utf-8', buffering=1) as log:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(log), redirect_stderr(log):
                watch(suite, args.interval)
    else:
        metadata = export_losses(suite)
        print(json.dumps(dict(output=str(suite / 'loss_curves'), cases=metadata), ensure_ascii=False))


if __name__ == '__main__':
    main()

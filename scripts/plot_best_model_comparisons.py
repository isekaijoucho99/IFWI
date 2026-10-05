"""Render four wide IFWI comparisons from a frozen selected-model archive.

Input: selection.json, selected_metrics.csv and source_data/ from the original
94x288 Marmousi parameter study. This script never trains or reselects models.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
import numpy as np

ACCENT = '#B54708'
NEUTRAL = '#333333'
EXTENT = (-.0075, 4.3125, 1.4025, -.0075)
PANEL_ASPECT = 1.395 / 4.305
FAMILIES = [('shots', 'Shot Count', ['baseline', 'shots_25', 'shots_49']),
            ('depth', 'Network Depth', ['baseline', 'depth_6', 'depth_8']),
            ('width', 'Network Width', ['baseline', 'width_256', 'width_512']),
            ('omega', 'Omega', ['baseline', 'omega_10', 'omega_20', 'omega_50'])]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def model_header(ax, case, factor):
    label = TextArea(case['label'], textprops=dict(fontsize=12, fontweight='bold',
        color=ACCENT if case['name'] != 'baseline' else NEUTRAL))
    update = TextArea(f' | Update {case["update"]}', textprops=dict(fontsize=12, color=NEUTRAL))
    config = [(f'{case["shots"]} shots', 'shots'), (' | ', None),
              (str(case['layers']), 'depth'), (' x ', None),
              (str(case['width']), 'width'), (' | ', None),
              (f'Omega = {case["omega"]}', 'omega')]
    chunks = [TextArea(text, textprops=dict(fontsize=12,
              color=ACCENT if key == factor else NEUTRAL,
              fontweight='bold' if key == factor else 'normal')) for text, key in config]
    header = VPacker(children=[HPacker(children=[label, update], align='baseline', pad=0, sep=0),
                               HPacker(children=chunks, align='baseline', pad=0, sep=0)],
                     align='center', pad=0, sep=4)
    ax.add_artist(AnchoredOffsetbox(loc='lower center', child=header, frameon=False,
        bbox_to_anchor=(.5, 1.025), bbox_transform=ax.transAxes, pad=0, borderpad=0))


def archive_path(root, relative):
    """Accept Windows/POSIX manifest separators, keeping inputs inside root."""
    normalized = PurePosixPath(str(relative).replace('\\', '/'))
    if normalized.is_absolute() or '..' in normalized.parts or ':' in str(normalized):
        raise ValueError(f'Invalid archive-relative path: {relative}')
    path = root.joinpath(*normalized.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Archive path escapes its directory: {relative}')
    return path


def main(input_dir, output_dir, preview=False, overwrite=False):
    PREVIOUS, OUT = input_dir.resolve(), output_dir.resolve()
    if OUT == PREVIOUS or OUT.is_relative_to(PREVIOUS) or PREVIOUS.is_relative_to(OUT):
        raise ValueError('Input and output must be separate directories, without nesting.')
    previous = read(PREVIOUS / 'selection.json')
    if previous['policy'] != 'minimum_full_rmse_among_retained_final_and_loss_best':
        raise ValueError('Input selection must use minimum full RMSE among retained endpoints.')
    expected_names = {name for _, _, names in FAMILIES for name in names}
    if {row['name'] for row in previous['rows']} != expected_names or len(previous['rows']) != len(expected_names):
        raise ValueError('Input must contain the ten distinct cases of the original parameter study.')
    manifest_paths = {str(item['copy']).replace('\\', '/') for item in previous['sources']}
    required_paths = {'source_data/v_true.npy'} | {
        f'source_data/{row["name"]}/' + {'final': 'last_velocity.npy', 'loss_best': 'best_velocity.npy'}[row['endpoint']]
        for row in previous['rows']}
    if not required_paths.issubset(manifest_paths):
        raise ValueError('Source manifest must include the truth and every selected velocity array.')
    for item in previous['sources']:
        source = archive_path(PREVIOUS, item['copy'])
        if not source.is_relative_to(PREVIOUS / 'source_data'):
            raise ValueError('All manifest sources must be under source_data/.')
        if digest(source) != item['sha256']:
            raise ValueError(f'Frozen input changed: {item["copy"]}')
    if OUT.exists() and (not OUT.is_dir() or (any(OUT.iterdir()) and not overwrite)):
        raise ValueError('Output is not an empty directory; choose a new path or use --overwrite.')
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []
    for item in previous['sources']:
        source = archive_path(PREVIOUS, item['copy'])
        destination = archive_path(OUT, item['copy'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(destination) != item['sha256']:
            raise ValueError(f'Copied input changed: {item["copy"]}')
        sources.append(dict(copy=destination.relative_to(OUT).as_posix(), sha256=item['sha256']))
    shutil.copyfile(PREVIOUS / 'selected_metrics.csv', OUT / 'selected_metrics.csv')
    rows = copy.deepcopy(previous['rows'])
    truth = np.load(OUT / 'source_data/v_true.npy')
    by_name = {}
    for row in rows:
        filename = {'final': 'last_velocity.npy', 'loss_best': 'best_velocity.npy'}[row['endpoint']]
        array = np.load(OUT / 'source_data' / row['name'] / filename)
        assert array.shape == truth.shape == (94, 288) and np.isfinite(array).all()
        error = array.astype(np.float64) - truth
        np.testing.assert_allclose(np.sqrt(np.mean(error ** 2)), row['full_rmse'], rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.sqrt(np.mean(error[47:] ** 2)), row['deep_rmse'], rtol=0, atol=1e-6)
        by_name[row['name']] = dict(row, array=array)
    if previous['baseline'] != next(row for row in rows if row['name'] == 'baseline'):
        raise ValueError('Baseline record differs from the selected baseline row.')
    baseline = by_name['baseline']
    vmin, vmax = previous['velocity_limits_kmps']
    error_min, error_max = previous['error_limits_mps']
    profile_limits = previous['profile_xlim_mps']
    depth = np.arange(94) * .015
    colors = ['#333333', '#D55E00', '#0072B2', '#009E73']
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'path.simplify': False, 'legend.frameon': False})
    figures = []

    def image_panel(ax, array, title, error=False):
        values = array.astype(np.float64) - truth if error else array / 1000
        im = ax.imshow(values, extent=EXTENT, aspect='equal', interpolation='nearest', cmap='RdBu_r',
            vmin=error_min if error else vmin, vmax=error_max if error else vmax)
        ax.set(title=title, xlabel='Horizontal Distance (km)', ylabel='Depth (km)',
               xlim=(0, 4.305), ylim=(1.395, 0))
        ax.axhline(.705, color='#444444', linestyle='--', linewidth=.8)
        return im

    def comparison(factor, title, names, book):
        current = [by_name[name] for name in names]
        columns = len(current) + 1
        fig, axes = plt.subplots(3, columns, figsize=(5.8 * columns, 7.4), layout='constrained',
                                gridspec_kw={'height_ratios': [1, 1, .85]})
        fig.suptitle(title + ': Best Available Reconstructions', fontsize=18,
                     color=ACCENT, fontweight='bold')
        image_panel(axes[0, 0], truth, 'Ground Truth')
        axes[2, 0].axis('off')
        reference_text = axes[2, 0].text(.02, .95,
                        f'Reference Configuration\n{baseline["shots"]} shots; {baseline["layers"]} x {baseline["width"]}; Omega = {baseline["omega"]}\n'
                        'Seed = 3; Adam LR = 1e-4\nSelection: Minimum Full RMSE\nCandidates: Final / Loss-Best\nDeep Region: Depth >= 0.705 km',
                        transform=axes[2, 0].transAxes, va='top', fontsize=11, linespacing=1.5)
        for index, case in enumerate(current):
            array = case['array']
            vim = image_panel(axes[0, index + 1], array, '')
            model_header(axes[0, index + 1], case, factor)
            eim = image_panel(axes[1, index + 1], array, case['label'] + ' - Ground Truth', error=True)
            profile = np.sqrt(np.mean((array.astype(np.float64) - truth) ** 2, axis=1))
            axes[1, 0].plot(profile, depth, color=colors[index], label=case['label'], linewidth=1.3)
            axes[2, index + 1].axis('off')
            values = (f'{case["label"]} | Update {case["update"]}\n'
                      f'Full RMSE: {case["full_rmse"]:.2f} m/s\n'
                      f'Deep RMSE: {case["deep_rmse"]:.2f} m/s\n'
                      f'Deep SSIM: {case["deep_ssim"]:.4f}\n'
                      f'Waveform MSE: {case["data_mse"]:.3e}\n'
                      f'Full RMSE vs. Baseline: {case["full_rmse_vs_baseline_pct"]:+.2f}%')
            axes[2, index + 1].text(.02, .95, values, transform=axes[2, index + 1].transAxes,
                                  va='top', fontsize=11, linespacing=1.5,
                                  fontweight='bold' if case['name'] == 'baseline' else 'normal')
        axes[1, 0].set(title='RMSE by Depth', xlabel='RMSE (m/s)', ylabel='Depth (km)',
                       xlim=profile_limits, ylim=(1.395, 0))
        axes[1, 0].set_box_aspect(PANEL_ASPECT)
        axes[1, 0].axhline(.705, color='#444444', linestyle='--', linewidth=.8)
        handles, labels = axes[1, 0].get_legend_handles_labels()
        profile_legend = fig.legend(handles, labels, fontsize=10, loc='upper left', ncols=2,
                                    borderaxespad=0, handlelength=1.6, columnspacing=1)
        profile_legend.set_in_layout(False)
        axes[1, 0].grid(alpha=.2)
        velocity_bar = fig.colorbar(vim, ax=list(axes[0]), label='Velocity (km/s)', fraction=.018, pad=.015)
        error_bar = fig.colorbar(eim, ax=list(axes[1, 1:]), label='Signed Error (m/s)', fraction=.022, pad=.015)
        footer = fig.supxlabel('Signed error: blue = underestimate; red = overestimate.\n'
                      'Minimum ground-truth RMSE among retained endpoints; update counts differ. Shared scales; no smoothing.', fontsize=11)
        fig.canvas.draw()
        # Freeze the solved layout, then match each colorbar to its image row.
        fig.set_layout_engine('none')
        for bar, reference in [(velocity_bar, axes[0, -1]), (error_bar, axes[1, -1])]:
            position = bar.ax.get_position()
            image_position = reference.get_position()
            bar.ax.set_axes_locator(None)
            bar.ax.set_box_aspect(None)
            bar.ax.set_position([position.x0, image_position.y0, position.width, image_position.height])
        renderer = fig.canvas.get_renderer()
        reference_box = reference_text.get_window_extent(renderer)
        reference_position = axes[2, 0].get_position()
        profile_legend.set_bbox_to_anchor((reference_position.x0 + .02 * reference_position.width,
            (reference_box.y0 - 6 * fig.dpi / 72) / fig.bbox.height), transform=fig.transFigure)
        fig.canvas.draw()
        panel_ratios = [float(ax.get_window_extent().width / ax.get_window_extent().height)
                        for ax in [*axes[0], *axes[1, 1:]]]
        np.testing.assert_allclose(panel_ratios, 1 / PANEL_ASPECT, rtol=0, atol=1e-6)
        renderer = fig.canvas.get_renderer()
        row_gaps_pt = [float((top.xaxis.label.get_window_extent(renderer).y0
                              - bottom.title.get_window_extent(renderer).y1) * 72 / fig.dpi)
                       for top, bottom in zip(axes[0], axes[1])]
        assert min(row_gaps_pt) >= 3, f'Insufficient label/title spacing: {row_gaps_pt}'
        legend_footer_gap_pt = float((profile_legend.get_window_extent(renderer).y0
                                     - footer.get_window_extent(renderer).y1) * 72 / fig.dpi)
        assert legend_footer_gap_pt >= 3, 'Legend must fit between the reference block and footer.'
        destination = OUT / 'families' / factor
        destination.parent.mkdir(parents=True, exist_ok=True)
        for extension in ('.png', '.pdf', '.svg'):
            fig.savefig(destination.with_suffix(extension), dpi=300, facecolor='white')
        book.savefig(fig, dpi=300, facecolor='white')
        figures.append(dict(path='families/' + factor, names=names, kind='velocity_comparison',
                            figsize_inches=list(fig.get_size_inches()), panel_width_height=panel_ratios,
                            row_label_title_gaps_pt=row_gaps_pt, legend_footer_gap_pt=legend_footer_gap_pt))
        plt.close(fig)

    with PdfPages(OUT / ('preview.pdf' if preview else 'all_comparisons.pdf')) as book:
        for family in FAMILIES[:1] if preview else FAMILIES:
            comparison(*family, book)
    if preview:
        print('Preview rendered: families/shots.png')
        return
    selection = {key: copy.deepcopy(previous[key]) for key in (
        'policy', 'tie_policy', 'typography', 'language', 'baseline', 'rows', 'audited_endpoints',
        'velocity_limits_kmps', 'error_limits_mps', 'profile_xlim_mps', 'caveats')}
    selection.update(figures=figures, sources=sources,
                     layout_revision=dict(equal_spatial_aspect=True,
                                          change='Four wider landscape family figures only.'))
    assert selection['rows'] == previous['rows'] and selection['baseline'] == previous['baseline']
    (OUT / 'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    qa = dict(unchanged_selection=True, unchanged_csv=digest(OUT / 'selected_metrics.csv') == digest(PREVIOUS / 'selected_metrics.csv'),
              unchanged_source_count=len(sources), shared_scales_preserved=True, figures=figures)
    (OUT / 'qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    cards = ''.join(f'<h2>{title}</h2><img src="families/{factor}.png" alt="{title}">'
                    f'<p><a href="families/{factor}.pdf">PDF</a> · <a href="families/{factor}.svg">SVG</a></p>'
                    for factor, title, _ in FAMILIES)
    (OUT / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>IFWI 四组最佳模型对比 · 宽版</title><style>body{font-family:Arial,Microsoft YaHei,sans-serif;'
        'margin:30px 20px;line-height:1.7}img{max-width:100%;display:block}</style>'
        '<h1>四组最佳模型对比 · 宽版</h1><p>沿用此前选定的模型与全部指标，只调整绘图布局。'
        '模型横纵轴按实际空间距离等比例显示；英文标题、变量高亮和Baseline数值保留。</p>'
        '<p><a href="all_comparisons.pdf">四组PDF</a> · <a href="all_images.zip">图片包</a></p>' + cards + '</html>', encoding='utf-8')
    with zipfile.ZipFile(OUT / 'all_images.zip', 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for factor, _, _ in FAMILIES:
            for extension in ('.png', '.pdf', '.svg'):
                path = OUT / 'families' / (factor + extension)
                bundle.write(path, path.relative_to(OUT).as_posix())
        for name in ['all_comparisons.pdf', 'selected_metrics.csv', 'selection.json', 'qa.json', 'index.html']:
            bundle.write(OUT / name, name)
        bundle.write(Path(__file__), 'plot_best_model_comparisons.py')
        for item in sources:
            path = archive_path(OUT, item['copy'])
            bundle.write(path, item['copy'])
    (OUT / 'all_images.zip.sha256').write_text(digest(OUT / 'all_images.zip') + '\n', encoding='ascii')
    print(json.dumps(dict(output=str(OUT), figures=len(figures), unchanged_rows=len(rows),
                          panel_aspect=1 / PANEL_ASPECT, unchanged_sources=len(sources))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='Archive containing selection.json, selected_metrics.csv and source_data/.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Separate directory for wide figures and copied source arrays.')
    parser.add_argument('--preview', action='store_true', help='Render only the shot-count figure for layout review.')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing generated files in an existing output directory.')
    args = parser.parse_args()
    try:
        main(args.input_dir, args.output_dir, args.preview, args.overwrite)
    except (OSError, ValueError, KeyError, AssertionError) as error:
        parser.exit(1, f'Error: {error}\n')

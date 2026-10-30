"""Accuracy and signed forgetting maps for completed QA-only task orders."""
import argparse
import csv
import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
COMMON_TASKS = ['AGNEWS', 'AMAZON', 'DBPEDIA', 'YAHOO']
PAPER_ORDERS = [('DBPEDIA', 'AMAZON', 'YAHOO', 'AGNEWS'),
                ('DBPEDIA', 'AMAZON', 'AGNEWS', 'YAHOO'),
                ('YAHOO', 'AMAZON', 'AGNEWS', 'DBPEDIA')]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, signed=False):
    rounded = Decimal(str(round(float(value), 10))).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    if rounded == 0:
        rounded = abs(rounded)
    return ('+' if signed and rounded > 0 else '') + str(rounded)


def nullable(values):
    return [[None if not np.isfinite(value) else float(value) for value in row] for row in values]


def collect_completed():
    records = []
    for folder in sorted((ROOT / 'results').glob('standard_cl_*')):
        summary_path = folder / 'sequential_results.json'
        if not summary_path.is_file():
            continue
        summary = read(summary_path)
        tasks = summary['tasks']
        if len(summary['stages']) != 4 or len(tasks) != 4 or set(tasks) != set(COMMON_TASKS):
            continue
        launch = read(folder / 'launch.json')
        command = launch['command']
        if not any(value.endswith('/train_sequential_tasks.py') for value in command):
            continue
        candidates = []
        for matrix_path in sorted(folder.rglob('accuracy_matrix.json')):
            payload = read(matrix_path)
            if payload.get('complete') and payload['tasks'] == tasks:
                candidates.append((matrix_path, payload))
        if not candidates:
            continue
        matrix_path, payload = candidates[0]
        matrix = np.asarray(payload['accuracy'], dtype=float)
        if matrix.shape != (5, 4) or not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
            raise ValueError(f'Invalid accuracy matrix: {matrix_path}')
        for alternate, value in candidates[1:]:
            if not np.array_equal(matrix, np.asarray(value['accuracy'])):
                raise ValueError(f'Duplicate accuracy matrices disagree: {alternate}')
        settings = payload['evaluation']
        assert settings['split'] == 'test' and settings['max_input_length'] == 1536
        assert settings['max_new_tokens'] == 8 and settings['seed'] == 1
        adapters = [folder / f'{stage:02d}_{task}' / 'best_model' for stage, task in enumerate(tasks, 1)]
        actual = [None if item is None else str(Path(item).resolve()) for item in payload['adapter_dirs']]
        assert actual == [None, *map(str, adapters)]
        assert all((adapter / 'adapter_model.safetensors').is_file() for adapter in adapters)
        records.append({'tasks': tasks, 'run_dir': folder, 'summary_path': summary_path, 'summary': summary,
                        'matrix_path': matrix_path, 'matrix': matrix * 100, 'payload': payload,
                        'completed_time': summary_path.stat().st_mtime})
    def key(record):
        tasks = tuple(record['tasks'])
        return (0, PAPER_ORDERS.index(tasks)) if tasks in PAPER_ORDERS else (1, record['completed_time'])
    records.sort(key=key)
    if not records:
        raise ValueError('No completed QA-only task orders')
    assert len({tuple(record['tasks']) for record in records}) == len(records)
    return records


def audit_predictions(records):
    datasets, modules, data_hashes = {}, {}, {}
    for task in COMMON_TASKS:
        path = ROOT / 'data' / task / 'test.jsonl'
        datasets[task] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        modules[task] = importlib.import_module(f'tasks.{task}')
        data_hashes[task] = digest(path)
    sources = []
    base = None
    for record in records:
        task_order, matrix = record['tasks'], record['matrix']
        base_common = [matrix[0, task_order.index(task)] for task in COMMON_TASKS]
        if base is None:
            base = base_common
        assert np.allclose(base, base_common, atol=1e-12, rtol=0)
        for stage in range(5):
            row_name = '00_base' if stage == 0 else f'{stage:02d}_{task_order[stage - 1]}'
            for col, task in enumerate(task_order):
                path = record['matrix_path'].parent / row_name / f'{col + 1:02d}_{task}.json'
                raw = path.read_bytes()
                saved = json.loads(raw)
                results, metrics = saved['results'], saved['metrics']
                assert len(results) == len(datasets[task]) == metrics['total']
                correct = 0
                for index, (result, example) in enumerate(zip(results, datasets[task])):
                    label = modules[task].extract_label_from_example(example)
                    prediction = result['predicted_answer']
                    success = prediction is not None and prediction == label
                    if (result['index'] != index or result['ground_truth_answer'] != label
                            or bool(result['correct']) != success):
                        raise ValueError(f'Inconsistent prediction: {path}, example {index}')
                    correct += int(success)
                score = correct / len(results)
                assert metrics['correct'] == correct and abs(metrics['accuracy'] - score) < 1e-12
                assert abs(matrix[stage, col] - score * 100) < 1e-10
                if stage > col:
                    assert abs(record['summary']['stages'][stage - 1]['test_metrics'][task]['accuracy'] - score) < 1e-12
                sources.append({'run_dir': str(record['run_dir']), 'stage': stage, 'task': task,
                                'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
    return sources, data_hashes


def grid(ax, values, xlabels, ylabels, title, *, forgetting=False, limit=35, fontsize=13):
    values = np.asarray(values, dtype=float)
    if forgetting:
        cmap = LinearSegmentedColormap.from_list('forgetting', ['#B64342', '#fafafa', '#3775BA']).with_extremes(bad='#e9edf2')
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect='auto')
    else:
        image = ax.imshow(np.ma.masked_invalid(values), cmap='Blues', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=fontsize)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 3, pad=12, fontweight='bold')
    ax.set_xticks(np.arange(-.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, values.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row, col in np.ndindex(values.shape):
        value = values[row, col]
        if not np.isfinite(value):
            ax.text(col, row, '—', ha='center', va='center', fontsize=fontsize, color='#a7afbb')
            continue
        color = image.cmap(image.norm(value))
        luminance = .299 * color[0] + .587 * color[1] + .114 * color[2]
        ax.text(col, row, number(value, forgetting), ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white' if luminance < .55 else '#17233b')
    return image


def export(fig, stem, output, pdf=None):
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{extension}', dpi=300, facecolor='white')
    if pdf:
        pdf.savefig(fig)


def overview(records, output, limit, forgetting=False):
    columns = min(4, len(records))
    rows = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 5.7, rows * 5.2), squeeze=False)
    for ax, record in zip(axes.flat, records):
        tasks = record['tasks']
        title = record['display_id'] + ' | ' + ' → '.join(tasks[:2]) + ' →\n' + ' → '.join(tasks[2:])
        values = record['forgetting'] if forgetting else record['matrix']
        image = grid(ax, values, tasks, ['M0 base', 'M1', 'M2', 'M3', 'M4'], title,
                     forgetting=forgetting, limit=limit, fontsize=11)
        ax.tick_params(axis='x', labelrotation=25)
        if not forgetting:
            for index in range(4):
                ax.add_patch(Rectangle((index - .47, index + .53), .94, .94, fill=False, edgecolor='#163755', linewidth=1.3))
    for ax in axes.flat[len(records):]:
        ax.set_axis_off()
    title = 'Signed forgetting (pp) · learned tasks only' if forgetting else 'Accuracy (%) · base + four adapters on all four tasks'
    fig.suptitle(f'QA only · {len(records)} completed orders\n{title}', fontsize=23, y=.985, fontweight='bold')
    note = 'Mi is the saved adapter after the first i tasks in its panel; M0 is the base model.'
    if forgetting:
        note += '\nCurrent accuracy minus the same task just-trained accuracy; negative = forgetting. Base/unlearned cells are blank.'
    else:
        note += '\nOutlined cells are just-trained references. Upper-triangle and base-model evaluations are included.'
    fig.text(.5, .02, note, ha='center', fontsize=13)
    fig.tight_layout(rect=(0, .075, .955, .90), pad=1.6, h_pad=2.8, w_pad=2.2)
    cax = fig.add_axes([.965, .16, .012, .67])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label('Change (pp)' if forgetting else 'Accuracy (%)', fontsize=14)
    colorbar.outline.set_visible(False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = collect_completed()
    sources, data_hashes = audit_predictions(records)
    final_accuracy, final_forgetting = [], []
    cells, summaries, results = [], [], []
    for index, record in enumerate(records, 1):
        record['display_id'] = f'R{index}'
        matrix = record['matrix']
        references = np.diag(matrix[1:])
        raw_change = matrix - references[None, :]
        learned = np.arange(5)[:, None] >= (np.arange(4)[None, :] + 1)
        forgetting = np.where(learned, raw_change, np.nan)
        assert np.allclose(np.diag(forgetting[1:]), 0, atol=1e-12)
        record['forgetting'] = forgetting
        positions = [record['tasks'].index(task) for task in COMMON_TASKS]
        a = matrix[-1, positions].tolist()
        f = forgetting[-1, positions].tolist()
        final_accuracy.append([*a, float(np.mean(a))])
        final_forgetting.append([*f, float(np.mean(f))])
        summaries.append({'display_id': record['display_id'], 'task_order': ' -> '.join(record['tasks']),
                          **{f'{task}_accuracy_pct': a[i] for i, task in enumerate(COMMON_TASKS)},
                          'mean_accuracy_pct': float(np.mean(a)),
                          **{f'{task}_forgetting_pp': f[i] for i, task in enumerate(COMMON_TASKS)},
                          'mean_forgetting_all4_pp': float(np.mean(f)),
                          'mean_forgetting_old3_pp': float(np.mean(forgetting[-1, :3])),
                          'run_dir': str(record['run_dir'])})
        for stage in range(5):
            for col, task in enumerate(record['tasks']):
                cells.append({'display_id': record['display_id'], 'task_order': ' -> '.join(record['tasks']),
                              'model': f'M{stage}', 'trained_through': 'BASE' if stage == 0 else record['tasks'][stage - 1],
                              'evaluated_task': task, 'accuracy_pct': float(matrix[stage, col]),
                              'reference_model': f'M{col + 1}', 'just_trained_accuracy_pct': float(references[col]),
                              'change_vs_just_trained_pp': float(raw_change[stage, col]),
                              'forgetting_applicable': bool(learned[stage, col]),
                              'forgetting_pp': None if not learned[stage, col] else float(forgetting[stage, col])})
        results.append({'display_id': record['display_id'], 'tasks': record['tasks'], 'run_dir': str(record['run_dir']),
                        'matrix_source': str(record['matrix_path']), 'matrix_sha256': digest(record['matrix_path']),
                        'summary_sha256': digest(record['summary_path']), 'accuracy_pct': matrix.tolist(),
                        'forgetting_pp': nullable(forgetting), 'reference_models': [f'M{i}' for i in range(1, 5)],
                        'just_trained_accuracy_pct': references.tolist()})
    limit = math.ceil(max(1, max(float(np.nanmax(np.abs(record['forgetting']))) for record in records)) / 5) * 5
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    report = output / 'accuracy_forgetting_maps.pdf'
    with PdfPages(report) as pdf:
        labels = [record['display_id'] + '  ' + ' → '.join(record['tasks']) for record in records]
        columns = [*COMMON_TASKS, 'Mean (4 tasks)']
        fig, axes = plt.subplots(1, 2, figsize=(22, max(6, len(records) * .58 + 2)))
        for ax, values, title, is_forgetting in zip(axes, [final_accuracy, final_forgetting], ['Final accuracy (%)', 'Final signed forgetting (pp)'], [False, True]):
            row_labels = labels if not is_forgetting else [record['display_id'] for record in records]
            image = grid(ax, values, columns, row_labels, title, forgetting=is_forgetting, limit=limit, fontsize=12)
            ax.tick_params(axis='x', labelrotation=20)
            bar = fig.colorbar(image, ax=ax, fraction=.04, pad=.035)
            bar.outline.set_visible(False)
        fig.suptitle(f'QA only · {len(records)} completed orders · final adapter M4', fontsize=23, fontweight='bold')
        fig.text(.5, .025, 'Forgetting: current accuracy minus each task just-trained accuracy (pp). Negative = forgetting; positive = improvement. Mean includes all 4 tasks.', ha='center', fontsize=12)
        fig.tight_layout(rect=(0, .08, 1, .91), pad=1.8, w_pad=2.8)
        export(fig, 'final_accuracy_forgetting_comparison', output, pdf)
        plt.close(fig)
        for values, title, stem, is_forgetting in [(final_accuracy, 'Final accuracy (%)', 'final_accuracy_map', False),
                                                   (final_forgetting, 'Final signed forgetting (pp)', 'final_forgetting_map', True)]:
            fig, ax = plt.subplots(figsize=(14, max(6, len(records) * .6 + 2)))
            image = grid(ax, values, columns, labels, title, forgetting=is_forgetting, limit=limit, fontsize=12)
            ax.tick_params(axis='x', labelrotation=15)
            fig.colorbar(image, ax=ax, fraction=.035, pad=.035)
            fig.text(.5, .03, f'R1–R{len(records)} are display IDs; exact training orders appear beside each row. Mean includes all 4 tasks.', ha='center', fontsize=11)
            fig.tight_layout(rect=(0, .075, 1, .98), pad=1.8)
            export(fig, stem, output)
            plt.close(fig)
        for is_forgetting, stem in [(False, 'all_orders_accuracy_maps'), (True, 'all_orders_forgetting_maps')]:
            fig = overview(records, output, limit, is_forgetting)
            export(fig, stem, output, pdf)
            plt.close(fig)
        for record in records:
            tasks = record['tasks']
            rows = ['M0 · base', *[f'M{i} · after {task}' for i, task in enumerate(tasks, 1)]]
            fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
            for ax, values, title, is_forgetting in zip(axes, [record['matrix'], record['forgetting']],
                                                       ['Accuracy (%) · full 5×4', 'Signed forgetting (pp)'], [False, True]):
                xlabels = tasks if not is_forgetting else [f'{task}\nref: M{i}' for i, task in enumerate(tasks, 1)]
                image = grid(ax, values, xlabels, rows, title, forgetting=is_forgetting, limit=limit, fontsize=12)
                fig.colorbar(image, ax=ax, fraction=.04, pad=.035)
            fig.suptitle(record['display_id'] + ' | ' + ' → '.join(tasks), fontsize=21, fontweight='bold')
            fig.text(.5, .035, 'Forgetting is current accuracy minus the same task just-trained accuracy. Negative = forgetting. Base/unlearned cells are blank.', ha='center', fontsize=11)
            fig.tight_layout(rect=(0, .09, 1, .90), pad=1.6, w_pad=2)
            stem = record['display_id'] + '_' + '_'.join(tasks) + '_accuracy_forgetting'
            export(fig, stem, output, pdf)
            plt.close(fig)
    for name, rows in [('final_metrics.csv', summaries), ('stage_task_metrics.csv', cells)]:
        with (output / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    snapshot = {'complete': True, 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'orders': len(records), 'performance_data': 'QA test sets only', 'text_mix_included': False,
                'accuracy_shape_per_order': [5, 4], 'forgetting_definition': 'Current accuracy minus the same task just-trained accuracy, in percentage points',
                'forgetting_mask': 'Only stage >= task just-trained stage; base and not-yet-learned cells are null',
                'display_ids_are_not_paper_order_numbers': True, 'task_columns_in_final_comparison': COMMON_TASKS,
                'data_sha256': data_hashes, 'validated_accuracy_cells': len(sources),
                'results': results, 'prediction_sources': sources, 'final_metrics': summaries}
    (output / 'maps_and_provenance.json').write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    raw = report.read_bytes()
    pages = len(re.findall(rb'/Type\s*/Page\b', raw))
    assert pages == len(records) + 3 and raw.rstrip().endswith(b'%%EOF')
    from PIL import Image
    for path in output.glob('*.png'):
        with Image.open(path) as image:
            image.verify()
    (output / 'README.md').write_text(f'''# Completed QA-only accuracy and forgetting maps

{len(records)} fully trained orders with complete 5×4 QA test accuracy matrices.
All {len(sources)} cells were checked against saved predictions and current test labels.
No model inference was performed. The active experiment queue was not changed.

Accuracy includes M0 base and M1–M4 on all four tasks, including future tasks.
Forgetting is current performance minus the same task just-trained performance,
in percentage points. Negative means degradation, positive improvement. The
just-trained reference for the i-th task is Mi, not M0. Base and unlearned cells
are blank because forgetting is evaluated only after learning a task.
Final comparison columns use AGNEWS, AMAZON, DBPEDIA, YAHOO consistently.
Mean (4 tasks) includes the zero for the last task; final_metrics.csv also gives
the mean over only the three older tasks. R IDs are display IDs, not paper orders.

accuracy_forgetting_maps.pdf has {pages} pages: final summary, all accuracy maps,
all forgetting maps, then one enlarged page per order. PNG and PDF figures,
CSV values, source checksums and prediction audit are provided alongside it.
''')
    with zipfile.ZipFile(output / 'accuracy_forgetting_maps_bundle.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.name)
    print(json.dumps({'orders': len(records), 'validated_accuracy_cells': len(sources), 'report_pages': pages,
                      'output_dir': str(output), 'final_metrics': summaries}, indent=2))


if __name__ == '__main__':
    main()

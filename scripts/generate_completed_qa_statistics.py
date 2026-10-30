"""Generate task and per-option statistics from completed QA-only predictions."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
STYLE = runpy.run_path(str(ROOT / 'scripts/plot_completed_qa_order_maps.py'), run_name='map_helpers')
grid = STYLE['grid']
export = STYLE['export']
nullable = STYLE['nullable']
TASKS = STYLE['COMMON_TASKS']


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing statistics snapshot: {output}')
    output.mkdir(parents=True)
    with (output / 'generation.log').open('w') as log:
        subprocess.run([sys.executable, str(ROOT / 'scripts/plot_completed_qa_order_maps.py'),
                        '--output_dir', str(output)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    snapshot = json.loads((output / 'maps_and_provenance.json').read_text())
    records = snapshot['results']
    print(f"Validated {len(records)} completed orders; calculating per-option statistics.", flush=True)
    modules = {task: importlib.import_module(f'tasks.{task}') for task in TASKS}
    labels, names = {}, {}
    mapping = []
    import re
    for task, module in modules.items():
        names[task] = dict(re.findall(r'^([A-N])\.\s+(.+)$', module._prompt_format_after, re.M))
        labels[task] = sorted(names[task])
        assert len(labels[task]) == {'AGNEWS': 4, 'AMAZON': 5, 'DBPEDIA': 14, 'YAHOO': 10}[task]
        mapping.extend({'task': task, 'option': label, 'class_name': names[task][label]} for label in labels[task])
    checksums = {item['path']: item['sha256'] for item in snapshot['prediction_sources']}
    options, tasks, confusion, plots = [], [], [], {}
    for record in records:
        order = record['tasks']
        plot_data = {}
        for task in order:
            ref_stage = order.index(task) + 1
            accuracies = np.zeros((5, len(labels[task])))
            counts_by_stage = []
            predictions_by_stage = []
            for stage in range(5):
                folder = '00_base' if stage == 0 else f'{stage:02d}_{order[stage - 1]}'
                path = Path(record['matrix_source']).parent / folder / f'{order.index(task) + 1:02d}_{task}.json'
                raw = path.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == checksums[str(path)]
                saved = json.loads(raw)
                rows = saved['results']
                true_count = Counter(row['ground_truth_answer'] for row in rows)
                predicted_count = Counter(row['predicted_answer'] for row in rows)
                correct_count = Counter(row['ground_truth_answer'] for row in rows if row['correct'])
                invalid_count = Counter(row['ground_truth_answer'] for row in rows if row['predicted_answer'] is None)
                pairs = Counter((row['ground_truth_answer'], row['predicted_answer']) for row in rows)
                assert set(true_count) == set(labels[task])
                assert sum(correct_count.values()) == saved['metrics']['correct']
                assert len(rows) == saved['metrics']['total']
                assert abs(100 * sum(correct_count.values()) / len(rows) - record['accuracy_pct'][stage][order.index(task)]) < 1e-10
                for col, label in enumerate(labels[task]):
                    accuracies[stage, col] = 100 * correct_count[label] / true_count[label]
                counts_by_stage.append((true_count, predicted_count, correct_count, invalid_count, len(rows)))
                predictions_by_stage.append(pairs)
            changes = accuracies - accuracies[ref_stage]
            changes[:ref_stage] = np.nan
            assert np.array_equal(accuracies[ref_stage] - accuracies[ref_stage], changes[ref_stage])
            plot_data[task] = {'accuracy': accuracies, 'forgetting': changes, 'reference_stage': ref_stage}
            for stage in range(5):
                true_count, predicted_count, correct_count, invalid_count, total = counts_by_stage[stage]
                reference_accuracy = record['accuracy_pct'][ref_stage][order.index(task)]
                accuracy = record['accuracy_pct'][stage][order.index(task)]
                info = {'display_id': record['display_id'], 'task_order': ' -> '.join(order), 'model': f'M{stage}',
                        'trained_through': 'BASE' if stage == 0 else order[stage - 1], 'evaluated_task': task,
                        'total': total, 'correct': sum(correct_count.values()), 'incorrect': total - sum(correct_count.values()),
                        'invalid_predictions': sum(invalid_count.values()), 'accuracy_pct': accuracy,
                        'reference_model': f'M{ref_stage}', 'just_trained_accuracy_pct': reference_accuracy,
                        'forgetting_pp': accuracy - reference_accuracy if stage >= ref_stage else None,
                        'change_vs_previous_stage_pp': accuracy - record['accuracy_pct'][stage - 1][order.index(task)] if stage else None}
                tasks.append(info)
                class_forgetting = []
                for col, label in enumerate(labels[task]):
                    predicted = predicted_count[label]
                    correct = correct_count[label]
                    item = {'display_id': record['display_id'], 'task_order': ' -> '.join(order), 'model': f'M{stage}',
                            'trained_through': info['trained_through'], 'evaluated_task': task, 'option': label,
                            'class_name': names[task][label], 'true_samples': true_count[label], 'correct': correct,
                            'incorrect': true_count[label] - correct, 'class_accuracy_pct': float(accuracies[stage, col]),
                            'predicted_as_option': predicted, 'prediction_precision_pct': 100 * correct / predicted if predicted else None,
                            'false_positive': predicted - correct, 'prediction_share_pct': 100 * predicted / total,
                            'invalid_predictions_with_this_true_label': invalid_count[label],
                            'reference_model': f'M{ref_stage}', 'just_trained_class_accuracy_pct': float(accuracies[ref_stage, col]),
                            'forgetting_pp': float(changes[stage, col]) if stage >= ref_stage else None,
                            'change_vs_previous_stage_pp': float(accuracies[stage, col] - accuracies[stage - 1, col]) if stage else None}
                    options.append(item)
                    if stage >= ref_stage:
                        class_forgetting.append(changes[stage, col] * true_count[label] / total)
                weighted = sum(accuracies[stage, col] * true_count[label] / total for col, label in enumerate(labels[task]))
                assert abs(weighted - accuracy) < 1e-10
                if stage >= ref_stage:
                    assert abs(sum(class_forgetting) - info['forgetting_pp']) < 1e-10
                for (gold, prediction), count in sorted(predictions_by_stage[stage].items(), key=lambda item: (item[0][0], item[0][1] or '')):
                    confusion.append({'display_id': record['display_id'], 'task_order': ' -> '.join(order),
                                      'model': f'M{stage}', 'evaluated_task': task, 'true_option': gold,
                                      'predicted_option': prediction or 'INVALID', 'count': count})
        plots[record['display_id']] = plot_data
    write_csv(output / 'accuracy_forgetting_statistics.csv', tasks)
    write_csv(output / 'per_option_statistics.csv', options)
    write_csv(output / 'final_per_option_statistics.csv', [item for item in options if item['model'] == 'M4'])
    write_csv(output / 'option_label_mapping.csv', mapping)
    write_csv(output / 'confusion_counts.csv', confusion)
    assert len(options) == len(records) * 5 * 33
    assert len(tasks) == len(records) * 20
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    report = output / 'per_option_accuracy_forgetting.pdf'
    with PdfPages(report) as pdf:
        for task in TASKS:
            values = [plots[record['display_id']][task]['accuracy'][-1] for record in records]
            fig, ax = plt.subplots(figsize=(max(15, len(labels[task]) * .8 + 7), max(7.5, len(records) * .56 + 3.0)))
            ylabels = [record['display_id'] + '  ' + ' → '.join(record['tasks']) for record in records]
            image = grid(ax, values, labels[task], ylabels, f'{task} · final adapter M4 · per-option accuracy (%)', fontsize=12)
            fig.colorbar(image, ax=ax, fraction=.025, pad=.025, label='Class accuracy (%)')
            legend = '\n'.join(textwrap.wrap('   |   '.join(label + ': ' + names[task][label] for label in labels[task]), width=145))
            fig.text(.5, .075, legend, ha='center', va='center', fontsize=10)
            fig.text(.5, .025, 'Denominator: examples whose TRUE label is that option. All values use held-out QA test predictions.', ha='center', fontsize=11)
            fig.tight_layout(rect=(0, .17, 1, .98), pad=1.8)
            export(fig, f'final_{task}_option_accuracy', output, pdf)
            plt.close(fig)
        for record in records:
            for kind in ('accuracy', 'forgetting'):
                fig, axes = plt.subplots(2, 2, figsize=(22, 12.8))
                for ax, task in zip(axes.flat, record['tasks']):
                    values = plots[record['display_id']][task][kind]
                    fontsize = 9 if len(labels[task]) > 10 else 11
                    image = grid(ax, values, labels[task], ['M0 base', *[f'M{i} {name}' for i, name in enumerate(record['tasks'], 1)]],
                                 task + (' · class accuracy (%)' if kind == 'accuracy' else ' · signed class forgetting (pp)'),
                                 forgetting=kind == 'forgetting', limit=100, fontsize=fontsize)
                    fig.colorbar(image, ax=ax, fraction=.035, pad=.025)
                    legend = '\n'.join(textwrap.wrap(' | '.join(label + ': ' + names[task][label] for label in labels[task]), width=110))
                    ax.text(0, -.15, legend, transform=ax.transAxes, va='top', fontsize=9, linespacing=1.5)
                fig.suptitle(record['display_id'] + ' | ' + ' → '.join(record['tasks']), fontsize=23, fontweight='bold', y=.975)
                note = 'Accuracy per TRUE option class; M0 base and every saved stage adapter, including future tasks.' if kind == 'accuracy' else 'Current class accuracy minus its accuracy in the same task just-trained adapter. Negative = forgetting; base/unlearned cells are blank.'
                fig.text(.5, .025, note, ha='center', fontsize=12)
                fig.tight_layout(rect=(0, .085, 1, .92), pad=1.8, h_pad=8.0, w_pad=2.2)
                export(fig, record['display_id'] + '_per_option_' + kind, output, pdf)
                plt.close(fig)
    from PIL import Image
    for path in output.glob('*.png'):
        with Image.open(path) as picture:
            picture.verify()
    raw = report.read_bytes()
    pages = len(re.findall(rb'/Type\s*/Page\b', raw))
    assert pages == 4 + len(records) * 2 and raw.rstrip().endswith(b'%%EOF')
    definitions = '''# Accuracy、Forgetting 与逐选项统计

本快照只包括完整完成训练和 5×4 评估的无 text_mix 顺序，数据来自已保存的 QA 测试集预测。
这些是测试集结果，不是旧训练样本的记忆轨迹；本次没有重新运行模型。

- Accuracy：该任务答对样本数除以全部测试样本数。
- Forgetting：当前准确率减去该任务刚训练后的准确率，单位为百分点；负值表示遗忘。
- 任务级和类别级 Forgetting 均只在该任务训练完成后计算；base、尚未训练的任务留空。
- 每个选项的类别正确率（class_accuracy_pct）：真实标签为该选项的样本中被正确分类的比例，即 recall。
- 预测选项准确率（prediction_precision_pct）：所有被预测为该选项的回答中，真实标签也是该选项的比例，即 precision。没有预测为该选项时留空。
- prediction_share_pct：该选项在全部预测中的比例。INVALID 为无法解析成当前任务合法选项的回答。
- 类别名称由当前任务提示中的选项定义读取，见 option_label_mapping.csv。
- 每个类别同时提供正确数、错误数、真实样本数、预测数量、类别遗忘及相邻阶段变化。
- 最终任务平均 Forgetting 同时提供四任务平均（含最后任务的零）和前三个旧任务平均。
- R1、R2 等只是图表显示编号，不是论文 order 编号；完整顺序见各行 task_order。

文件：
- final_metrics.csv：每个顺序最终 accuracy 和 forgetting 汇总。
- accuracy_forgetting_statistics.csv：base 和四个 adapter 在全部四任务上的详细统计。
- per_option_statistics.csv：所有顺序、所有模型阶段、所有任务的逐选项统计。
- final_per_option_statistics.csv：只保留最终 M4 的逐选项统计。
- accuracy_forgetting_maps.pdf：任务级完整热图和顺序对比。
- per_option_accuracy_forgetting.pdf：各任务最终选项正确率，以及每个顺序的五模型类别正确率和类别遗忘。
- confusion_counts.csv：每个阶段的真实选项→预测选项计数。
- maps_and_provenance.json：数据、结果来源与逐样本审核校验信息。
'''
    (output / '统计说明.md').write_text(definitions)
    validation = {'complete': True, 'orders': len(records), 'task_model_cells': len(tasks),
                  'per_option_rows': len(options), 'final_per_option_rows': sum(item['model'] == 'M4' for item in options),
                  'options_per_task': {task: len(labels[task]) for task in TASKS},
                  'per_option_report_pages': pages, 'all_class_counts_reconstruct_task_accuracy': True,
                  'weighted_class_forgetting_matches_task_forgetting': True, 'source_predictions_verified': True,
                  'performance_data': 'QA test sets only', 'model_inference_performed': False}
    (output / 'statistics_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    with zipfile.ZipFile(output / 'accuracy_forgetting_option_statistics.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.name)
    print(json.dumps({**validation, 'output_dir': str(output)}, indent=2))


if __name__ == '__main__':
    main()

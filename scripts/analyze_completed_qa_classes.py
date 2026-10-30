"""Describe class retention and observed later-task effects in a saved snapshot."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import runpy

ROOT = Path(__file__).resolve().parents[1]
HELPERS = runpy.run_path(str(ROOT / 'scripts/plot_completed_qa_order_maps.py'), run_name='class_helpers')
TASKS = HELPERS['COMMON_TASKS']
number = HELPERS['number']


def read_csv(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics_dir', type=Path, required=True)
    args = parser.parse_args()
    source = args.statistics_dir.resolve()
    out = source / 'class_analysis'
    out.mkdir(exist_ok=True)
    final = read_csv(source / 'final_per_option_statistics.csv')
    stages = read_csv(source / 'per_option_statistics.csv')
    confusion = read_csv(source / 'confusion_counts.csv')
    groups = defaultdict(list)
    old_pairs = set()
    for row in final:
        if row['reference_model'] != 'M4':
            groups[(row['evaluated_task'], row['option'], row['class_name'])].append(row)
            old_pairs.add((row['display_id'], row['evaluated_task']))
    summary = []
    for (task, label, name), values in groups.items():
        losses = [float(row['forgetting_pp']) for row in values]
        final_scores = [float(row['class_accuracy_pct']) for row in values]
        wrong = Counter()
        ids = {row['display_id'] for row in values}
        for item in confusion:
            if (item['display_id'] in ids and item['model'] == 'M4' and item['evaluated_task'] == task
                    and item['true_option'] == label and item['predicted_option'] != label):
                wrong[item['predicted_option']] += int(item['count'])
        destination, count = wrong.most_common(1)[0] if wrong else ('NONE', 0)
        summary.append({'task': task, 'option': label, 'class_name': name, 'old_task_orders': len(values),
                        'mean_just_trained_accuracy_pct': statistics.mean(float(row['just_trained_class_accuracy_pct']) for row in values),
                        'mean_final_accuracy_pct': statistics.mean(final_scores),
                        'mean_forgetting_pp': statistics.mean(losses), 'min_forgetting_pp': min(losses),
                        'max_forgetting_pp': max(losses), 'orders_with_decrease': sum(value < -1e-9 for value in losses),
                        'orders_with_increase': sum(value > 1e-9 for value in losses),
                        'min_final_accuracy_pct': min(final_scores), 'max_final_accuracy_pct': max(final_scores),
                        'most_common_wrong_option': destination,
                        'top_wrong_option_share_of_errors_pct': 100 * count / sum(wrong.values()) if wrong else None,
                        'total_error_predictions_across_old_orders': sum(wrong.values())})
    summary.sort(key=lambda row: (TASKS.index(row['task']), row['mean_forgetting_pp']))
    prefixes = defaultdict(list)
    for row in stages:
        stage, reference = int(row['model'][1:]), int(row['reference_model'][1:])
        if stage <= reference:
            continue
        order = tuple(row['task_order'].split(' -> '))
        prefixes[(order[:stage], row['evaluated_task'], row['option'])].append(row)
    unique = []
    for (prefix, task, label), values in prefixes.items():
        changes = [float(row['change_vs_previous_stage_pp']) for row in values]
        unique.append({'training_prefix': ' -> '.join(prefix), 'new_task': prefix[-1], 'old_task': task,
                       'option': label, 'class_name': values[0]['class_name'],
                       'mean_change_pp': statistics.mean(changes), 'duplicate_order_observations': len(values),
                       'within_prefix_range_pp': max(changes) - min(changes),
                       'source_display_ids': ','.join(row['display_id'] for row in values)})
    effects = defaultdict(list)
    for row in unique:
        effects[(row['new_task'], row['old_task'], row['option'], row['class_name'])].append(row['mean_change_pp'])
    transitions = [{'new_task': new, 'old_task': old, 'option': label, 'class_name': name,
                    'unique_training_prefixes': len(values), 'mean_adjacent_change_pp': statistics.mean(values),
                    'min_adjacent_change_pp': min(values), 'max_adjacent_change_pp': max(values)}
                   for (new, old, label, name), values in effects.items()]
    transitions.sort(key=lambda row: row['mean_adjacent_change_pp'])
    write_csv(out / 'class_retention_summary.csv', summary)
    write_csv(out / 'later_task_class_changes.csv', transitions)
    write_csv(out / 'deduplicated_prefix_observations.csv', unique)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    with PdfPages(out / 'class_analysis.pdf') as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(21, 14))
        for ax, task in zip(axes.flat, TASKS):
            rows = [row for row in summary if row['task'] == task]
            y = np.arange(len(rows))
            means = [row['mean_forgetting_pp'] for row in rows]
            ax.barh(y, means, color=['#B64342' if value < 0 else '#3775BA' for value in means], alpha=.8, height=.62)
            for index, row in enumerate(rows):
                ax.plot([row['min_forgetting_pp'], row['max_forgetting_pp']], [index, index], color='#17233b', linewidth=1.5)
                ax.scatter(row['mean_forgetting_pp'], index, s=25, color='#17233b', zorder=3)
                ax.text(1.01, index, number(row['mean_forgetting_pp'], signed=True), transform=ax.get_yaxis_transform(), va='center', fontsize=11)
            ax.set_yticks(y, [row['option'] + ' · ' + row['class_name'] for row in rows], fontsize=11)
            ax.invert_yaxis()
            ax.axvline(0, color='#777777', linewidth=1)
            ax.set_xlim(-105, 40)
            ax.set_xlabel('Final minus just-trained class accuracy (pp)')
            ax.set_title(f"{task} · {rows[0]['old_task_orders']} orders where task is old", fontsize=16, fontweight='bold')
        fig.suptitle('Class retention across completed QA-only orders', fontsize=24, fontweight='bold')
        fig.text(.5, .035, 'Bars/dots: mean change. Lines: observed minimum–maximum across orders, not confidence intervals.\nTasks trained last are excluded; results are descriptive for this incomplete, single-seed order set.', ha='center', fontsize=12)
        fig.tight_layout(rect=(0, .09, .98, .94), pad=2.0, w_pad=5, h_pad=3)
        fig.savefig(out / 'class_retention_summary.png', dpi=300)
        fig.savefig(out / 'class_retention_summary.pdf')
        pdf.savefig(fig)
        plt.close(fig)
        ordered = [(row['task'], row['option'], row['class_name']) for row in summary]
        lookup = {(row['new_task'], row['old_task'], row['option']): row for row in transitions}
        matrix = np.full((len(ordered), 4), np.nan)
        for index, (task, label, name) in enumerate(ordered):
            for col, new in enumerate(TASKS):
                entry = lookup.get((new, task, label))
                if entry:
                    matrix[index, col] = entry['mean_adjacent_change_pp']
        fig, ax = plt.subplots(figsize=(13, 19))
        image = HELPERS['grid'](ax, matrix, TASKS, [task + ' ' + label + ' · ' + name for task, label, name in ordered],
                                'Class changes immediately after each later task', forgetting=True, limit=100, fontsize=10)
        ax.set_xlabel('Newly trained task')
        for row, (task, label, _) in enumerate(ordered):
            for col, new in enumerate(TASKS):
                entry = lookup.get((new, task, label))
                if entry:
                    ax.text(col + .36, row + .32, f"n={entry['unique_training_prefixes']}", fontsize=7, ha='right', color='#555555')
        fig.colorbar(image, ax=ax, fraction=.035, pad=.04, label='Mean adjacent class-accuracy change (pp)')
        fig.text(.5, .025, 'Shared training prefixes are counted once. n = distinct observed training prefixes.\nCells describe observed histories; they do not isolate a causal effect of task identity.', ha='center', fontsize=11)
        fig.tight_layout(rect=(0, .07, 1, .98), pad=2)
        fig.savefig(out / 'later_task_class_changes.png', dpi=300)
        pdf.savefig(fig)
        plt.close(fig)
    lines = ['# 按类别分析已完成的无 text_mix 顺序', '',
             '数据来源：上一级目录的九个已完成顺序、相同测试集和保存预测。',
             '最终遗忘只对旧任务统计，排除最后一个任务自比得到的零。负值表示准确率下降，单位为百分点。',
             '各类别跨顺序等权；相邻阶段分析对相同训练前缀去重，避免重复计入相同训练路径。', '',
             '|任务|选项|类别|旧任务顺序数|刚训练平均正确率|最终平均正确率|平均变化|下降顺序数|最常见错误选项|',
             '|---|---|---|---:|---:|---:|---:|---:|---|']
    for row in summary:
        lines.append('|'+ '|'.join([row['task'], row['option'], row['class_name'], str(row['old_task_orders']),
                                    number(row['mean_just_trained_accuracy_pct']), number(row['mean_final_accuracy_pct']),
                                    number(row['mean_forgetting_pp'], signed=True), str(row['orders_with_decrease']),
                                    row['most_common_wrong_option']]) + '|')
    lines += ['', '## 后续训练任务对应的主要类别变化', '',
              '|新训练任务|旧任务|类别|不同训练前缀数|平均相邻变化|最差相邻变化|',
              '|---|---|---|---:|---:|---:|']
    for row in transitions[:20]:
        lines.append('|'+ '|'.join([row['new_task'], row['old_task'], row['option']+' '+row['class_name'],
                                    str(row['unique_training_prefixes']), number(row['mean_adjacent_change_pp'], signed=True),
                                    number(row['min_adjacent_change_pp'], signed=True)])+'|')
    lines += ['', '## 解释边界', '',
              '九个顺序并非全部二十四种排列，各任务所在位置及后续任务组合不均衡，且只使用 seed 1。',
              '跨顺序均值和相邻变化是描述性结果，不能直接断言某一任务本身造成固定幅度的损害。',
              '正确率低需结合刚训练后的水平判断：原本就较难学的类别，与训练后明显退化的类别应区分。',
              '少数选项的召回率升高还应结合 precision 和预测分布观察，可能伴随其他类别被吸收到该选项。',
              '本次仅分析测试集分类表现，不构成对训练样本逐条记忆或知识被永久抹除的证明。', '']
    (out / 'class_analysis.md').write_text('\n'.join(lines))
    assert len(summary) == 33
    assert len(old_pairs) == len({row['display_id'] for row in final}) * 3
    result = {'complete': True, 'source_orders': len({row['display_id'] for row in final}), 'classes': len(summary),
              'old_task_order_pairs': len(old_pairs), 'distinct_class_prefix_observations': len(unique),
              'tasks_trained_last_excluded': True, 'shared_prefixes_deduplicated': True,
              'max_within_prefix_range_pp': max(row['within_prefix_range_pp'] for row in unique),
              'source_statistics_dir': str(source)}
    (out / 'analysis_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({**result, 'output_dir': str(out)}, indent=2))


if __name__ == '__main__':
    main()

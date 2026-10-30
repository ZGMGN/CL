"""Run balanced class diagnostics, then relate class drift to retained performance."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import runpy

ROOT = Path(__file__).resolve().parents[1]


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--analysis_only', action='store_true', help='Build reports from completed class metrics without model inference')
    args = parser.parse_args()
    scope = args.config.resolve().parent
    if not args.analysis_only:
        subprocess.run([sys.executable, '-u', str(ROOT / 'src/evaluation/sequential_classwise_eval.py'), '--config', str(args.config.resolve())], check=True)
    if not json.loads((scope / 'status.json').read_text()).get('complete'):
        raise ValueError('Classwise inference must be complete before building the aggregate report')
    helpers = runpy.run_path(str(ROOT / 'scripts/plot_completed_qa_order_maps.py'), run_name='class_plot_helpers')
    with (scope / 'final_old_class_metrics.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['evaluated_task'], row['option'], row['class_name'], row['pooling'], int(row['layer']))].append(row)
    metrics = ['cka', 'rep_drift', 'kl_reference_to_current', 'kl_current_to_reference', 'symmetric_kl', 'test_forgetting_pp']
    summaries = []
    for (task, label, name, pooling, layer), values in groups.items():
        assert len({row['task_order'] for row in values}) == len(values)
        item = {'evaluated_task': task, 'option': label, 'class_name': name, 'pooling': pooling,
                'layer': layer, 'old_task_orders': len(values), 'probes_per_class_per_order': 64}
        for key in metrics:
            item['mean_' + key] = float(np.mean([float(row[key]) for row in values]))
        summaries.append(item)
    assert len(summaries) == 33 * 4
    save_csv(scope / 'class_drift_retention_summary.csv', summaries)
    correlations = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42})
    colors = dict(zip(helpers['COMMON_TASKS'], ['#3775BA', '#B64342', '#42949E', '#9A4D8E']))
    settings = [(pooling, layer) for pooling in ('last', 'mean') for layer in (16, 32)]
    with PdfPages(scope / 'all_orders_classwise_analysis.pdf') as pdf:
        for metric, title in [('cka', 'Linear CKA'), ('rep_drift', 'Representation drift'), ('symmetric_kl', 'Symmetric KL (nats)')]:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            for ax, (pooling, layer) in zip(axes.flat, settings):
                values = [row for row in rows if row['pooling'] == pooling and int(row['layer']) == layer]
                x = np.asarray([float(row[metric]) for row in values])
                y = np.asarray([float(row['test_forgetting_pp']) for row in values])
                coefficient = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None
                correlations.append({'pooling': pooling, 'layer': layer, 'metric': metric,
                                     'class_order_observations': len(values), 'pearson_r': coefficient})
                for task in helpers['COMMON_TASKS']:
                    selected = [i for i, row in enumerate(values) if row['evaluated_task'] == task]
                    ax.scatter(x[selected], y[selected], color=colors[task], alpha=.6, s=28, label=task)
                ax.axhline(0, color='#777777', linewidth=1)
                ax.set_xlabel(title)
                ax.set_ylabel('Final minus just-trained class accuracy (pp)')
                coefficient_label = 'N/A' if coefficient is None else f'{coefficient:.3f}'
                ax.set_title(f'{pooling.upper()} · layer {layer} · descriptive r={coefficient_label}')
                ax.legend(frameon=False, fontsize=10)
            fig.suptitle(title + ' versus class performance retention', fontsize=21)
            fig.text(.5, .025, 'Each point is one old class in one order (M4 versus its just-trained model).\nShared classes/orders are not independent replicates; correlations are descriptive, not causal evidence.', ha='center', fontsize=11)
            fig.tight_layout(rect=(0, .085, 1, .94), pad=2)
            fig.savefig(scope / f'class_retention_vs_{metric}.png', dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
        displayed = [('mean_cka', 'CKA', 0, 1), ('mean_rep_drift', 'Rep drift', 0, 2),
                     ('mean_kl_reference_to_current', 'KL(ref || M4)', 0, None),
                     ('mean_kl_current_to_reference', 'KL(M4 || ref)', 0, None),
                     ('mean_test_forgetting_pp', 'Test change (pp)', None, None)]
        for pooling, layer in settings:
            values = [row for row in summaries if row['pooling'] == pooling and row['layer'] == layer]
            values.sort(key=lambda row: (helpers['COMMON_TASKS'].index(row['evaluated_task']), row['option']))
            labels = [f"{row['evaluated_task']} {row['option']} · {row['class_name']} (n={row['old_task_orders']})" for row in values]
            fig, axes = plt.subplots(1, len(displayed), figsize=(24, 19), sharey=True)
            for ax, (key, title, low, high) in zip(axes, displayed):
                image = helpers['grid'](ax, [[row[key]] for row in values], [title], labels,
                                         title, forgetting=key == 'mean_test_forgetting_pp', limit=100, fontsize=10)
                if key != 'mean_test_forgetting_pp':
                    upper = high if high is not None else max(1e-12, max(row[key] for row in values))
                    image.set_clim(low, upper)
                for text, row in zip(ax.texts, values):
                    rgba = image.cmap(image.norm(row[key]))
                    luminance = .299 * rgba[0] + .587 * rgba[1] + .114 * rgba[2]
                    text.set_color('white' if luminance < .5 else '#162033')
                fig.colorbar(image, ax=ax, fraction=.045, pad=.04)
            fig.suptitle(f'Class-conditioned means across completed orders · {pooling.upper()} · layer {layer}', fontsize=20)
            fig.text(.5, .025, 'Only tasks followed by later training are included; n is the number of included orders for that class.\nValidation probes: 64 per class, shared across models/orders. Performance change: full held-out test sets.', ha='center', fontsize=11)
            fig.tight_layout(rect=(0, .08, 1, .94), pad=2, w_pad=2)
            pdf.savefig(fig)
            plt.close(fig)
    save_csv(scope / 'metric_retention_associations.csv', correlations)
    lines = ['# 类别级 CKA、表征漂移与输出变化分析', '',
             '所有结果使用每类固定的 64 条验证样本。比较最终 M4 与该任务刚训练后的模型；最后任务的自比较不纳入均值。',
             'CKA、表征漂移、KL 来自相同验证输入，类别正确率变化来自完整测试集。',
             '每个 order 目录另含全部模型对、相邻阶段和任务刚训练模型作为参考的完整 CSV。',
             'CKA 衡量类内样本之间的表征几何关系，不能单独判定该类别是否遗忘；需要与类别表现变化一起看。',
             '以下均值为当前顺序集合的描述，n 并非独立随机种子的重复数。', '']
    for pooling, layer in settings:
        lines += [f'## {pooling} pooling，第 {layer} 层', '',
                  '|任务|类别|order数|平均CKA|平均表征漂移|KL(ref→M4)|KL(M4→ref)|平均类别表现变化(pp)|',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        values = sorted([row for row in summaries if row['pooling'] == pooling and row['layer'] == layer], key=lambda row: row['mean_test_forgetting_pp'])
        for row in values:
            lines.append('|'+ '|'.join([row['evaluated_task'], row['option']+' '+row['class_name'], str(row['old_task_orders']),
                                        f"{row['mean_cka']:.4f}", f"{row['mean_rep_drift']:.4f}",
                                        f"{row['mean_kl_reference_to_current']:.4f}", f"{row['mean_kl_current_to_reference']:.4f}",
                                        f"{row['mean_test_forgetting_pp']:.2f}"])+ '|')
        lines.append('')
    (scope / 'classwise_analysis.md').write_text('\n'.join(lines))
    status = json.loads((scope / 'status.json').read_text())
    status.update(analysis_report_complete=True, class_summary_rows=len(summaries), analysis_report=str(scope / 'all_orders_classwise_analysis.pdf'))
    temporary = scope / 'status.tmp'
    temporary.write_text(json.dumps(status, indent=2)+'\n')
    temporary.replace(scope / 'status.json')
    print('Completed class-conditioned metrics, plots and retention analysis.', flush=True)


if __name__ == '__main__':
    main()

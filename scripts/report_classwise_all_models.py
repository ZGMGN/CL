"""Report every base/adapter pair separately for each true-label class; CPU only."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import textwrap
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import numpy as np

SETTINGS = [('last', 16), ('last', 32), ('mean', 16), ('mean', 32)]
PAIRS = list(itertools.combinations(range(5), 2))


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heatmap(ax, values, xlabels, ylabels, title, *, kind, high=None, fontsize=10, show_y=True):
    values = np.asarray(values, dtype=float)
    if kind == 'change':
        bound = high or max(1., float(np.nanmax(np.abs(values))))
        image = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap='RdBu',
                          norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    else:
        upper = {'cka': 1., 'rep': 2., 'accuracy': 100.}.get(kind, high)
        image = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap='YlGnBu', vmin=0, vmax=upper)
    image.cmap.set_bad('#e9edf2')
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=fontsize)
    ax.set_yticks(range(len(ylabels)), ylabels if show_y else [''] * len(ylabels), fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 3, pad=12)
    ax.set_xticks(np.arange(-.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.8)
    ax.tick_params(which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, j in np.ndindex(values.shape):
        value = values[i, j]
        if np.isnan(value):
            label, color = 'pre', '#707984'
        else:
            label = f'{value:+.1f}' if kind == 'change' else (f'{value:.1f}' if kind == 'accuracy' else f'{value:.3f}')
            rgba = image.cmap(image.norm(value))
            luminance = .299 * rgba[0] + .587 * rgba[1] + .114 * rgba[2]
            color = 'white' if luminance < .5 else '#162033'
        ax.text(j, i, label, ha='center', va='center', fontsize=fontsize, color=color)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    if not data.get('complete'):
        raise ValueError('Completed classwise model-pair metrics are required')
    tasks = data['tasks']
    assert len(tasks) == 4 and len(set(tasks)) == 4
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'axes.spines.top': False, 'axes.spines.right': False})
    mapping = ['M0 = base'] + [f'M{i} = after {task}' for i, task in enumerate(tasks, 1)]
    mapping_line = '    |    '.join(mapping)
    order_line = 'Training order: ' + ' → '.join(tasks)
    pair_lookup, reference_lookup, labels, class_names = {}, {}, {}, {}
    for row in data['pair_results']:
        task, option = row['evaluated_task'], row['option']
        a, b = int(row['model_a'][1:]), int(row['model_b'][1:])
        key = (task, option, row['pooling'], row['layer'], a, b)
        assert key not in pair_lookup and a < b and row['probe_samples'] == 64
        pair_lookup[key] = row
        class_names[(task, option)] = row['class_name']
    for task in tasks:
        labels[task] = sorted({label for name, label in class_names if name == task})
    assert sum(map(len, labels.values())) == 33 and len(pair_lookup) == 1320
    for row in data['task_learned_reference_results']:
        key = (row['evaluated_task'], row['option'], row['pooling'], row['layer'], int(row['current_model'][1:]))
        assert key not in reference_lookup
        reference_lookup[key] = row
    assert len(reference_lookup) == 660

    def metric(task, option, pool, layer, a, b, key):
        if a == b:
            return 1. if key == 'cka' else 0.
        first, second = sorted((a, b))
        row = pair_lookup[(task, option, pool, layer, first, second)]
        if key == 'kl':
            return row['kl_a_to_b'] if a < b else row['kl_b_to_a']
        return row[key]

    def matrix(task, option, pool, layer, key):
        return [[metric(task, option, pool, layer, a, b, key) for b in range(5)] for a in range(5)]

    all_rows = []
    accuracy_rows = []
    for task in tasks:
        for option in labels[task]:
            learned = tasks.index(task) + 1
            for stage in range(5):
                row = reference_lookup[(task, option, 'last', 16, stage)]
                for pool, layer in SETTINGS:
                    other = reference_lookup[(task, option, pool, layer, stage)]
                    assert other['test_class_accuracy_pct'] == row['test_class_accuracy_pct']
                accuracy_rows.append({'task': task, 'option': option, 'class_name': class_names[task, option],
                                     'model': f'M{stage}', 'model_after': 'BASE' if stage == 0 else tasks[stage - 1],
                                     'test_class_accuracy_pct': row['test_class_accuracy_pct'],
                                     'task_learned_model': f'M{learned}', 'signed_forgetting_pp': row['test_forgetting_pp'],
                                     'phase': row['comparison_phase']})
            for pool, layer in SETTINGS:
                for a in range(5):
                    for b in range(5):
                        current = reference_lookup[(task, option, pool, layer, b)]
                        reference_accuracy = reference_lookup[(task, option, pool, layer, a)]['test_class_accuracy_pct']
                        all_rows.append({'task': task, 'option': option, 'class_name': class_names[task, option],
                                         'samples_per_class': 64, 'pooling': pool, 'layer': layer,
                                         'row_model': f'M{a}', 'row_model_after': 'BASE' if a == 0 else tasks[a - 1],
                                         'column_model': f'M{b}', 'column_model_after': 'BASE' if b == 0 else tasks[b - 1],
                                         'cka': metric(task, option, pool, layer, a, b, 'cka'),
                                         'representation_drift': metric(task, option, pool, layer, a, b, 'rep_drift'),
                                         'kl_row_to_column_nats': metric(task, option, pool, layer, a, b, 'kl'),
                                         'kl_column_to_row_nats': metric(task, option, pool, layer, b, a, 'kl'),
                                         'symmetric_kl_nats': metric(task, option, pool, layer, a, b, 'symmetric_kl'),
                                         'row_test_class_accuracy_pct': reference_accuracy,
                                         'column_test_class_accuracy_pct': current['test_class_accuracy_pct'],
                                         'test_change_column_minus_row_pp': current['test_class_accuracy_pct'] - reference_accuracy,
                                         'column_signed_forgetting_pp': current['test_forgetting_pp']})
    assert len(all_rows) == 3300 and len(accuracy_rows) == 165
    save_csv(output / 'all_model_class_matrices.csv', all_rows)
    save_csv(output / 'class_accuracy_all_models.csv', accuracy_rows)
    # These two views retain all controls, including models before the task was learned.
    save_csv(output / 'base_reference_class_metrics.csv', [row for row in all_rows if row['row_model'] == 'M0'])
    save_csv(output / 'task_learned_reference_class_metrics.csv',
             [row for row in all_rows if row['row_model'] == f"M{tasks.index(row['task']) + 1}"])
    kl_high = max(row['kl_row_to_column_nats'] for row in all_rows)
    sym_high = max(row['symmetric_kl_nats'] for row in all_rows)
    page_count = 0
    appendix_count = 0

    def export(fig, pdf, stem, *, appendix=False):
        nonlocal page_count, appendix_count
        if appendix:
            appendix_count += 1
            number = appendix_count
        else:
            page_count += 1
            number = page_count
        fig.text(.98, .012, f'{number}', ha='right', fontsize=10, color='#707984')
        pdf.savefig(fig)
        fig.savefig(figures / f'{stem}.png', dpi=300)
        plt.close(fig)

    def header(fig, title, subtitle=''):
        fig.suptitle(title, fontsize=20, y=.985)
        fig.text(.5, .945, mapping_line, ha='center', fontsize=11)
        if subtitle:
            fig.text(.5, .916, subtitle, ha='center', fontsize=11)

    pair_ticks = [f'M{a} vs M{b}' for a, b in PAIRS]
    with PdfPages(output / 'all_models_by_class_report.pdf') as pdf:
        sections = [
            ('All models, all tasks, all labels', [
                order_line, '', *mapping, '',
                'Scope: AGNEWS 4 labels; YAHOO 10; AMAZON 5; DBPEDIA 14 (33 labels total).',
                'Every class uses 64 fixed validation inputs, identically ordered for all five models.',
                'All ten distinct model pairs are shown; the first four columns compare with base M0.',
                'The remaining six columns compare intermediate and final cumulative adapters.',
                'No training or hidden-state collection was repeated to produce this report.', '',
                'Pages 3–6: full-test class accuracy for M0–M4 and signed forgetting.',
                'Pages 7–10: class-conditioned forward KL, reverse KL and symmetric KL for all pairs.',
                'Pages 11–18: LAST pooling, layers 16 then 32, all four tasks.',
                'Pages 19–26: MEAN pooling, layers 16 then 32, all four tasks.',
                'Companion PDF: one page per label, with full 5 x 5 model matrices.',
                'Companion CSV: 3300 rows = 33 labels x 4 extraction settings x 25 ordered model pairs.',
            ]),
            ('Calculation and interpretation', [
                '1. Inputs: validation samples are grouped by their true label, not by model prediction.',
                '   Each model sees the same prompt-only text, left-truncated to at most 1536 tokens.',
                '   Evaluation mode disables dropout. No answer text is included in the probe input.', '',
                '2. Representation: hidden_states[16] and hidden_states[32] are collected.',
                '   Index 16 follows transformer block 16. Index 32 is the final model representation',
                '   returned by this Llama implementation, including its final normalization.',
                '   LAST uses the last non-padding prompt token. MEAN averages all non-padding tokens.',
                '   Each class therefore yields a 64 by hidden-size matrix per model and setting.', '',
                '3. Linear CKA: feature columns are centered across these 64 examples; sample-similarity',
                '   matrices are formed, centered and compared with a normalized Frobenius inner product.',
                '   Larger CKA means more similar within-class geometry. It does not measure class separation.',
                '   Representation drift averages one minus cosine similarity over the 64 aligned examples.', '',
                '4. Functional drift: softmax over the complete vocabulary at the first answer position.',
                '   KL(A || B) weights log-probability changes by A; KL(B || A) weights them by B.',
                '   Compute per input and then average within the label; symmetric KL averages both directions.',
                '   KL is in nats and does not depend on representation layer or pooling.', '',
                '5. Performance: each label uses its full held-out test subset, rather than the 64 probes.',
                '   Signed forgetting is current accuracy minus accuracy immediately after that task was learned.',
                '   Negative values mean a decline; models before task training are marked pre, not forgetting.',
                '   Diagonal self-comparisons: CKA 1; representation drift and KL 0.',
                '   Drift alone does not establish that training examples or knowledge were erased.',
            ]),
        ]
        for index, (title, lines) in enumerate(sections):
            fig, ax = plt.subplots(figsize=(17, 12))
            ax.axis('off')
            fig.suptitle(title, fontsize=23, y=.96)
            ax.text(.015, .95, '\n'.join(lines), va='top', fontsize=13, linespacing=1.5, transform=ax.transAxes)
            fig.subplots_adjust(left=.035, right=.98, top=.91, bottom=.04)
            export(fig, pdf, f'{index + 1:02d}_guide')
        for task in tasks:
            ylabels = [f'{label} · {class_names[task, label]}' for label in labels[task]]
            selected = [[reference_lookup[(task, label, 'last', 16, stage)] for stage in range(5)] for label in labels[task]]
            acc = [[row['test_class_accuracy_pct'] for row in group] for group in selected]
            change = [[np.nan if row['test_forgetting_pp'] is None else row['test_forgetting_pp'] for row in group] for group in selected]
            fig, axes = plt.subplots(1, 2, figsize=(18, max(7.5, len(ylabels) * .45 + 3.5)))
            header(fig, f'{task}: each true label on all five models', 'Full held-out TEST subsets; pre = model before that task was trained')
            for ax, values, title, kind in zip(axes, [acc, change], ['Class accuracy (%)', f'Signed forgetting vs M{tasks.index(task)+1} (pp)'], ['accuracy', 'change']):
                im = heatmap(ax, values, [f'M{i}' for i in range(5)], ylabels, title, kind=kind, fontsize=11)
                fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
            fig.tight_layout(rect=(0, .05, 1, .87), pad=2, w_pad=3)
            export(fig, pdf, f'accuracy_{task}')
        for task in tasks:
            ylabels = [f'{label} · {class_names[task, label]}' for label in labels[task]]
            fig, axes = plt.subplots(3, 1, figsize=(19, max(12, len(ylabels) * .9 + 5)))
            header(fig, f'{task}: output-distribution change for every model pair', 'Column Mx vs My: top KL(Mx || My), middle KL(My || Mx), bottom their average')
            for ax, key, title in zip(axes, ['kl_a_to_b', 'kl_b_to_a', 'symmetric_kl'],
                                       ['KL(earlier model || later model), nats', 'KL(later model || earlier model), nats', 'Symmetric KL, nats']):
                values = [[pair_lookup[(task, label, 'last', 16, a, b)][key] for a, b in PAIRS] for label in labels[task]]
                im = heatmap(ax, values, pair_ticks, ylabels, title, kind='kl', high=sym_high if key == 'symmetric_kl' else kl_high, fontsize=10)
                ax.axvline(3.5, color='#B64342', linewidth=2)
                fig.colorbar(im, ax=ax, fraction=.018, pad=.02)
            fig.text(.5, .027, 'First 4 columns: base comparisons. Remaining 6: adapter comparisons. Same 64 validation prompts per label; full vocabulary.', ha='center', fontsize=11)
            fig.tight_layout(rect=(0, .05, 1, .88), pad=2, h_pad=3)
            export(fig, pdf, f'functional_{task}')
        for pool, layer in SETTINGS:
            for task in tasks:
                ylabels = [f'{label} · {class_names[task, label]}' for label in labels[task]]
                fig, axes = plt.subplots(2, 1, figsize=(19, max(10, len(ylabels) * .72 + 4)))
                header(fig, f'{task}: {pool.upper()} pooling, hidden-state index {layer}', 'Each row is one true-label class; each column is an explicitly named model pair')
                for ax, key, title, kind in zip(axes, ['cka', 'rep_drift'], ['Linear CKA (higher = more similar)', 'Representation drift (higher = more changed)'], ['cka', 'rep']):
                    values = [[pair_lookup[(task, label, pool, layer, a, b)][key] for a, b in PAIRS] for label in labels[task]]
                    im = heatmap(ax, values, pair_ticks, ylabels, title, kind=kind, fontsize=11)
                    ax.axvline(3.5, color='#B64342', linewidth=2)
                    fig.colorbar(im, ax=ax, fraction=.022, pad=.02)
                fig.text(.5, .027, 'First 4 columns: M0 base vs each adapter. Remaining 6: all adapter pairs. 64 identical validation examples per label.', ha='center', fontsize=11)
                fig.tight_layout(rect=(0, .05, 1, .88), pad=2, h_pad=3)
                export(fig, pdf, f'representation_{pool}_{layer}_{task}')
    assert page_count == 26
    print('Main report complete: 26 pages.', flush=True)

    model_ticks = [f'M{i}' for i in range(5)]
    with PdfPages(output / 'per_label_5x5_model_matrices.pdf') as pdf:
        for task in tasks:
            for option in labels[task]:
                fig, axes = plt.subplots(3, 4, figsize=(22, 16))
                header(fig, f'{task} · {option} {class_names[task, option]}: all five models', 'Matrix ROW = reference model; COLUMN = comparison model. KL panel is KL(row || column).')
                for column, (pool, layer) in enumerate(SETTINGS):
                    for row_index, key, name, kind in [(0, 'cka', 'CKA', 'cka'), (1, 'rep_drift', 'Representation drift', 'rep')]:
                        ax = axes[row_index, column]
                        im = heatmap(ax, matrix(task, option, pool, layer, key), model_ticks, model_ticks,
                                     f'{name}\n{pool.upper()} · layer {layer}', kind=kind, fontsize=9)
                        ax.set_xlabel('Comparison model', fontsize=10)
                        ax.set_ylabel('Reference model', fontsize=10)
                        fig.colorbar(im, ax=ax, fraction=.04, pad=.03)
                for ax, key, name, high in zip(axes[2, :2], ['kl', 'symmetric_kl'], ['Directional KL(row || column), nats', 'Symmetric KL, nats'], [kl_high, sym_high]):
                    im = heatmap(ax, matrix(task, option, 'last', 16, key), model_ticks, model_ticks, name, kind='kl', high=high, fontsize=9)
                    ax.set_xlabel('Comparison model', fontsize=10)
                    ax.set_ylabel('Reference model', fontsize=10)
                    fig.colorbar(im, ax=ax, fraction=.04, pad=.03)
                rows = [reference_lookup[(task, option, 'last', 16, stage)] for stage in range(5)]
                for ax, key, title in [(axes[2, 2], 'test_class_accuracy_pct', 'Full-test class accuracy (%)'), (axes[2, 3], 'test_forgetting_pp', 'Signed forgetting vs just-trained (pp)')]:
                    values = [np.nan if row[key] is None else row[key] for row in rows]
                    colors = ['#3775BA' if key == 'test_class_accuracy_pct' or value >= 0 else '#B64342' for value in values]
                    ax.bar(range(5), values, color=colors, width=.65)
                    for stage, value in enumerate(values):
                        if np.isfinite(value):
                            ax.annotate(f'{value:.1f}', (stage, value), xytext=(0, 5 if value >= 0 else -13), textcoords='offset points', ha='center', fontsize=10)
                        else:
                            ax.text(stage, 0, 'pre', ha='center', va='bottom', color='#707984', fontsize=10)
                    ax.axhline(0, color='#707984', linewidth=.8)
                    ax.set_xticks(range(5), model_ticks)
                    ax.set_title(title, fontsize=12, pad=12)
                    ax.set_xlabel('Model')
                    if key == 'test_class_accuracy_pct':
                        ax.set_ylim(0, 110)
                    else:
                        ax.margins(y=.25)
                fig.text(.5, .022, 'Hidden-state / KL metrics: 64 aligned validation inputs. Performance: full test subset. Self CKA = 1; self drift = 0. KL does not depend on pooling/layer.', ha='center', fontsize=11)
                fig.tight_layout(rect=(0, .05, 1, .885), pad=2, w_pad=2.5, h_pad=3)
                export(fig, pdf, f'label_{task}_{option}_5x5', appendix=True)
    assert appendix_count == 33
    provenance = {'complete': True, 'tasks': tasks, 'model_mapping': mapping, 'source': str(args.source.resolve()),
                  'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  'probe_manifest': data['probe_manifest'], 'accuracy_source': data['accuracy_source'], 'settings': data['settings'],
                  'classes': {task: {label: class_names[task, label] for label in labels[task]} for task in tasks},
                  'main_report_pages': page_count, 'per_label_report_pages': appendix_count,
                  'matrix_csv_rows': len(all_rows), 'class_accuracy_rows': len(accuracy_rows),
                  'all_model_pairs_present': True, 'new_model_inference': False}
    (output / 'validation.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    methods = '''# AGNEWS → YAHOO → AMAZON → DBPEDIA：全部模型的逐类别分析

模型对应关系：M0 是 base；M1 是训练完 AGNEWS；M2 是继续训练完 YAHOO；M3 是继续训练完 AMAZON；M4 是继续训练完 DBPEDIA。每个 adapter 都是该阶段的累计训练结果。

- `all_models_by_class_report.pdf`：26 页。先给模型说明和计算过程，再依次展示四任务的所有类别在 M0–M4 的测试正确率、相对该任务刚训练后模型的表现变化、全部 10 个模型对的双向 KL 和 symmetric KL、last pooling 的第 16/32 层 CKA 与表征漂移，最后展示 mean pooling。每张模型对热图的前 4 列是 base 比较，后 6 列是 adapter 之间的全部比较。
- `per_label_5x5_model_matrices.pdf`：33 页，每个真实 label 单独一页。行与列均包含 M0–M4，直接查看 base、中间 adapter 和最终 adapter 两两之间的指标。每页包含 4 种设置的 CKA、4 种设置的表征漂移、完整双向 KL 矩阵、symmetric KL、类别正确率和有符号 forgetting。KL 矩阵的行是第一个分布，列是第二个分布，转置位置即相反方向。
- `all_model_class_matrices.csv`：3300 行，保留所有有向模型对及自身比较，并给出两个方向的 KL。`base_reference_class_metrics.csv` 只保留以 M0 为参考的比较；`task_learned_reference_class_metrics.csv` 以该任务刚训练后的模型为参考，覆盖包括 base 在内的全部 5 个模型。
- `class_accuracy_all_models.csv`：165 行，即 33 个类别乘 5 个模型。正确率是按真实类别条件计算的召回率，使用完整测试集。
- `figures/`：所有报告页的 300 dpi PNG，可直接单独使用。

## 输入和表征如何获得

四个任务的 33 个真实类别各固定抽取 64 条验证样本，同一类别在全部模型之间使用完全相同且顺序一致的输入。抽样 seed 为 1，样本索引保存在 probe_manifest 中。本报告复用已完成的隐藏状态评估，不重新推理，也不更新模型。

输入只有任务提示和原文本，不包含正确答案，最多保留 1536 个 token。使用左侧补齐和截断。关闭 dropout，以推理模式收集表征。第 16 层使用 hidden_states[16]；第 32 层使用 hidden_states[32]，后者在当前 Llama 实现中包含最终归一化。

last pooling 取每条输入最后一个有效 token 的隐藏状态；mean pooling 将非补齐 token 的隐藏状态平均，补齐位置由 attention mask 排除。因此每个类别、每个模型、每种设置都得到一张 64 行、hidden_size 列的矩阵。

## 指标的计算过程

**CKA**：先在类别的 64 个样本上对每个表征特征中心化，再形成样本之间的线性相似度矩阵并中心化。将两模型的相似度矩阵进行归一化内积，得到线性 CKA。越接近 1，表示类内样本的相对表征几何关系越相似。这不是类别间分离程度，也不能单独说明该类别是否遗忘。对类别 CKA 求平均也不等同于对整个任务一次计算 CKA。

**Representation drift（表征漂移）**：对每一条对应样本计算两模型表征的余弦相似度，用 1 减去相似度，再对同一类别的 64 条样本求平均。越大说明对应样本的表征方向变化越大，范围为 0 到 2。它与 CKA 衡量的性质不同。

**双向 KL**：对相同输入，在第一个答案 token 位置，经模型最终输出头得到全词表概率分布。每条样本分别计算两个方向的 KL，再按真实 label 求平均。KL(Ma‖Mb) 由 Ma 的概率加权，KL(Mb‖Ma) 由 Mb 的概率加权，两者一般不相等。symmetric KL 是两个方向的均值，单位均为 nats。KL 使用完整词表，并不是只使用类别选项。它只来自答案位置的最终输出分布，不随表征层号和 pooling 改变，因此不重复绘制四份。

**类别正确率与 forgetting**：类别正确率使用完整测试集中真实 label 相同的所有样本，分母是该 label 的测试样本数。当前类别正确率减去同一任务刚训练后模型的类别正确率，得到有符号 forgetting，单位是百分点。负值表示下降；任务尚未训练的阶段标为 pre，不称为遗忘。类别正确率与表征指标来自不同的留出集。

所有矩阵的自比较位置定义为 CKA 等于 1、表征漂移和 KL 等于 0。基于 drift 的解释应结合测试表现一起看，不把表征改变直接等同于训练数据被删除。
'''
    (output / 'README.md').write_text(methods, encoding='utf-8')
    with zipfile.ZipFile(output / 'all_models_classwise_reports.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in ('.pdf', '.csv', '.md', '.json'):
                archive.write(path, path.name)
    print(f'Complete: {output}; main report 26 pages, label appendix 33 pages, 3300 matrix rows.', flush=True)


if __name__ == '__main__':
    main()

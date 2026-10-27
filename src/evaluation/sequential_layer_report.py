"""Derive layer-16 reports from saved layer metrics; no model inference."""
import csv
from pathlib import Path

import numpy as np


def derive_layer_results(payload, layer_rows, diagonal_fallback=None):
    if 16 not in payload['resolved_layers']:
        return {}, []
    selected = list(dict.fromkeys([16, payload['summary_layer']]))
    tasks, models = payload['tasks'], payload['models']
    size = len(models)
    indices = {model['model']: model['stage'] for model in models}
    matrices = {}
    for layer in selected:
        cka = payload['layer_cka_matrices'][str(layer)]
        if layer == payload['summary_layer']:
            representation = payload['pairwise_matrices']['rep_drift']
        else:
            representation = {task: [[None] * size for _ in models] for task in tasks}
            seen = set()
            for row in layer_rows:
                if int(row['layer']) != layer:
                    continue
                task = row['evaluated_task']
                first, second = indices[row['model_a']], indices[row['model_b']]
                key = (task, min(first, second), max(first, second))
                if first == second or key in seen:
                    raise ValueError('Expected distinct, unique model pairs in layer CSV')
                seen.add(key)
                value = row['rep_drift']
                value = None if value is None or value == '' else float(value)
                representation[task][first][second] = value
                representation[task][second][first] = value
            expected = {(task, first, second) for task in tasks for first in range(size) for second in range(first + 1, size)}
            if seen != expected:
                raise ValueError(f'Layer {layer}: layer CSV is missing model-pair/task results')
            for task in tasks:
                for model in models:
                    stage = model['stage']
                    # Defined self-CKA proves nonzero features, hence self-distance = 0.
                    if cka[task][stage][stage] is not None:
                        representation[task][stage][stage] = 0.0
                    elif diagonal_fallback is not None:
                        representation[task][stage][stage] = diagonal_fallback(stage, task, layer)
                    else:
                        raise ValueError('Undefined self-CKA: cached features are required to determine self-distance')
        matrices[str(layer)] = {'cka': cka, 'rep_drift': representation}
    counts = {row['evaluated_task']: row['validation_probe_count'] for row in payload['model_task_results']}
    results = []
    for layer in selected:
        for index, task in enumerate(tasks):
            for reference_kind, reference_stage in [('base', 0), ('task_learned', index + 1)]:
                reference = models[reference_stage]
                for model in models:
                    stage = model['stage']
                    results.append({
                        'layer': layer, 'evaluated_task': task, 'model': model['model'],
                        'adapter_dir': model['adapter_dir'], 'reference_kind': reference_kind,
                        'reference_model': reference['model'], 'reference_adapter_dir': reference['adapter_dir'],
                        'validation_probe_count': counts[task], 'pooling': payload['settings']['pooling'],
                        'cka': matrices[str(layer)]['cka'][task][reference_stage][stage],
                        'representation_drift': matrices[str(layer)]['rep_drift'][task][reference_stage][stage],
                        'representation_drift_measure': 'Mean per-example cosine distance',
                    })
    return matrices, results


def layer_analysis_markdown(payload):
    if not payload.get('selected_layer_results'):
        return ''
    layers = payload['selected_representation_layers']
    final = payload['models'][-1]['model']
    summary = payload['summary_layer']
    lines = [f"## {payload['settings']['pooling'].title()} pooling：第 16 层表征与 CKA 分析", '',
             f'分析层为 hidden_states[16]；另与汇总层 hidden_states[{summary}] 对照。'
             '各层独立计算，不将第 16 层与第 32 层的向量直接互算 CKA。', '',
             'Representation drift（表征漂移）的具体度量仍为逐样本余弦距离的均值。'
             '所有模型使用同一任务相同的固定验证样本与 pooling。准确率与 KL 来自模型输出，不单独定义一个“第 16 层准确率或 KL”。', '',
             f'下表固定被比较模型为最终 {final}，分别以 base 和该任务刚训练后的模型为参照。', '',
             '| 任务 | 参照 | 层 | CKA | Representation drift |', '| --- | --- | --- | --- | --- |']
    def number(value):
        return 'N/A' if value is None else f'{value:.4f}'
    for row in payload['selected_layer_results']:
        if row['model'] == final:
            lines.append(f"| {row['evaluated_task']} | {row['reference_model']} ({row['reference_kind']}) | {row['layer']} | {number(row['cka'])} | {number(row['representation_drift'])} |")
    lines += ['', '### 第 16 层：全部模型与任务', '',
              '| 任务 | 被比较模型 | 参照模型 | CKA | Representation drift |', '| --- | --- | --- | --- | --- |']
    for row in payload['selected_layer_results']:
        if row['layer'] == 16:
            lines.append(f"| {row['evaluated_task']} | {row['model']} | {row['reference_model']} | {number(row['cka'])} | {number(row['representation_drift'])} |")
    lines += ['', '### 两层对比的含义', '']
    if summary != 16:
        lookup = {(row['layer'], row['evaluated_task'], row['reference_kind'], row['model']): row for row in payload['selected_layer_results']}
        for task in payload['tasks']:
            left = lookup[16, task, 'task_learned', final]
            right = lookup[summary, task, 'task_learned', final]
            lines.append(f"- {task}，{final} vs {left['reference_model']}：第 16 层 CKA={number(left['cka'])}、表征漂移={number(left['representation_drift'])}；第 {summary} 层 CKA={number(right['cka'])}、表征漂移={number(right['representation_drift'])}。")
    lines += ['', 'CKA 越高表示样本间几何结构越相似；表征漂移越低表示同一样本的向量方向变化越小。'
              '跨层数值对照描述各层受到的影响，不能仅凭其中一个指标推断因果关系或遗忘。'
              '参照之前的模型仍为训练前对照，之后的模型用于观察后续变化。', '']
    return '\n'.join(lines)


def persist_layer_results(payload, output_dir):
    if 16 not in payload['resolved_layers']:
        return
    with (output_dir / 'all_model_pair_layer_metrics.csv').open() as stream:
        source = list(csv.DictReader(stream))
    def diagonal(stage, task, layer):
        import torch
        from evaluation.drift_metrics import representation_cosine_drift
        cache = output_dir / 'probe_cache' / f'{stage:02d}_{task}.pt'
        feature = torch.load(cache, map_location='cpu', weights_only=True)['features'][layer]
        return representation_cosine_drift(feature, feature)
    matrices, results = derive_layer_results(payload, source, diagonal)
    payload['selected_representation_layers'] = list(map(int, matrices))
    payload['selected_layer_matrices'] = matrices
    payload['selected_layer_results'] = results
    from evaluation.sequential_drift_eval import save_json
    save_json(output_dir / 'selected_layer_metrics.json', {
        'complete': payload['complete'], 'layers': payload['selected_representation_layers'],
        'definition': 'Same-layer, same-task, aligned-example comparisons; base and task-just-learned references are explicit.',
        'matrices': matrices, 'results': results,
    })
    for filename, rows in [('selected_layer_metrics.csv', results), ('layer_16_metrics.csv', [row for row in results if row['layer'] == 16])]:
        with (output_dir / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / 'layer_16_analysis.md').write_text(layer_analysis_markdown(payload), encoding='utf-8')


def plot_layer_pages(payload, grid, focus_layer=16, include_comparison=True):
    if not payload.get('selected_layer_results') or str(focus_layer) not in payload.get('selected_layer_matrices', {}):
        return []
    import matplotlib.pyplot as plt
    tasks, models = payload['tasks'], payload['models']
    labels = [item['label'] for item in models]
    matrices = payload['selected_layer_matrices'][str(focus_layer)]
    pooling = payload['settings']['pooling'].upper()
    pages = []
    for kind in ('base', 'task_learned'):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8.5))
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.34, top=0.80, wspace=0.48)
        title = 'M0 (Base)' if kind == 'base' else 'the task-specific model named in each column'
        fig.suptitle(f'{pooling} pooling | layer {focus_layer} | compare row model with {title}', fontsize=17, y=0.96)
        fig.text(0.5, 0.90, f'Same validation prompts and pooling for both models | hidden_states[{focus_layer}]', ha='center', fontsize=12)
        references = [0 if kind == 'base' else index + 1 for index in range(len(tasks))]
        xlabels = [f"{task}\nref: {models[reference]['label']}" for task, reference in zip(tasks, references)]
        for ax, key, title, upper in [(axes[0], 'cka', 'Linear CKA (higher = more similar)', 1),
                                     (axes[1], 'rep_drift', 'Representation drift (lower = more similar)', None)]:
            values = [[matrices[key][task][reference][model['stage']] for task, reference in zip(tasks, references)] for model in models]
            grid(ax, values, xlabels, labels, title, upper=upper)
        fig.text(0.17, 0.08, 'Representation drift measure: mean per-example cosine distance. CKA uses the complete aligned probe matrix.', fontsize=11)
        fig.text(0.17, 0.045, f'Each cell compares two models at layer {focus_layer} on the named task; models before a task reference are pre-training controls.', fontsize=10)
        pages.append((fig, f'layer_{focus_layer}_vs_{kind}'))
    other = payload['summary_layer']
    if include_comparison and other != focus_layer:
        final = models[-1]
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.80, hspace=0.70, wspace=0.25)
        fig.suptitle(f"{pooling} pooling | layer {focus_layer} vs layer {other} | evaluated model: {final['label']}", fontsize=19, y=0.97)
        fig.text(0.5, 0.92, 'Same model pairs and task probes; metrics are computed independently within each layer.', ha='center', fontsize=12)
        for row, kind in enumerate(('base', 'task_learned')):
            references = [0 if kind == 'base' else index + 1 for index in range(len(tasks))]
            for column, (key, label) in enumerate((('cka', 'Linear CKA'), ('rep_drift', 'Representation drift'))):
                ax = axes[row, column]
                maximum = 0.0
                for offset, (layer, color) in enumerate(((focus_layer, '#2563eb'), (other, '#d97706'))):
                    values = [payload['selected_layer_matrices'][str(layer)][key][task][reference][final['stage']] for task, reference in zip(tasks, references)]
                    values = np.array([np.nan if value is None else value for value in values])
                    positions = np.arange(len(tasks)) + (offset - 0.5) * 0.34
                    ax.bar(positions, values, width=0.31, color=color, label=f'Layer {layer}')
                    for x, value in zip(positions, values):
                        if np.isfinite(value):
                            maximum = max(maximum, value)
                            ax.annotate(f'{value:.3f}', (x, value), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
                        else:
                            ax.text(x, 0.01, 'N/A', ha='center', fontsize=9)
                ax.set_xticks(range(len(tasks)), [f"{task}\nref: {models[reference]['model']}" for task, reference in zip(tasks, references)])
                ax.set_ylim(0, 1.16 if key == 'cka' else max(0.05, maximum * 1.25))
                ax.set_title(label + (' | reference: M0' if kind == 'base' else ' | reference: task just learned'), fontsize=12)
                ax.set_ylabel('Similarity' if key == 'cka' else 'Mean cosine distance')
                ax.grid(axis='y', color='#e2e8f0')
                ax.set_axisbelow(True)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.88), ncol=2, frameon=False)
        fig.text(0.09, 0.055, 'CKA: higher = more similar. Representation drift: lower = more similar. Self-comparisons have CKA 1 and drift 0 when defined.', fontsize=10)
        pages.append((fig, f'layer_{focus_layer}_vs_summary_layer'))
    return pages

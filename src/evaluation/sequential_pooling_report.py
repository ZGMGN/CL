"""Append matched mean-pooling analyses after the last-pooling report section."""
import csv
import json
from pathlib import Path

import numpy as np

from evaluation.sequential_layer_report import plot_layer_pages


def validate_pooling_inputs(last, mean, last_probes, mean_probes):
    if not last.get('complete') or not mean.get('complete'):
        raise ValueError('Both pooling evaluations must be complete')
    if last['settings']['pooling'] != 'last' or mean['settings']['pooling'] != 'mean':
        raise ValueError('Expected last pooling first, mean pooling second')
    for key in ('run_dir', 'tasks', 'models', 'accuracy'):
        if last[key] != mean[key]:
            raise ValueError(f'Pooling comparisons differ for {key}')
    for key in ('max_input_length', 'max_new_tokens', 'seed', 'batch_size'):
        if last['settings'][key] != mean['settings'][key]:
            raise ValueError(f'Pooling comparisons differ for {key}')
    if last_probes != mean_probes:
        raise ValueError('Pooling comparisons must use identical validation probe manifests')
    for payload in (last, mean):
        if not {16, 32}.issubset(payload.get('selected_representation_layers', [])):
            raise ValueError('Both pooling evaluations require layers 16 and 32')


def join_pooling_results(last, mean):
    def identity(row):
        return row['layer'], row['evaluated_task'], row['model'], row['reference_kind'], row['reference_model']
    mean_lookup = {identity(row): row for row in mean['selected_layer_results']}
    last_lookup = {identity(row): row for row in last['selected_layer_results']}
    if len(mean_lookup) != len(mean['selected_layer_results']) or len(last_lookup) != len(last['selected_layer_results']):
        raise ValueError('Duplicate layer/model/task/reference rows')
    if mean_lookup.keys() != last_lookup.keys():
        raise ValueError('Pooling comparisons must cover the same layers/models/tasks/references')
    rows = []
    for key, before in last_lookup.items():
        after = mean_lookup[key]
        if before['validation_probe_count'] != after['validation_probe_count']:
            raise ValueError('Pooling probe counts differ')
        def difference(a, b):
            return None if a is None or b is None else b - a
        rows.append({
            'layer': before['layer'], 'evaluated_task': before['evaluated_task'], 'model': before['model'],
            'adapter_dir': before['adapter_dir'], 'reference_kind': before['reference_kind'],
            'reference_model': before['reference_model'], 'reference_adapter_dir': before['reference_adapter_dir'],
            'validation_probe_count': before['validation_probe_count'],
            'last_cka': before['cka'], 'mean_cka': after['cka'],
            'cka_change_mean_minus_last': difference(before['cka'], after['cka']),
            'last_representation_drift': before['representation_drift'],
            'mean_representation_drift': after['representation_drift'],
            'representation_drift_change_mean_minus_last': difference(before['representation_drift'], after['representation_drift']),
        })
    return rows


def load_mean_results(last, output_dir, mean_dir):
    mean_dir = Path(mean_dir).resolve()
    mean = json.loads((mean_dir / 'all_model_task_metrics.json').read_text())
    last_probes = json.loads((output_dir / 'probe_manifest.json').read_text())
    mean_probes = json.loads((mean_dir / 'probe_manifest.json').read_text())
    validate_pooling_inputs(last, mean, last_probes, mean_probes)
    for task in last['tasks']:
        if not np.allclose(last['pairwise_matrices']['symmetric_kl'][task], mean['pairwise_matrices']['symmetric_kl'][task], atol=1e-7, rtol=1e-7):
            raise ValueError(f'{task}: output distributions differ between pooling runs')
    # The two directed means, not only their symmetric average, must also agree.
    from evaluation.sequential_directional_kl_report import directional_kl_matrices
    first_kl = directional_kl_matrices(last)
    second_kl = directional_kl_matrices(mean)
    for task in last['tasks']:
        if not np.allclose(first_kl[task], second_kl[task], atol=1e-7, rtol=1e-7):
            raise ValueError(f'{task}: directed KL differs between pooling runs')
    last_contract = output_dir / 'probe_cache/manifest.json'
    mean_contract = mean_dir / 'probe_cache/manifest.json'
    if last_contract.is_file() and mean_contract.is_file():
        first, second = json.loads(last_contract.read_text()), json.loads(mean_contract.read_text())
        for key in ('models', 'probes', 'max_input_length', 'batch_size', 'seed', 'base_config_sha256'):
            if first[key] != second[key]:
                raise ValueError(f'Pooling cache identities differ for {key}')
    return mean


def pooling_markdown(last, mean, rows):
    final = last['models'][-1]['model']
    lines = ['## Mean pooling：第 16、32 层分析（接在 last pooling 之后）', '',
             '本节与前面的 last pooling 使用相同模型、adapter、验证样本、顺序、提示和截断设置。'
             '两种 pooling 均分别分析第 16、32 层，并与 M0 或各任务刚训练后的 Mi 比较。', '',
             'Mean pooling 按 attention_mask 排除 padding，对剩余 token 的向量取均值。'
             '平均范围包含完整有效提示中的文本、指令、类别选项和提示模板；没有另外屏蔽指令或选项。', '',
             '先对每条样本求 token 均值，再计算样本之间的 CKA 或逐样本 cosine distance；'
             '不是先计算每个 token 的 drift 再平均。准确率与 KL 来自相同模型输出，不随 pooling 重新定义。', '',
             f'下表固定被比较模型为最终 {final}。更高 CKA 或更低表征漂移描述不同 pooling 下观测到的差异，不能据此直接判定哪种 pooling 更好。', '',
             '| 层 | 任务 | 参照 | Last CKA | Mean CKA | Last drift | Mean drift |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    def number(value):
        return 'N/A' if value is None else f'{value:.4f}'
    for row in rows:
        if row['model'] == final:
            lines.append(f"| {row['layer']} | {row['evaluated_task']} | {row['reference_model']} ({row['reference_kind']}) | {number(row['last_cka'])} | {number(row['mean_cka'])} | {number(row['last_representation_drift'])} | {number(row['mean_representation_drift'])} |")
    lines += ['', '### Mean pooling：全部模型的指标', '',
              '| 层 | 任务 | 被比较模型 | 参照 | CKA | Representation drift |', '| --- | --- | --- | --- | --- | --- |']
    for row in mean['selected_layer_results']:
        lines.append(f"| {row['layer']} | {row['evaluated_task']} | {row['model']} | {row['reference_model']} ({row['reference_kind']}) | {number(row['cka'])} | {number(row['representation_drift'])} |")
    lines += ['', '参照模型之前的阶段是训练前对照，不解释为遗忘；之后的阶段用于观察后续训练影响。'
              '两种 pooling 的结论若不同，说明结论对表征汇总方式敏感，应结合准确率与 KL 分析。', '']
    return '\n'.join(lines)


def persist_pooling_comparison(last, output_dir):
    mean_dir = last.get('mean_pooling_results_dir')
    if not mean_dir:
        return None, ''
    mean = load_mean_results(last, output_dir, mean_dir)
    rows = join_pooling_results(last, mean)
    from evaluation.sequential_drift_eval import save_json
    save_json(output_dir / 'pooling_comparison_metrics.json', {
        'complete': True, 'last_results_dir': str(output_dir.resolve()), 'mean_results_dir': str(Path(mean_dir).resolve()),
        'layers': [16, 32], 'same_validation_probes': True,
        'mean_mask': 'All attention_mask=1 prompt tokens, including instructions/options; exclude padding only.',
        'row_order': 'last pooling analyses, then mean pooling analyses, then directional KL, then calculation methods', 'results': rows,
    })
    with (output_dir / 'pooling_comparison_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = pooling_markdown(last, mean, rows)
    (output_dir / 'mean_pooling_analysis.md').write_text(markdown, encoding='utf-8')
    return mean, markdown


def plot_mean_pages(last, mean, grid):
    # Four full model/task heatmaps, followed by the two-layer final-model view.
    layer16 = plot_layer_pages(mean, grid, focus_layer=16)
    layer32 = plot_layer_pages(mean, grid, focus_layer=32, include_comparison=False)
    pages = [(fig, 'mean_' + name) for fig, name in layer16[:2] + layer32 + layer16[2:]]
    import matplotlib.pyplot as plt
    tasks, models = last['tasks'], last['models']
    final = models[-1]
    for layer in (16, 32):
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.80, hspace=0.70, wspace=0.25)
        fig.suptitle(f"Last vs mean pooling | layer {layer} | evaluated model: {final['label']}", fontsize=18, y=0.97)
        fig.text(0.5, 0.92, 'Same model pairs, prompts and validation probes; only pooling changes.', ha='center', fontsize=12)
        for row, kind in enumerate(('base', 'task_learned')):
            references = [0 if kind == 'base' else index + 1 for index in range(len(tasks))]
            for column, (key, label) in enumerate((('cka', 'Linear CKA'), ('rep_drift', 'Representation drift'))):
                ax = axes[row, column]
                maximum = 0.0
                for offset, (payload, name, color) in enumerate(((last, 'Last pooling', '#2563eb'), (mean, 'Mean pooling', '#d97706'))):
                    matrix = payload['selected_layer_matrices'][str(layer)][key]
                    values = np.array([np.nan if matrix[task][reference][final['stage']] is None else matrix[task][reference][final['stage']]
                                       for task, reference in zip(tasks, references)])
                    positions = np.arange(len(tasks)) + (offset - 0.5) * 0.34
                    ax.bar(positions, values, width=0.31, color=color, label=name)
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
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.88), ncol=2, frameon=False)
        fig.text(0.09, 0.055, 'Mean pooling excludes padding and averages all other prompt tokens. Higher CKA/lower drift does not establish a better pooling method.', fontsize=10)
        pages.append((fig, f'last_vs_mean_layer_{layer}'))
    return pages

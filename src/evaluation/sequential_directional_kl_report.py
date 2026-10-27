"""Directional full-vocabulary KL reports derived from saved model-pair means."""
import csv
import json
import math

import numpy as np


def directional_kl_matrices(payload):
    """Entry [task][a][b] is KL(P_Ma || P_Mb); never symmetrize it."""
    tasks, models = payload['tasks'], payload['models']
    indices = {model['model']: model['stage'] for model in models}
    size = len(models)
    matrices = {task: [[0.0 if first == second else None for second in range(size)] for first in range(size)] for task in tasks}
    seen = set()
    for row in payload['pairwise_results']:
        task = row['evaluated_task']
        first, second = indices[row['model_a']], indices[row['model_b']]
        identity = (task, min(first, second), max(first, second))
        if first == second or identity in seen:
            raise ValueError('Directional KL source contains duplicate/self model pairs')
        seen.add(identity)
        forward, reverse = row['kl_a_to_b'], row['kl_b_to_a']
        if any(value is None or not math.isfinite(value) or value < 0 for value in (forward, reverse)):
            raise ValueError('Directional KL requires finite nonnegative saved values')
        symmetric = payload['pairwise_matrices']['symmetric_kl'][task][first][second]
        if not math.isclose((forward + reverse) / 2, symmetric, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Directional KL does not reproduce saved symmetric KL')
        matrices[task][first][second] = forward
        matrices[task][second][first] = reverse
    expected = {(task, first, second) for task in tasks for first in range(size) for second in range(first + 1, size)}
    if seen != expected:
        raise ValueError('Directional KL source is missing model-pair/task results')
    return matrices


def add_directional_kl_results(payload):
    matrices = directional_kl_matrices(payload)
    tasks, models = payload['tasks'], payload['models']
    counts = {row['evaluated_task']: row['validation_probe_count'] for row in payload['model_task_results']}
    rows = []
    for task_index, task in enumerate(tasks):
        for kind, reference_stage in (('base', 0), ('task_learned', task_index + 1)):
            reference = models[reference_stage]
            for model in models:
                stage = model['stage']
                rows.append({
                    'evaluated_task': task, 'reference_kind': kind,
                    'reference_model': reference['model'], 'reference_adapter_dir': reference['adapter_dir'],
                    'model': model['model'], 'adapter_dir': model['adapter_dir'],
                    'model_relative_to_reference': 'same_model' if stage == reference_stage else ('earlier_model' if stage < reference_stage else 'later_model'),
                    'validation_probe_count': counts[task],
                    'kl_reference_to_model': matrices[task][reference_stage][stage],
                    'kl_model_to_reference': matrices[task][stage][reference_stage],
                    'symmetric_kl': payload['pairwise_matrices']['symmetric_kl'][task][reference_stage][stage],
                    'unit': 'nats',
                })
    for item in payload['model_task_results']:
        stage = next(model['stage'] for model in models if model['model'] == item['model'])
        task = item['evaluated_task']
        reference = tasks.index(task) + 1
        item.update(kl_base_to_model=matrices[task][0][stage], kl_model_to_base=matrices[task][stage][0],
                    kl_task_learned_to_model=matrices[task][reference][stage], kl_model_to_task_learned=matrices[task][stage][reference])
    for item in payload.get('task_learned_reference_results', []):
        stage = next(model['stage'] for model in models if model['model'] == item['model'])
        reference, task = item['reference_stage'], item['evaluated_task']
        item.update(kl_reference_to_model=matrices[task][reference][stage], kl_model_to_reference=matrices[task][stage][reference])
    payload['directional_kl_matrices'] = matrices
    payload['directional_kl_reference_results'] = rows
    payload['directional_kl_definition'] = 'Matrix[task][a][b] is mean KL(P_Ma || P_Mb) over identical task validation prompts, summed over the full vocabulary at the first answer position, in nats.'
    return rows


def directional_kl_markdown(payload):
    rows = payload['directional_kl_reference_results']
    final = payload['models'][-1]['model']
    lines = ['## 双向 KL：Functional drift 的方向性分析', '',
             '记 R 为参照模型，M 为被比较模型。以下两个方向分别统计，不将它们合并或互换：', '',
             '- `KL(R || M)`：以参照分布加权，衡量被比较模型对参照行为的偏离。',
             '- `KL(M || R)`：以被比较模型分布加权，衡量其当前偏好的输出与参照分布的差异。',
             '- `Symmetric KL = (KL(R || M) + KL(M || R))/2`，保留作为整体变化的补充。', '',
             '参照由具体任务和图表决定：base 图用 M0；任务参照图用该任务刚训练后的 Mi。'
             '即使被比较模型早于 Mi，第一列方向仍是 Mi 到被比较模型，不会按保存顺序自动交换。'
             '训练前模型只是回溯对照，不能把其 KL 解释为后续遗忘。', '',
             '计算口径为每条样本在首个答案位置对完整词表求和，再平均固定验证样本；单位 nats。'
             'KL 不使用 hidden-state pooling，因此不区分第 16/32 层或 last/mean pooling。'
             '两个 pooling 运行的正反 KL 已分别核对；本节只展示一份共享结果。', '',
             f'### 最终模型 {final} 相对两种参照', '',
             '| 任务 | 参照 R | 被比较模型 M | KL(R ‖ M) | KL(M ‖ R) | Symmetric KL |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        if row['model'] == final:
            lines.append(f"| {row['evaluated_task']} | {row['reference_model']} ({row['reference_kind']}) | {row['model']} | {row['kl_reference_to_model']:.4f} | {row['kl_model_to_reference']:.4f} | {row['symmetric_kl']:.4f} |")
    lines += ['', '### 全部模型与任务', '',
              '| 任务 | 参照 R | 被比较模型 M | 相对参照时序 | KL(R ‖ M) | KL(M ‖ R) | Symmetric KL |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['evaluated_task']} | {row['reference_model']} ({row['reference_kind']}) | {row['model']} | {row['model_relative_to_reference']} | {row['kl_reference_to_model']:.4f} | {row['kl_model_to_reference']:.4f} | {row['symmetric_kl']:.4f} |")
    lines += ['', '双向 KL 均非负，方向不表示变化的正负，也不是百分比。'
              'KL 没有通用大小阈值；较大 KL 可以伴随学习改善。判断遗忘仍需结合该任务准确率变化与模型/样本设置。', '']
    return '\n'.join(lines)


def persist_directional_kl(payload, output_dir):
    rows = add_directional_kl_results(payload)
    from evaluation.sequential_drift_eval import save_json
    save_json(output_dir / 'directional_kl_metrics.json', {
        'complete': payload['complete'], 'tasks': payload['tasks'], 'models': payload['models'],
        'definition': payload['directional_kl_definition'], 'matrices': payload['directional_kl_matrices'],
        'results': rows,
    })
    def csv_file(filename, entries):
        with (output_dir / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    csv_file('directional_kl_metrics.csv', rows)
    directed_pairs = []
    for task in payload['tasks']:
        for first in payload['models']:
            for second in payload['models']:
                if first['stage'] != second['stage']:
                    directed_pairs.append({'evaluated_task': task, 'first_model_P': first['model'], 'second_model_Q': second['model'],
                                           'kl_P_to_Q': payload['directional_kl_matrices'][task][first['stage']][second['stage']], 'unit': 'nats'})
    csv_file('directional_kl_all_pairs.csv', directed_pairs)
    csv_file('all_model_task_metrics.csv', payload['model_task_results'])
    if payload.get('task_learned_reference_results'):
        csv_file('task_learned_reference_metrics.csv', payload['task_learned_reference_results'])
        target = output_dir / 'task_learned_reference_metrics.json'
        reference_payload = json.loads(target.read_text())
        reference_payload['results'] = payload['task_learned_reference_results']
        reference_payload['kl_direction_definition'] = 'Reference is always the task-just-learned model Mi, regardless of chronological order of the compared model.'
        save_json(target, reference_payload)
    save_json(output_dir / 'all_model_task_metrics.json', payload)
    text = directional_kl_markdown(payload)
    (output_dir / 'directional_kl_analysis.md').write_text(text, encoding='utf-8')
    return text


def plot_directional_kl_pages(payload, grid):
    import matplotlib.pyplot as plt
    tasks, models = payload['tasks'], payload['models']
    matrices = payload['directional_kl_matrices']
    labels = [model['label'] for model in models]
    pages = []
    # Use the same range for both directions and the symmetric mean.
    maximum = max(value for task in tasks for row in matrices[task] for value in row)
    for kind in ('base', 'task_learned'):
        fig, axes = plt.subplots(1, 3, figsize=(21, 8))
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.28, top=0.76, wspace=0.62)
        description = 'M0 (Base)' if kind == 'base' else 'task-just-learned model named in each column'
        fig.suptitle('Functional drift | reference R = ' + description, fontsize=18, y=0.97)
        fig.text(0.5, 0.90, 'M = evaluated model on the row. Both directions use the same fixed validation prompts; units: nats.', ha='center', fontsize=12)
        refs = [0 if kind == 'base' else index + 1 for index in range(len(tasks))]
        xlabels = [f"{task}\nR: {models[reference]['model']}" for task, reference in zip(tasks, refs)]
        forward = [[matrices[task][reference][model['stage']] for task, reference in zip(tasks, refs)] for model in models]
        reverse = [[matrices[task][model['stage']][reference] for task, reference in zip(tasks, refs)] for model in models]
        symmetric = [[payload['pairwise_matrices']['symmetric_kl'][task][reference][model['stage']] for task, reference in zip(tasks, refs)] for model in models]
        for ax, values, title in zip(axes, (forward, reverse, symmetric),
                                     ('KL(R || M)\nReference first', 'KL(M || R)\nEvaluated model first', 'Symmetric KL\nMean of both directions')):
            grid(ax, values, xlabels, labels, title, upper=maximum)
        fig.text(0.12, 0.11, 'Directions are determined by the named reference, not by which model was trained earlier. Diagonal/self-comparison = 0.', fontsize=11)
        fig.text(0.12, 0.065, 'One shared functional metric section: changing hidden-state pooling or selecting layer 16/32 does not change these output probabilities.', fontsize=10)
        pages.append((fig, f'directional_kl_vs_{kind}'))
    columns = min(2, len(tasks))
    nrows = (len(tasks) + columns - 1) // columns
    fig, axes = plt.subplots(nrows, columns, figsize=(17, max(6, nrows * 5.0)), squeeze=False)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.85, hspace=0.65, wspace=0.42)
    fig.suptitle('Directed KL between every model pair, separately for each validation task', fontsize=18, y=0.97)
    fig.text(0.5, 0.92, 'Cell = KL(P || Q): row model supplies P, column model supplies Q. Matrices need not be symmetric.', ha='center', fontsize=12)
    for index, task in enumerate(tasks):
        ax = axes.flat[index]
        grid(ax, matrices[task], [model['model'] for model in models], labels, f'Validation data: {task}', upper=maximum)
        ax.set_xlabel('Q: second distribution / column model')
        ax.set_ylabel('P: first distribution')
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)
    fig.text(0.14, 0.035, 'Summed over the full vocabulary at the first answer position, then averaged across fixed task probes. Shared color scale; units: nats.', fontsize=10)
    pages.append((fig, 'directional_kl_all_pairs'))
    return pages

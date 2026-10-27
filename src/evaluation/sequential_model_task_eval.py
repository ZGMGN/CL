"""All saved models on every task: test accuracy and within-task pairwise drift.

Reuse a complete test accuracy matrix. Each model is forwarded once per validation
probe; CPU caches allow recomputing every model pair without rerunning inference.
"""
import argparse
import csv
import gc
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from peft import PeftModel

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / 'src'))

from evaluation import sequential_drift_eval as drift
from evaluation import sequential_heatmap_eval as heatmap
from evaluation.drift_metrics import distribution_drift, linear_cka, representation_cosine_drift
from evaluation.general_task_eval import build_eval_model
from model.utils import GENERAL_TEMPLATE, MODEL_DIR, set_seed


METRICS = ('cka', 'rep_drift', 'symmetric_kl')


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def model_mapping(run_dir, tasks, adapters):
    return [
        {'model': f'M{stage}', 'stage': stage,
         'label': 'M0 (Base)' if stage == 0 else f'M{stage} (after {tasks[stage - 1]})',
         'base_model': str(Path(MODEL_DIR).resolve()),
         'adapter_dir': None if stage == 0 else str(adapters[stage - 1].resolve()),
         'cumulative_tasks': tasks[:stage]}
        for stage in range(len(tasks) + 1)
    ]


def prepare_fixed_probes(tasks, args, output_dir, source_dir):
    source_path = source_dir / 'probe_manifest.json'
    if not source_path.is_file():
        return drift.prepare_probes(tasks, args, output_dir)
    manifest = json.loads(source_path.read_text())
    if set(manifest) != set(tasks):
        raise ValueError('Source probe manifest must contain exactly the evaluation tasks')
    probes, modules = {}, {}
    for task in tasks:
        entry = manifest[task]
        path = ROOT_DIR / 'data' / task / 'val.jsonl'
        dataset = heatmap.load_dataset('json', data_files=str(path), split='train')
        indices = entry['indices']
        if (entry['split'] != 'val' or len(indices) < 3 or len(set(indices)) != len(indices)
                or len(indices) != min(args.probe_samples, len(dataset))
                or any(not isinstance(i, int) or not 0 <= i < len(dataset) for i in indices)):
            raise ValueError(f'{task}: incompatible source validation probes')
        probes[task] = [dataset[i] for i in indices]
        digest = hashlib.sha256(json.dumps(probes[task], sort_keys=True).encode()).hexdigest()
        if digest != entry['examples_sha256']:
            raise ValueError(f'{task}: validation example contents changed since the source evaluation')
        modules[task] = importlib.import_module(f'tasks.{task}')
    drift.save_json(output_dir / 'probe_manifest.json', manifest)
    return probes, modules


def collect_cached_outputs(args, tasks, models, probes, modules, layers, output_dir):
    cache_dir = output_dir / 'probe_cache'
    cache_dir.mkdir(exist_ok=True)
    manifest = json.loads((output_dir / 'probe_manifest.json').read_text())
    # Bind cached outputs to adapter contents as well as probe/preprocessing settings.
    identities = []
    for item in models:
        weights = None
        if item['adapter_dir']:
            folder = Path(item['adapter_dir'])
            weights = next(folder / name for name in ('adapter_model.safetensors', 'adapter_model.bin') if (folder / name).is_file())
        identities.append({
            **item,
            'adapter_config_sha256': None if weights is None else hashlib.sha256((weights.parent / 'adapter_config.json').read_bytes()).hexdigest(),
            'adapter_weights_sha256': None if weights is None else hashlib.sha256(weights.read_bytes()).hexdigest(),
        })
    signature = {
        'models': identities, 'probes': manifest, 'layers': layers,
        'pooling': args.pooling, 'max_input_length': args.max_input_length,
        'batch_size': args.batch_size, 'seed': args.seed,
        'base_config_sha256': hashlib.sha256((Path(MODEL_DIR) / 'config.json').read_bytes()).hexdigest(),
    }
    contract = cache_dir / 'manifest.json'
    if contract.exists() and json.loads(contract.read_text()) != signature:
        raise ValueError('Probe cache metadata differs; choose a new output directory')
    drift.save_json(contract, signature)
    all_outputs = []
    model = tokenizer = device = None
    active = None
    try:
        for item in models:
            stage = item['stage']
            paths = {task: cache_dir / f'{stage:02d}_{task}.pt' for task in tasks}
            if any(not path.is_file() for path in paths.values()):
                if model is None:
                    set_seed(args.seed)
                    model, tokenizer, device, _ = build_eval_model(argparse.Namespace(peft_model_folder=None), GENERAL_TEMPLATE)
                    tokenizer.padding_side = tokenizer.truncation_side = 'left'
                if stage:
                    name = f'model_task_{stage}'
                    if active is None:
                        model = PeftModel.from_pretrained(model, item['adapter_dir'], adapter_name=name, is_trainable=False)
                    else:
                        model.load_adapter(item['adapter_dir'], adapter_name=name, is_trainable=False)
                    model.set_adapter(name)
                    if active is not None:
                        model.delete_adapter(active)
                    active = name
                elif active is not None:
                    raise ValueError('Base probes must be collected before loading any adapter')
                for task, path in paths.items():
                    if path.is_file():
                        continue
                    print(f"Collect {item['label']} | validation task {task} | {len(probes[task])} examples", flush=True)
                    outputs = drift.collect_outputs(model, tokenizer, probes[task], modules[task], layers, args, device)
                    temporary = path.with_suffix('.tmp')
                    torch.save(outputs, temporary)
                    temporary.replace(path)
            all_outputs.append({task: torch.load(path, map_location='cpu', weights_only=True) for task, path in paths.items()})
            drift.save_json(output_dir / 'status.json', {'state': 'collecting', 'models_cached': stage + 1, 'models_total': len(models)})
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_outputs


def compare_all_models(outputs, tasks, layers, accuracy, models):
    """Symmetric metrics indexed [task][model_a][model_b]; KL directions are explicit."""
    count = len(models)
    matrices = {key: {task: [[None] * count for _ in models] for task in tasks} for key in METRICS}
    layer_matrices = {str(layer): {task: [[None] * count for _ in models] for task in tasks} for layer in layers}
    pair_rows, layer_rows = [], []
    for task_index, task in enumerate(tasks):
        print(f'Compare all {count * (count - 1) // 2} model pairs | validation task {task}', flush=True)
        for first, second in itertools.combinations_with_replacement(range(count), 2):
            before, after = outputs[first][task], outputs[second][task]
            layer_values = {}
            for layer in layers:
                cka = linear_cka(before['features'][layer], after['features'][layer])
                rep = representation_cosine_drift(before['features'][layer], after['features'][layer])
                layer_values[layer] = {'cka': cka, 'rep_drift': rep}
                layer_matrices[str(layer)][task][first][second] = cka
                layer_matrices[str(layer)][task][second][first] = cka
                if first != second:
                    layer_rows.append({'model_a': models[first]['model'], 'model_b': models[second]['model'],
                                       'evaluated_task': task, 'layer': layer, **layer_values[layer]})
            kl = distribution_drift(before['log_probs'], after['log_probs']) if first != second else {
                'kl_forward': 0.0, 'kl_reverse': 0.0, 'symmetric_kl': 0.0}
            values = {**layer_values[layers[-1]], 'symmetric_kl': kl['symmetric_kl']}
            for metric, value in values.items():
                matrices[metric][task][first][second] = value
                matrices[metric][task][second][first] = value
            if first != second:
                pair_rows.append({
                    'model_a': models[first]['model'], 'adapter_a': models[first]['adapter_dir'],
                    'model_b': models[second]['model'], 'adapter_b': models[second]['adapter_dir'],
                    'evaluated_task': task, 'validation_probe_count': len(before['log_probs']),
                    'summary_layer': layers[-1], **values,
                    'kl_a_to_b': kl['kl_forward'], 'kl_b_to_a': kl['kl_reverse'],
                    'test_accuracy_a': float(accuracy[first, task_index]),
                    'test_accuracy_b': float(accuracy[second, task_index]),
                    'test_accuracy_change_pp_b_minus_a': float(100 * (accuracy[second, task_index] - accuracy[first, task_index])),
                })
    model_rows = []
    for item in models:
        stage = item['stage']
        for index, task in enumerate(tasks):
            learned = index + 1
            acc = float(accuracy[stage, index])
            model_rows.append({
                'model': item['model'], 'adapter_dir': item['adapter_dir'], 'cumulative_tasks': ' -> '.join(item['cumulative_tasks']),
                'evaluated_task': task, 'task_seen_in_training': stage >= learned, 'test_accuracy': acc,
                'test_accuracy_change_vs_base_pp': 100 * (acc - float(accuracy[0, index])),
                'test_accuracy_change_vs_previous_pp': None if stage == 0 else 100 * (acc - float(accuracy[stage - 1, index])),
                'forgetting_since_task_learned_pp': None if stage < learned else 100 * (float(accuracy[learned, index]) - acc),
                'validation_probe_count': len(outputs[stage][task]['log_probs']), 'summary_layer': layers[-1],
                'cka_vs_base': matrices['cka'][task][0][stage],
                'rep_drift_vs_base': matrices['rep_drift'][task][0][stage],
                'symmetric_kl_vs_base': matrices['symmetric_kl'][task][0][stage],
            })
    return matrices, layer_matrices, model_rows, pair_rows, layer_rows



def add_task_learned_reference(payload):
    """Derive all-model comparisons using each task's just-trained model as reference.

    Stages before that reference are retrospective controls, not forgetting.
    This transformation only reads completed accuracy and pairwise matrices.
    """
    tasks, models = payload['tasks'], payload['models']
    accuracy = np.array(payload['accuracy'], dtype=float)
    references, rows, layer_rows = [], [], []
    for index, task in enumerate(tasks):
        reference_stage = index + 1
        reference = models[reference_stage]
        references.append({'evaluated_task': task, 'reference_stage': reference_stage,
                           'reference_model': reference['model'], 'reference_label': reference['label'],
                           'reference_adapter_dir': reference['adapter_dir']})
        for model in models:
            stage = model['stage']
            phase = 'before_task_training' if stage < reference_stage else ('task_just_learned' if stage == reference_stage else 'after_task_training')
            change = float(100 * (accuracy[stage, index] - accuracy[reference_stage, index]))
            row = {
                'evaluated_task': task, 'model': model['model'], 'model_label': model['label'],
                'adapter_dir': model['adapter_dir'], 'reference_model': reference['model'],
                'reference_label': reference['label'], 'reference_adapter_dir': reference['adapter_dir'],
                'reference_stage': reference_stage, 'comparison_phase': phase,
                'test_accuracy': float(accuracy[stage, index]),
                'reference_test_accuracy': float(accuracy[reference_stage, index]),
                'test_accuracy_change_vs_task_learned_pp': change,
                'forgetting_since_task_learned_pp': None if stage < reference_stage else -change,
                'summary_layer': payload['summary_layer'],
                **{f'{key}_vs_task_learned': payload['pairwise_matrices'][key][task][reference_stage][stage] for key in METRICS},
            }
            rows.append(row)
            for layer in payload['resolved_layers']:
                layer_rows.append({'evaluated_task': task, 'model': model['model'],
                                   'reference_model': reference['model'], 'layer': layer,
                                   'cka_vs_task_learned': payload['layer_cka_matrices'][str(layer)][task][reference_stage][stage]})
    payload['task_learned_references'] = references
    payload['task_learned_reference_results'] = rows
    lookup = {(row['model'], row['evaluated_task']): row for row in rows}
    for item in payload['model_task_results']:
        reference = lookup[item['model'], item['evaluated_task']]
        for key in ('reference_model', 'reference_adapter_dir', 'comparison_phase',
                    'test_accuracy_change_vs_task_learned_pp', 'cka_vs_task_learned',
                    'rep_drift_vs_task_learned', 'symmetric_kl_vs_task_learned'):
            item[key] = reference[key]
    return rows, layer_rows


def persist_task_learned_reference(payload, output_dir):
    rows, layer_rows = add_task_learned_reference(payload)
    drift.save_json(output_dir / 'task_learned_reference_metrics.json', {
        'complete': payload['complete'], 'tasks': payload['tasks'],
        'references': payload['task_learned_references'],
        'definition': 'For each task Ti, compare every model Mj to Mi, the saved cumulative model immediately after training Ti. Identical task probes for both models.',
        'accuracy_change_definition': '100*(accuracy(Mj,Ti)-accuracy(Mi,Ti)); stages j<i are pretraining controls, not forgetting.',
        'results': rows,
    })
    write_csv(output_dir / 'task_learned_reference_metrics.csv', rows)
    write_csv(output_dir / 'task_learned_reference_layer_metrics.csv', layer_rows)
    drift.save_json(output_dir / 'all_model_task_metrics.json', payload)
    write_csv(output_dir / 'all_model_task_metrics.csv', payload['model_task_results'])


def refresh_reports(output_dir):
    """Rebuild derived comparisons and figures without loading datasets or models."""
    payload = json.loads((output_dir / 'all_model_task_metrics.json').read_text())
    if not payload.get('complete'):
        raise ValueError('Report-only mode requires complete model-pair metrics')
    persist_task_learned_reference(payload, output_dir)
    write_analysis(payload, output_dir)
    plot_report(payload, output_dir)
    print(f'Updated task-trained reference reports: {output_dir}', flush=True)
    return payload


def write_analysis(payload, output_dir):
    tasks, models = payload['tasks'], payload['models']
    accuracy = np.array(payload['accuracy']) * 100
    labels = [item['model'] for item in models]
    lines = ['# 全部模型 × 全部任务分析', '',
             '训练顺序：' + ' → '.join(tasks), '',
             f"共 {len(models)} 个模型、{len(tasks)} 个任务，{len(payload['model_task_results'])} 组测试准确率，"
             f"{len(payload['pairwise_results'])} 组同任务模型对比较。", '',
             'M0 是 base；后续模型均为同一 base 加该阶段累计训练得到的 adapter。', '',
             '| 模型 | 累计训练任务 | adapter |', '| --- | --- | --- |']
    for item in models:
        relative = '无' if item['adapter_dir'] is None else str(Path(item['adapter_dir']).relative_to(Path(payload['run_dir'])))
        lines.append(f"| {item['model']} | {' → '.join(item['cumulative_tasks']) or '无'} | {relative} |")
    lines += ['', '## 完整测试准确率（%）', '', '| 模型 | ' + ' | '.join(tasks) + ' |',
              '| --- | ' + ' | '.join('---' for _ in tasks) + ' |']
    for index, label in enumerate(labels):
        lines.append('| ' + label + ' | ' + ' | '.join(f'{value:.2f}' for value in accuracy[index]) + ' |')
    lines += ['', '## 逐任务结果', '']
    for index, task in enumerate(tasks):
        learned = index + 1
        best = int(np.argmax(accuracy[:, index]))
        delta_base = accuracy[-1, index] - accuracy[0, index]
        forgetting = accuracy[learned, index] - accuracy[-1, index]
        values = [f'{label} {accuracy[stage, index]:.2f}%' for stage, label in enumerate(labels)]
        lines += [f'### {task}', '', ' → '.join(values), '',
                  f'最佳准确率为 {labels[best]} 的 {accuracy[best, index]:.2f}%；最终模型相对 base 改变 {delta_base:+.2f} 个百分点。',
                  f'相对刚学完该任务的 {labels[learned]}，最终遗忘量为 {forgetting:+.2f} 个百分点（负值表示进一步改善）。', '']
        for reference, name in ((0, 'base'), (learned, '刚学完该任务的模型')):
            pair = payload['pairwise_matrices']
            def number(value):
                return 'N/A' if value is None else f'{value:.4f}'
            lines.append(f"最终模型 {labels[-1]} 与{ name } {labels[reference]}，在同一 {task} 验证集样本上的比较："
                         f"CKA={number(pair['cka'][task][reference][-1])}，"
                         f"表征 cosine drift={number(pair['rep_drift'][task][reference][-1])}，"
                         f"symmetric KL={number(pair['symmetric_kl'][task][reference][-1])} nats。")
        lines += ['', f'以 {task} 刚训练完成的模型 {labels[learned]} 为参照的全阶段比较：', '',
                  '| 被比较模型 | 相对参照阶段 | 准确率变化（百分点） | CKA | 表征漂移 | symmetric KL（nats） |',
                  '| --- | --- | --- | --- | --- | --- |']
        phases = {'before_task_training': '该任务训练前对照', 'task_just_learned': '参照模型自身', 'after_task_training': '该任务训练后'}
        for result in payload['task_learned_reference_results']:
            if result['evaluated_task'] != task:
                continue
            lines.append(f"| {result['model']} | {phases[result['comparison_phase']]} | {result['test_accuracy_change_vs_task_learned_pp']:+.2f} | "
                         f"{number(result['cka_vs_task_learned'])} | {number(result['rep_drift_vs_task_learned'])} | {number(result['symmetric_kl_vs_task_learned'])} |")
        lines += ['', '准确率变化 = 被比较模型减参照模型；参照之前的模型仅作训练前对照，不将差值解释为遗忘。', '']
    lines += ['## 计算口径', '',
              f"- 测试准确率复用完整测试集矩阵：`{payload['accuracy_source']}`。",
              f"- CKA/表征漂移汇总使用 hidden_states[{payload['summary_layer']}]，pooling={payload['settings']['pooling']}；逐层结果另存 CSV。",
              '- 对同一任务，所有模型使用完全相同的验证样本、顺序、提示和截断设置；样本索引与 hash 见 probe_manifest.json。',
              '- KL 对相同提示的首个答案位置、整个词表计算；symmetric KL 为正反 KL 的平均值。',
              '- CKA 和 cosine drift 是两个模型之间的指标，并非某个模型单独的分数。原 5×4 总览与 M0 比较；新增总览每列以该任务刚训练后的 Mi 为参照；5×5 图显示所有模型对。',
              '- 未学任务、已学任务均纳入评估；高 CKA 不直接代表准确率高或没有遗忘。',
              '- 验证样本数量沿用已有设置，并未因为增加模型对而增加独立样本数；这些是点估计，未给出置信区间。',
              '- 梯度 cosine 与 LoRA 更新 cosine 描述任务梯度/训练更新方向，仍在原 transition 报告中，不冒充单模型任务指标。', '']
    (output_dir / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')


def run_evaluation(args):
    if args.report_only:
        destination = Path(args.output_dir).resolve() if args.output_dir else Path(args.run_dir).resolve() / 'evaluation_all_models'
        return refresh_reports(destination)
    if min(args.batch_size, args.eval_batch_size, args.max_input_length, args.max_new_tokens, args.cpu_threads) <= 0 or args.probe_samples < 3:
        raise ValueError('Positive limits and at least three validation probes are required')
    torch.set_num_threads(args.cpu_threads)
    run_dir = Path(args.run_dir).expanduser().resolve()
    tasks = heatmap.resolve_tasks(run_dir, args.tasks)
    if len(set(tasks)) != len(tasks):
        raise ValueError('Tasks must be unique')
    adapters = heatmap.find_adapters(run_dir, tasks)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / 'evaluation_all_models'
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_eval_dir).resolve() if args.source_eval_dir else next(
        (run_dir / name for name in ('evaluation_adjacent', 'evaluation_drift') if (run_dir / name / 'probe_manifest.json').is_file()), run_dir / 'evaluation_drift')
    source_metadata = source_dir / 'transition_metrics.json'
    if source_metadata.is_file():
        source = json.loads(source_metadata.read_text())
        if source['tasks'] != tasks or not source.get('complete'):
            raise ValueError('Source evaluation must be complete and have the same task order')
        for key in ('pooling', 'max_input_length', 'seed'):
            if source['settings'][key] != getattr(args, key):
                raise ValueError(f'Source evaluation differs for {key}; pass matching settings')
    if not args.accuracy_matrix:
        candidates = [source_dir / 'accuracy/accuracy_matrix.json', run_dir / 'evaluation_drift/accuracy/accuracy_matrix.json', run_dir / 'evaluation_heatmap/accuracy_matrix.json']
        if source_metadata.is_file() and source['settings'].get('accuracy_matrix'):
            candidates.insert(0, Path(source['settings']['accuracy_matrix']))
        args.accuracy_matrix = next((str(path) for path in candidates if path.is_file()), None)
    accuracy = drift.load_accuracy(args, tasks, output_dir)
    models = model_mapping(run_dir, tasks, adapters)
    config = json.loads((Path(MODEL_DIR) / 'config.json').read_text())
    layers = drift.resolve_layers(args.layers, config['num_hidden_layers'] + 1)
    probes, modules = prepare_fixed_probes(tasks, args, output_dir, source_dir)
    outputs = collect_cached_outputs(args, tasks, models, probes, modules, layers, output_dir)
    drift.save_json(output_dir / 'status.json', {'state': 'comparing', 'models_cached': len(models)})
    matrices, layer_matrices, model_rows, pair_rows, layer_rows = compare_all_models(outputs, tasks, layers, accuracy, models)
    payload = {
        'complete': True, 'run_dir': str(run_dir), 'tasks': tasks, 'models': models,
        'settings': vars(args), 'source_eval_dir': str(source_dir), 'accuracy_source': args.accuracy_matrix or str(output_dir / 'accuracy/accuracy_matrix.json'),
        'summary_layer': layers[-1], 'resolved_layers': layers, 'accuracy': accuracy.tolist(),
        'accuracy_split': 'test', 'drift_split': 'val',
        'pairwise_definition': 'Compare model_a and model_b on identical validation prompts from evaluated_task. Diagonal is self-comparison; undefined CKA remains null.',
        'pairwise_matrices': matrices, 'layer_cka_matrices': layer_matrices,
        'model_task_results': model_rows, 'pairwise_results': pair_rows,
    }
    drift.save_json(output_dir / 'all_model_task_metrics.json', payload)
    drift.save_json(output_dir / 'model_adapter_mapping.json', {'models': models})
    write_csv(output_dir / 'all_model_task_metrics.csv', model_rows)
    write_csv(output_dir / 'all_model_pair_metrics.csv', pair_rows)
    write_csv(output_dir / 'all_model_pair_layer_metrics.csv', layer_rows)
    persist_task_learned_reference(payload, output_dir)
    write_analysis(payload, output_dir)
    plot_report(payload, output_dir)
    drift.save_json(output_dir / 'status.json', {'state': 'complete', 'model_task_count': len(model_rows), 'model_pair_task_count': len(pair_rows)})
    print(f'Completed: {len(model_rows)} model/task results and {len(pair_rows)} model-pair/task comparisons -> {output_dir}', flush=True)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--tasks')
    parser.add_argument('--report_only', action='store_true', help='Rebuild reports from complete metric JSON; no model loading or inference')
    parser.add_argument('--output_dir')
    parser.add_argument('--source_eval_dir', help='Reuse this evaluation\'s exact validation probe manifest')
    parser.add_argument('--accuracy_matrix', help='Reuse a complete all-model/all-task test accuracy matrix')
    parser.add_argument('--probe_samples', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_input_length', type=int, default=1536)
    parser.add_argument('--max_new_tokens', type=int, default=8)
    parser.add_argument('--layers', default='all')
    parser.add_argument('--pooling', choices=['last', 'mean'], default='last')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu_threads', type=int, default=4)
    return parser


def plot_report(payload, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm

    tasks, models = payload['tasks'], payload['models']
    labels = [item['label'] for item in models]
    short = [item['model'] for item in models]
    stages = np.arange(len(models))
    accuracy = np.array(payload['accuracy']) * 100
    pairs = payload['pairwise_matrices']
    summary_layer = payload['summary_layer']
    order = 'Training order: ' + ' → '.join(tasks)
    counts = {row['evaluated_task']: row['validation_probe_count'] for row in payload['model_task_results']}
    figures = []

    def grid(ax, values, xlabels, ylabels, title, *, cmap='YlGnBu', lower=0, upper=None, fmt='.3f', signed=False):
        values = np.array([[np.nan if item is None else item for item in row] for row in values], dtype=float)
        colormap = plt.get_cmap(cmap).with_extremes(bad='#e2e8f0')
        options = {'vmin': lower, 'vmax': upper}
        if signed:
            bound = max(1.0, float(np.nanmax(np.abs(values))))
            options = {'norm': TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)}
        im = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap=colormap, **options)
        ax.set_xticks(range(len(xlabels)), xlabels, rotation=25, ha='right')
        ax.set_yticks(range(len(ylabels)), ylabels)
        ax.set_title(title, fontsize=12, pad=12)
        for row, column in np.ndindex(values.shape):
            value = values[row, column]
            text = 'N/A' if not np.isfinite(value) else format(value, fmt)
            rgba = im.cmap(im.norm(value)) if np.isfinite(value) else (0.9, 0.9, 0.9, 1)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(column, row, text, ha='center', va='center', fontsize=10,
                    color='white' if luminance < 0.48 else '#0f172a')
        ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
        return im

    def page(fig, name):
        fig.savefig(output_dir / f'{name}.png', dpi=180)
        fig.savefig(output_dir / f'{name}.pdf')
        figures.append(fig)

    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.axis('off')
        fig.suptitle('All models on every task: evaluation guide', fontsize=21, y=0.95)
        fig.text(0.07, 0.86, order, fontsize=13)
        table_rows = [[item['model'], ' → '.join(item['cumulative_tasks']) or 'Base model; no task training',
                       str(Path(item['adapter_dir']).relative_to(Path(payload['run_dir']))) if item['adapter_dir'] else 'No adapter'] for item in models]
        table = ax.table(cellText=table_rows, colLabels=['Model ID', 'Training history (cumulative)', 'Adapter relative to run directory'],
                         cellLoc='left', colLoc='left', colWidths=[0.1, 0.52, 0.38], bbox=[0, 0.48, 1, 0.42])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor('#cbd5e1')
            cell.set_facecolor('#e2e8f0' if row == 0 else ('#f8fafc' if row % 2 else 'white'))
            if row == 0:
                cell.set_text_props(weight='bold')
        lines = [
            f"Test accuracy: all {len(models)} models × all {len(tasks)} tasks, using the complete test sets.",
            f"Pairwise drift: {len(payload['pairwise_results'])} distinct model-pair/task comparisons, including unseen tasks.",
            'Validation probes per task: ' + ', '.join(f'{task}={counts[task]}' for task in tasks),
            f"Representations: hidden_states[{summary_layer}], pooling={payload['settings']['pooling']}; same examples for both models.",
            'Base overview: row model vs M0. Task-trained overview: row model vs the reference named in its column.',
            'CKA: higher = more similar. Cosine drift / symmetric KL: lower = more similar. Gray = undefined.',
            'KL: full vocabulary at the first answer position; symmetric KL = half the sum of both directions.',
            'Model IDs always refer to cumulative adapters, rather than independently trained task-only models.',
        ]
        for index, text in enumerate(lines):
            fig.text(0.07, 0.43 - index * 0.042, text, fontsize=11)
        page(fig, 'model_task_guide')

        fig, axes = plt.subplots(2, 2, figsize=(17, 12))
        fig.subplots_adjust(left=0.17, right=0.96, bottom=0.11, top=0.86, hspace=0.55, wspace=0.45)
        fig.suptitle(f'All {len(models)} models × all {len(tasks)} evaluation tasks', fontsize=20, y=0.97)
        fig.text(0.5, 0.93, order, ha='center', fontsize=12)
        grid(axes[0, 0], accuracy, tasks, labels, 'Complete test accuracy (%)', upper=100, fmt='.2f')
        for ax, key, title, upper in (
            (axes[0, 1], 'cka', f'CKA: row model vs M0 | layer {summary_layer}', 1),
            (axes[1, 0], 'rep_drift', f'Cosine drift: row model vs M0 | layer {summary_layer}', None),
            (axes[1, 1], 'symmetric_kl', 'Symmetric KL: row model vs M0 (nats)', None),
        ):
            values = [[pairs[key][task][0][stage] for task in tasks] for stage in stages]
            grid(ax, values, tasks, labels, title, upper=upper)
        fig.text(0.17, 0.025, 'Columns identify the data task. Accuracy uses test data; CKA/drift/KL use aligned validation probes. See guide for exact adapters.', fontsize=10)
        page(fig, 'all_model_task_overview')

        reference_labels = [f"{item['evaluated_task']}\nref: {item['reference_label']}" for item in payload['task_learned_references']]
        reference_accuracy = accuracy[np.arange(1, len(tasks) + 1), np.arange(len(tasks))]
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.subplots_adjust(left=0.17, right=0.96, bottom=0.16, top=0.85, hspace=0.75, wspace=0.48)
        fig.suptitle('Compare each model with the model just trained on the evaluated task', fontsize=19, y=0.97)
        fig.text(0.5, 0.92, order, ha='center', fontsize=12)
        grid(axes[0, 0], accuracy - reference_accuracy, reference_labels, labels,
             'Test accuracy change: row model minus task reference (pp)', cmap='RdBu', signed=True, fmt='+.2f')
        for ax, key, title, upper in (
            (axes[0, 1], 'cka', f'CKA: row model vs task reference | layer {summary_layer}', 1),
            (axes[1, 0], 'rep_drift', f'Cosine drift: row model vs task reference | layer {summary_layer}', None),
            (axes[1, 1], 'symmetric_kl', 'Symmetric KL: row model vs task reference (nats)', None),
        ):
            values = [[pairs[key][task][index + 1][stage] for index, task in enumerate(tasks)] for stage in stages]
            grid(ax, values, reference_labels, labels, title, upper=upper)
        fig.text(0.17, 0.065, 'Each column names its evaluation task and exact reference model. CKA/drift/KL use identical validation prompts.', fontsize=11)
        fig.text(0.17, 0.035, 'Models before the reference stage are pre-training controls. Only stages after the reference measure subsequent retention/forgetting.', fontsize=10)
        page(fig, 'task_learned_reference_overview')

        columns = min(2, len(tasks))
        row_count = (len(tasks) + columns - 1) // columns
        fig, axes = plt.subplots(row_count, columns, figsize=(16, max(5, row_count * 4.5)), squeeze=False)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.85, hspace=0.65, wspace=0.22)
        fig.suptitle('Test accuracy throughout sequential training', fontsize=20, y=0.97)
        fig.text(0.5, 0.92, order, ha='center', fontsize=12)
        for index, task in enumerate(tasks):
            ax = axes.flat[index]
            ax.plot(stages, accuracy[:, index], '-o', color='#2563eb', linewidth=2, markersize=7)
            ax.axvline(index + 1, color='#d97706', linestyle='--', linewidth=1.4, label=f'{task} training completed')
            ax.axhline(accuracy[0, index], color='#64748b', linestyle=':', label='Base accuracy')
            for stage, value in enumerate(accuracy[:, index]):
                ax.annotate(f'{value:.2f}', (stage, value), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
            ax.set_ylim(0, 108)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xticks(stages, labels, rotation=25, ha='right', fontsize=9)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Test data: {task}', loc='left', fontsize=13)
            ax.grid(axis='y', color='#e2e8f0')
            ax.legend(fontsize=9, loc='lower left')
        for ax in axes.flat[len(tasks):]:
            ax.set_visible(False)
        page(fig, 'all_model_accuracy_trajectories')

        for task in tasks:
            fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
            fig.subplots_adjust(left=0.10, right=0.97, bottom=0.23, top=0.74, wspace=0.45)
            fig.suptitle(f'Validation task: {task} | all model pairs', fontsize=20, y=0.97)
            fig.text(0.5, 0.89, f'Same {counts[task]} prompts for every model | hidden-state layer {summary_layer}', ha='center', fontsize=12)
            for ax, key, title, upper in (
                (axes[0], 'cka', 'Linear CKA (higher = similar)', 1),
                (axes[1], 'rep_drift', 'Cosine drift (lower = similar)', None),
                (axes[2], 'symmetric_kl', 'Symmetric KL, nats (lower = similar)', None),
            ):
                grid(ax, pairs[key][task], short, labels, title, upper=upper)
                ax.set_xlabel('Second model (see guide for exact adapter)')
            axes[0].set_ylabel('First model')
            fig.text(0.10, 0.045, 'Each cell compares its row model and column model on this task only. Diagonal = self-comparison; matrices are symmetric.', fontsize=11)
            page(fig, f'pairwise_{task}')

        fig, axes = plt.subplots(row_count, columns, figsize=(18, max(6, row_count * 4.7)), squeeze=False)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.10, top=0.85, hspace=0.70, wspace=0.4)
        fig.suptitle('Layerwise CKA: each saved model vs M0, separately for each task', fontsize=18, y=0.97)
        layers = payload['resolved_layers']
        for index, task in enumerate(tasks):
            ax = axes.flat[index]
            values = [[payload['layer_cka_matrices'][str(layer)][task][0][stage] for layer in layers] for stage in stages]
            # Avoid labelling every layer when all 33 hidden states were requested.
            grid(ax, values, [''] * len(layers), labels, f'Validation data: {task}', upper=1)
            for text in list(ax.texts):
                text.remove()
            ticks = list(range(0, len(layers), max(1, len(layers) // 8)))
            if ticks[-1] != len(layers) - 1:
                ticks.append(len(layers) - 1)
            ax.set_xticks(ticks, [layers[tick] for tick in ticks], rotation=0)
            ax.set_xlabel('hidden_states index (0 = embedding output)')
        for ax in axes.flat[len(tasks):]:
            ax.set_visible(False)
        fig.text(0.14, 0.025, 'Each cell compares the row model with M0 at the same layer on the named validation task. Gray = zero-variance / undefined CKA.', fontsize=10)
        page(fig, 'all_model_layerwise_cka')

        fig, axes = plt.subplots(row_count, columns, figsize=(18, max(6, row_count * 4.7)), squeeze=False)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.10, top=0.85, hspace=0.70, wspace=0.4)
        fig.suptitle("Layerwise CKA: each model vs the evaluated task's just-trained model", fontsize=18, y=0.97)
        for index, task in enumerate(tasks):
            ax = axes.flat[index]
            reference = models[index + 1]
            values = [[payload['layer_cka_matrices'][str(layer)][task][index + 1][stage] for layer in layers] for stage in stages]
            grid(ax, values, [''] * len(layers), labels,
                 f"Validation data: {task} | reference: {reference['label']}", upper=1)
            for text in list(ax.texts):
                text.remove()
            ticks = list(range(0, len(layers), max(1, len(layers) // 8)))
            if ticks[-1] != len(layers) - 1:
                ticks.append(len(layers) - 1)
            ax.set_xticks(ticks, [layers[tick] for tick in ticks], rotation=0)
            ax.set_xlabel('hidden_states index (0 = embedding output)')
        for ax in axes.flat[len(tasks):]:
            ax.set_visible(False)
        fig.text(0.14, 0.025, 'Reference changes with the named validation task. Same task/layer for both models; gray = undefined CKA.', fontsize=10)
        page(fig, 'task_learned_reference_layerwise_cka')

        with PdfPages(output_dir / 'all_model_task_report.pdf') as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)


if __name__ == '__main__':
    options = build_parser().parse_args()
    try:
        run_evaluation(options)
    except Exception as error:
        destination = Path(options.output_dir) if options.output_dir else Path(options.run_dir) / 'evaluation_all_models'
        drift.save_json(destination / 'status.json', {'state': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise

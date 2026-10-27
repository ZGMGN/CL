"""Analyze sequential transitions with KL, cosine drift, linear CKA and LoRA updates.

For every i < j, compare M_(j-1) with M_j on identical validation probes from T_i.
Gradient cosine compares T_i and T_j at M_(j-1), in the same active LoRA coordinates.
LoRA cosine compares task increments U_i = W_i-W_(i-1) and U_j = W_j-W_(j-1).
This does not claim to reconstruct an independently trained T_j-only model.
"""
import argparse
import csv
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from peft import PeftModel

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / 'src'))

from continual_learning.train_single_task import AnswerOnlyDataCollator, _format_answer_only_pairs
from evaluation import sequential_heatmap_eval as heatmap
from evaluation.drift_metrics import (
    cosine, distribution_drift, linear_cka, load_lora_factors,
    lora_update_cosine, representation_cosine_drift,
)
from evaluation.general_task_eval import build_eval_model
from model.utils import GENERAL_TEMPLATE, set_seed


def resolve_layers(value, count):
    layers = list(range(count)) if value.lower() == 'all' else [int(x.strip()) for x in value.split(',')]
    layers = [index if index >= 0 else count + index for index in layers]
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= index < count for index in layers):
        raise ValueError(f'Layers must be unique hidden-state indices in 0..{count - 1}')
    return layers
#把层转换为数字索引列表

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    #这里创建父目录
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    #把文本写入path,dumps是转为json文件，dump是写成json文件

def prepare_probes(tasks, args, output_dir):
    probes, modules, manifest = {}, {}, {}
    for index, task in enumerate(tasks):
        path = ROOT_DIR / 'data' / task / 'val.jsonl'
        dataset = heatmap.load_dataset('json', data_files=str(path), split='train')
        if len(dataset) < 3:
            raise ValueError(f'{task}: at least 3 validation examples are required for CKA')
        indices = sorted(np.random.default_rng(args.seed + index).choice(
            len(dataset), size=min(args.probe_samples, len(dataset)), replace=False
        ).tolist())
        probes[task] = [dataset[i] for i in indices]
        modules[task] = importlib.import_module(f'tasks.{task}')
        manifest[task] = {
            'path': str(path), 'split': 'val', 'indices': indices,
            'examples_sha256': hashlib.sha256(json.dumps(probes[task], sort_keys=True).encode()).hexdigest(),
        }
        #manifest留取样本指纹，分为三步
        #1.转为json
        #2.转为字节，默认使用utf-8
        #3.计算 SHA-256，并转换成十六进制字符串
    save_json(output_dir / 'probe_manifest.json', manifest)
    return probes, modules


@torch.inference_mode()
def collect_outputs(model, tokenizer, examples, task_module, layers, args, device):
    """Keep prompt-only pooled states and the first answer-position vocabulary distribution."""
    model.eval()
    features = {layer: [] for layer in layers}
    log_probs = []
    for start in range(0, len(examples), args.batch_size):
        prompts = [task_module.format_prompt(row, tokenizer, include_answer=False)
                   for row in examples[start:start + args.batch_size]]
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,
                           max_length=args.max_input_length, add_special_tokens=False).to(device)
        output = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        #logits_to_keep=1表示只计算最后一个位置的logit，因为(batch_size,sequence_length,1,vocab_size)input的每个token都会计算一次logit
        log_probs.append(output.logits[:, -1].float().log_softmax(-1).cpu())
        #先用logits[:,-1]把[B,1,V]->[B,V],然后.float()转为浮点数，最后按最后一维计算softmax分数，然后再转到cpu上
        for layer in layers:
        #遍历需要收集表征的层
            hidden = output.hidden_states[layer]
            #B,L,H
            if args.pooling == 'last':
                pooled = hidden[:, -1]
            #last pooling就是取整个输入的最后一个token的hidden向量作为表征
            #B,L,H -> B,H
            else:
            #不用last就用mean
                mask = inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                #B,L -> B,L,1 为了和hidden进行广播运算
                pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
                #B,L,H sum后转为 B,H 再除mask.sum(1)(shape:B,1) 得到 B,H
            features[layer].append(pooled.float().cpu())
        del output
    return {'features': {layer: torch.cat(rows) for layer, rows in features.items()},
            'log_probs': torch.cat(log_probs)}
            #返回表征计算和散度计算输入


def mean_lora_gradient(model, tokenizer, examples, task_module, args, device):
    """Token-averaged answer-only loss gradient, eval mode, without any optimizer step."""
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    was_training = model.training
    try:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('lora_A.' in name or 'lora_B.' in name)
            #如果name里还有lora_A，或者lora_B，就开启梯度计算
        selected = sorted((name, p) for name, p in model.named_parameters() if p.requires_grad)
        #统计需要计算的参数
        if not selected:
            raise ValueError('No LoRA parameters available for gradient cosine')
        model.eval()  # disable dropout for repeatable task-gradient diagnostics
        #关闭dropout
        model.zero_grad(set_to_none=True)
        #梯度清零
        collator = AnswerOnlyDataCollator(
            tokenizer, args.max_input_length,
            lambda batch: _format_answer_only_pairs(batch, tokenizer, task_module.example_to_messages),
        )
        total_tokens = 0
        for example in examples[:args.grad_samples]:
            #只计算grad_samples个
            batch = {key: value.to(device) for key, value in collator([example]).items()}
            tokens = int((batch['labels'][:, 1:] != -100).sum())
            #去掉标签的第一个位置。因为因果语言模型是当前位置预测下一个位置，第一个标签位置没有前一个位置来预测它。
            if not tokens:
                raise ValueError('Gradient probe has no answer targets')
            loss = model(**batch, use_cache=False).loss
            #这里返回的是mean cross entropy
            (loss * tokens).backward()
            #平均乘数量 计算总的，backward只计算grad 不反向传播
            total_tokens += tokens
        vector = torch.cat([
            (torch.zeros_like(p) if p.grad is None else p.grad).detach().float().cpu().flatten()
            for _, p in selected
        ]) / total_tokens
        if not torch.isfinite(vector).all():
            raise ValueError('Nonfinite diagnostic gradient')
        return vector
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(flags[name])
        model.train(was_training)


def load_accuracy(args, tasks, output_dir):
    if args.accuracy_matrix:
        payload = json.loads(Path(args.accuracy_matrix).read_text())
    else:
        payload = heatmap.run_evaluation(argparse.Namespace(
            run_dir=args.run_dir, tasks=','.join(tasks), output_dir=str(output_dir / 'accuracy'),
            batch_size=args.eval_batch_size, max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens, seed=args.seed,
        ))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not payload.get('complete') or payload['tasks'] != tasks:
        raise ValueError('Accuracy matrix must be complete and have the identical task order')
    matrix = np.array(payload['accuracy'], dtype=float)
    if matrix.shape != (len(tasks) + 1, len(tasks)) or not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
        raise ValueError('Expected a finite (n+1) x n accuracy matrix in [0,1]')
    settings = payload.get('evaluation', {})
    for key in ('max_input_length', 'max_new_tokens'):
        if settings.get(key) != getattr(args, key):
            raise ValueError(f'Accuracy matrix {key} differs from this run')
    expected = [None, *[str(path.resolve()) for path in heatmap.find_adapters(Path(args.run_dir), tasks)]]
    actual = payload.get('adapter_dirs')
    if actual is None or [None if path is None else str(Path(path).resolve()) for path in actual] != expected:
        raise ValueError('Accuracy matrix adapter paths do not match the current run')
    return matrix


def persist_results(output_dir, metadata, rows):
    save_json(output_dir / 'transition_metrics.json', {**metadata, 'transitions': rows})
    if rows:
        with (output_dir / 'transition_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = [key for key in rows[0] if key != 'layers']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        with (output_dir / 'layer_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['transition', 'before_stage', 'after_stage', 'layer', 'cosine_drift', 'cka', 'cka_drift'])
            writer.writeheader()
            for row in rows:
                for layer, values in row['layers'].items():
                    writer.writerow({'transition': row['transition'], 'before_stage': row['before_stage'],
                                     'after_stage': row['after_stage'], 'layer': layer, **values})



def build_adjacent_comparison(stage, tasks, task_index, accuracy, layer_values, functional,
                              summary_layer, grad_cosine, lora_cosine, probe_count):
    """One fixed task evaluated before/after the current training stage."""
    if task_index not in (stage - 1, stage - 2) or not 0 <= task_index < len(tasks):
        raise ValueError('Adjacent report accepts only the current or immediately previous task')
    role = 'current' if task_index == stage - 1 else 'previous'
    before = float(accuracy[stage - 1, task_index])
    after = float(accuracy[stage, task_index])
    change = 100 * (after - before)
    summary = layer_values[str(summary_layer)]
    return {
        'trained_task': tasks[stage - 1], 'evaluated_task': tasks[task_index], 'task_role': role,
        'before_stage': stage - 1, 'after_stage': stage,
        'accuracy_before': before, 'accuracy_after': after, 'accuracy_change_pp': change,
        'learning_gain_pp': change if role == 'current' else None,
        'forgetting_pp': -change if role == 'previous' else None,
        'grad_cosine': grad_cosine, 'lora_cosine': lora_cosine,
        'rep_drift': summary['cosine_drift'], 'cka': summary['cka'], 'cka_drift': summary['cka_drift'],
        'func_drift': functional['symmetric_kl'], **functional,
        'summary_layer': summary_layer, 'probe_count': probe_count, 'layers': layer_values,
    }


def persist_adjacent_results(output_dir, metadata, rows):
    save_json(output_dir / 'adjacent_task_metrics.json', {
        **metadata,
        'comparison_definition': 'For training Tj, compare M_(j-1) and M_j on each of Tj and T(j-1), separately.',
        'accuracy_change_definition': '100*(accuracy_after-accuracy_before); positive means improvement.',
        'pair_metric_definition': 'Gradient/LoRA cosines describe the previous/current task pair and are shared by its two rows; first stage is null.',
        'comparisons': rows,
    })
    if not rows:
        return
    with (output_dir / 'adjacent_task_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key in rows[0] if key != 'layers'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'adjacent_layer_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['trained_task', 'evaluated_task', 'task_role', 'before_stage', 'after_stage', 'layer', 'cosine_drift', 'cka', 'cka_drift']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for layer, values in row['layers'].items():
                writer.writerow({**{key: row[key] for key in fields[:5]}, 'layer': layer, **values})


def plot_adjacent_comparisons(rows, output_dir, run_dir=None):
    """Readable two-page report: accuracy changes, then representation/function drift."""
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    positions = np.arange(count)
    stage_tasks = {row['after_stage']: row['trained_task'] for row in rows}
    stage_order = [stage_tasks[index] for index in sorted(stage_tasks)]
    palette = {'YAHOO': '#2563eb', 'AMAZON': '#d97706', 'AGNEWS': '#7c3aed',
               'DBPEDIA': '#059669', 'YELP': '#dc2626', 'SST2': '#0891b2', 'QQP': '#9333ea'}
    colors = [palette.get(row['evaluated_task'], '#475569') for row in rows]

    def model_label(index):
        return 'M0 (Base)' if index == 0 else f'M{index} (after {stage_tasks[index]})'

    labels = [f"{model_label(row['before_stage'])} → {model_label(row['after_stage'])}\nTest data: {row['evaluated_task']}" for row in rows]
    validation_labels = [f"{model_label(row['before_stage'])} ↔ {model_label(row['after_stage'])}\nValidation data: {row['evaluated_task']}" for row in rows]
    task_legend = [Line2D([], [], marker='o', linestyle='none', color=palette.get(task, '#475569'),
                          label=f'Data: {task}', markersize=7) for task in stage_order]
    order_label = 'Training order: ' + '  →  '.join(stage_order)
    height = max(5.5, 0.85 * count + 2.3)
    legend = [
        Line2D([], [], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor='#64748b', label='Earlier model: open circle', markersize=7),
        Line2D([], [], marker='o', linestyle='none', color='#334155', label='Later model: filled circle', markersize=7),
    ]
    style = {'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
             'xtick.labelsize': 10, 'ytick.labelsize': 10, 'pdf.fonttype': 42}

    def decorate(ax, show_labels, split='test'):
        begin = 0
        for index in range(1, count + 1):
            if index == count or rows[index]['after_stage'] != rows[begin]['after_stage']:
                if rows[begin]['after_stage'] % 2 == 0:
                    ax.axhspan(begin - 0.5, index - 0.5, color='#f1f5f9', zorder=0)
                if index < count:
                    ax.axhline(index - 0.5, color='#cbd5e1', linewidth=0.9, zorder=0)
                begin = index
        ax.set_ylim(count - 0.5, -0.5)
        ax.set_yticks(positions, labels if split == 'test' else validation_labels)
        ax.tick_params(axis='y', length=0, pad=12, labelleft=show_labels)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e2e8f0', linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    with plt.rc_context(style):
        accuracy_fig, (accuracy_ax, change_ax) = plt.subplots(
            1, 2, figsize=(16, height), gridspec_kw={'width_ratios': [2.4, 1.25]}, sharey=True
        )
        accuracy_fig.subplots_adjust(left=0.34, right=0.98, bottom=0.14, top=0.72, wspace=0.22)
        accuracy_fig.suptitle('Test accuracy: explicit model pairs and evaluation tasks', fontsize=17, x=0.50, y=0.975)
        accuracy_fig.text(0.5, 0.93, order_label, ha='center', color='#475569', fontsize=11)
        accuracy_fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.65, 0.90), ncol=2, frameon=False, fontsize=10)
        accuracy_fig.legend(handles=task_legend, loc='upper center', bbox_to_anchor=(0.65, 0.85), ncol=min(4, len(task_legend)), frameon=False, fontsize=10)
        before = np.array([row['accuracy_before'] * 100 for row in rows])
        after = np.array([row['accuracy_after'] * 100 for row in rows])
        change = np.array([row['accuracy_change_pp'] for row in rows])
        accuracy_ax.set_title('Earlier model vs later model', loc='left', pad=17)
        accuracy_ax.text(106, -0.85, 'Earlier → Later', color='#475569', fontsize=10, clip_on=False)
        accuracy_ax.hlines(positions, before, after, color='#94a3b8', linewidth=3, zorder=2)
        accuracy_ax.scatter(before, positions, facecolors='white', edgecolors='#64748b', s=66, linewidths=1.5, zorder=3)
        accuracy_ax.scatter(after, positions, c=colors, s=66, zorder=4)
        for index, (old, new) in enumerate(zip(before, after)):
            accuracy_ax.text(106, index, f'{old:.2f} → {new:.2f}', va='center', fontsize=11, color='#0f172a')
        accuracy_ax.set_xlim(0, 144)
        accuracy_ax.set_xticks([0, 25, 50, 75, 100])
        accuracy_ax.set_xlabel('Accuracy (%)')
        decorate(accuracy_ax, True)
        limit = max(5.0, float(np.max(np.abs(change))) * 1.3)
        change_colors = ['#0f766e' if value >= 0 else '#dc2626' for value in change]
        change_ax.barh(positions, change, color=change_colors, height=0.38, zorder=2)
        change_ax.axvline(0, color='#64748b', linewidth=1.2, zorder=1)
        for index, value in enumerate(change):
            change_ax.text(value + (limit * 0.035 if value >= 0 else -limit * 0.035), index,
                           f'{value:+.2f}', ha='left' if value >= 0 else 'right', va='center',
                           color=change_colors[index], fontsize=11, fontweight='bold')
        change_ax.set_xlim(-limit, limit)
        change_ax.set_title('Accuracy change', loc='left', pad=17)
        change_ax.set_xlabel('Percentage points (after - before)')
        decorate(change_ax, False)
        accuracy_fig.text(0.34, 0.045, 'M0 = base. Mj = base + the saved cumulative stage-j adapter. Exact adapter mapping is on page 3.',
                          fontsize=10, color='#475569')

        drift_fig, axes = plt.subplots(1, 3, figsize=(16, height + 0.8), sharey=True)
        drift_fig.subplots_adjust(left=0.34, right=0.98, bottom=0.15, top=0.72, wspace=0.30)
        drift_fig.suptitle('Model-pair similarity and drift on the named validation task', fontsize=16, y=0.975)
        drift_fig.text(0.5, 0.93, order_label, ha='center', color='#475569', fontsize=11)
        drift_fig.legend(handles=task_legend, loc='upper center', bbox_to_anchor=(0.66, 0.88),
                         ncol=min(4, len(task_legend)), frameon=False, fontsize=10)
        specifications = [
            ('cka', 'Linear CKA', 'Higher = more similar', 'Similarity', 1.10),
            ('rep_drift', 'Representation cosine drift', 'Higher = more change', 'Cosine distance (observed range)', None),
            ('func_drift', 'Output distribution drift', 'Higher = more change', 'Symmetric KL (nats)', None),
        ]
        for column, (ax, (key, title, direction, xlabel, fixed_limit)) in enumerate(zip(axes, specifications)):
            values = np.array([np.nan if row[key] is None else row[key] for row in rows], dtype=float)
            finite = np.isfinite(values)
            observed = float(np.max(values[finite])) if finite.any() else 0.0
            upper = fixed_limit if fixed_limit is not None else max(0.05, observed * 1.28)
            ax.set_xlim(0, upper)
            for index, value in enumerate(values):
                if not np.isfinite(value):
                    ax.text(upper * 0.04, index, 'N/A', va='center', color='#64748b')
                    continue
                ax.barh(index, value, color=colors[index], alpha=0.15, height=0.38, zorder=1)
                ax.scatter(value, index, color=colors[index], s=52, zorder=3)
                ax.text(value + upper * 0.035, index, f'{value:.3f}', va='center', fontsize=11, color='#0f172a')
            ax.set_title(title + '\n' + direction, loc='left', pad=14, fontsize=12, linespacing=1.4)
            ax.set_xlabel(xlabel)
            if key == 'cka':
                ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
            decorate(ax, column == 0, split='validation')
        layers = sorted(set(row['summary_layer'] for row in rows))
        counts = sorted(set(row['probe_count'] for row in rows))
        drift_fig.text(0.34, 0.055, f"Hidden-state layer: {', '.join(map(str, layers))} | Validation probes/task: {', '.join(map(str, counts))}",
                       fontsize=10, color='#475569')
        drift_fig.text(0.34, 0.025, 'Same validation examples within each model pair. KL uses the first answer token. All adapters are cumulative; see page 3.',
                       fontsize=9.5, color='#475569')

        model_mapping = [{'model': 'M0', 'cumulative_tasks': [], 'adapter_dir': None}]
        for index in sorted(stage_tasks):
            relative = f'{index:02d}_{stage_tasks[index]}/best_model'
            model_mapping.append({
                'model': f'M{index}', 'cumulative_tasks': [stage_tasks[i] for i in sorted(stage_tasks) if i <= index],
                'adapter_dir': str((Path(run_dir) / relative).resolve()) if run_dir is not None else relative,
            })
        save_json(output_dir / 'model_adapter_mapping.json', {
            'base_model': str(ROOT_DIR / 'Llama-3.1-8B-Instruct'),
            'run_dir': None if run_dir is None else str(Path(run_dir).resolve()),
            'models': model_mapping,
            'comparison': 'Each row names two models and a single fixed evaluation task; Mj uses the saved cumulative stage-j adapter.',
        })
        with (output_dir / 'model_adapter_mapping.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['model', 'cumulative_tasks', 'adapter_dir'])
            writer.writeheader()
            writer.writerows([{**item, 'cumulative_tasks': ' → '.join(item['cumulative_tasks'])} for item in model_mapping])
        mapping_fig, mapping_ax = plt.subplots(figsize=(16, max(5.6, len(model_mapping) * 0.7 + 2)))
        mapping_ax.axis('off')
        mapping_fig.suptitle('Which model and adapter does each identifier mean?', fontsize=18, y=0.96)
        mapping_fig.text(0.08, 0.85, 'All models share the same base weights. Each saved adapter includes the full training history shown below.', fontsize=12)
        table_rows = [['M0', 'No task training (base model)', 'No adapter']]
        for index in sorted(stage_tasks):
            table_rows.append([f'M{index}', ' → '.join(stage_tasks[i] for i in sorted(stage_tasks) if i <= index),
                               f'{index:02d}_{stage_tasks[index]}/best_model'])
        table = mapping_ax.table(cellText=table_rows,
            colLabels=['Model', 'Training completed (cumulative)', 'Saved adapter / relative path'],
            cellLoc='left', colLoc='left', colWidths=[0.09, 0.49, 0.42], bbox=[0.02, 0.20, 0.96, 0.56])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row_index, _), cell in table.get_celld().items():
            cell.set_edgecolor('#cbd5e1')
            cell.set_facecolor('#e2e8f0' if row_index == 0 else ('#f8fafc' if row_index % 2 else 'white'))
            if row_index == 0:
                cell.set_text_props(weight='bold')
        mapping_fig.text(0.08, 0.13, 'Example: M1 ↔ M2, Validation data: YAHOO compares both models on exactly the same YAHOO prompts.', fontsize=11)
        mapping_fig.text(0.08, 0.075, 'Full paths and cumulative task lists are saved in model_adapter_mapping.json and .csv.', fontsize=10, color='#475569')
        mapping_fig.savefig(output_dir / 'model_adapter_mapping.png', dpi=220)

        accuracy_fig.savefig(output_dir / 'adjacent_task_metrics.png', dpi=220)
        accuracy_fig.savefig(output_dir / 'adjacent_accuracy_metrics.pdf')
        drift_fig.savefig(output_dir / 'adjacent_representation_metrics.png', dpi=220)
        drift_fig.savefig(output_dir / 'adjacent_representation_metrics.pdf')
        with PdfPages(output_dir / 'adjacent_task_metrics.pdf') as pdf:
            pdf.savefig(accuracy_fig)
            pdf.savefig(drift_fig)
            pdf.savefig(mapping_fig)
        plt.close(accuracy_fig)
        plt.close(drift_fig)
        plt.close(mapping_fig)


def plot_transitions(rows, output_dir):
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [('grad_cosine', 'Gradient cosine', -1, 1), ('lora_cosine', 'LoRA update cosine', -1, 1),
               ('rep_drift', 'Representation cosine drift', 0, 2), ('cka', 'Linear CKA', 0, 1),
               ('func_drift', 'Symmetric KL (nats)', 0, None), ('forgetting_pp', 'Forgetting (pp)', None, None)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, max(4, len(rows) * 0.55 + 2)), sharey=True)
    for ax, (key, label, low, high) in zip(axes, metrics):
        values = np.array([np.nan if row[key] is None else row[key] for row in rows]).reshape(-1, 1)
        im = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap='coolwarm', vmin=low, vmax=high)
        ax.set_xticks([0], [label], rotation=45, ha='right')
        ax.set_yticks(range(len(rows)), [row['transition'] for row in rows])
        for index, value in enumerate(values[:, 0]):
            ax.text(0, index, 'N/A' if np.isnan(value) else f'{value:.3f}', ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle('Sequential task transitions (each metric has its own scale)')
    fig.tight_layout()
    fig.savefig(output_dir / 'transition_metrics.png', dpi=200)
    fig.savefig(output_dir / 'transition_metrics.pdf')
    plt.close(fig)



def plot_model_matrices(matrices, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key, title, vmax in (
        ('cka_vs_base', 'Linear CKA vs base model', 1),
        ('rep_drift_vs_base', 'Representation cosine drift vs base model', 2),
        ('symmetric_kl_vs_base', 'Symmetric KL vs base model (nats)', None),
    ):
        values = np.array([[np.nan if value is None else value for value in row]
                           for row in matrices[key]], dtype=float)
        fig, ax = plt.subplots(figsize=(max(7, len(matrices['tasks']) * 1.5), max(4, len(values) * 0.7 + 1.5)))
        im = ax.imshow(np.ma.masked_invalid(values), cmap='YlGnBu', aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(matrices['tasks'])), matrices['tasks'], rotation=30, ha='right')
        ax.set_yticks(range(len(matrices['models'])), matrices['models'])
        ax.set_title(title + f" (hidden-state layer {matrices['summary_layer']})")
        ax.set_xlabel('Validation probe task')
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, 'N/A' if np.isnan(values[i, j]) else f'{values[i, j]:.3f}', ha='center', va='center', fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / f'{key}_heatmap.png', dpi=200)
        fig.savefig(output_dir / f'{key}_heatmap.pdf')
        plt.close(fig)


def run_evaluation(args):
    if min(args.batch_size, args.eval_batch_size, args.grad_samples, args.max_input_length, args.max_new_tokens) <= 0 or args.probe_samples < 3:
        raise ValueError('Positive batch/token/gradient limits and at least 3 probe samples are required')
    tasks = heatmap.resolve_tasks(Path(args.run_dir), args.tasks)
    if len(tasks) < 2 or len(set(tasks)) != len(tasks):
        raise ValueError('Transition analysis requires at least two distinct, nonrepeated tasks')
    adapters = heatmap.find_adapters(Path(args.run_dir), tasks)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.run_dir) / 'evaluation_drift'
    output_dir.mkdir(parents=True, exist_ok=True)
    probes, modules = prepare_probes(tasks, args, output_dir)
    # Factor-space products compute effective BA updates without allocating dense 8B-model deltas.
    factors = [{}] + [load_lora_factors(path) for path in adapters]
    if any(set(stage) != set(factors[1]) for stage in factors[2:]):
        raise ValueError('All stages must use identical LoRA target modules')
    accuracy = load_accuracy(args, tasks, output_dir)
    set_seed(args.seed)
    model, tokenizer, device, _ = build_eval_model(argparse.Namespace(peft_model_folder=None), GENERAL_TEMPLATE)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    layers = resolve_layers(args.layers, model.config.num_hidden_layers + 1)
    metadata = {
        'tasks': tasks, 'complete': False, 'settings': vars(args), 'resolved_layers': layers,
        'definitions': {
            'reference': 'For Ti -> Tj (i<j), compare M_(j-1) to M_j on fixed Ti validation probes.',
            'gradient': 'Token-mean answer-only loss gradients for Ti and Tj at M_(j-1), active LoRA only, eval mode.',
            'functional': 'First-answer-position full-vocabulary KL in nats; symmetric KL is half the sum of both directions.',
            'representation': 'Mean sample cosine distance and biased centered linear CKA on aligned prompt representations.',
            'lora': 'Global Frobenius cosine of incremental effective updates Ui and Uj; Ui=Wi-W(i-1).',
            'forgetting': '100*(test_accuracy(M_(j-1),Ti)-test_accuracy(M_j,Ti)); negative values mean improvement.',
            'cka_reference': 'https://proceedings.mlr.press/v97/kornblith19a.html',
            'undefined': 'Zero-norm cosine or zero-variance CKA is null, never replaced with zero.',
        },
    }
    rows = []
    adjacent_rows = []
    previous = {task: collect_outputs(model, tokenizer, probes[task], modules[task], layers, args, device) for task in tasks}
    baseline = previous
    matrices = {
        'tasks': tasks, 'models': ['Base model'] + [f'{index:02d} After {task}' for index, task in enumerate(tasks, 1)],
        'summary_layer': layers[-1], 'accuracy': accuracy.tolist(), 'complete': False,
        'cka_vs_base': [[linear_cka(baseline[task]['features'][layers[-1]], baseline[task]['features'][layers[-1]]) for task in tasks]],
        'rep_drift_vs_base': [[0.0] * len(tasks)],
        'symmetric_kl_vs_base': [[0.0] * len(tasks)],
        'layer_cka_vs_base': {str(layer): [[linear_cka(baseline[task]['features'][layer], baseline[task]['features'][layer]) for task in tasks]] for layer in layers},
    }
    active = None
    for stage, task in enumerate(tasks, 1):
        gradient_cosines = {}
        if stage > 1 and not args.skip_grad_cosine:
            print(f'Stage {stage}: diagnostic gradients at stage {stage - 1}', flush=True)
            new_gradient = mean_lora_gradient(model, tokenizer, probes[task], modules[task], args, device)
            for old_task in tasks[:stage - 1]:
                old_gradient = mean_lora_gradient(model, tokenizer, probes[old_task], modules[old_task], args, device)
                gradient_cosines[old_task] = cosine(old_gradient, new_gradient)
                del old_gradient
            del new_gradient
        name = f'drift_stage_{stage}'
        if active is None:
            model = PeftModel.from_pretrained(model, str(adapters[stage - 1]), adapter_name=name, is_trainable=False)
        else:
            model.load_adapter(str(adapters[stage - 1]), adapter_name=name, is_trainable=False)
        model.set_adapter(name)
        if active is not None:
            model.delete_adapter(active)
        active = name
        model.eval()
        current = {}
        for index, old_task in enumerate(tasks):
            print(f'Stage {stage}/{len(tasks)}: probes for {old_task}', flush=True)
            current[old_task] = collect_outputs(model, tokenizer, probes[old_task], modules[old_task], layers, args, device)
            if index > stage - 1:
                continue
            before, after = previous[old_task], current[old_task]
            layer_values = {}
            for layer in layers:
                cka = linear_cka(before['features'][layer], after['features'][layer])
                layer_values[str(layer)] = {
                    'cosine_drift': representation_cosine_drift(before['features'][layer], after['features'][layer]),
                    'cka': cka, 'cka_drift': None if cka is None else 1 - cka,
                }
            functional = distribution_drift(before['log_probs'], after['log_probs'])
            last = layer_values[str(layers[-1])]
            if index in (stage - 2, stage - 1):
                pair_previous_task = tasks[stage - 2] if stage > 1 else None
                pair_lora_cosine = (
                    lora_update_cosine(factors[stage - 1], factors[stage - 2], factors[stage], factors[stage - 1])
                    if stage > 1 else None
                )
                adjacent_rows.append(build_adjacent_comparison(
                    stage, tasks, index, accuracy, layer_values, functional, layers[-1],
                    gradient_cosines.get(pair_previous_task), pair_lora_cosine, len(probes[old_task]),
                ))
                persist_adjacent_results(output_dir, metadata, adjacent_rows)
            if index == stage - 1:
                continue
            row = {
                'transition': f'{old_task} -> {task}', 'old_task': old_task, 'new_task': task,
                'before_stage': stage - 1, 'after_stage': stage,
                'grad_cosine': gradient_cosines.get(old_task),
                'lora_cosine': lora_update_cosine(factors[index + 1], factors[index], factors[stage], factors[stage - 1]),
                'rep_drift': last['cosine_drift'], 'cka': last['cka'], 'cka_drift': last['cka_drift'],
                'summary_layer': layers[-1], 'func_drift': functional['symmetric_kl'], **functional,
                'accuracy_before': float(accuracy[stage - 1, index]),
                'accuracy_after': float(accuracy[stage, index]),
                'forgetting_pp': float(100 * (accuracy[stage - 1, index] - accuracy[stage, index])),
                'cumulative_forgetting_pp': float(100 * (accuracy[index + 1, index] - accuracy[stage, index])),
                'probe_count': len(probes[old_task]), 'gradient_probe_count': 0 if args.skip_grad_cosine else min(args.grad_samples, len(probes[old_task])),
                'layers': layer_values,
            }
            rows.append(row)
            persist_results(output_dir, metadata, rows)
        for layer in layers:
            matrices['layer_cka_vs_base'][str(layer)].append([
                linear_cka(baseline[probe_task]['features'][layer], current[probe_task]['features'][layer])
                for probe_task in tasks
            ])
        matrices['cka_vs_base'].append(matrices['layer_cka_vs_base'][str(layers[-1])][-1])
        matrices['rep_drift_vs_base'].append([
            representation_cosine_drift(baseline[probe_task]['features'][layers[-1]], current[probe_task]['features'][layers[-1]])
            for probe_task in tasks
        ])
        matrices['symmetric_kl_vs_base'].append([
            distribution_drift(baseline[probe_task]['log_probs'], current[probe_task]['log_probs'])['symmetric_kl']
            for probe_task in tasks
        ])
        save_json(output_dir / 'model_drift_matrices.json', matrices)
        previous = current
    matrices['complete'] = True
    save_json(output_dir / 'model_drift_matrices.json', matrices)
    plot_model_matrices(matrices, output_dir)
    metadata['complete'] = True
    persist_adjacent_results(output_dir, metadata, adjacent_rows)
    plot_adjacent_comparisons(adjacent_rows, output_dir, run_dir=args.run_dir)
    persist_results(output_dir, metadata, rows)
    plot_transitions(rows, output_dir)
    print(f'Completed {len(rows)} transitions: {output_dir}', flush=True)
    return {**metadata, 'transitions': rows}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--tasks', help='Training order; inferred from run metadata by default')
    parser.add_argument('--output_dir')
    parser.add_argument('--accuracy_matrix', help='Reuse a complete accuracy_matrix.json; otherwise evaluate all models/tasks')
    parser.add_argument('--probe_samples', type=int, default=128)
    parser.add_argument('--grad_samples', type=int, default=8)
    parser.add_argument('--skip_grad_cosine', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2, help='Hidden-state probe batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_input_length', type=int, default=1536)
    parser.add_argument('--max_new_tokens', type=int, default=8)
    parser.add_argument('--layers', default='-1', help='Hidden-state indices, e.g. 8,16,24,32; -1 for last, all for every layer')
    parser.add_argument('--pooling', choices=['last', 'mean'], default='last')
    parser.add_argument('--seed', type=int, default=1)
    return parser


if __name__ == '__main__':
    run_evaluation(build_parser().parse_args())

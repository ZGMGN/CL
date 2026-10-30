"""Class-conditioned CKA, representation drift and KL on aligned balanced probes."""
import argparse
from collections import Counter
import csv
import gc
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.drift_metrics import linear_cka, representation_cosine_drift
from evaluation.general_task_eval import build_eval_model
from model.utils import GENERAL_TEMPLATE, MODEL_DIR, set_seed

TASKS = ['AGNEWS', 'AMAZON', 'DBPEDIA', 'YAHOO']
LAYERS = [16, 32]
POOLINGS = ['last', 'mean']


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def balanced_probes(samples_per_class, seed):
    """Independent of task order; select the same examples for every adapter."""
    probes, modules, manifest = {}, {}, {}
    for task_index, task in enumerate(TASKS):
        module = importlib.import_module(f'tasks.{task}')
        path = ROOT / 'data' / task / 'val.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        groups = {}
        for index, row in enumerate(rows):
            label = module.extract_label_from_example(row)
            groups.setdefault(label, []).append(index)
        indices = []
        for label in sorted(groups):
            available = groups[label]
            if len(available) < samples_per_class:
                raise ValueError(f'{task}/{label}: insufficient validation samples')
            rng = np.random.default_rng(seed + 1000 * task_index + ord(label))
            indices.extend(rng.choice(available, samples_per_class, replace=False).tolist())
        indices.sort()
        selected = [rows[index] for index in indices]
        labels = [module.extract_label_from_example(row) for row in selected]
        assert set(Counter(labels).values()) == {samples_per_class}
        probes[task], modules[task] = selected, module
        manifest[task] = {'path': str(path), 'split': 'val', 'indices': indices, 'labels': labels,
                          'samples_per_class': samples_per_class, 'class_counts': dict(Counter(labels)),
                          'dataset_sha256': digest(path),
                          'examples_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()}
    return probes, modules, manifest


def per_example_kl(first, second):
    if first.shape != second.shape or first.ndim != 2 or len(first) == 0:
        raise ValueError('Aligned nonempty [examples,vocabulary] log probabilities required')
    forward, reverse = [], []
    for start in range(0, len(first), 8):
        a, b = first[start:start + 8].double(), second[start:start + 8].double()
        forward.append((a.exp() * (a - b)).sum(1).clamp_min(0))
        reverse.append((b.exp() * (b - a)).sum(1).clamp_min(0))
    return torch.cat(forward), torch.cat(reverse)


@torch.inference_mode()
def collect_outputs(model, tokenizer, examples, module, batch_size, max_length, device):
    model.eval()
    features = {pooling: {layer: [] for layer in LAYERS} for pooling in POOLINGS}
    log_probs = []
    for start in range(0, len(examples), batch_size):
        prompts = [module.format_prompt(row, tokenizer, include_answer=False)
                   for row in examples[start:start + batch_size]]
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True,
                           max_length=max_length, add_special_tokens=False).to(device)
        assert bool(inputs['attention_mask'][:, -1].all())
        output = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        log_probs.append(output.logits[:, -1].float().log_softmax(-1).cpu())
        for layer in LAYERS:
            hidden = output.hidden_states[layer]
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            features['last'][layer].append(hidden[:, -1].float().cpu())
            features['mean'][layer].append(((hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
        del output, inputs
    return {'features': {pooling: {layer: torch.cat(rows) for layer, rows in values.items()}
                         for pooling, values in features.items()}, 'log_probs': torch.cat(log_probs)}


class Collector:
    def __init__(self, config, probes, modules, manifest, output):
        self.config, self.probes, self.modules, self.manifest = config, probes, modules, manifest
        self.output = output
        self.model = self.tokenizer = self.device = None
        self.active = None
        self.loaded_identity = None
        self.adapter_counter = 0

    def paths(self, folder, identity):
        cache = folder / 'probe_cache'
        cache.mkdir(parents=True, exist_ok=True)
        signature = {'identity': {key: identity[key] for key in ('weights_sha256', 'config_sha256')},
                     'probes': self.manifest, 'layers': LAYERS, 'poolings': POOLINGS,
                     'max_input_length': self.config['max_input_length'], 'batch_size': self.config['batch_size'],
                     'seed': self.config['seed'], 'prompt_only': True,
                     'base_config_sha256': digest(Path(MODEL_DIR) / 'config.json'),
                     'task_source_sha256': {task: digest(ROOT / 'src/tasks' / f'{task}.py') for task in TASKS}}
        contract = cache / 'manifest.json'
        if contract.exists() and read(contract) != signature:
            raise ValueError(f'Cache belongs to different inputs or models: {cache}')
        save(contract, signature)
        return {task: cache / f'{task}.pt' for task in TASKS}

    def collect(self, folder, identity):
        paths = self.paths(folder, identity)
        missing = [task for task, path in paths.items() if not path.is_file()]
        if missing:
            if self.model is None:
                set_seed(self.config['seed'])
                self.model, self.tokenizer, self.device, _ = build_eval_model(argparse.Namespace(peft_model_folder=None), GENERAL_TEMPLATE)
                self.tokenizer.padding_side = self.tokenizer.truncation_side = 'left'
            if self.loaded_identity != identity:
                adapter = identity['adapter_dir']
                if adapter is None:
                    if self.active is not None:
                        raise ValueError('Shared base probes must be collected before adapters')
                else:
                    self.adapter_counter += 1
                    name = f'classwise_{self.adapter_counter}'
                    if self.active is None:
                        self.model = PeftModel.from_pretrained(self.model, adapter, adapter_name=name, is_trainable=False)
                    else:
                        self.model.load_adapter(adapter, adapter_name=name, is_trainable=False)
                    self.model.set_adapter(name)
                    if self.active is not None:
                        self.model.delete_adapter(self.active)
                    self.active = name
                self.loaded_identity = identity
            for task in missing:
                print(f"Collect {identity['name']} | {task} | {len(self.probes[task])} balanced probes", flush=True)
                outputs = collect_outputs(self.model, self.tokenizer, self.probes[task], self.modules[task],
                                          self.config['batch_size'], self.config['max_input_length'], self.device)
                assert len(outputs['log_probs']) == len(self.probes[task])
                temporary = paths[task].with_suffix('.tmp')
                torch.save(outputs, temporary)
                temporary.replace(paths[task])
                del outputs
        return {task: str(path) for task, path in paths.items()}

    def close(self):
        self.model = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_class_accuracies(spec, modules):
    matrix_path = Path(spec['accuracy_matrix'])
    payload = read(matrix_path)
    assert payload['complete'] and payload['tasks'] == spec['tasks']
    expected = [None, *[str(Path(spec['run_dir']) / f'{stage:02d}_{task}' / 'best_model')
                       for stage, task in enumerate(spec['tasks'], 1)]]
    assert payload['adapter_dirs'] == expected
    results = {}
    for task in spec['tasks']:
        test_path = ROOT / 'data' / task / 'test.jsonl'
        examples = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
        label_counts = None
        task_values = []
        for stage in range(5):
            folder = '00_base' if stage == 0 else f"{stage:02d}_{spec['tasks'][stage - 1]}"
            path = matrix_path.parent / folder / f"{spec['tasks'].index(task) + 1:02d}_{task}.json"
            saved = read(path)
            assert len(saved['results']) == len(examples)
            for index, (row, example) in enumerate(zip(saved['results'], examples)):
                assert row['index'] == index
                assert row['ground_truth_answer'] == modules[task].extract_label_from_example(example)
                prediction = modules[task].extract_answer_letter(row['full_response'])
                assert prediction == row['predicted_answer']
                assert row['correct'] == (prediction is not None and prediction == row['ground_truth_answer'])
            true = Counter(row['ground_truth_answer'] for row in saved['results'])
            correct = Counter(row['ground_truth_answer'] for row in saved['results'] if row['correct'])
            if label_counts is None:
                label_counts = true
            assert label_counts == true
            assert sum(correct.values()) == saved['metrics']['correct']
            assert abs(sum(correct.values()) / sum(true.values()) - payload['accuracy'][stage][spec['tasks'].index(task)]) < 1e-12
            task_values.append({label: 100 * correct[label] / true[label] for label in true})
        results[task] = task_values
    return results


def compare_order(spec, paths, manifest, class_accuracy):
    pair_rows, references, adjacent = [], [], []
    import re
    for task in spec['tasks']:
        labels = manifest[task]['labels']
        names = dict(re.findall(r'^([A-N])\.\s+(.+)$', importlib.import_module(f'tasks.{task}')._prompt_format_after, re.M))
        positions = {label: torch.tensor([i for i, value in enumerate(labels) if value == label]) for label in sorted(set(labels))}
        outputs = [torch.load(paths[stage][task], weights_only=True, mmap=True, map_location='cpu') for stage in range(5)]
        learned = spec['tasks'].index(task) + 1
        lookup = {}
        for first, second in itertools.combinations(range(5), 2):
            print(f"Compare {'->'.join(spec['tasks'])} | {task} | M{first} vs M{second}", flush=True)
            forward, reverse = per_example_kl(outputs[first]['log_probs'], outputs[second]['log_probs'])
            for label, index in positions.items():
                assert len(index) == manifest[task]['samples_per_class']
                kl_a_b, kl_b_a = float(forward[index].mean()), float(reverse[index].mean())
                for pooling in POOLINGS:
                    for layer in LAYERS:
                        a = outputs[first]['features'][pooling][layer][index]
                        b = outputs[second]['features'][pooling][layer][index]
                        cka, rep = linear_cka(a, b), representation_cosine_drift(a, b)
                        if cka is None or rep is None or not math.isfinite(cka + rep + kl_a_b + kl_b_a):
                            raise ValueError(f'Degenerate class metric: {task}/{label}/{pooling}/{layer}')
                        row = {'task_order': ' -> '.join(spec['tasks']), 'evaluated_task': task,
                               'option': label, 'class_name': names[label], 'probe_samples': len(index),
                               'model_a': f'M{first}', 'model_b': f'M{second}',
                               'model_a_after': 'BASE' if first == 0 else spec['tasks'][first - 1],
                               'model_b_after': spec['tasks'][second - 1],
                               'pooling': pooling, 'layer': layer, 'cka': cka, 'rep_drift': rep,
                               'kl_a_to_b': kl_a_b, 'kl_b_to_a': kl_b_a, 'symmetric_kl': (kl_a_b + kl_b_a) / 2,
                               'test_class_accuracy_a_pct': class_accuracy[task][first][label],
                               'test_class_accuracy_b_pct': class_accuracy[task][second][label],
                               'test_class_change_b_minus_a_pp': class_accuracy[task][second][label] - class_accuracy[task][first][label],
                               'task_learned_model': f'M{learned}',
                               'test_forgetting_b_pp': class_accuracy[task][second][label] - class_accuracy[task][learned][label] if second >= learned else None}
                        pair_rows.append(row)
                        lookup[(first, second, label, pooling, layer)] = row
                        if second == first + 1 and second > learned:
                            adjacent.append({**row, 'newly_trained_task': spec['tasks'][second - 1]})
        for label, index in positions.items():
            for pooling in POOLINGS:
                for layer in LAYERS:
                    for current in range(5):
                        if current == learned:
                            values = {'cka': 1.0, 'rep_drift': 0.0, 'kl_reference_to_current': 0.0,
                                      'kl_current_to_reference': 0.0, 'symmetric_kl': 0.0}
                        else:
                            a, b = sorted((current, learned))
                            row = lookup[(a, b, label, pooling, layer)]
                            values = {key: row[key] for key in ('cka', 'rep_drift', 'symmetric_kl')}
                            values['kl_reference_to_current'] = row['kl_a_to_b'] if learned == a else row['kl_b_to_a']
                            values['kl_current_to_reference'] = row['kl_b_to_a'] if learned == a else row['kl_a_to_b']
                        references.append({'task_order': ' -> '.join(spec['tasks']), 'evaluated_task': task,
                                           'option': label, 'class_name': names[label], 'probe_samples': len(index),
                                           'reference_model': f'M{learned}', 'reference_after': task,
                                           'current_model': f'M{current}',
                                           'current_after': 'BASE' if current == 0 else spec['tasks'][current - 1],
                                           'comparison_phase': 'before_task_training' if current < learned else ('just_trained' if current == learned else 'after_task_training'),
                                           'pooling': pooling, 'layer': layer, **values,
                                           'test_class_accuracy_pct': class_accuracy[task][current][label],
                                           'reference_test_class_accuracy_pct': class_accuracy[task][learned][label],
                                           'test_forgetting_pp': class_accuracy[task][current][label] - class_accuracy[task][learned][label] if current >= learned else None})
        del outputs
        gc.collect()
    assert len(pair_rows) == 33 * 10 * 4 and len(references) == 33 * 5 * 4
    return pair_rows, references, adjacent


def plot_report(spec, references, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm
    selected = [row for row in references if row['current_model'] == 'M4' and row['comparison_phase'] == 'after_task_training']
    metrics = [('cka', 'Linear CKA', 0, 1), ('rep_drift', 'Representation drift', 0, 2),
               ('kl_reference_to_current', 'KL(reference || M4)', 0, None),
               ('kl_current_to_reference', 'KL(M4 || reference)', 0, None),
               ('test_forgetting_pp', 'Test class change (pp)', None, None)]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42}):
        with PdfPages(folder / 'classwise_report.pdf') as pdf:
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.axis('off')
            fig.suptitle('Class-conditioned representation and output analysis', fontsize=22)
            text = ['Order: ' + ' → '.join(spec['tasks']), '',
                    'Models: M0 base; ' + '; '.join(f'M{i} after {task}' for i, task in enumerate(spec['tasks'], 1)), '',
                    'Each class uses the same 64 validation examples across every model and order.',
                    'Prompt only; left padding/truncation; max input 1536; evaluation mode.',
                    'Hidden states 16 and 32; last and masked mean pooling collected in one forward pass.',
                    'CKA is centered linear CKA computed separately within each true-label class.',
                    'Representation drift is mean per-example (1 minus cosine similarity).',
                    'KL uses the entire vocabulary at the first answer position, in nats.',
                    'Reference for a task is its just-trained cumulative adapter, not the base.',
                    'All ten model pairs and all five just-trained-reference comparisons are in CSV.',
                    'Figures focus on M4 versus each old task just-trained model; last-task self-comparisons omitted.',
                    'Accuracy changes use the full held-out test set; drift metrics use balanced validation probes.',
                    'Class CKA values are descriptive; a mean over classes is not the whole-task CKA.',
                    'Within-class CKA measures geometry inside a class, not separation from other classes.',
                    'These metrics describe changes; they do not by themselves establish knowledge deletion.']
            ax.text(.025, .93, '\n'.join(text), transform=ax.transAxes, va='top', fontsize=13, linespacing=1.7)
            pdf.savefig(fig)
            plt.close(fig)
            for pooling in POOLINGS:
                for layer in LAYERS:
                    rows = [row for row in selected if row['pooling'] == pooling and row['layer'] == layer]
                    labels = [f"{row['evaluated_task']} {row['option']} · {row['class_name']} | ref {row['reference_model']}" for row in rows]
                    fig, axes = plt.subplots(1, len(metrics), figsize=(23, max(9, len(rows) * .39 + 3)), sharey=True)
                    for ax, (key, title, low, high) in zip(axes, metrics):
                        values = np.asarray([row[key] for row in rows], dtype=float).reshape(-1, 1)
                        settings = {'vmin': low, 'vmax': high}
                        cmap = 'YlGnBu'
                        if key == 'test_forgetting_pp':
                            bound = max(1, float(np.max(np.abs(values))))
                            settings = {'norm': TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)}
                            cmap = 'RdBu'
                        image = ax.imshow(values, aspect='auto', cmap=cmap, **settings)
                        ax.set_xticks([0], [title], rotation=20, ha='right')
                        ax.set_yticks(range(len(rows)), labels, fontsize=10)
                        for index, value in enumerate(values[:, 0]):
                            rgba = image.cmap(image.norm(value))
                            color = 'white' if .299 * rgba[0] + .587 * rgba[1] + .114 * rgba[2] < .5 else '#162033'
                            ax.text(0, index, f'{value:.3f}', ha='center', va='center', fontsize=11, color=color)
                        fig.colorbar(image, ax=ax, fraction=.04, pad=.04)
                    fig.suptitle(' → '.join(spec['tasks']) + f'\n{pooling.upper()} pooling · layer {layer} · final M4 versus class task reference', fontsize=19)
                    fig.text(.5, .025, '64 identical validation probes per class. KL in nats; test change in percentage points (negative = forgetting). Each metric has its own scale.', ha='center', fontsize=11)
                    fig.tight_layout(rect=(0, .075, 1, .91), pad=1.8, w_pad=2)
                    fig.savefig(folder / f'final_classwise_{pooling}_layer{layer}.png', dpi=200)
                    pdf.savefig(fig)
                    plt.close(fig)


def run(args):
    config_path = Path(args.config).resolve()
    config, output = read(config_path), config_path.parent
    torch.set_num_threads(config.get('cpu_threads', 4))
    for item in config.get('adapter_identities', []):
        folder = Path(item['adapter_dir'])
        assert digest(folder / 'adapter_model.safetensors') == item['weights_sha256']
        assert digest(folder / 'adapter_config.json') == item['config_sha256']
    probes, modules, manifest = balanced_probes(config['samples_per_class'], config['seed'])
    assert config['samples_per_class'] == 64 and config['layers'] == LAYERS and config['poolings'] == POOLINGS
    source = output / 'probe_manifest.json'
    if source.exists() and read(source) != manifest:
        raise ValueError('Balanced probes differ from the saved classwise evaluation')
    save(source, manifest)
    collector = Collector(config, probes, modules, manifest, output)
    complete = []
    try:
        base = {'name': 'M0 base', 'adapter_dir': None, 'weights_sha256': None, 'config_sha256': None}
        base_paths = collector.collect(output / 'shared_base', base)
        for index, spec in enumerate(config['orders'], 1):
            folder = output / ('order_' + '_'.join(spec['tasks']))
            folder.mkdir(exist_ok=True)
            contract = {'spec': spec, 'probes': manifest, 'settings': {key: config[key] for key in ('samples_per_class', 'seed', 'batch_size', 'max_input_length', 'layers', 'poolings')}}
            if (folder / 'contract.json').exists() and read(folder / 'contract.json') != contract:
                raise ValueError('Order output belongs to different inputs')
            save(folder / 'contract.json', contract)
            if (folder / 'classwise_metrics.json').exists() and read(folder / 'classwise_metrics.json').get('complete') and (folder / 'classwise_report.pdf').exists():
                complete.append(spec['tasks'])
                continue
            save(output / 'status.json', {'complete': False, 'state': 'collecting', 'current_order': spec['tasks'], 'order_index': index, 'orders_completed': complete, 'orders_total': len(config['orders'])})
            paths = [base_paths]
            for stage, task in enumerate(spec['tasks'], 1):
                adapter = Path(spec['run_dir']) / f'{stage:02d}_{task}' / 'best_model'
                identity = {'name': f'M{stage} after {task}', 'adapter_dir': str(adapter),
                            'weights_sha256': digest(adapter / 'adapter_model.safetensors'),
                            'config_sha256': digest(adapter / 'adapter_config.json')}
                cache_key = hashlib.sha256((identity['weights_sha256'] + identity['config_sha256']).encode()).hexdigest()
                paths.append(collector.collect(output / 'shared_adapters' / cache_key, identity))
            save(output / 'status.json', {'complete': False, 'state': 'comparing', 'current_order': spec['tasks'], 'order_index': index, 'orders_completed': complete, 'orders_total': len(config['orders'])})
            class_accuracy = test_class_accuracies(spec, modules)
            pairs, references, adjacent = compare_order(spec, paths, manifest, class_accuracy)
            write_csv(folder / 'all_model_pair_class_metrics.csv', pairs)
            write_csv(folder / 'task_learned_reference_class_metrics.csv', references)
            write_csv(folder / 'adjacent_old_class_metrics.csv', adjacent)
            plot_report(spec, references, folder)
            save(folder / 'classwise_metrics.json', {'complete': True, 'tasks': spec['tasks'], 'settings': contract['settings'],
                 'accuracy_source': spec['accuracy_matrix'], 'probe_manifest': str(source), 'model_cache_paths': paths,
                 'pair_results': pairs, 'task_learned_reference_results': references, 'adjacent_old_task_results': adjacent})
            complete.append(spec['tasks'])
            save(output / 'status.json', {'complete': False, 'state': 'order_complete', 'orders_completed': complete, 'orders_total': len(config['orders'])})
        all_pairs, all_references, all_adjacent = [], [], []
        for spec in config['orders']:
            data = read(output / ('order_' + '_'.join(spec['tasks'])) / 'classwise_metrics.json')
            assert data['complete']
            all_pairs.extend(data['pair_results'])
            all_references.extend(data['task_learned_reference_results'])
            all_adjacent.extend(data['adjacent_old_task_results'])
        write_csv(output / 'all_orders_pair_class_metrics.csv', all_pairs)
        write_csv(output / 'all_orders_task_learned_reference_class_metrics.csv', all_references)
        write_csv(output / 'all_orders_adjacent_old_class_metrics.csv', all_adjacent)
        final = [row for row in all_references if row['current_model'] == 'M4' and row['comparison_phase'] == 'after_task_training']
        write_csv(output / 'final_old_class_metrics.csv', final)
        save(output / 'status.json', {'complete': True, 'state': 'complete', 'orders_completed': complete, 'orders_total': len(config['orders']),
             'classes_per_order': 33, 'samples_per_class': 64, 'pair_class_layer_pooling_rows': len(all_pairs),
             'final_old_class_rows': len(final), 'settings': {key: config[key] for key in ('layers', 'poolings', 'max_input_length')}})
    finally:
        collector.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        save(Path(args.config).resolve().parent / 'status.json', {'complete': False, 'state': 'failed', 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})
        raise

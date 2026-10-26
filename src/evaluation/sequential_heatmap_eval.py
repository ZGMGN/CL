"""Evaluate the base model and every stage adapter on every task's test split."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from datasets import load_dataset
from peft import PeftModel

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / 'src'))

from evaluation.general_task_eval import build_eval_model, save_json_results
from model.utils import GENERAL_TEMPLATE, set_seed

SUPPORTED_TASKS = ('AGNEWS', 'AMAZON', 'YELP', 'DBPEDIA', 'YAHOO', 'SST2', 'QQP')


def resolve_tasks(run_dir, tasks=None):
    #用于从command中获取order
    """Recover the exact training order from the training summary or launch command."""
    if tasks is None:
        summary = run_dir / 'sequential_results.json'
        launch = run_dir / 'launch.json'
        if summary.is_file():
            tasks = json.loads(summary.read_text())['tasks']
        elif launch.is_file():
            command = json.loads(launch.read_text())['command']
            if '--tasks' not in command:
                raise ValueError('launch.json has no --tasks; supply --tasks explicitly')
            tasks = command[command.index('--tasks') + 1]
        else:
            raise ValueError('Cannot infer task order; supply --tasks explicitly')
    if isinstance(tasks, str):
        tasks = [task.strip().upper() for task in tasks.split(',') if task.strip()]
    else:
        tasks = [task.strip().upper() for task in tasks]
    if not tasks or any(task not in SUPPORTED_TASKS for task in tasks):
        raise ValueError(f'Tasks must be a nonempty sequence from {SUPPORTED_TASKS}')
    return tasks


def find_adapters(run_dir, tasks):
    adapters = [run_dir / f'{index:02d}_{task}' / 'best_model' for index, task in enumerate(tasks, 1)]
    missing = [str(folder) for folder in adapters if not (
        (folder / 'adapter_config.json').is_file()
        and any((folder / name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin'))
    )]
    if missing:
        raise FileNotFoundError('Stage adapters are not ready. Finish training first:\n' + '\n'.join(missing))
    return adapters


def save_matrix(output_dir, payload):
    """Persist completed cells immediately, including a mask for any unfinished cells."""
    with (output_dir / 'accuracy_matrix.json').open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (output_dir / 'accuracy_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', *payload['tasks']])
        for label, row in zip(payload['models'], payload['accuracy']):
            writer.writerow([label, *['' if value is None else value for value in row]])


def plot_heatmap(payload, output_dir):
    # Use a noninteractive backend so plotting also works on the training server.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = np.array([
        [np.nan if value is None else value * 100 for value in row]
        for row in payload['accuracy']
    ], dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, len(payload['tasks']) * 1.5), max(4, len(values) * 0.7 + 1.5)))
    im = ax.imshow(np.ma.masked_invalid(values), cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(payload['tasks'])), payload['tasks'], rotation=30, ha='right')
    ax.set_yticks(range(len(payload['models'])), payload['models'])
    ax.set_xlabel('Test task')
    ax.set_ylabel('Model / training stage')
    ax.set_title('Test accuracy: base model and all stage adapters')
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(j, i, '—' if np.isnan(value) else f'{value:.1f}',
                    ha='center', va='center', color='white' if value >= 60 else 'black')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')
    fig.tight_layout()
    fig.savefig(output_dir / 'accuracy_heatmap.png', dpi=200)
    fig.savefig(output_dir / 'accuracy_heatmap.pdf')
    plt.close(fig)


def run_evaluation(args):
    if args.batch_size <= 0 or args.max_input_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError('Batch size and token limits must be positive')
    run_dir = Path(args.run_dir).expanduser().resolve()
    tasks = resolve_tasks(run_dir, args.tasks)
    adapters = find_adapters(run_dir, tasks)
    datasets, evaluators = {}, {}
    for task in dict.fromkeys(tasks):
        path = ROOT_DIR / 'data' / task / 'test.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'Test dataset does not exist: {path}')
        datasets[task] = load_dataset('json', data_files=str(path), split='train')
        if not len(datasets[task]):
            raise ValueError(f'Test dataset is empty: {task}')
        module = importlib.import_module(f'evaluation.{task}_eval')
        evaluators[task] = getattr(module, f'evaluate_{task}_QA')

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / 'evaluation_heatmap'
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ['Base model', *[f'{index:02d} After {task}' for index, task in enumerate(tasks, 1)]]
    payload = {
        'tasks': tasks,
        'models': labels,
        'adapter_dirs': [None, *map(str, adapters)],
        'accuracy': [[None] * len(tasks) for _ in labels],
        'accuracy_unit': 'fraction',
        'complete': False,
        'evaluation': {
            'split': 'test', 'batch_size': args.batch_size,
            'max_input_length': args.max_input_length, 'max_new_tokens': args.max_new_tokens,
            'seed': args.seed,
        },
    }
    set_seed(args.seed)
    # Always start from the untouched base; never merge stage adapters into it.
    model_args = argparse.Namespace(peft_model_folder=None)
    model, tokenizer, device, _ = build_eval_model(model_args, GENERAL_TEMPLATE)
    active_adapter = None
    for row, label in enumerate(labels):
        if row:
            name = f'stage_{row}'
            if active_adapter is None:
                model = PeftModel.from_pretrained(model, str(adapters[row - 1]), adapter_name=name, is_trainable=False)
            else:
                model.load_adapter(str(adapters[row - 1]), adapter_name=name, is_trainable=False)
            model.set_adapter(name)
            if active_adapter is not None:
                model.delete_adapter(active_adapter)
            active_adapter = name
            model.eval()
        row_dir = output_dir / ('00_base' if row == 0 else f'{row:02d}_{tasks[row - 1]}')
        for col, task in enumerate(tasks):
            print(f'[{row + 1}/{len(labels)}] {label} -> {task}', flush=True)
            results, metrics = evaluators[task](
                model=model, tokenizer=tokenizer, test_dataset=datasets[task], device=device,
                batch_size=args.batch_size, max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens, print_sample=False,
            )
            accuracy = float(metrics['accuracy'])
            if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError(f'Invalid accuracy from {label} / {task}: {accuracy}')
            save_json_results(str(row_dir / f'{col + 1:02d}_{task}.json'), results, metrics)
            payload['accuracy'][row][col] = accuracy
            save_matrix(output_dir, payload)
        plot_heatmap(payload, output_dir)
    payload['complete'] = True
    save_matrix(output_dir, payload)
    print(f'Completed {len(labels)} x {len(tasks)} evaluations. Results: {output_dir}', flush=True)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True, help='Sequential training output directory')
    parser.add_argument('--tasks', help='Exact comma-separated training order; inferred from run metadata by default')
    parser.add_argument('--output_dir', help='Default: <run_dir>/evaluation_heatmap')
    parser.add_argument('--batch_size', '--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_input_length', type=int, default=1536)
    parser.add_argument('--max_new_tokens', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    return parser


if __name__ == '__main__':
    run_evaluation(build_parser().parse_args())

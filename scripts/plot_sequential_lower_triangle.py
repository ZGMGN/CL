"""Plot QA test accuracy on tasks seen so far, checking the saved predictions."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    summary = json.loads((run / 'sequential_results.json').read_text())
    launch = json.loads((run / 'launch.json').read_text())
    tasks, stages = summary['tasks'], summary['stages']
    if tasks != launch['tasks'] or len(stages) != len(tasks):
        raise ValueError('All training stages must be complete and match the launch order')
    n = len(tasks)
    matrix = np.full((n, n), np.nan)
    sources, records, datasets = [], [], {}
    for task in tasks:
        path = ROOT / 'data' / task / 'test.jsonl'
        datasets[task] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for index, stage in enumerate(stages):
        if stage['stage'] != index + 1 or stage['task'] != tasks[index]:
            raise ValueError('Training stages are not in the declared order')
        folder = Path(stage['best_model_dir']).resolve()
        if not folder.is_relative_to(run) or not (folder / 'adapter_model.safetensors').is_file():
            raise ValueError('Stage adapter does not belong to this experiment')
        for col, task in enumerate(tasks[:index + 1]):
            path = folder.parent / f'test_{task}.json'
            saved = json.loads(path.read_text())
            results, metrics = saved['results'], saved['metrics']
            examples = datasets[task]
            module = importlib.import_module(f'tasks.{task}')
            if len(results) != len(examples) or metrics['total'] != len(examples):
                raise ValueError(f'{path}: prediction count mismatch')
            correct = 0
            for number, (prediction, example) in enumerate(zip(results, examples)):
                label = module.extract_label_from_example(example)
                actual = prediction['predicted_answer']
                success = actual is not None and actual == label
                if (prediction['index'] != number or prediction['ground_truth_answer'] != label
                        or bool(prediction['correct']) != success):
                    raise ValueError(f'{path}: inconsistent prediction at row {number}')
                correct += int(success)
            accuracy = correct / len(examples)
            if (metrics['correct'] != correct or abs(metrics['accuracy'] - accuracy) > 1e-12
                    or abs(stage['test_metrics'][task]['accuracy'] - accuracy) > 1e-12):
                raise ValueError(f'{path}: summary accuracy mismatch')
            matrix[index, col] = accuracy * 100
            sources.append({'stage': index + 1, 'task': task, 'path': str(path), 'sha256': digest(path),
                            'test_data_sha256': digest(ROOT / 'data' / task / 'test.jsonl')})
            records.append({'model': f'M{index + 1}', 'after_training': tasks[index], 'evaluated_task': task,
                            'accuracy_pct': accuracy * 100, 'correct': correct, 'total': len(examples)})
    output = run / 'evaluation_lower_triangle'
    output.mkdir(exist_ok=True)
    final_mean = float(matrix[-1].mean())
    signed = matrix - np.diag(matrix)[None, :]
    payload = {'complete': True, 'run_dir': str(run), 'tasks': tasks, 'metric': 'QA test accuracy',
               'unit': 'percent', 'accuracy_pct': [[None if np.isnan(v) else float(v) for v in row] for row in matrix],
               'final_macro_accuracy_pct': final_mean, 'validated_cells': len(records),
               'final_change_vs_just_trained_pp': signed[-1].tolist(),
               'training_mode': launch.get('training_mode', 'sequential_qa_only'),
               'sources': sources, 'summary_sha256': digest(run / 'sequential_results.json')}
    (output / 'lower_triangle_accuracy.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    with (output / 'lower_triangle_accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 14, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(10.8, 8.1))
        cmap = plt.get_cmap('Blues').with_extremes(bad='#f0f2f5')
        im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=100, cmap=cmap, aspect='equal')
        ax.set_xticks(range(n), tasks, fontsize=14)
        ax.set_yticks(range(n), [f'M{i + 1} · after {task}' for i, task in enumerate(tasks)], fontsize=13)
        ax.set_xlabel('Evaluated task (QA test set)', labelpad=13)
        ax.set_ylabel('Saved adapter after training', labelpad=12)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=3)
        ax.tick_params(which='both', length=0)
        for row, col in np.ndindex(matrix.shape):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f'{value:.2f}', ha='center', va='center', fontsize=19,
                        fontweight='bold', color='white' if value >= 58 else '#17233b')
            else:
                ax.text(col, row, '—', ha='center', va='center', color='#a5adb9', fontsize=18)
        for index in range(n):
            ax.add_patch(Rectangle((index - .47, index - .47), .94, .94, fill=False,
                                   edgecolor='#163755', linewidth=1.6))
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.045)
        colorbar.set_label('Accuracy (%)', labelpad=10)
        colorbar.outline.set_visible(False)
        fig.suptitle('Sequential LoRA · QA only', fontsize=22, fontweight='bold', y=.965)
        fig.text(.5, .91, ' → '.join(tasks), ha='center', fontsize=16)
        fig.text(.5, .057, f'Final mean accuracy: {final_mean:.2f}%   |   Diagonal: just-trained task performance',
                 ha='center', fontsize=12)
        fig.text(.5, .026, 'Rows show successive saved adapters; blank cells are tasks not yet trained.',
                 ha='center', fontsize=11, color='#56616f')
        fig.tight_layout(rect=(0, .09, 1, .875), pad=1.3)
        for suffix in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'lower_triangle_accuracy.{suffix}', dpi=300, facecolor='white')
        plt.close(fig)
    print(json.dumps({k: v for k, v in payload.items() if k != 'sources'}, indent=2))


if __name__ == '__main__':
    main()

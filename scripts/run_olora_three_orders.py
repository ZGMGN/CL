"""Run local ordinary LoRA on O-LoRA's three orders, then full evaluations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / 'results/olora_three_orders_label_seed_1'
ORDERS = [
    ['DBPEDIA', 'AMAZON', 'YAHOO', 'AGNEWS'],
    ['DBPEDIA', 'AMAZON', 'AGNEWS', 'YAHOO'],
    ['YAHOO', 'AMAZON', 'AGNEWS', 'DBPEDIA'],
]

def status(state, **extra):
    payload = dict(state=state, updated_at=datetime.now(timezone.utc).isoformat(),
                   algorithm='ordinary_lora', order_source='cmnfriend/O-LoRA',
                   label_format='category_names', orders=ORDERS, **extra)
    temporary = QUEUE / 'status.tmp'
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(QUEUE / 'status.json')

def run(command, logfile):
    print('Starting:', ' '.join(map(str, command)), flush=True)
    with logfile.open('a') as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, env=os.environ.copy(),
                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=True)

def main():
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / 'experiment_config.json').write_text(json.dumps({
        'orders': ORDERS, 'algorithm': 'ordinary_lora', 'seed': 1,
        'instruction_and_labels': 'O-LoRA style; local chat template and local dataset splits',
        'evaluation': 'all base/stage models on all full test sets; all-layer last/mean pooling drift on 128 validation probes',
    }, indent=2))
    for index, tasks in enumerate(ORDERS, 1):
        folder = QUEUE / f'order_{index}'
        folder.mkdir(exist_ok=True)
        status('training', order=index, tasks=tasks)
        run([sys.executable, '-u', ROOT / 'src/continual_learning/train_sequential_tasks.py',
             '--tasks', ','.join(tasks), '--output_dir', folder,
             '--seed', '1', '--num_workers', '4', '--num_train_epochs', '1',
             '--per_device_train_batch_size', '8', '--gradient_accumulation_steps', '2',
             '--lora_r', '16', '--lora_alpha', '64', '--lora_dropout', '0.05',
             '--learning_rate', '0.0002', '--lr_scheduler_type', 'cosine',
             '--warmup_ratio', '0.01', '--logging_steps', '25', '--save_total_limit', '5',
             '--weight_decay', '0.0001', '--eval_batch_size', '16',
             '--max_input_length', '1536', '--max_new_tokens', '8'], folder / 'train.log')
    for index, tasks in enumerate(ORDERS, 1):
        folder = QUEUE / f'order_{index}'
        for pooling in ('last', 'mean'):
            status('evaluating', order=index, pooling=pooling)
            output = folder / ('evaluation_all_models' if pooling == 'last' else 'evaluation_all_models_mean')
            command = [sys.executable, '-u', ROOT / 'src/evaluation/sequential_model_task_eval.py',
                       '--run_dir', folder, '--tasks', ','.join(tasks), '--output_dir', output,
                       '--probe_samples', '128', '--batch_size', '2', '--eval_batch_size', '16',
                       '--max_input_length', '1536', '--max_new_tokens', '8', '--layers', 'all',
                       '--pooling', pooling, '--seed', '1']
            if pooling == 'mean':
                command += ['--source_eval_dir', folder / 'evaluation_all_models',
                            '--accuracy_matrix', folder / 'evaluation_all_models/accuracy/accuracy_matrix.json']
            run(command, folder / f'evaluation_{pooling}.log')
    status('complete')

if __name__ == '__main__':
    try:
        main()
    except Exception:
        status('failed', error=traceback.format_exc())
        raise

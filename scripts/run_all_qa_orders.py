"""Run all remaining four-task permutations, preserving the active training job."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue_dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Reuse verified completed stages after a planned coordinator restart')
    args = parser.parse_args()
    queue = args.queue_dir.resolve()
    config = read(queue / 'queue_config.json')
    root = Path(config['cwd'])
    lock = (queue / '.runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = read(queue / 'queue_status.json') if (queue / 'queue_status.json').exists() else {}
    completed = previous.get('completed_jobs', []).copy() if args.resume else []
    assert set(completed) <= {job['job_id'] for job in config['jobs']}
    if args.resume:
        for job in config['jobs']:
            if job['job_id'] in completed:
                verified = read(Path(job['run_dir']) / 'evaluation_heatmap/validation.json')
                assert verified['complete'] and verified['tasks'] == job['tasks'] and verified['validated_cells'] == 20
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    current, phase = None, 'preflight'

    def status(state, **extra):
        value = {'status': state, 'phase': phase, 'runner_pid': os.getpid(), 'started_at_utc': started,
                 'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'completed_jobs': completed.copy(), 'total_jobs': len(config['jobs']),
                 'already_complete_orders': len(config['already_complete']), 'all_orders': 24,
                 'evaluation_scope': 'full_5x4_QA_accuracy_only', **extra}
        if current:
            value.update(job_id=current['job_id'], tasks=current['tasks'], run_dir=current['run_dir'])
            save(Path(current['run_dir']) / 'pipeline_status.json', value)
        if phase == 'classwise_evaluation':
            value['evaluation_scope'] = 'classwise_CKA_representation_and_KL_for_completed_orders'
        save(queue / 'queue_status.json', value)
        print(json.dumps(value), flush=True)

    def verify_inputs():
        for collection in ('source_hashes', 'input_hashes'):
            for name, checksum in config[collection].items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum:
                    raise ValueError(f'Experiment source/data changed: {name}')

    def execute(command, log):
        verify_inputs()
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        with log.open('w') as stream:
            child = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            status('running', child_pid=child.pid, command=command, log_path=str(log))
            result = child.wait()
        if result:
            raise RuntimeError(f'{phase} exited with {result}; inspect {log}')

    def summarize():
        records = []
        for item in config['already_complete']:
            payload = read(Path(item['accuracy_matrix']))
            records.append({'tasks': item['tasks'], 'run_dir': item['run_dir'], 'complete': True,
                            'accuracy_matrix': item['accuracy_matrix'], 'accuracy': payload['accuracy']})
        for job in config['jobs']:
            path = Path(job['run_dir']) / 'evaluation_heatmap/validation.json'
            if job['job_id'] in completed and path.exists():
                result = read(path)
                records.append({'tasks': job['tasks'], 'run_dir': job['run_dir'], 'complete': True,
                                'accuracy_matrix': str(path.parent / 'accuracy_matrix.json'),
                                'accuracy': result['accuracy']})
        summary = {'complete': len(records) == 24, 'complete_orders': len(records), 'all_orders': 24,
                   'evaluation_scope': 'full_5x4_QA_accuracy_only', 'orders': records}
        save(queue / 'all_orders_summary.json', summary)
        with (queue / 'final_accuracy.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['task_order', 'AGNEWS', 'AMAZON', 'DBPEDIA', 'YAHOO', 'final_macro_accuracy_pct', 'run_dir'])
            writer.writeheader()
            for item in records:
                scores = dict(zip(item['tasks'], (100 * value for value in item['accuracy'][-1])))
                writer.writerow({'task_order': ' -> '.join(item['tasks']), **scores,
                                 'final_macro_accuracy_pct': sum(scores.values()) / 4, 'run_dir': item['run_dir']})

    def run_requested_diagnostics():
        nonlocal current, phase
        request_path = queue / 'classwise_request.json'
        if not request_path.is_file():
            return
        request = read(request_path)
        if request.get('complete') or request['after_job_id'] not in completed:
            return
        saved_current = current
        current = None
        phase = 'classwise_evaluation'
        execute(request['command'], Path(request['log']))
        assert read(Path(request['result_status']))['complete']
        save(request_path, {**request, 'complete': True,
                           'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        current = saved_current

    try:
        verify_inputs()
        summarize()
        run_requested_diagnostics()
        for job in config['jobs']:
            if job['job_id'] in completed:
                continue
            current = job
            folder = Path(job['run_dir'])
            if job['mode'] == 'existing_training_and_heatmap':
                phase = 'waiting_for_current_training_and_heatmap'
                original = Path(config['original_queue'])
                ready_path = original / 'accuracy_only_handoff_ready.json'
                last_update = 0
                while not ready_path.exists():
                    old_state = read(original / 'queue_status.json')
                    if old_state['status'] == 'failed':
                        raise RuntimeError(f'Existing training/heatmap failed: {old_state}')
                    runner = Path(f"/proc/{config['original_runner_pid']}/stat")
                    if not runner.exists() or runner.read_text().split(') ', 1)[1].split()[0] == 'Z':
                        raise RuntimeError('Original coordinator exited before the accuracy-only handoff')
                    if time.monotonic() - last_update >= 60:
                        status('waiting', existing_runner_pid=config['original_runner_pid'], existing_phase=old_state.get('phase'),
                               existing_log=old_state.get('log_path'))
                        last_update = time.monotonic()
                    time.sleep(2)
                ready = read(ready_path)
                assert ready['run_dir'] == str(folder) and ready['parent_pid'] == config['original_runner_pid']
                assert read(folder / 'evaluation_heatmap/accuracy_matrix.json')['complete']
                assert (folder / 'evaluation_heatmap/accuracy_heatmap.pdf').is_file()
                pid = config['original_runner_pid']
                stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
                assert stat[19] == config['original_runner_start_ticks']
                assert str(original / 'run_queue.py') in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
                assert os.getpgid(pid) == pid
                os.killpg(pid, signal.SIGTERM)
                for _ in range(100):
                    live = []
                    for process in (pid, ready['pid']):
                        path = Path(f'/proc/{process}/stat')
                        if path.exists() and path.read_text().split(') ', 1)[1].split()[0] != 'Z':
                            live.append(process)
                    if not live:
                        break
                    time.sleep(.1)
                if live:
                    raise RuntimeError(f'Old coordinator did not exit: {live}')
                previous = read(original / 'queue_status.json')
                save(original / 'queue_status.json', {**previous, 'status': 'transferred_accuracy_only',
                     'replacement_queue': str(queue), 'reason': 'User requested remaining orders with QA heatmaps only'})
            else:
                if job['mode'] == 'training_and_heatmap':
                    phase = 'training'
                    summary_path = folder / 'sequential_results.json'
                    finished_training = False
                    if args.resume and summary_path.is_file():
                        result = read(summary_path)
                        finished_training = result['tasks'] == job['tasks'] and len(result['stages']) == 4
                        if finished_training:
                            assert all((folder / f'{i:02d}_{task}' / 'best_model/adapter_model.safetensors').is_file()
                                       for i, task in enumerate(job['tasks'], 1))
                    if not finished_training and any(folder.glob('[0-9][0-9]_*')):
                        raise FileExistsError(f'Refusing to overwrite training outputs: {folder}')
                    if not finished_training:
                        execute(job['command'], folder / 'train.log')
                phase = 'qa_matrix'
                matrix_path = folder / 'evaluation_heatmap/accuracy_matrix.json'
                finished_matrix = args.resume and matrix_path.is_file() and read(matrix_path).get('complete')
                if not finished_matrix:
                    execute([sys.executable, '-u', str(queue / 'report_accuracy.py'), str(folder)], folder / 'evaluation_heatmap/evaluation.log')
            phase = 'validation'
            execute([sys.executable, '-u', str(queue / 'validate_accuracy.py'), str(folder)], folder / 'evaluation_heatmap/validation.log')
            completed.append(job['job_id'])
            phase = 'complete'
            status('complete', heatmap=str(folder / 'evaluation_heatmap/accuracy_heatmap.pdf'))
            summarize()
            run_requested_diagnostics()
        current = None
        phase = 'all_orders_complete'
        assert read(queue / 'all_orders_summary.json')['complete']
        status('complete', complete_orders=24, summary=str(queue / 'all_orders_summary.json'))
    except BaseException as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()

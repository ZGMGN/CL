"""Refresh step/loss plots from running order logs without touching training."""
import argparse
import ast
import csv
import json
import re
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOKEN = re.compile(r"(?P<progress>\d+%\|[^\r\n]*?\|\s*(?P<step>\d+)/(?P<total>\d+)\s*\[)|(?P<record>\{[^\r\n{}]*['\"]loss['\"][^\r\n{}]*\})")


def read_points(log):
    points = []
    step = 0
    stage = 0
    previous_epoch = -1
    for match in TOKEN.finditer(log.read_text(errors='replace')):
        if match['progress']:
            step = int(match['step'])
            continue
        try:
            item = ast.literal_eval(match['record'])
            loss = float(item['loss'])
            epoch = float(item['epoch'])
        except (ValueError, SyntaxError, KeyError, TypeError):
            continue
        if epoch < previous_epoch:
            stage += 1
        previous_epoch = epoch
        points.append(dict(stage=stage + 1, step=step, loss=loss, epoch=epoch))
    return points


def refresh(queue):
    config_path = queue / 'experiment_config.json'
    if not config_path.exists():
        return
    orders = json.loads(config_path.read_text())['orders']
    for index, tasks in enumerate(orders, 1):
        folder = queue / f'order_{index}'
        log = folder / 'train.log'
        if not log.exists():
            continue
        points = read_points(log)
        if not points:
            continue
        with (folder / 'training_loss.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['stage', 'task', 'step', 'loss', 'epoch'])
            writer.writeheader()
            writer.writerows(dict(point, task=tasks[point['stage']-1]) for point in points if point['stage'] <= len(tasks))
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for stage, (task, ax) in enumerate(zip(tasks, axes.flat), 1):
            rows = [point for point in points if point['stage'] == stage]
            ax.set_title(f'{stage}. {task}')
            ax.set_xlabel('Optimizer step (within task)')
            ax.set_ylabel('Training loss')
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            if rows:
                ax.plot([r['step'] for r in rows], [r['loss'] for r in rows], color='#277DA8', marker='o', markersize=3, linewidth=1.6)
                stage_dir = folder / f'{stage:02d}_{task}'
                stage_dir.mkdir(exist_ok=True)
                single, single_ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
                single_ax.plot([r['step'] for r in rows], [r['loss'] for r in rows], color='#277DA8', marker='o', markersize=3)
                single_ax.set(xlabel='Optimizer step', ylabel='Training loss', title=f'Order {index} / {task}')
                single_ax.grid(alpha=.2)
                single_ax.spines[['top', 'right']].set_visible(False)
                single.savefig(stage_dir / 'training_loss.png', dpi=180)
                plt.close(single)
            else:
                ax.text(.5, .5, 'Pending', transform=ax.transAxes, ha='center', color='gray')
        fig.suptitle(f'Order {index}: training loss (logged every 25 steps)')
        for suffix in ('png', 'svg'):
            fig.savefig(folder / f'training_loss.{suffix}', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue_dir', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        refresh(args.queue_dir)
        status_path = args.queue_dir / 'status.json'
        state = json.loads(status_path.read_text()).get('state') if status_path.exists() else None
        if not args.watch or state in ('complete', 'failed'):
            break
        time.sleep(30)

if __name__ == '__main__':
    main()

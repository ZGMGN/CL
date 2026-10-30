"""Audit class retention and read actual error transitions in the ten frozen orders."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / 'results/standard_cl_all_orders_seed_1_bs8_ga2/classwise_eval_20261030_10orders'
OUT = SCOPE / 'semantic_patterns_analysis'


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    config = json.loads((SCOPE / 'config.json').read_text())
    events, finals, names, source_hashes = [], [], {}, {}
    examples = defaultdict(list)
    wanted = {('AGNEWS', 'D', 'DBPEDIA', 'C'), ('YAHOO', 'E', 'AGNEWS', 'D'),
              ('YAHOO', 'B', 'AGNEWS', 'D'), ('YAHOO', 'J', 'AMAZON', 'A'),
              ('YAHOO', 'H', 'AMAZON', 'A'), ('YAHOO', 'I', 'AMAZON', 'A'),
              ('AMAZON', 'E', 'AGNEWS', 'D'), ('AMAZON', 'A', 'AGNEWS', 'B'),
              ('DBPEDIA', 'L', 'AGNEWS', 'C'), ('DBPEDIA', 'N', 'AGNEWS', 'C')}
    datasets = {task: [json.loads(line) for line in (ROOT / 'data' / task / 'test.jsonl').read_text().splitlines()]
                for task in ['AGNEWS', 'YAHOO', 'AMAZON', 'DBPEDIA']}
    for spec in config['orders']:
        tasks = spec['tasks']
        detail = json.loads((SCOPE / ('order_' + '_'.join(tasks)) / 'classwise_metrics.json').read_text())
        for r in detail['task_learned_reference_results']:
            if r['pooling'] == 'last' and r['layer'] == 32:
                names[r['evaluated_task'], r['option']] = r['class_name']
                if r['current_model'] == 'M4' and r['comparison_phase'] == 'after_task_training':
                    finals.append(r)
        base = Path(spec['accuracy_matrix']).parent
        for task in tasks:
            learned = tasks.index(task) + 1
            predictions = {}
            for stage in range(learned, 5):
                folder = f'{stage:02d}_{tasks[stage - 1]}'
                path = base / folder / f'{tasks.index(task) + 1:02d}_{task}.json'
                raw = path.read_bytes()
                source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
                predictions[stage] = json.loads(raw)['results']
                assert len(predictions[stage]) == len(datasets[task])
            for stage in range(learned + 1, 5):
                new, prefix = tasks[stage - 1], ' -> '.join(tasks[:stage])
                groups = defaultdict(list)
                for i, (a, b) in enumerate(zip(predictions[stage - 1], predictions[stage])):
                    assert a['index'] == b['index'] == i
                    assert a['ground_truth_answer'] == b['ground_truth_answer']
                    groups[a['ground_truth_answer']].append((a, b))
                    target = b['predicted_answer'] or 'INVALID'
                    key = (task, a['ground_truth_answer'], new, target)
                    if key in wanted and a['correct'] and not b['correct'] and len(examples[key]) < 3:
                        if not any(e['test_index'] == i for e in examples[key]):
                            examples[key].append({'task': task, 'true_label': a['ground_truth_answer'],
                                                  'newly_trained_task': new, 'predicted_label_after': target,
                                                  'order': tasks, 'training_prefix': prefix, 'test_index': i,
                                                  'before_response': a['full_response'], 'after_response': b['full_response'],
                                                  'original_example': datasets[task][i]})
                for label, pairs in groups.items():
                    before = sum(a['correct'] for a, b in pairs)
                    after = sum(b['correct'] for a, b in pairs)
                    lost = Counter((b['predicted_answer'] or 'INVALID') for a, b in pairs if a['correct'] and not b['correct'])
                    recovered = sum(not a['correct'] and b['correct'] for a, b in pairs)
                    assert after - before == recovered - sum(lost.values())
                    events.append({'task_order': ' -> '.join(tasks), 'training_prefix': prefix,
                                   'evaluated_task': task, 'option': label, 'class_name': names[task, label],
                                   'newly_trained_task': new, 'test_samples': len(pairs),
                                   'accuracy_before_pct': 100 * before / len(pairs), 'accuracy_after_pct': 100 * after / len(pairs),
                                   'change_pp': 100 * (after - before) / len(pairs),
                                   'correct_to_wrong': sum(lost.values()), 'wrong_to_correct': recovered,
                                   'correct_to_wrong_destinations': dict(lost)})
    final_groups = defaultdict(list)
    for r in finals:
        final_groups[r['evaluated_task'], r['option']].append(r)
    final_summary = []
    for (task, label), rows in final_groups.items():
        values = [r['test_forgetting_pp'] for r in rows]
        final_summary.append({'task': task, 'option': label, 'class_name': names[task, label],
                              'eligible_orders': len(rows), 'mean_signed_forgetting_pp': statistics.mean(values),
                              'median_signed_forgetting_pp': statistics.median(values),
                              'min_pp': min(values), 'max_pp': max(values),
                              'orders_with_decline': sum(v < -1e-8 for v in values),
                              'orders_with_decline_at_least_5pp': sum(v <= -5 for v in values)})
    prefix_groups = defaultdict(list)
    for r in events:
        prefix_groups[r['training_prefix'], r['evaluated_task'], r['option']].append(r)
    prefix_events = []
    for (prefix, task, label), rows in prefix_groups.items():
        destination_counts = defaultdict(list)
        labels = set().union(*(r['correct_to_wrong_destinations'] for r in rows))
        for target in labels:
            destination_counts[target] = [r['correct_to_wrong_destinations'].get(target, 0) for r in rows]
        prefix_events.append({'training_prefix': prefix, 'evaluated_task': task, 'option': label,
                              'class_name': names[task, label], 'newly_trained_task': rows[0]['newly_trained_task'],
                              'order_replicates': len(rows), 'change_pp': statistics.mean(r['change_pp'] for r in rows),
                              'correct_to_wrong': statistics.mean(r['correct_to_wrong'] for r in rows),
                              'wrong_to_correct': statistics.mean(r['wrong_to_correct'] for r in rows),
                              'correct_to_wrong_destinations': {target: statistics.mean(v) for target, v in destination_counts.items()}})
    trigger_groups = defaultdict(list)
    for r in prefix_events:
        trigger_groups[r['evaluated_task'], r['option'], r['newly_trained_task']].append(r)
    trigger_summary = []
    for (task, label, new), rows in trigger_groups.items():
        targets = Counter()
        for row in rows:
            targets.update(row['correct_to_wrong_destinations'])
        total = sum(targets.values())
        dominant, count = targets.most_common(1)[0] if targets else (None, 0)
        trigger_summary.append({'evaluated_task': task, 'option': label, 'class_name': names[task, label],
                                'newly_trained_task': new, 'unique_prefixes': len(rows),
                                'mean_change_pp': statistics.mean(r['change_pp'] for r in rows),
                                'min_change_pp': min(r['change_pp'] for r in rows), 'max_change_pp': max(r['change_pp'] for r in rows),
                                'declining_prefixes': sum(r['change_pp'] < -1e-8 for r in rows),
                                'dominant_lost_target': dominant, 'dominant_lost_target_fraction': count / total if total else None,
                                'lost_target_prefix_normalized_counts': dict(targets)})
    for name, rows in [('final_class_summary.csv', final_summary), ('adjacent_class_events.csv', events),
                       ('prefix_averaged_class_events.csv', prefix_events), ('new_task_class_effects.csv', trigger_summary)]:
        csv_write(OUT / name, [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v for k, v in r.items()} for r in rows])
    payload = {'complete': True, 'orders': [r['tasks'] for r in config['orders']],
               'scope': 'Ten frozen QA-only orders; held-out full test predictions; old tasks only',
               'caveat': 'One seed, selected unbalanced orders. Repeated logical prefixes averaged before trigger means. Descriptive, not causal or independent-seed significance.',
               'final_class_summary': final_summary, 'new_task_class_effects': trigger_summary,
               'examples': [e for rows in examples.values() for e in rows], 'prediction_source_sha256': source_hashes}
    (OUT / 'semantic_pattern_evidence.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('COMPLETE:',OUT)
    for r in sorted(trigger_summary, key=lambda r:r['mean_change_pp'])[:25]:
        print(r['evaluated_task'],r['option'],r['newly_trained_task'], 'change',round(r['mean_change_pp'],2),
              'prefixes',r['unique_prefixes'],'negative',r['declining_prefixes'],'target',r['dominant_lost_target'],
              'targetshare',round(r['dominant_lost_target_fraction'] or 0,3))
    print('Final frequencies:')
    for r in sorted(final_summary,key=lambda r:r['mean_signed_forgetting_pp'])[:15]:print(r)


if __name__ == '__main__':
    main()

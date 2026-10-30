"""Add complete label mappings and destination-level errors to the class analysis."""
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / 'results/standard_cl_all_orders_seed_1_bs8_ga2/classwise_eval_20261030_10orders'
OUT = SCOPE / 'semantic_patterns_analysis'
TASKS = ['AGNEWS', 'YAHOO', 'AMAZON', 'DBPEDIA']
FOCUS = ['AGNEWS', 'YAHOO', 'AMAZON', 'DBPEDIA']
ZH = {
    'AGNEWS': ['国际新闻', '体育', '商业', '科学与技术'],
    'YAHOO': ['社会与文化', '科学与数学', '健康', '教育与参考资料', '计算机与互联网', '体育', '商业与金融', '娱乐与音乐', '家庭与人际关系', '政治与政府'],
    'AMAZON': ['一星评价', '二星评价', '三星评价', '四星评价', '五星评价'],
    'DBPEDIA': ['公司', '教育机构', '艺术家', '运动员', '公职人员', '交通工具', '建筑物', '自然地理地点', '村庄', '动物', '植物', '音乐专辑', '电影', '文字作品'],
}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mappings():
    result, rows = {}, []
    for task in TASKS:
        tree = ast.parse((ROOT / 'src/tasks' / f'{task}.py').read_text())
        prompt = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == '_prompt_format_after' for target in node.targets))
        names = dict(re.findall(r'^([A-N])\.\s+(.+)$', prompt, re.M))
        assert len(names) == len(ZH[task])
        result[task] = {}
        for i, (option, name) in enumerate(names.items()):
            assert option == chr(65 + i)
            row = {'task': task, 'raw_label': i, 'option': option, 'english_class': name, 'chinese_class': ZH[task][i]}
            rows.append(row)
            result[task][option] = row
    return result, rows


def main():
    cfg = json.loads((SCOPE / 'config.json').read_text())
    evidence = json.loads((OUT / 'semantic_pattern_evidence.json').read_text())
    names, label_rows = mappings()
    assert len(label_rows) == 33 and len(cfg['orders']) == 10
    write_csv(OUT / 'label_mapping.csv', label_rows)
    model_confusions, adjacent, final_rows = [], [], []
    adjacent_groups, final_groups = defaultdict(list), defaultdict(list)
    sources, focus_rows = {}, {}
    raw = {task: [json.loads(line) for line in (ROOT / 'data' / task / 'test.jsonl').read_text().splitlines() if line.strip()] for task in TASKS}
    validated_cells = 0
    for spec in cfg['orders']:
        tasks = spec['tasks']
        order = ' -> '.join(tasks)
        parent = Path(spec['accuracy_matrix']).parent
        matrix = json.loads(Path(spec['accuracy_matrix']).read_text())
        assert matrix['complete'] and matrix['tasks'] == tasks
        for task in TASKS:
            learned = tasks.index(task) + 1
            cache = {}
            for stage in range(5):
                folder = '00_base' if stage == 0 else f'{stage:02d}_{tasks[stage - 1]}'
                path = parent / folder / f'{tasks.index(task) + 1:02d}_{task}.json'
                blob = path.read_bytes()
                checksum = hashlib.sha256(blob).hexdigest()
                if str(path) in evidence['prediction_source_sha256']:
                    assert checksum == evidence['prediction_source_sha256'][str(path)]
                sources[str(path)] = checksum
                payload = json.loads(blob)
                rows = payload['results']
                assert len(rows) == len(raw[task])
                for i, row in enumerate(rows):
                    assert row['index'] == i and row['ground_truth_answer'] == chr(65 + raw[task][i]['label'])
                    assert row['predicted_answer'] is None or row['predicted_answer'] in names[task]
                    assert row['correct'] == (row['predicted_answer'] == row['ground_truth_answer'])
                assert sum(r['correct'] for r in rows) == payload['metrics']['correct']
                assert abs(sum(r['correct'] for r in rows) / len(rows) - matrix['accuracy'][stage][tasks.index(task)]) < 1e-12
                validated_cells += 1
                cache[stage] = rows
                for label in names[task]:
                    selected = [r for r in rows if r['ground_truth_answer'] == label]
                    counts = Counter(r['predicted_answer'] or 'INVALID' for r in selected)
                    errors = len(selected) - counts[label]
                    assert sum(counts.values()) == len(selected)
                    for target in [*names[task], 'INVALID']:
                        count = counts[target]
                        model_confusions.append({'task_order': order, 'evaluated_task': task, 'true_option': label,
                                                 'true_class': names[task][label]['english_class'], 'true_class_zh': names[task][label]['chinese_class'],
                                                 'model': f'M{stage}', 'model_after': 'BASE' if stage == 0 else tasks[stage - 1],
                                                 'predicted_option': target, 'predicted_class': names[task][target]['english_class'] if target != 'INVALID' else 'Invalid/unparsed response',
                                                 'predicted_class_zh': names[task][target]['chinese_class'] if target != 'INVALID' else '无效或未解析输出',
                                                 'is_correct_destination': target == label, 'count': count,
                                                 'true_class_test_samples': len(selected), 'class_errors': errors,
                                                 'fraction_of_class': count / len(selected),
                                                 'fraction_of_errors': count / errors if target != label and errors else None})
            for label in names[task]:
                indices = [i for i, r in enumerate(cache[4]) if r['ground_truth_answer'] == label]
                n = len(indices)
                all_errors = Counter(cache[4][i]['predicted_answer'] or 'INVALID' for i in indices if not cache[4][i]['correct'])
                lost = Counter(cache[4][i]['predicted_answer'] or 'INVALID' for i in indices if cache[learned][i]['correct'] and not cache[4][i]['correct'])
                recovered = sum(not cache[learned][i]['correct'] and cache[4][i]['correct'] for i in indices)
                baseline_correct = sum(cache[learned][i]['correct'] for i in indices)
                final_correct = sum(cache[4][i]['correct'] for i in indices)
                assert final_correct - baseline_correct == recovered - sum(lost.values())
                item = {'task_order': order, 'evaluated_task': task, 'option': label,
                        'class_name': names[task][label]['english_class'], 'samples': n,
                        'reference_model': f'M{learned}', 'eligible_old_task': learned < 4,
                        'reference_accuracy_pct': 100 * baseline_correct / n, 'final_accuracy_pct': 100 * final_correct / n,
                        'change_pp': 100 * (final_correct - baseline_correct) / n,
                        'all_final_errors': all_errors, 'reference_correct_final_wrong': lost,
                        'wrong_to_correct': recovered}
                if tasks == FOCUS:
                    focus_rows[task, label] = item
                if learned < 4:
                    final_groups[task, label].append(item)
                for target in [x for x in names[task] if x != label] + ['INVALID']:
                    final_rows.append({'task_order': order, 'evaluated_task': task, 'true_option': label,
                                       'true_class': names[task][label]['english_class'], 'reference_model': f'M{learned}',
                                       'eligible_old_task': learned < 4, 'target_option': target,
                                       'target_class': names[task][target]['english_class'] if target != 'INVALID' else 'Invalid/unparsed response',
                                       'true_class_test_samples': n, 'reference_accuracy_pct': item['reference_accuracy_pct'],
                                       'final_accuracy_pct': item['final_accuracy_pct'], 'signed_change_pp': item['change_pp'],
                                       'all_final_errors': all_errors[target], 'total_final_errors': sum(all_errors.values()),
                                       'fraction_of_final_errors': all_errors[target] / sum(all_errors.values()) if all_errors else None,
                                       'reference_correct_final_wrong': lost[target], 'total_reference_correct_final_wrong': sum(lost.values()),
                                       'fraction_of_new_errors': lost[target] / sum(lost.values()) if lost else None,
                                       'wrong_to_correct_total': recovered})
                for stage in range(learned + 1, 5):
                    before, after = cache[stage - 1], cache[stage]
                    new_errors = Counter(after[i]['predicted_answer'] or 'INVALID' for i in indices if before[i]['correct'] and not after[i]['correct'])
                    gained = sum(not before[i]['correct'] and after[i]['correct'] for i in indices)
                    delta = 100 * (gained - sum(new_errors.values())) / n
                    prefix = ' -> '.join(tasks[:stage])
                    group = {'task_order': order, 'training_prefix': prefix, 'evaluated_task': task,
                             'true_option': label, 'newly_trained_task': tasks[stage - 1], 'before_model': f'M{stage-1}',
                             'after_model': f'M{stage}', 'test_samples': n, 'new_error_counts': new_errors,
                             'wrong_to_correct': gained, 'change_pp': delta}
                    adjacent_groups[prefix, task, label].append(group)
                    for target in [x for x in names[task] if x != label] + ['INVALID']:
                        adjacent.append({k: v for k, v in group.items() if k != 'new_error_counts'} |
                                        {'true_class': names[task][label]['english_class'], 'target_option': target,
                                         'target_class': names[task][target]['english_class'] if target != 'INVALID' else 'Invalid/unparsed response',
                                         'new_errors_to_target': new_errors[target], 'total_new_errors': sum(new_errors.values()),
                                         'fraction_of_new_errors': new_errors[target] / sum(new_errors.values()) if new_errors else None})
    assert validated_cells == 200 and len(focus_rows) == 33
    write_csv(OUT / 'all_models_class_confusion.csv', model_confusions)
    write_csv(OUT / 'adjacent_new_error_destinations.csv', adjacent)
    write_csv(OUT / 'reference_to_final_error_destinations.csv', final_rows)
    # Average equivalent logical prefixes before pooling by the new task.
    trigger_counts, trigger_changes, trigger_sizes = defaultdict(Counter), defaultdict(list), Counter()
    for (prefix, task, label), group in adjacent_groups.items():
        key = (task, label, group[0]['newly_trained_task'])
        trigger_counts[key]  # retain class/new-task groups with no new errors
        all_targets = set().union(*(g['new_error_counts'] for g in group))
        for target in all_targets:
            trigger_counts[key][target] += statistics.mean(g['new_error_counts'][target] for g in group)
        trigger_changes[key].append(statistics.mean(g['change_pp'] for g in group))
        trigger_sizes[key] += 1
    trigger_rows = []
    for (task, label, new), counts in trigger_counts.items():
        total = sum(counts.values())
        for target in [x for x in names[task] if x != label] + ['INVALID']:
            trigger_rows.append({'evaluated_task': task, 'true_option': label, 'true_class': names[task][label]['english_class'],
                                 'newly_trained_task': new, 'unique_prefixes': trigger_sizes[task, label, new],
                                 'mean_change_pp': statistics.mean(trigger_changes[task, label, new]),
                                 'target_option': target, 'target_class': names[task][target]['english_class'] if target != 'INVALID' else 'Invalid/unparsed response',
                                 'prefix_normalized_new_errors': counts[target], 'prefix_normalized_total_new_errors': total,
                                 'fraction_of_new_errors': counts[target] / total if total else None})
    assert len(trigger_counts) == 99
    old_effects = {(r['evaluated_task'], r['option'], r['newly_trained_task']): r for r in evidence['new_task_class_effects']}
    for key, counts in trigger_counts.items():
        expected = old_effects[key]
        assert trigger_sizes[key] == expected['unique_prefixes']
        assert abs(statistics.mean(trigger_changes[key]) - expected['mean_change_pp']) < 1e-10
        expected_counts = expected['lost_target_prefix_normalized_counts']
        for option in set(counts) | set(expected_counts):
            assert abs(counts[option] - expected_counts.get(option, 0)) < 1e-10
    write_csv(OUT / 'new_task_error_destinations.csv', trigger_rows)

    def option_name(task, option):
        return 'INVALID（无效或未解析输出）' if option == 'INVALID' else f"{option} {names[task][option]['english_class']}（{names[task][option]['chinese_class']}）"

    def distribution(task, counts, fractional=False):
        total = sum(counts.values())
        if not total:
            return '无'
        bits = []
        for option, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if count:
                count_text = f'{count:g}' if fractional else str(int(count))
                bits.append(f'{option_name(task, option)}：{count_text}（{100*count/total:.1f}%）')
        return '；'.join(bits)

    label_lines = ['<!-- COMPLETE_LABEL_MAPPING_START -->', '## 四个任务的全部 label 对照', '',
                   '原始数值 label 从 0 开始，与 A、B、C……依次对应。下文错误去向的字母均按**被评估任务**解释，不按新训练任务解释。', '']
    for task in TASKS:
        label_lines += [f'### {task}（{len(names[task])} 类）', '', '|原始label|选项|英文类别|中文含义|', '|---:|---|---|---|']
        for label, row in names[task].items():
            label_lines.append(f"|{row['raw_label']}|{label}|{row['english_class']}|{row['chinese_class']}|")
        label_lines.append('')
    label_lines.append('<!-- COMPLETE_LABEL_MAPPING_END -->')

    lines = ['<!-- COMPLETE_ERROR_DESTINATIONS_START -->', '## 全部类别的错误流向', '',
             '下列表格覆盖全部 33 个类别及所有实际出现的错误目标，未省略为 top-1 或 top-3；零计数目标在 CSV 中保留。INVALID 单列，不强行归入任何选项。', '',
             '**统计口径**：最终错误包括 M4 所有错分；新增遗忘错误只包括参考模型原本正确、后续模型变错的样本。两者分开，避免把原本就不会的样本当成遗忘。每个目标后的百分比分母是当前表格该列对应的错误总数，不是该类别的全部测试样本数。', '',
             '### AGNEWS → YAHOO → AMAZON → DBPEDIA：逐类最终错分与新增遗忘', '',
             'M0=base；M1=AGNEWS后；M2=YAHOO后；M3=AMAZON后；M4=DBPEDIA后。新增遗忘以该任务刚训练后的模型为参考。DBPEDIA 刚训练模型就是 M4，没有后续训练；其最终错误仍列出，但新增遗忘为 0。', '']
    for task in TASKS:
        lines += [f'#### {task}', '', '|真实类别|测试数|参考→M4正确率|M4全部错误去向：次数（占该类错误比例）|参考正确→M4错误去向：次数（占新增错误比例）|', '|---|---:|---|---|---|']
        for label in names[task]:
            r = focus_rows[task, label]
            lines.append(f"|{option_name(task,label)}|{r['samples']}|{r['reference_model']} {r['reference_accuracy_pct']:.1f}% → M4 {r['final_accuracy_pct']:.1f}%|{distribution(task,r['all_final_errors'])}|{distribution(task,r['reference_correct_final_wrong'])}|")
        lines.append('')
    lines += ['### 十个 order 汇总：所有旧任务类别的最终错误流向', '',
              '只纳入该任务后面仍有训练任务的顺序。先逐顺序计数，再合并同类错误；因此这里是“样本—模型评估次数”，同一测试文本可能在不同 order 重复出现，不是不同文本的数量，也不是独立随机种子重复。最终错误目标比例按合并后的错误次数计算；它不同于对各 order 的错误比例直接求均值。', '']
    aggregate_rows = []
    for task in TASKS:
        lines += [f'#### {task}', '', '|真实类别|可分析order数|M4全部错误去向|刚训练正确→M4错误去向|', '|---|---:|---|---|']
        for label in names[task]:
            group = final_groups[task, label]
            counts, lost = Counter(), Counter()
            for row in group:
                counts.update(row['all_final_errors']);lost.update(row['reference_correct_final_wrong'])
            lines.append(f'|{option_name(task,label)}|{len(group)}|{distribution(task,counts)}|{distribution(task,lost)}|')
            for target in [x for x in names[task] if x != label] + ['INVALID']:
                aggregate_rows.append({'evaluated_task': task, 'true_option': label, 'true_class': names[task][label]['english_class'],
                                       'eligible_orders': len(group), 'target_option': target,
                                       'target_class': names[task][target]['english_class'] if target != 'INVALID' else 'Invalid/unparsed response',
                                       'pooled_final_errors': counts[target], 'pooled_total_final_errors': sum(counts.values()),
                                       'fraction_of_final_errors': counts[target] / sum(counts.values()) if counts else None,
                                       'pooled_reference_correct_final_wrong': lost[target], 'pooled_total_reference_correct_final_wrong': sum(lost.values()),
                                       'fraction_of_new_errors': lost[target] / sum(lost.values()) if lost else None})
        lines.append('')
    write_csv(OUT / 'all_old_orders_final_error_destinations.csv', aggregate_rows)
    lines += ['### 训练哪个新任务之后，旧任务的每个类别转到哪些选项', '',
              '以下包含全部 99 个“被评估任务×真实类别×新训练任务”组合，包括没有新增错误的组合。相同逻辑训练前缀内先平均，再合并不同前缀的错误目标计数，故归一计数可能出现小数；百分比是这些计数在该组合新增错误中的占比。即时变化包含错误→正确的恢复，因此即使平均表现提高，也可能仍出现少量新增错误。', '']
    for task in TASKS:
        lines += [f'#### 被评估任务：{task}', '', '|真实类别|新训练任务|不同训练前缀数|平均即时变化(pp)|原本正确→错误的全部去向：归一计数（占比）|', '|---|---|---:|---:|---|']
        for label in names[task]:
            for new in TASKS:
                key = task, label, new
                if key not in trigger_counts:
                    continue
                lines.append(f'|{option_name(task,label)}|{new}|{trigger_sizes[key]}|{statistics.mean(trigger_changes[key]):+.2f}|{distribution(task,trigger_counts[key],True)}|')
        lines.append('')
    lines += ['### 完整明细文件', '',
              '- `label_mapping.csv`：33 个类别的数值 label、选项、英文名和中文含义。',
              '- `all_models_class_confusion.csv`：十个 order、M0–M4、四任务全部类别的完整混淆计数，含正确项、零计数和 INVALID。',
              '- `reference_to_final_error_destinations.csv`：每个 order 的每类 M4 错分目标，以及刚训练正确→最终错误的目标。',
              '- `adjacent_new_error_destinations.csv`：每个 order 每个旧任务相邻阶段的新增错误去向。',
              '- `new_task_error_destinations.csv`：按新训练任务分组、先平均重复训练前缀后的全部错误目标。',
              '- `all_old_orders_final_error_destinations.csv`：十个 order 中旧任务的最终错误目标汇总。',
              '- `error_destination_validation.json`：200 个原始评估单元的来源哈希、标签和准确率一致性校验。', '', '<!-- COMPLETE_ERROR_DESTINATIONS_END -->']
    report_path = OUT / 'semantic_patterns_report.md'
    report = report_path.read_text()
    report = re.sub(r'\n<!-- COMPLETE_LABEL_MAPPING_START -->.*?<!-- COMPLETE_LABEL_MAPPING_END -->\n', '\n', report, flags=re.S)
    report = re.sub(r'\n<!-- COMPLETE_ERROR_DESTINATIONS_START -->.*?<!-- COMPLETE_ERROR_DESTINATIONS_END -->\n', '\n', report, flags=re.S)
    report = report.replace('## 一、哪些类别反复下降，哪些只在特定顺序崩溃', '\n'.join(label_lines) + '\n\n## 一、哪些类别反复下降，哪些只在特定顺序崩溃', 1)
    report += '\n\n' + '\n'.join(lines) + '\n'
    assert report.count('<!-- COMPLETE_LABEL_MAPPING_START -->') == report.count('<!-- COMPLETE_ERROR_DESTINATIONS_START -->') == 1
    report_path.write_text(report, encoding='utf-8')
    validation = {'complete': True, 'frozen_orders': len(cfg['orders']), 'classes': len(label_rows),
                  'validated_model_task_cells': validated_cells, 'focus_order': FOCUS,
                  'all_model_confusion_rows': len(model_confusions), 'per_order_adjacent_destination_rows': len(adjacent),
                  'reference_to_final_destination_rows': len(final_rows), 'new_task_class_groups': len(trigger_counts),
                  'new_task_destination_rows': len(trigger_rows), 'old_task_final_aggregate_rows': len(aggregate_rows),
                  'all_target_options_including_zero_and_invalid': True,
                  'new_errors_plus_recoveries_match_accuracy_changes': True,
                  'prefix_statistics_match_existing_semantic_analysis': True, 'prediction_source_sha256': sources}
    (OUT / 'error_destination_validation.json').write_text(json.dumps(validation, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    with zipfile.ZipFile(OUT / 'semantic_analysis_with_error_destinations.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.suffix in ('.csv', '.json', '.md'):
                archive.write(path, path.name)
    print('Updated:',report_path,flush=True)
    print(json.dumps({k:v for k,v in validation.items() if k!='prediction_source_sha256'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

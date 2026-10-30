"""Expose M0--M4 errors and every directed model pair for all true-label classes."""
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / 'results/standard_cl_all_orders_seed_1_bs8_ga2/classwise_eval_20261030_10orders'
OUT = SCOPE / 'semantic_patterns_analysis'
TASKS = ['AGNEWS', 'YAHOO', 'AMAZON', 'DBPEDIA']
FOCUS = tuple(TASKS)
CATEGORIES = ['correct_to_correct', 'correct_to_wrong', 'wrong_to_correct', 'wrong_to_different_wrong', 'same_wrong']


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def category(true, source, target):
    if source == true:
        return 'correct_to_correct' if target == true else 'correct_to_wrong'
    if target == true:
        return 'wrong_to_correct'
    return 'same_wrong' if source == target else 'wrong_to_different_wrong'


def main():
    cfg = json.loads((SCOPE / 'config.json').read_text())
    prior = json.loads((OUT / 'error_destination_validation.json').read_text())
    label_rows = list(csv.DictReader((OUT / 'label_mapping.csv').open(encoding='utf-8-sig')))
    names = {task: {r['option']: r for r in label_rows if r['task'] == task} for task in TASKS}
    with (OUT / 'all_models_class_confusion.csv').open(encoding='utf-8-sig') as stream:
        expected = {(r['task_order'], r['evaluated_task'], r['true_option'], r['model'], r['predicted_option']): int(r['count']) for r in csv.DictReader(stream)}
    folder = OUT / 'all_model_order_reports'
    folder.mkdir(exist_ok=True)
    all_summaries, all_flows, all_states, validation_orders, sources = [], [], [], [], {}
    focus_text = None

    def label_name(task, label):
        return 'INVALID（无效输出）' if label == 'INVALID' else f"{label} {names[task][label]['english_class']}（{names[task][label]['chinese_class']}）"

    def count_list(task, counts):
        total = sum(counts.values())
        return '无' if not total else '；'.join(f'{label_name(task,key)}：{count}（{100*count/total:.1f}%）'
                                              for key,count in sorted(counts.items(), key=lambda kv:(-kv[1],kv[0])) if count)

    def changed_wrong_list(task, flows):
        total = sum(flows.values())
        return '无' if not total else '；'.join(f'{label_name(task,a)} → {label_name(task,b)}：{n}（{100*n/total:.1f}%）'
                                              for (a,b),n in sorted(flows.items(), key=lambda kv:(-kv[1],kv[0])))

    for spec in cfg['orders']:
        tasks = spec['tasks'];order = ' -> '.join(tasks)
        parent = Path(spec['accuracy_matrix']).parent
        data = {}
        state_lookup, pair_lookup = {}, {}
        samples = {}
        for task in TASKS:
            grouped = defaultdict(list)
            for stage in range(5):
                model = f'M{stage}'
                subfolder = '00_base' if not stage else f'{stage:02d}_{tasks[stage-1]}'
                path = parent / subfolder / f'{tasks.index(task)+1:02d}_{task}.json'
                blob = path.read_bytes();checksum = hashlib.sha256(blob).hexdigest()
                assert prior['prediction_source_sha256'][str(path)] == checksum
                sources[str(path)] = checksum
                rows = json.loads(blob)['results'];data[task,stage] = rows
                for i,r in enumerate(rows):
                    assert r['index'] == i
                    assert r['correct'] == (r['ground_truth_answer'] == r['predicted_answer'])
                    if stage == 0:
                        grouped[r['ground_truth_answer']].append(i)
                    else:
                        assert r['ground_truth_answer'] == data[task,0][i]['ground_truth_answer']
                assert len(rows) == len(data[task,0])
            for true, indices in sorted(grouped.items()):
                samples[task,true] = len(indices)
                for stage in range(5):
                    counts = Counter(data[task,stage][i]['predicted_answer'] or 'INVALID' for i in indices)
                    for option in [*names[task],'INVALID']:
                        assert counts[option] == expected[order,task,true,f'M{stage}',option]
                    state_lookup[task,true,stage] = counts
                    total = len(indices);correct = counts[true];errors = total-correct
                    all_states.append({'task_order':order,'evaluated_task':task,'true_option':true,
                                       'true_class':names[task][true]['english_class'],'model':f'M{stage}',
                                       'model_after':'BASE' if not stage else tasks[stage-1],
                                       'task_already_trained':stage>=tasks.index(task)+1,
                                       'test_samples':total,'correct':correct,'errors':errors,'accuracy_pct':100*correct/total,
                                       'all_error_destinations':json.dumps({k:v for k,v in counts.items() if k!=true},ensure_ascii=False)})
                for a,b in itertools.product(range(5),repeat=2):
                    flows = Counter((data[task,a][i]['predicted_answer'] or 'INVALID',data[task,b][i]['predicted_answer'] or 'INVALID') for i in indices)
                    kinds = Counter()
                    for (source,target),count in flows.items():
                        kinds[category(true,source,target)] += count
                    assert sum(kinds.values()) == len(indices)
                    source_counts = state_lookup[task,true,a];target_counts = state_lookup[task,true,b]
                    for option in [*names[task],'INVALID']:
                        assert sum(n for (x,y),n in flows.items() if x==option)==source_counts[option]
                        assert sum(n for (x,y),n in flows.items() if y==option)==target_counts[option]
                    assert target_counts[true]-source_counts[true]==kinds['wrong_to_correct']-kinds['correct_to_wrong']
                    if a==b:
                        assert kinds['correct_to_wrong']==kinds['wrong_to_correct']==kinds['wrong_to_different_wrong']==0
                    phase = ('self' if a==b else 'reverse_comparison' if a>b else
                             'old_task_forward' if a>=tasks.index(task)+1 else 'includes_pretraining_control')
                    item = {'task_order':order,'evaluated_task':task,'true_option':true,
                            'true_class':names[task][true]['english_class'],'source_model':f'M{a}',
                            'source_after':'BASE' if not a else tasks[a-1], 'target_model':f'M{b}',
                            'target_after':'BASE' if not b else tasks[b-1], 'comparison_phase':phase,
                            'test_samples':len(indices),'source_accuracy_pct':100*source_counts[true]/len(indices),
                            'target_accuracy_pct':100*target_counts[true]/len(indices),
                            'accuracy_change_pp':100*(target_counts[true]-source_counts[true])/len(indices),
                            **{key:kinds[key] for key in CATEGORIES}}
                    all_summaries.append(item)
                    pair_lookup[task,true,a,b] = (flows,item)
                    for (source,target),count in sorted(flows.items()):
                        cat=category(true,source,target)
                        all_flows.append({'task_order':order,'evaluated_task':task,'true_option':true,
                                          'true_class':names[task][true]['english_class'],
                                          'source_model':f'M{a}','target_model':f'M{b}','comparison_phase':phase,
                                          'source_prediction':source,'source_prediction_class':names[task][source]['english_class'] if source!='INVALID' else 'Invalid/unparsed response',
                                          'target_prediction':target,'target_prediction_class':names[task][target]['english_class'] if target!='INVALID' else 'Invalid/unparsed response',
                                          'transition_type':cat,'count':count,'true_class_samples':len(indices),
                                          'fraction_of_class':count/len(indices),'fraction_of_transition_type':count/kinds[cat],
                                          'fraction_of_source_prediction':count/source_counts[source]})
                for a,b in itertools.combinations(range(5),2):
                    forward,_=pair_lookup[task,true,a,b];reverse,_=pair_lookup[task,true,b,a]
                    assert all(count==reverse[y,x] for (x,y),count in forward.items())
        # All 5 states and all 20 distinct ordered model pairs are expanded in readable tables.
        lines = [f'# {" → ".join(tasks)}：全部模型逐类别错误与完整模型对转移', '',
                 '模型对应：M0=base；'+'；'.join(f'M{i}=训练完{task}' for i,task in enumerate(tasks,1))+'。', '',
                 '覆盖四个任务的 33 个真实类别：每类展示全部 5 个模型的错误，以及全部 20 个非自身有向模型对。CSV 另保留 5 个自身比较。', '',
                 '所有计数来自同一完整测试集上的逐样本配对。任务尚未训练阶段和反向模型对也保留作对照；只有任务已学习后的正向正确→错误转移才适合作为阶段遗忘分析。', '',
                 '百分比口径：单模型错误表按该模型该真实类别的错误总数计算；转移表每一列分别按该列转移类型的总数计算。错误→不同错误列同时列出原错误选项和新错误选项。未出现的流向为零，CSV 流向明细仅存非零项。INVALID 单列。', '',
                 '## M0–M4：每个模型在每个真实类别上的所有错误', '']
        for task in TASKS:
            lines += [f'### {task}', '', '|真实类别|模型|正确数/测试数（正确率）|全部错误目标：次数（占本类错误比例）|', '|---|---|---|---|']
            for true in names[task]:
                for stage in range(5):
                    counts=state_lookup[task,true,stage];n=samples[task,true]
                    errors=Counter({k:v for k,v in counts.items() if k!=true})
                    lines.append(f'|{label_name(task,true)}|M{stage}|{counts[true]}/{n}（{100*counts[true]/n:.1f}%）|{count_list(task,errors)}|')
            lines.append('')
        lines += ['## 全部 20 个非自身有向模型对：逐类别完整错误转移', '',
                  '保持同一错误单独计数；正确→错误、错误→正确、错误→不同错误均展示全部实际选项。每行五种状态的计数之和等于该类别测试样本数。', '']
        for task in TASKS:
            lines += [f'### {task}', '']
            for a,b in itertools.product(range(5),repeat=2):
                if a==b:continue
                phase=pair_lookup[task,next(iter(names[task])),a,b][1]['comparison_phase']
                desc={'reverse_comparison':'反向模型对照','old_task_forward':'已学任务的后续正向比较','includes_pretraining_control':'包含任务尚未训练阶段的对照'}[phase]
                lines += [f'#### M{a} → M{b}（{desc}）', '', '|真实类别|保持正确|正确→错误：新目标|错误→正确：原错误选项|错误→不同错误：原→新|保持同一错误|', '|---|---:|---|---|---|---:|']
                for true in names[task]:
                    flows,summary=pair_lookup[task,true,a,b]
                    lost=Counter({y:n for (x,y),n in flows.items() if x==true and y!=true})
                    gain=Counter({x:n for (x,y),n in flows.items() if x!=true and y==true})
                    changed=Counter({(x,y):n for (x,y),n in flows.items() if x!=true and y!=true and x!=y})
                    lines.append(f"|{label_name(task,true)}|{summary['correct_to_correct']}|{count_list(task,lost)}|{count_list(task,gain)}|{changed_wrong_list(task,changed)}|{summary['same_wrong']}|")
                lines.append('')
        report_path=folder/('order_'+'_'.join(tasks)+'.md')
        text='\n'.join(lines)+'\n';report_path.write_text(text,encoding='utf-8')
        if tuple(tasks)==FOCUS:focus_text=text
        validation_orders.append({'tasks':tasks,'report':str(report_path),'classes':33,'state_rows':165,
                                  'directed_nonself_class_comparisons':660,'including_self_class_comparisons':825})
    assert len(all_states)==1650 and len(all_summaries)==8250 and len(sources)==200 and focus_text
    write_csv(OUT/'all_model_class_error_states.csv',all_states)
    write_csv(OUT/'all_model_pair_transition_summary.csv',all_summaries)
    write_csv(OUT/'all_model_pair_prediction_transitions.csv',all_flows)
    index=['# 全部模型错误与模型对转移：报告索引','',
           '每个顺序均覆盖 M0–M4、四个任务全部33类、20个非自身有向模型对；包括任务未训练阶段的对照。','',
           '|训练顺序|逐类别完整报告|','|---|---|']
    for item in validation_orders:
        index.append('|'+ ' → '.join(item['tasks'])+'|[打开报告]('+item['report']+')|')
    index += ['', 'CSV：`all_model_class_error_states.csv` 为每类每模型错误汇总；`all_models_class_confusion.csv` 为完整混淆计数（含零）；`all_model_pair_transition_summary.csv` 为25个有向模型对的五种转移状态计数；`all_model_pair_prediction_transitions.csv` 保留全部非零“原预测→新预测”选项及比例。', '',
              '每个模型对的样本总数与源模型、目标模型的混淆矩阵边际均已逐项核对。反向比较是正向转移的转置；自身比较没有新增错误、恢复或错误选项改变。']
    (OUT/'all_model_error_reports_index.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    main_path=OUT/'semantic_patterns_report.md';main_text=main_path.read_text()
    marker='<!-- ALL_MODEL_ERROR_TRANSITIONS_START -->'
    main_text=re.sub(r'\n<!-- ALL_MODEL_ERROR_TRANSITIONS_START -->.*?<!-- ALL_MODEL_ERROR_TRANSITIONS_END -->\n','\n',main_text,flags=re.S)
    intro='\n'+marker+'\n## M0–M4 全阶段、全部模型对的逐类别错误\n\n'+('\n'.join(index[2:]))+'\n\n'
    intro += focus_text.replace('# AGNEWS', '### AGNEWS',1)+'\n<!-- ALL_MODEL_ERROR_TRANSITIONS_END -->\n'
    position='<!-- COMPLETE_ERROR_DESTINATIONS_START -->'
    assert position in main_text
    main_text=main_text.replace(position,intro+'\n'+position,1)
    main_text=main_text.replace('只分析任务已训练之后的阶段，不将未训练阶段称为遗忘。','现同时展示任务训练前后的所有模型和全部模型对；任务未训练阶段及反向比较保留作对照，不称为遗忘。',1)
    assert main_text.count(marker)==1
    main_path.write_text(main_text,encoding='utf-8')
    validation={'complete':True,'orders':validation_orders,'models_per_order':5,'classes_per_order':33,
                'state_summary_rows':len(all_states),'directed_model_pair_summary_rows':len(all_summaries),
                'nonzero_prediction_transition_rows':len(all_flows),'source_model_task_cells':len(sources),
                'every_directed_pair_including_self':True,'transition_counts_match_both_model_marginals':True,
                'correctness_change_matches_accuracy_change':True,'reverse_pairs_are_transposes':True,
                'self_comparisons_have_no_changed_predictions':True,'prediction_source_sha256':sources}
    (OUT/'all_model_transition_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with zipfile.ZipFile(OUT/'semantic_analysis_with_error_destinations.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.csv','.json','.md'):archive.write(p,str(p.relative_to(OUT)))
    print('Updated analysis and 10 full model-pair reports.',flush=True)
    print(json.dumps({k:v for k,v in validation.items() if k not in ('orders','prediction_source_sha256')},indent=2))


if __name__=='__main__':
    main()

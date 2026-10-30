"""Finalize readable attention rankings, separating special markers from input content."""
import gc
import json
import zipfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot_actual_answer_attention import OUT, ORIGINAL, STAGES, LAYERS, NAMES, COLORS, guide, export, write_csv
from model.utils import build_tokenizer, GENERAL_TEMPLATE


def main():
    tokenizer=build_tokenizer(type('Args',(),{})(),chat_template=GENERAL_TEMPLATE)
    special=set(tokenizer.all_special_ids)
    cases=json.loads((ORIGINAL/'selected_examples.json').read_text())['examples']
    rows=[];pages=0
    lines=['# 三个模型实际输出答案，最关注哪些输入token', '',
           '每个模型先实际贪心生成一个答案字母，再将这个实际输出接到输入后采集attention，没有输入gold。', '',
           'prediction query 是预测首个答案时的prompt末位置；answer query 是实际答案token接回输入后的位置。若关心为什么选D/E，应优先查看prediction。', '',
           '所有排名只对prompt key计算，排除输出token对自身的注意力。完整排名保留特殊token；内容/选项排名排除所有tokenizer special token、chat prefix和assistant header，不排除正常标点。评论词语排名进一步只保留标题/评论区域且包含字母或数字的token。权重不重新归一化；attention大小不等于因果贡献。', '']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'pdf.fonttype':42})
    with PdfPages(OUT/'answer_to_input_top_tokens.pdf') as pdf:
        guide(pdf,'Most-attended INPUT tokens: base / M1 / M2',[
              'Left column: prediction query before the first answer token. Right: actual generated answer query.',
              'Rows are M0 base, M1 after AGNEWS, M2 after YAHOO.',
              'Every model actually generated its own answer; no gold is supplied.',
              'Bars show top-10 content/option keys, excluding all special IDs, chat prefix and assistant header.',
              'Normal punctuation is retained. Weights are original 32-head means and are not renormalized.',
              'Full rankings including BOS/chat markers and review-word rankings are in the CSV and Markdown.',
              'The generated answer self-key is excluded. Attention is descriptive, not causal attribution.'])
        for case in cases:
            L=case['sequence_length'];captures={s:torch.load(OUT/'actual_answer_cache'/f"M{s}_{case['case_id']}.pt",weights_only=True,map_location='cpu') for s in STAGES}
            lines += [f"## {case['case_id']}：真实{case['true_stars']}星，参考标签{case['true_option']}（未输入）", '',
                      '实际输出：'+'；'.join(f"M{s}→{captures[s]['generated_token']}" for s in STAGES)+'。', '',
                      '标题：'+case['example']['title'], '',
                      '|层|模型输出|query类型|全部输入token Top5|排除特殊标记的内容/选项 Top5|标题/评论词语 Top5|', '|---:|---|---|---|---|---|']
            for layer in LAYERS:
                fig,axes=plt.subplots(3,2,figsize=(28,19))
                for row,stage in enumerate(STAGES):
                    mean=captures[stage]['attentions'][layer].float().mean(0).numpy()
                    for col,(mode,q) in enumerate([('prediction',L-1),('answer',L)]):
                        values=mean[q,:L];rank=np.argsort(-values)
                        eligible=[int(i) for i in rank if case['regions'][i] not in ('chat prefix','assistant header') and case['input_ids'][i] not in special]
                        words=[i for i in eligible if case['regions'][i] in ('title','review') and any(c.isalnum() for c in tokenizer.decode([case['input_ids'][i]]))]
                        word_rank={pos:i+1 for i,pos in enumerate(words)}
                        for i,pos in enumerate(rank,1):
                            rows.append({'case':case['case_id'],'model':NAMES[stage],'layer':layer,'query_type':mode,'query_position':q,
                                         'actual_answer':captures[stage]['generated_token'],'input_token_rank':i,'input_position':int(pos),
                                         'token_id':case['input_ids'][pos],'token':case['tokens'][pos],'region':case['regions'][pos],
                                         'head_mean_attention':float(values[pos]),'is_special_token':case['input_ids'][pos] in special,
                                         'content_or_options_scope':int(pos) in eligible,'title_review_word_rank':word_rank.get(int(pos))})
                        def fmt(indices):return '；'.join(f"{i}: `{case['tokens'][i]}` ({values[i]:.4f})" for i in indices).replace('|',r'\|')
                        lines.append(f"|{layer}|M{stage}→{captures[stage]['generated_token']}|{mode} pos {q}|{fmt(rank[:5])}|{fmt(eligible[:5])}|{fmt(words[:5])}|")
                        top=eligible[:10];ax=axes[row,col];v=values[top];y=np.arange(len(top))
                        ax.barh(y,v,color=COLORS[stage]);ax.invert_yaxis();ax.set_yticks(y,[f"{i}: {case['tokens'][i]}" for i in top],fontsize=13)
                        for j,x in enumerate(v):ax.text(x,j,f' {x:.4f}',va='center',fontsize=12)
                        ax.set_xlim(0,max(.001,float(v.max())*1.25));ax.set_xlabel('Original head-mean attention weight')
                        ax.set_title(f"{NAMES[stage]} | {mode} query | actual answer {captures[stage]['generated_token']}",fontsize=16)
                fig.suptitle(f"{case['case_id']} | block {layer} | most-attended INPUT tokens\nLeft: predicting the answer. Right: processing the actually generated answer.",fontsize=22)
                fig.text(.5,.015,'Top-10 prompt content/option keys; special IDs/chat prefix/assistant header excluded. Output self-key excluded; weights not renormalized. Not causal attribution.',ha='center',fontsize=12)
                fig.tight_layout(rect=(0,.045,1,.91),pad=3,w_pad=5,h_pad=4)
                export(fig,f"{case['case_id']}_layer{layer}_top_input_tokens",pdf);pages+=1
            lines.append('')
            del captures;gc.collect()
    assert pages==6
    write_csv(OUT/'all_input_token_attention_rankings.csv',rows)
    (OUT/'which_input_tokens_are_attended.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    v=json.loads((OUT/'validation.json').read_text());v['ranking_refinement']={'all_special_token_ids_excluded_from_content_bars':True,'full_rankings_keep_all_prompt_keys':True,'review_word_rank_also_saved':True,'rank_rows':len(rows)}
    (OUT/'validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'actual_answers_base_M1_M2_figures_and_data.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.pdf','.png','.csv','.json','.md','.pt'):z.write(p,str(p.relative_to(OUT)))
    print('Refined 6 ranking figures and complete token rankings, with special markers separated.',flush=True)


if __name__=='__main__':main()

"""M0/M1/M2: greedy generated answer token, prediction-query attention, and answer-query attention."""
from collections import defaultdict
import gc
import json
from pathlib import Path
import re
import traceback
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from plot_amazon_DE_probability_attention import OUT as ORIGINAL, RUN, SCOPE, MODEL_DIR, clean_token, save_json, write_csv
from plot_large_gold_attention import guide
from plot_prompt_only_base_attention import plot_curves
from model.utils import build_tokenizer, GENERAL_TEMPLATE

OUT=ORIGINAL/'actual_output_with_base'
PROMPT=ORIGINAL/'prompt_only_with_base'
STAGES=[0,1,2];LAYERS=[16,32]
NAMES={0:'M0 base',1:'M1 after AGNEWS',2:'M2 after YAHOO'}
COLORS={0:'#596773',1:'#3775BA',2:'#B64342'}
NORM=LogNorm(vmin=1e-5,vmax=1)


def export(fig,stem,pdf):
    fig.savefig(OUT/f'{stem}.pdf',dpi=140,facecolor='white')
    fig.savefig(OUT/f'{stem}.png',dpi=140,facecolor='white');pdf.savefig(fig,dpi=140);plt.close(fig)


def ticks(ax,labels,with_y=False):
    ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=10,ha='center',va='top')
    if with_y:ax.set_yticks(range(len(labels)),labels,fontsize=10)
    for text in [*ax.get_xticklabels(),*(ax.get_yticklabels() if with_y else [])]:
        if '[OUTPUT]' in text.get_text():text.set_color('#B64342');text.set_fontweight('bold')
    ax.tick_params(length=2,pad=4)


def main():
    OUT.mkdir(exist_ok=True);(OUT/'actual_answer_cache').mkdir(exist_ok=True)
    torch.set_num_threads(8)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})
    cases=json.loads((ORIGINAL/'selected_examples.json').read_text())['examples']
    tokenizer=build_tokenizer(type('Args',(),{})(),chat_template=GENERAL_TEMPLATE,padding_side='left',truncation_side='left')
    option_ids=[tokenizer.encode(c,add_special_tokens=False)[0] for c in 'ABCDE']
    model=None;active=None
    try:
        for stage in STAGES:
            missing=[c for c in cases if not (OUT/'actual_answer_cache'/f"M{stage}_{c['case_id']}.pt").exists()]
            if not missing:continue
            if model is None:
                save_json(OUT/'status.json',{'complete':False,'state':'loading_CPU_model','stage':stage})
                model=AutoModelForCausalLM.from_pretrained(MODEL_DIR,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,device_map={'':'cpu'})
                model.config.use_cache=False;model.config.pad_token_id=tokenizer.pad_token_id
                if stage:
                    model=PeftModel.from_pretrained(model,str(RUN/('01_AGNEWS' if stage==1 else '02_YAHOO')/'best_model'),adapter_name=f'M{stage}',is_trainable=False)
                active=stage
            if active!=stage:
                if active==0:model=PeftModel.from_pretrained(model,str(RUN/'01_AGNEWS/best_model'),adapter_name='M1',is_trainable=False)
                else:model.load_adapter(str(RUN/'02_YAHOO/best_model'),adapter_name='M2',is_trainable=False);model.set_adapter('M2')
                active=stage
            model.eval()
            for case in missing:
                print('Generate actual answer and collect attention:',NAMES[stage],case['case_id'],flush=True)
                save_json(OUT/'status.json',{'complete':False,'state':'CPU_generation_and_attention','stage':stage,'case':case['case_id']})
                prompt_ids=torch.tensor([case['input_ids']]);length=case['sequence_length']
                with torch.inference_mode():
                    generated=model.generate(input_ids=prompt_ids,attention_mask=torch.ones_like(prompt_ids),max_new_tokens=1,do_sample=False,use_cache=False,pad_token_id=tokenizer.pad_token_id)
                    assert torch.equal(generated[0,:length],prompt_ids[0]) and generated.shape[1]==length+1
                    answer_id=int(generated[0,-1]);answer=tokenizer.decode([answer_id]);assert answer in 'ABCDE' and len(answer)==1
                    result=model(input_ids=generated,attention_mask=torch.ones_like(generated),output_attentions=True,use_cache=False,logits_to_keep=2)
                assert result.logits.shape[1]==2 and len(result.attentions)==32
                probabilities=result.logits[0,0].float().softmax(-1).cpu();assert int(probabilities.argmax())==answer_id
                captures={layer:result.attentions[layer-1][0].detach().cpu() for layer in LAYERS}
                old=torch.load(PROMPT/'attention_cache'/f"M{stage}_{case['case_id']}.pt",weights_only=True,map_location='cpu')
                assert old['input_ids']==case['input_ids'] and int(old['full_vocab_probabilities'].argmax())==answer_id
                prefix_checks=[]
                for layer,values in captures.items():
                    assert values.shape==(32,length+1,length+1)
                    assert float((values.float().sum(-1)-1).abs().max())<.012
                    assert float(values.float().triu(1).abs().max())==0
                    assert float(values[:,:length,-1].abs().max())==0
                    prefix_checks.append({'layer':layer,'prompt_prefix_max_attention_difference':float((values[:,:length,:length].float()-old['attentions'][layer].float()).abs().max())})
                save={'model':NAMES[stage],'prompt_input_ids':case['input_ids'],'input_ids':generated[0].tolist(),
                      'generated_token_id':answer_id,'generated_token':answer,'answer_position':length,
                      'generation_method':'Actual greedy model.generate(max_new_tokens=1); single-letter task answer; no forced answer or gold.',
                      'attentions':captures,'prediction_position_probabilities':probabilities,'prefix_checks':prefix_checks,
                      'mode':'prompt + actual model-generated answer; CPU BF16 eager','gold_was_used_as_input':False}
                torch.save(save,OUT/'actual_answer_cache'/f"M{stage}_{case['case_id']}.pt")
                del generated,result,captures,old,save;gc.collect()
    finally:del model;gc.collect()
    save_json(OUT/'status.json',{'complete':False,'state':'rendering'})
    common=['Models: M0 unadapted base; M1 after AGNEWS; M2 after YAHOO.',
            'The prompt is identical across models. Each model ACTUALLY generates its own one-letter answer greedily.',
            'That model-generated token is appended for the second forward pass; no gold answer is supplied.',
            'Prediction query = last prompt position. It predicts the answer and cannot attend to the future output.',
            'Answer query = appended generated answer. It describes processing the chosen answer, not why it was chosen.',
            'For explaining the choice of D/E, inspect prediction-query rows first. Answer-query rows are a separate view.',
            'All comparisons retain every token and all 32 query heads at transformer blocks 16 and 32.',
            'Reference labels are metadata only. M0 may independently generate a token equal to the true label.',
            'Attention weights are descriptive and are not equivalent to causal contribution or gradient attribution.',
            'Rankings are restricted to prompt keys, excluding the appended answer itself.',
            'Both unrestricted rankings and content/option rankings excluding chat markers are supplied.',
            'CPU inference avoids the active training GPU. Raw tensors and exact token IDs are retained.']
    report=['# 模型实际输出与哪些输入token关联：base、M1、M2', '',
            '本版本先让模型真实贪心生成一个答案token，再把该实际答案接到原prompt后重新计算attention。没有拼接gold。任务要求单字母答案，因此只生成1个token。', '',
            '**看“为什么预测这个答案”优先读 prediction query（原prompt最后位置）；answer query 是读入实际输出答案后的位置，它不能单独解释此前选答案的原因。**', '',
            '所有排名使用32头均值、只针对prompt中的key，不把输出token对自身的注意力算作输入关联。完整排名含BOS和聊天标记；“内容/选项排名”只排除chat prefix和assistant header，保留标题、评论、问题、所有选项及指令，权重不重新归一化。', '']
    ranking_rows=[];prediction_rows=[];checks=[];counts={'full':0,'heads':0,'zoom':0,'top':0}
    with PdfPages(OUT/'actual_output_full_attention.pdf') as full_pdf, PdfPages(OUT/'actual_output_all_heads.pdf') as heads_pdf, PdfPages(OUT/'actual_output_token_zoom.pdf') as zoom_pdf, PdfPages(OUT/'answer_to_input_top_tokens.pdf') as top_pdf:
        for pdf,title in [(full_pdf,'Actual model output: full attention, all tokens'),(heads_pdf,'Actual answer token query: all 32 heads'),(zoom_pdf,'Prediction query and actual-answer query: token zoom'),(top_pdf,'Which prompt tokens receive the most attention?')]:guide(pdf,title,common)
        for case in cases:
            print('Render actual-output figures:',case['case_id'],flush=True)
            captures={s:torch.load(OUT/'actual_answer_cache'/f"M{s}_{case['case_id']}.pt",weights_only=True,map_location='cpu') for s in STAGES}
            length=case['sequence_length'];n=length+1;prompt_q=length-1;answer_q=length
            answer_names=' | '.join(f"M{s}={captures[s]['generated_token']}" for s in STAGES)
            report += [f"## {case['case_id']}：真实{case['true_stars']}星，参考标签{case['true_option']}", '',
                       f'实际生成：{answer_names}。', '', '原文标题：'+case['example']['title'], '',
                       '|层|模型|query|全部输入token中注意力最高的5项|排除聊天标记后最高的5项|', '|---:|---|---|---|---|']
            for stage in STAGES:
                c=captures[stage];assert c['prompt_input_ids']==case['input_ids'] and c['input_ids']==case['input_ids']+[c['generated_token_id']]
                assert not c['gold_was_used_as_input']
                prediction_rows.append({'case':case['case_id'],'reference_stars':case['true_stars'],'reference_label':case['true_option'],
                                        'model':NAMES[stage],'actual_generated_answer':c['generated_token'],'actual_generated_token_id':c['generated_token_id'],
                                        **{f'P_{letter}':float(c['prediction_position_probabilities'][t]) for letter,t in zip('ABCDE',option_ids)}})
                checks.append({'case':case['case_id'],'stage':stage,'actual_answer':c['generated_token'],'prefix_checks':c['prefix_checks']})
                write_csv(OUT/f"{case['case_id']}_M{stage}_tokens_with_actual_output.csv",[{'position':i,'token_id':t,'decoded_token':clean_token(tokenizer.decode([t])),
                          'role':'actual generated output' if i==length else 'prompt','is_prediction_query':i==prompt_q,'is_answer_query':i==answer_q}
                          for i,t in enumerate(c['input_ids'])])
            for layer in LAYERS:
                means={s:captures[s]['attentions'][layer].float().mean(0).numpy() for s in STAGES}
                top_views={}
                for stage in STAGES:
                    for mode,query in [('prediction',prompt_q),('answer',answer_q)]:
                        weights=means[stage][query,:length]
                        all_rank=np.argsort(-weights)
                        eligible=[i for i in all_rank if case['regions'][i] not in ('chat prefix','assistant header') and case['input_ids'][i] not in set(tokenizer.all_special_ids)]
                        top_views[stage,mode]=eligible[:10]
                        for rank,i in enumerate(all_rank,1):
                            ranking_rows.append({'case':case['case_id'],'model':NAMES[stage],'layer':layer,'query_type':mode,
                                                 'query_position':query,'actual_answer':captures[stage]['generated_token'],
                                                 'input_token_rank':rank,'input_position':int(i),'token_id':case['input_ids'][i],
                                                 'token':case['tokens'][i],'region':case['regions'][i],
                                                 'head_mean_attention':float(weights[i]),'content_or_options_scope':case['regions'][i] not in ('chat prefix','assistant header') and case['input_ids'][i] not in set(tokenizer.all_special_ids)})
                        def describe(indices):return '；'.join(f"{i}: `{case['tokens'][i]}` ({weights[i]:.4f})" for i in indices).replace('|', r'\|')
                        report.append(f"|{layer}|M{stage} 输出{captures[stage]['generated_token']}|{mode} pos {query}|{describe(all_rank[:5])}|{describe(eligible[:5])}|")
                    labels=[f'{i}: {t}' for i,t in enumerate(case['tokens'])]+[f"{length}: {captures[stage]['generated_token']} [OUTPUT]"]
                    left=4.4;bottom=4.4;size=n*.18;w=left+size+1.8;h=bottom+size+2.1
                    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,size/w,size/h])
                    im=ax.imshow(np.ma.masked_less(means[stage],1e-5),cmap='viridis',norm=NORM,aspect='equal',interpolation='nearest',rasterized=True)
                    ticks(ax,labels,True);ax.axhline(length-.5,color='#B64342',linewidth=1.4);ax.axvline(length-.5,color='#B64342',linewidth=1.4)
                    cax=fig.add_axes([(left+size+.35)/w,bottom/h,.3/w,size/h]);fig.colorbar(im,cax=cax)
                    ax.set_xlabel('Key token: all prompt tokens + actual model output',fontsize=16,labelpad=14);ax.set_ylabel('Query token',fontsize=16,labelpad=14)
                    fig.suptitle(f"{NAMES[stage]} | block {layer} | {case['case_id']}\nActual generated answer={captures[stage]['generated_token']} | reference label={case['true_option']} is NOT supplied",fontsize=20,y=1-.35/h)
                    fig.text(.5,.35/h,f'Query {prompt_q} predicts the answer. Query {answer_q} processes the actual generated answer. No gold token.',ha='center',fontsize=13)
                    export(fig,f"{case['case_id']}_M{stage}_layer{layer}_actual_full",full_pdf);counts['full']+=1
                    left=2.3;bottom=4.4;width=n*.18;height=7.2;w=left+width+1.8;h=bottom+height+2.1
                    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,width/w,height/h])
                    head_values=captures[stage]['attentions'][layer].float()[:,answer_q,:].numpy()
                    im=ax.imshow(np.ma.masked_less(head_values,1e-5),cmap='viridis',norm=NORM,aspect='auto',interpolation='nearest',rasterized=True)
                    ticks(ax,labels);ax.set_yticks(range(32),range(1,33),fontsize=12);ax.set_ylabel('All 32 query heads',fontsize=16)
                    cax=fig.add_axes([(left+width+.35)/w,bottom/h,.3/w,height/h]);fig.colorbar(im,cax=cax)
                    fig.suptitle(f"{NAMES[stage]} | block {layer} | {case['case_id']}\nActual answer query: {captures[stage]['generated_token']} at position {answer_q}",fontsize=20,y=1-.35/h)
                    fig.text(.5,.35/h,'This is the actual answer-token query; the appended key is generated by this model, not the gold answer.',ha='center',fontsize=13)
                    export(fig,f"{case['case_id']}_M{stage}_layer{layer}_actual_heads",heads_pdf);counts['heads']+=1
                # Readable ranking bars: prediction and actual-answer views separated.
                fig,axes=plt.subplots(3,2,figsize=(28,19))
                for row,stage in enumerate(STAGES):
                    for col,mode in enumerate(['prediction','answer']):
                        ax=axes[row,col];top=top_views[stage,mode];query=prompt_q if mode=='prediction' else answer_q
                        vals=means[stage][query,top];y=np.arange(len(top))
                        ax.barh(y,vals,color=COLORS[stage]);ax.invert_yaxis()
                        ax.set_yticks(y,[f"{i}: {case['tokens'][i]}" for i in top],fontsize=13)
                        for j,v in enumerate(vals):ax.text(v,j,f' {v:.4f}',va='center',fontsize=12)
                        ax.set_xlim(0,max(.001,float(vals.max())*1.25));ax.set_xlabel('Original head-mean attention weight')
                        ax.set_title(f"{NAMES[stage]} | {mode} query | actual answer {captures[stage]['generated_token']}",fontsize=16)
                fig.suptitle(f"{case['case_id']} | block {layer} | most-attended INPUT tokens\nLeft: while predicting the answer. Right: after reading the actual generated answer.",fontsize=22)
                fig.text(.5,.015,'Top-10 content/option prompt keys, excluding chat prefix and assistant header; output self-attention excluded; weights not renormalized. Not causal attribution.',ha='center',fontsize=12)
                fig.tight_layout(rect=(0,.045,1,.91),pad=3,w_pad=5,h_pad=4)
                export(fig,f"{case['case_id']}_layer{layer}_top_input_tokens",top_pdf);counts['top']+=1
                query_rows=np.stack([means[s][q,:] for q in [prompt_q,answer_q] for s in STAGES]);query_rows[:3,-1]=np.nan
                row_labels=[f'M{s} prediction query {prompt_q}\nnext={captures[s]["generated_token"]}' for s in STAGES]+[f'M{s} actual answer {captures[s]["generated_token"]}\nquery {answer_q}' for s in STAGES]
                shared_labels=[f'{i}: {t}' for i,t in enumerate(case['tokens'])]+[f'{length}: {answer_names} [OUTPUT]']
                for start in range(0,n,32):
                    stop=min(n,start+32);width=max(5.,(stop-start)*.46);left=3.5;bottom=5.;height=4.;w=left+width+1.7;h=bottom+height+1.9
                    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,width/w,height/h]);values=query_rows[:,start:stop]
                    im=ax.imshow(np.ma.masked_invalid(np.ma.masked_less(values,1e-5)),cmap='viridis',norm=NORM,aspect='auto',interpolation='nearest')
                    ax.set_xticks(range(stop-start),shared_labels[start:stop],rotation=90,fontsize=11,ha='center',va='top');ax.set_yticks(range(6),row_labels,fontsize=11)
                    for text in ax.get_xticklabels():
                        if '[OUTPUT]' in text.get_text():text.set_color('#B64342');text.set_fontweight('bold')
                    for i,j in np.ndindex(values.shape):
                        value=values[i,j]
                        if np.isnan(value):label='MASK';color='#707984'
                        else:
                            label=f'{value:.3f}' if value>=.001 else f'{value:.0e}';rgba=im.cmap(im.norm(max(value,1e-5)));color='white' if .299*rgba[0]+.587*rgba[1]+.114*rgba[2]<.5 else '#162033'
                        ax.text(j,i,label,ha='center',va='center',fontsize=10,color=color)
                    cax=fig.add_axes([(left+width+.3)/w,bottom/h,.23/w,height/h]);fig.colorbar(im,cax=cax)
                    fig.suptitle(f"{case['case_id']} | block {layer} | keys {start}–{stop-1}\nActual output tokens: {answer_names} | no gold supplied",fontsize=17,y=.967)
                    fig.text(.5,.025,'Top 3 rows: predicting the answer (future output masked). Bottom 3: actual generated answer query. Last output token is model-specific.',ha='center',fontsize=11)
                    export(fig,f"{case['case_id']}_layer{layer}_actual_zoom_{start:03d}_{stop-1:03d}",zoom_pdf);counts['zoom']+=1
            report.append('')
    write_csv(OUT/'all_input_token_attention_rankings.csv',ranking_rows)
    write_csv(OUT/'actual_generated_answers_and_probabilities.csv',prediction_rows)
    (OUT/'which_input_tokens_are_attended.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    # Base is added to both aggregate and case probability curves.
    meta=json.loads((SCOPE/'order_AGNEWS_YAHOO_AMAZON_DBPEDIA/classwise_metrics.json').read_text());manifest=json.loads(Path(meta['probe_manifest']).read_text())['AMAZON']
    probs={s:torch.load(meta['model_cache_paths'][s]['AMAZON'],weights_only=True,mmap=True,map_location='cpu')['log_probs'].exp().numpy() for s in STAGES};gold=np.array([ord(x)-65 for x in manifest['labels']])
    with PdfPages(OUT/'base_M1_M2_token_probabilities.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(18,8));plot_curves(ax,{s:probs[s][:,option_ids].mean(0) for s in STAGES},[f'{c} ({i+1} star)' for i,c in enumerate('ABCDE')],'Mean probability over identical 320 validation inputs')
        fig.suptitle('M0 base / M1 after AGNEWS / M2 after YAHOO',fontsize=22);fig.text(.5,.02,'First answer position, before output is appended. Full-vocabulary probabilities, no A–E renormalization.',ha='center',fontsize=13)
        fig.tight_layout(rect=(0,.05,1,.93),pad=2);export(fig,'base_M1_M2_probability_overview',pdf)
        fig,axes=plt.subplots(2,3,figsize=(20,12))
        for k,ax in enumerate(axes.flat[:5]):plot_curves(ax,{s:probs[s][gold==k][:,option_ids].mean(0) for s in STAGES},list('ABCDE'),f'True {k+1} star: 64 inputs')
        axes.flat[5].axis('off');axes.flat[5].text(.05,.8,'M0 = base\nM1 = after AGNEWS\nM2 = after YAHOO\n\nLabels are metadata, never supplied.\nAttention uses actual generated answers.',va='top',fontsize=16)
        fig.suptitle('Token probabilities by true class, with base included',fontsize=22);fig.tight_layout(rect=(0,.03,1,.93),pad=2);export(fig,'base_M1_M2_probabilities_by_class',pdf)
        for case in cases:
            fig,ax=plt.subplots(figsize=(18,8));values={s:torch.load(OUT/'actual_answer_cache'/f"M{s}_{case['case_id']}.pt",weights_only=True,map_location='cpu')['prediction_position_probabilities'][option_ids].numpy() for s in STAGES}
            plot_curves(ax,values,[f'{c} | id {t}\n{i+1} star' for i,(c,t) in enumerate(zip('ABCDE',option_ids))],f"Reference {case['true_stars']} star / label {case['true_option']}, not supplied")
            fig.suptitle(case['case_id']+' | actual greedy answers from all three models',fontsize=22);ax.set_ylim(0,1.05);fig.tight_layout(rect=(0,.04,1,.93),pad=2);export(fig,case['case_id']+'_base_M1_M2_probability',pdf)
    assert counts=={'full':18,'heads':18,'zoom':32,'top':6},counts
    validation={'complete':True,'gold_input':False,'input':'same prompt + each model actual greedy generated one-token answer',
                'actual_generate_called':True,'models':NAMES,'layers':LAYERS,'case_count':3,'checks':checks,'figure_counts':counts,
                'probability_pages':5,'rankings_scope':'Prompt keys only; full rankings and content/option rankings. Answer self-attention excluded from rankings.',
                'prefix_causally_blind_to_output':True,'attention_device':'cpu','reference_label_used_only_as_metadata':True}
    save_json(OUT/'validation.json',validation)
    (OUT/'README.md').write_text('''# 不用gold，加入base：模型实际答案与输入token

三个模型：M0 base、M1（AGNEWS后）、M2（YAHOO后）。先对相同prompt真实调用greedy generate生成一个答案token，再将各自实际生成的token接回输入，采集第16/32层attention。实际生成字母不是人为指定的gold。

优先查看 answer_to_input_top_tokens.pdf：每层每个样本一页，左列是预测答案前的query，右列是读入实际输出后的query，三行分别为base、M1、M2。每个条形图显示内容/选项输入区域中注意力最高的10个token及原始权重。完整排名见all_input_token_attention_rankings.csv；原始完整排名包含BOS等特殊token，条形图排除chat prefix和assistant header。输出自身不参与输入token排名。

actual_output_token_zoom.pdf逐token放大，前3行是预测时query，后3行是实际答案query；末列标出每个模型实际的输出字母。actual_output_full_attention.pdf每页一张全部token完整大图。actual_output_all_heads.pdf显示实际答案位置的全部32头。base_M1_M2_token_probabilities.pdf包含base的总体、分星级及三个样本概率曲线。

预测前query更直接对应选答案时读取的信息；实际答案query描述模型读入它已经生成的答案后的状态，不能用来证明此前为什么选该答案。注意力大小不是因果贡献。真实标签只在标题作为参考信息，没有输入模型。
''',encoding='utf-8')
    save_json(OUT/'status.json',{'complete':True,'state':'complete','gold_input':False,'models':NAMES})
    with zipfile.ZipFile(OUT/'actual_answers_base_M1_M2_figures_and_data.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.pdf','.png','.csv','.json','.md','.pt'):archive.write(p,str(p.relative_to(OUT)))
    print('Complete:',OUT,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(exist_ok=True);save_json(OUT/'status.json',{'complete':False,'state':'failed','error':str(error),'traceback':traceback.format_exc()});raise

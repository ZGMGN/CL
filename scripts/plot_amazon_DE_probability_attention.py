"""Compare M1/M2 answer-token probabilities and CPU eager attention on identical AMAZON inputs."""
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import textwrap
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from model.utils import build_tokenizer, GENERAL_TEMPLATE, MODEL_DIR
from tasks import AMAZON

SCOPE = ROOT/'results/standard_cl_all_orders_seed_1_bs8_ga2/classwise_eval_20261030_10orders'
RUN = ROOT/'results/standard_cl_AGNEWS_YAHOO_AMAZON_DBPEDIA_seed_1_bs8_ga2'
OUT = RUN/'evaluation_DE_prob_attention'
STAGES = [1,2]
LAYERS = [16,32]
COLORS = {1:'#3775BA',2:'#B64342'}
NAMES = {1:'M1 after AGNEWS',2:'M2 after YAHOO'}
REGIONS = ['chat prefix','title','review','question','A: 1 star','B: 2 stars','C: 3 stars','D: 4 stars','E: 5 stars','instruction','assistant header']


def save_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def clean_token(token):
    return token.replace('\n','\\n').replace('\r','\\r').replace('\t','\\t').replace(' ','·')


def region_map(prompt,offsets):
    markers=[(0,'chat prefix'),(prompt.index('###Title:'),'title'),(prompt.index('###Review:'),'review'),
             (prompt.index('What is the star rating'),'question')]
    for letter,stars in zip('ABCDE',range(1,6)):
        markers.append((prompt.index(f'{letter}. {stars} star'),f'{letter}: {stars} star'+('s' if stars!=1 else '')))
    markers += [(prompt.index('Answer with A, B, C, D or E only.'),'instruction'),
                (prompt.rindex('<|start_header_id|>assistant'),'assistant header')]
    markers.sort()
    regions=[]
    for start,end in offsets:
        region=markers[0][1]
        for position,name in markers:
            if position<=start:region=name
            else:break
        regions.append(region)
    return regions


def export(fig,stem,pdf):
    fig.tight_layout(rect=(0,.045,1,.93),pad=2)
    fig.savefig(OUT/f'{stem}.png',dpi=300,facecolor='white')
    fig.savefig(OUT/f'{stem}.pdf',facecolor='white')
    pdf.savefig(fig);plt.close(fig)


def curves(ax,values,labels,title,stages=STAGES):
    x=np.arange(len(labels))
    for stage in stages:
        ax.plot(x,values[stage],marker='o',linewidth=2.2,markersize=6,color=COLORS[stage],label=NAMES[stage])
    ax.set_xticks(x,labels,rotation=35 if len(labels)>5 else 0,ha='right' if len(labels)>5 else 'center')
    ax.set_ylabel('Full-vocabulary token probability');ax.set_ylim(bottom=0)
    ax.set_title(title,fontsize=14);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'attention_cache').mkdir(exist_ok=True)
    torch.set_num_threads(8)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    meta=json.loads((SCOPE/'order_AGNEWS_YAHOO_AMAZON_DBPEDIA/classwise_metrics.json').read_text())
    manifest=json.loads(Path(meta['probe_manifest']).read_text())['AMAZON']
    tokenizer=build_tokenizer(type('Args',(),{})(),chat_template=GENERAL_TEMPLATE,padding_side='left',truncation_side='left')
    assert '<|start_header_id|>assistant' in AMAZON.format_prompt({'title':'test','content':'text','label':0},tokenizer,False)
    option_ids=[tokenizer.encode(letter,add_special_tokens=False) for letter in 'ABCDE'];assert all(len(x)==1 for x in option_ids)
    option_ids=[x[0] for x in option_ids]
    dataset=[json.loads(line) for line in (ROOT/'data/AMAZON/val.jsonl').read_text().splitlines() if line.strip()]
    indices=manifest['indices'];gold=np.array([ord(label)-65 for label in manifest['labels']])
    probs={};cache={};argmax={}
    for stage in STAGES:
        cache[stage]=torch.load(meta['model_cache_paths'][stage]['AMAZON'],weights_only=True,mmap=True,map_location='cpu')
        lp=cache[stage]['log_probs'];assert len(lp)==len(indices)==320
        probs[stage]=lp.exp().numpy();argmax[stage]=lp.argmax(1).numpy()
        assert np.max(np.abs(probs[stage].sum(1)-1))<2e-5
    selections=[]
    for label in [0,2,4]:
        candidates=[]
        for i in np.where(gold==label)[0]:
            if argmax[1][i]!=option_ids[3] or argmax[2][i]!=option_ids[4]:continue
            row=dataset[indices[i]];prompt=AMAZON.format_prompt(row,tokenizer,include_answer=False)
            inputs=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
            length=len(inputs['input_ids'])
            if 45<=length<=320:
                confidence=min(float(probs[1][i,option_ids[3]]),float(probs[2][i,option_ids[4]]))
                candidates.append((confidence>=.65,confidence,length,int(i),prompt,inputs))
        assert candidates,f'No suitable D-to-E example for true star {label+1}'
        candidates.sort(key=lambda x:(not x[0],abs(x[2]-140),-x[1],x[3]))
        _,confidence,length,i,prompt,inputs=candidates[0]
        case={'case_id':f'{label+1}star_val{indices[i]}','probe_row':i,'validation_index':indices[i],
              'true_option':'ABCDE'[label],'true_stars':label+1,'example':dataset[indices[i]],'prompt':prompt,
              'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_ids':inputs['input_ids'],
              'tokens':[clean_token(tokenizer.decode([t])) for t in inputs['input_ids']],
              'offsets':inputs['offset_mapping'],'regions':region_map(prompt,inputs['offset_mapping']),
              'sequence_length':length,'min_cached_D_E_probability':confidence}
        selections.append(case)
    save_json(OUT/'selected_examples.json',{'selection':'For true stars 1,3,5 choose cached D-to-E cases, prefer confidence >=0.65 and length nearest 140 in [45,320]. These are illustrative cases, not a random accuracy sample.','examples':selections})
    for case in selections:
        write_csv(OUT/f"{case['case_id']}_tokens.csv",[{'position':i,'token_id':tid,'decoded_token':token,'region':region,'char_start':offset[0],'char_end':offset[1]}
                  for i,(tid,token,region,offset) in enumerate(zip(case['input_ids'],case['tokens'],case['regions'],case['offsets']))])
    save_json(OUT/'status.json',{'complete':False,'state':'probability_plots','device':'cpu','examples':[c['case_id'] for c in selections]})
    rows=[]
    for i,index in enumerate(indices):
        for stage in STAGES:
            for letter,tid in zip('ABCDE',option_ids):
                rows.append({'validation_index':index,'true_option':'ABCDE'[gold[i]],'model':NAMES[stage],
                             'token':letter,'token_id':tid,'full_vocab_probability':float(probs[stage][i,tid]),
                             'full_vocab_argmax_token':tokenizer.decode([int(argmax[stage][i])])})
    write_csv(OUT/'all_320_example_option_probabilities.csv',rows)
    means={stage:probs[stage].mean(0) for stage in STAGES}
    # Shared token coordinates: no M1-versus-M2 scatter, no probability renormalization.
    vocab_ids=np.argsort(-(means[1]+means[2])/2)
    vocab_ids=[int(t) for t in vocab_ids if max(means[1][t],means[2][t])>=.005][:16]
    probability_pdf=OUT/'token_probability_curves.pdf'
    with PdfPages(probability_pdf) as pdf:
        fig,axes=plt.subplots(1,2,figsize=(18,6))
        curves(axes[0],{s:means[s][option_ids] for s in STAGES},[f'{c} ({i+1} star)' for i,c in enumerate('ABCDE')],'Mean A-E probabilities over the same 320 inputs')
        curves(axes[1],{s:means[s][vocab_ids] for s in STAGES},[f'{clean_token(tokenizer.decode([t]))}\n#{t}' for t in vocab_ids],'Shared vocabulary tokens with appreciable mean probability')
        fig.suptitle('AMAZON: M1 after AGNEWS versus M2 after YAHOO',fontsize=20)
        fig.text(.5,.015,'First answer-token position; 64 validation examples per class. Full vocabulary softmax; A-E are not renormalized.',ha='center',fontsize=11)
        export(fig,'token_probability_overview',pdf)
        fig,axes=plt.subplots(2,3,figsize=(17,10))
        for label,ax in enumerate(axes.flat[:5]):
            curves(ax,{s:probs[s][gold==label][:,option_ids].mean(0) for s in STAGES},list('ABCDE'),f'True {label+1} star: 64 identical inputs')
        axes.flat[5].axis('off');axes.flat[5].text(.05,.8,'M1: after AGNEWS\nM2: after YAHOO\n\nD = 4 stars in AMAZON\nE = 5 stars in AMAZON\n\nNo answer token included in input.',va='top',fontsize=15)
        fig.suptitle('Token probability curves separated by true star rating',fontsize=20)
        export(fig,'token_probabilities_by_true_star',pdf)
        fig,axes=plt.subplots(2,1,figsize=(18,8))
        order=np.lexsort((np.asarray(indices),gold));x=np.arange(320)
        for ax,letter,col in zip(axes,['D','E'],[3,4]):
            for stage in STAGES:ax.plot(x,probs[stage][order,option_ids[col]],linewidth=1,color=COLORS[stage],alpha=.8,label=NAMES[stage])
            for boundary in [64,128,192,256]:ax.axvline(boundary-.5,color='#777777',linewidth=.7)
            ax.set_xticks([31.5+64*i for i in range(5)],[f'True {i+1} star' for i in range(5)])
            ax.set_ylabel(f'P({letter})');ax.set_ylim(0,1);ax.legend(frameon=False)
        fig.suptitle('Paired inputs: raw D and E probabilities, grouped by true class',fontsize=20)
        fig.text(.5,.015,'Within each class, inputs follow ascending saved validation index; lines are raw, unsmoothed values.',ha='center',fontsize=11)
        export(fig,'paired_input_D_E_probability_curves',pdf)
        fig,axes=plt.subplots(1,3,figsize=(18,6))
        for case,ax in zip(selections,axes):
            i=case['probe_row'];curves(ax,{s:probs[s][i,option_ids] for s in STAGES},list('ABCDE'),f"True {case['true_stars']} star | val index {case['validation_index']}")
        fig.suptitle('The same D-to-E examples used for attention maps',fontsize=20)
        export(fig,'selected_case_token_probabilities',pdf)
    print('Probability plots complete. Selected cases:',[(c['case_id'],c['sequence_length']) for c in selections],flush=True)
    del cache;gc.collect()
    model=None
    active_stage=None
    try:
        for stage in STAGES:
            missing=[c for c in selections if not (OUT/'attention_cache'/f"M{stage}_{c['case_id']}.pt").exists()]
            if not missing:continue
            if model is None:
                save_json(OUT/'status.json',{'complete':False,'state':'loading_CPU_model','device':'cpu','stage':stage})
                model=AutoModelForCausalLM.from_pretrained(MODEL_DIR,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,device_map={'':'cpu'})
                model.config.use_cache=False;model.config.pad_token_id=tokenizer.pad_token_id
                adapter=RUN/('01_AGNEWS' if stage==1 else '02_YAHOO')/'best_model'
                model=PeftModel.from_pretrained(model,str(adapter),adapter_name=f'M{stage}',is_trainable=False)
                active_stage=stage
            if stage==2 and active_stage!=2:
                model.load_adapter(str(RUN/'02_YAHOO/best_model'),adapter_name='M2',is_trainable=False)
                model.set_adapter('M2')
                active_stage=2
            model.eval()
            for case in missing:
                print(f"CPU attention {NAMES[stage]} | {case['case_id']} | L={case['sequence_length']}",flush=True)
                save_json(OUT/'status.json',{'complete':False,'state':'CPU_attention','stage':stage,'example':case['case_id'],'device':'cpu'})
                inputs={'input_ids':torch.tensor([case['input_ids']]),'attention_mask':torch.ones(1,case['sequence_length'],dtype=torch.long)}
                with torch.inference_mode():
                    output=model(**inputs,output_attentions=True,use_cache=False,logits_to_keep=1)
                assert output.attentions is not None and len(output.attentions)==32
                full_probs=output.logits[0,-1].float().softmax(-1).cpu()
                predicted=int(full_probs.argmax());expected_id=option_ids[3 if stage==1 else 4]
                assert predicted==expected_id,(stage,case['case_id'],tokenizer.decode([predicted]))
                attentions={layer:output.attentions[layer-1][0].detach().cpu() for layer in LAYERS}
                for layer,att in attentions.items():
                    assert att.shape==(32,case['sequence_length'],case['sequence_length'])
                    assert float((att.float().sum(-1)-1).abs().max())<.012
                    assert float(att.float().triu(1).abs().max())==0
                saved={'model':NAMES[stage],'adapter_dir':str(RUN/('01_AGNEWS' if stage==1 else '02_YAHOO')/'best_model'),
                       'input_ids':case['input_ids'],'prompt_sha256':case['prompt_sha256'],'attentions':attentions,
                       'full_vocab_probabilities':full_probs,'layers':LAYERS,'head_aggregation':'Raw all 32 query heads retained; plots explicitly average heads where stated.',
                       'device':'cpu','dtype':'bfloat16','attention_backend':'eager','query_position':'Last prompt token immediately before first answer-token prediction; no generated answer in input.'}
                path=OUT/'attention_cache'/f"M{stage}_{case['case_id']}.pt";torch.save(saved,path)
                del output,attentions,saved;gc.collect()
    finally:
        del model;gc.collect()
    print('CPU attention collection complete.',flush=True)
    attention_rows=[];probability_checks=[]
    with PdfPages(OUT/'attention_maps.pdf') as pdf:
        for case in selections:
            captures={s:torch.load(OUT/'attention_cache'/f"M{s}_{case['case_id']}.pt",weights_only=True,map_location='cpu') for s in STAGES}
            length=case['sequence_length'];tick_positions=np.unique(np.linspace(0,length-1,11,dtype=int))
            tick_labels=[f"{i}: {case['tokens'][i][:16]}" for i in tick_positions]
            for s in STAGES:
                new=captures[s]['full_vocab_probabilities'].numpy();old=probs[s][case['probe_row']]
                probability_checks.append({'case':case['case_id'],'stage':s,'cached_backend':'prior GPU/batch2','new_backend':'CPU eager/batch1',
                                           'max_A_E_probability_difference':float(np.max(np.abs(new[option_ids]-old[option_ids]))),
                                           'cpu_predicted_token':tokenizer.decode([int(new.argmax())])})
            fig,axes=plt.subplots(2,2,figsize=(18,15))
            for row,layer in enumerate(LAYERS):
                for col,stage in enumerate(STAGES):
                    ax=axes[row,col];values=captures[stage]['attentions'][layer].float().mean(0).numpy()
                    im=ax.imshow(np.ma.masked_less(values,1e-5),cmap='viridis',norm=LogNorm(vmin=1e-5,vmax=1),aspect='auto',interpolation='nearest',rasterized=True)
                    ax.set_xticks(tick_positions,tick_labels,rotation=60,ha='right',fontsize=8);ax.set_yticks(tick_positions,tick_positions)
                    ax.set_xlabel('Key token position / decoded token');ax.set_ylabel('Query token position')
                    ax.set_title(f'{NAMES[stage]} | block {layer} | mean of 32 heads',fontsize=13)
                    fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label='Attention weight (log color scale)')
            fig.suptitle(f"Full causal attention | true {case['true_stars']} star | val {case['validation_index']}",fontsize=20)
            fig.text(.5,.012,'Identical prompt tokens in both stages. Masked values below 1e-5; upper triangle is causally masked. Full raw head tensors are saved.',ha='center',fontsize=10)
            export(fig,f"{case['case_id']}_full_attention",pdf)
            fig,axes=plt.subplots(2,2,figsize=(20,13))
            for row,layer in enumerate(LAYERS):
                for col,stage in enumerate(STAGES):
                    ax=axes[row,col];values=captures[stage]['attentions'][layer].float()[:,-1,:].numpy()
                    im=ax.imshow(np.ma.masked_less(values,1e-5),cmap='viridis',norm=LogNorm(vmin=1e-5,vmax=1),aspect='auto',interpolation='nearest',rasterized=True)
                    ax.set_xticks(tick_positions,tick_labels,rotation=55,ha='right',fontsize=8);ax.set_yticks([0,7,15,23,31],[1,8,16,24,32]);ax.set_xlabel('Input key token');ax.set_ylabel('Query head (1-32)')
                    ax.set_title(f'{NAMES[stage]} | block {layer} | last prompt query')
                    fig.colorbar(im,ax=ax,fraction=.025,pad=.025)
            fig.suptitle(f"Answer-prediction attention by head | true {case['true_stars']} star | val {case['validation_index']}",fontsize=20)
            fig.text(.5,.012,'The query is the last prompt token, before generating D or E. Both stages use the same query position and identical input.',ha='center',fontsize=11)
            export(fig,f"{case['case_id']}_last_query_all_heads",pdf)
            fig,axes=plt.subplots(2,2,figsize=(20,13))
            for row,layer in enumerate(LAYERS):
                values={s:captures[s]['attentions'][layer].float()[:,-1,:].mean(0).numpy() for s in STAGES}
                top=np.argsort(-np.maximum(values[1],values[2]))[:18];top=np.sort(top)
                for stage in STAGES:
                    axes[row,0].plot(np.arange(len(top)),values[stage][top],marker='o',color=COLORS[stage],label=NAMES[stage],linewidth=2)
                axes[row,0].set_xticks(range(len(top)),[f"{i}: {case['tokens'][i][:18]}" for i in top],rotation=65,ha='right',fontsize=9)
                axes[row,0].set_title(f'Block {layer}: shared top-18 key tokens, mean heads');axes[row,0].set_ylabel('Original attention weight');axes[row,0].legend(frameon=False)
                active=[r for r in REGIONS if r in case['regions']];x=np.arange(len(active))
                for stage,offset in zip(STAGES,[-.18,.18]):
                    mass=[]
                    for region in active:
                        mask=np.array([r==region for r in case['regions']]);total=float(values[stage][mask].sum());mass.append(total)
                        attention_rows.append({'case':case['case_id'],'true_stars':case['true_stars'],'stage':stage,'layer':layer,'region':region,
                                               'tokens_in_region':int(mask.sum()),'last_query_mean_head_attention_mass':total,
                                               'mean_attention_per_region_token':total/int(mask.sum())})
                    axes[row,1].bar(x+offset,mass,width=.36,color=COLORS[stage],label=NAMES[stage])
                axes[row,1].set_xticks(x,active,rotation=55,ha='right',fontsize=10);axes[row,1].set_ylabel('Attention mass (sum over region tokens)');axes[row,1].set_title(f'Block {layer}: all prompt regions, mean heads');axes[row,1].legend(frameon=False)
            fig.suptitle(f"Where the answer-prediction query attends | true {case['true_stars']} star | val {case['validation_index']}",fontsize=20)
            fig.text(.5,.012,'Top-token panels omit other tokens without renormalizing. Region panels retain all tokens; longer regions can accumulate more mass.',ha='center',fontsize=11)
            export(fig,f"{case['case_id']}_attention_tokens_and_regions",pdf)
    write_csv(OUT/'attention_region_masses.csv',attention_rows)
    write_csv(OUT/'cached_vs_CPU_probability_checks.csv',probability_checks)
    guide='''# AMAZON：从集中输出 D 到集中输出 E 的概率与 attention

比较同一训练顺序 AGNEWS→YAHOO→AMAZON→DBPEDIA 的 M1（AGNEWS 后）和 M2（YAHOO 后）。两者都尚未训练 AMAZON。M1 的 D/E 不能直接解释为经过训练的星级判断。

## 概率曲线

`token_probability_curves.pdf` 共4页，依次为320条固定验证输入的平均A–E曲线及共同词表token曲线、按真实星级分组的概率曲线、同一批输入上P(D)/P(E)的原始曲线、三个attention样本的A–E概率。完整softmax分母是全词表，没有把A–E重新归一化。横轴为明确命名的token或相同输入的索引，不是两模型概率的散点拟合。共同词表图只显示至少一个模型的平均概率达到0.005的前16个token，其他token仍包含在softmax中。

## attention map

`attention_maps.pdf` 共9页，每个样本3页：第16/32个transformer block的完整因果attention（32个query head的均值）；最后一个prompt位置对全部输入token的32头attention；共享top18 token曲线及各输入区域的attention总量。第16/32层attention指attentions[15]/attentions[31]，不是hidden_states索引或最终输出头。

两模型输入完全相同，query是生成首个答案前的最后一个prompt token，不是已经生成的D或E。图中使用统一的对数色标，低于1e-5的值不着色，原始权重未重归一化。所有头的完整矩阵保存在attention_cache，逐token位置、解码文本与区域在各样本_tokens.csv。区域总attention受token数量影响，CSV同时给出每token均值。

三个样本分别来自真实一星、三星和五星，按缓存D→E结果、置信度和长度规则选择，是用于检查现象的案例，不用于估计发生率或因果效应。attention是描述性权重，不能单凭热图确定模型作答原因。

attention使用CPU、BF16和eager实现，以避免占用训练显卡。概率总体曲线来自之前GPU评估缓存；CPU/后端/批量差异可能带来小的概率差异，cached_vs_CPU_probability_checks.csv列出逐样本差异，已确认新采集仍为M1输出D、M2输出E。
'''
    (OUT/'README.md').write_text(guide,encoding='utf-8')
    save_json(OUT/'validation.json',{'complete':True,'tasks':meta['tasks'],'stage_adapters':{s:str(RUN/('01_AGNEWS' if s==1 else '02_YAHOO')/'best_model') for s in STAGES},
                                   'probability_examples':320,'examples_per_true_class':64,'attention_examples':[c['case_id'] for c in selections],
                                   'layers':LAYERS,'attention_tensor_indices':[15,31],'query_heads':32,'attention_device':'cpu','backend':'eager',
                                   'no_model_training':True,'row_sums_checked':True,'causal_mask_checked':True,'CPU_predictions_match_D_to_E':True,
                                   'probability_checks':probability_checks,'source_probe_manifest':meta['probe_manifest']})
    with zipfile.ZipFile(OUT/'probability_attention_figures_and_data.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.pdf','.png','.csv','.json','.md','.pt'):archive.write(p,str(p.relative_to(OUT)))
    save_json(OUT/'status.json',{'complete':True,'state':'complete','probability_pdf':str(probability_pdf),'attention_pdf':str(OUT/'attention_maps.pdf')})
    print('Complete probability curves and real CPU attention maps:',OUT,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(exist_ok=True);save_json(OUT/'status.json',{'complete':False,'state':'failed','error':str(error),'traceback':traceback.format_exc()});raise

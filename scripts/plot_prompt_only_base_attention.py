"""M0 base/M1/M2 large attention and probabilities on identical prompt-only AMAZON inputs."""
import gc
import json
from pathlib import Path
import re
import shutil
import traceback
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from plot_amazon_DE_probability_attention import OUT as ORIGINAL, RUN, SCOPE, MODEL_DIR, clean_token, save_json, write_csv
from plot_large_gold_attention import guide
from model.utils import build_tokenizer, GENERAL_TEMPLATE
from tasks import AMAZON

OUT=ORIGINAL/'prompt_only_with_base'
STAGES=[0,1,2]
LAYERS=[16,32]
NAMES={0:'M0 base',1:'M1 after AGNEWS',2:'M2 after YAHOO'}
COLORS={0:'#596773',1:'#3775BA',2:'#B64342'}
NORM=LogNorm(vmin=1e-5,vmax=1)


def export(fig,stem,pdf):
    fig.savefig(OUT/f'{stem}.pdf',dpi=140,facecolor='white')
    fig.savefig(OUT/f'{stem}.png',dpi=140,facecolor='white')
    pdf.savefig(fig,dpi=140);plt.close(fig)


def all_tokens(ax,labels,with_y=False):
    ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=10,ha='center',va='top')
    if with_y:ax.set_yticks(range(len(labels)),labels,fontsize=10)
    for text in [*ax.get_xticklabels(),*(ax.get_yticklabels() if with_y else [])]:
        if '[QUERY]' in text.get_text():text.set_color('#3775BA');text.set_fontweight('bold')
    ax.tick_params(length=2,pad=4)


def plot_curves(ax,values,labels,title):
    for stage in STAGES:
        ax.plot(range(len(labels)),values[stage],color=COLORS[stage],label=NAMES[stage],linewidth=2.4,marker='o',markersize=7)
    ax.set_xticks(range(len(labels)),labels,rotation=35 if len(labels)>5 else 0,ha='right' if len(labels)>5 else 'center')
    ax.set_title(title,fontsize=15);ax.set_ylim(bottom=0);ax.set_ylabel('Full-vocabulary next-token probability');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)


def main():
    OUT.mkdir(exist_ok=True);(OUT/'attention_cache').mkdir(exist_ok=True)
    torch.set_num_threads(8)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})
    cases=json.loads((ORIGINAL/'selected_examples.json').read_text())['examples']
    tokenizer=build_tokenizer(type('Args',(),{})(),chat_template=GENERAL_TEMPLATE,padding_side='left',truncation_side='left')
    ids=[tokenizer.encode(x,add_special_tokens=False)[0] for x in 'ABCDE']
    meta=json.loads((SCOPE/'order_AGNEWS_YAHOO_AMAZON_DBPEDIA/classwise_metrics.json').read_text())
    manifest=json.loads(Path(meta['probe_manifest']).read_text())['AMAZON']
    checks=[]
    for case in cases:
        prompt=AMAZON.format_prompt(case['example'],tokenizer,include_answer=False)
        assert prompt==case['prompt'] and tokenizer(prompt,add_special_tokens=False)['input_ids']==case['input_ids']
        assert manifest['indices'][case['probe_row']]==case['validation_index']
        case['axis_labels']=[f'{i}: {token}' for i,token in enumerate(case['tokens'])]
        case['axis_labels'][-1]+=' [QUERY]'
        write_csv(OUT/f"{case['case_id']}_prompt_tokens.csv",[{'position':i,'token_id':t,'decoded_token':clean_token(tokenizer.decode([t])),
                  'axis_label':case['axis_labels'][i],'is_prediction_query':i==len(case['input_ids'])-1,'region':case['regions'][i]}
                  for i,t in enumerate(case['input_ids'])])
        for stage in [1,2]:
            source=ORIGINAL/'attention_cache'/f"M{stage}_{case['case_id']}.pt";target=OUT/'attention_cache'/source.name
            capture=torch.load(source,weights_only=True,map_location='cpu')
            assert capture['input_ids']==case['input_ids'] and len(capture['input_ids'])==case['sequence_length']
            if not target.exists():shutil.copy2(source,target)
    save_json(OUT/'selected_prompt_only_examples.json',{'input_mode':'prompt only; no answer or gold token appended','reference_labels_are_metadata_only':True,'examples':cases})
    missing=[c for c in cases if not (OUT/'attention_cache'/f"M0_{c['case_id']}.pt").exists()]
    if missing:
        save_json(OUT/'status.json',{'complete':False,'state':'loading_CPU_base_model'})
        base=AutoModelForCausalLM.from_pretrained(MODEL_DIR,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,device_map={'':'cpu'})
        base.config.use_cache=False;base.eval()
        assert not getattr(base,'peft_config',None)
        try:
            for case in missing:
                print('CPU prompt-only base attention:',case['case_id'],flush=True)
                save_json(OUT/'status.json',{'complete':False,'state':'CPU_base_attention','case':case['case_id']})
                input_ids=torch.tensor([case['input_ids']])
                with torch.inference_mode():result=base(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),output_attentions=True,use_cache=False,logits_to_keep=1)
                assert len(result.attentions)==32
                captures={layer:result.attentions[layer-1][0].detach().cpu() for layer in LAYERS}
                torch.save({'model':'M0 base','adapter_dir':None,'input_ids':case['input_ids'],'prompt_sha256':case['prompt_sha256'],
                            'attentions':captures,'full_vocab_probabilities':result.logits[0,-1].float().softmax(-1).cpu(),
                            'device':'cpu','dtype':'bfloat16','attention_backend':'eager','input_mode':'prompt only; no gold or answer appended'},
                           OUT/'attention_cache'/f"M0_{case['case_id']}.pt")
                del result,captures;gc.collect()
        finally:del base;gc.collect()
    save_json(OUT/'status.json',{'complete':False,'state':'rendering_prompt_only_figures'})
    common=['Order: AGNEWS -> YAHOO -> AMAZON -> DBPEDIA.',
            'M0 is the unadapted base; M1 is after AGNEWS; M2 is after YAHOO.',
            'All three models receive the exact same prompt tokens. No answer token, gold token or EOS is appended.',
            'True star/label is shown only as reference metadata in titles, never appended to input.',
            'The final prompt token is marked [QUERY]. It predicts the first answer token.',
            'Block 16/32 correspond to attentions[15]/attentions[31]. Raw 32 query heads are saved.',
            'All axis tokens are shown in full with positions. Spaces are middle dots; newline escapes are preserved.',
            'Color scale is the same log scale in all models/layers. Weights are not renormalized.',
            'Full/head maps are one heatmap per large PDF page; zoom maps split all keys into windows of 32.',
            'Three fixed illustrative examples have true 1, 3, 5 stars and were selected for M1 D -> M2 E.',
            'Attention is descriptive, not by itself causal evidence of a model decision.',
            'Base attention collected on CPU; original M1/M2 CPU prompt-only attention reused. No training GPU used.']
    count_full=count_head=count_zoom=0
    prediction_rows=[]
    with PdfPages(OUT/'prompt_only_full_attention_M0_M1_M2.pdf') as full_pdf, PdfPages(OUT/'prompt_only_heads_M0_M1_M2.pdf') as head_pdf, PdfPages(OUT/'prompt_only_token_zoom_M0_M1_M2.pdf') as zoom_pdf:
        guide(full_pdf,'Prompt-only full attention: M0 base, M1, M2',common)
        guide(head_pdf,'Prompt-only all-head maps: last prompt query',common)
        guide(zoom_pdf,'Prompt-only readable token zoom: all three models',common+['Every tile has three rows (M0/M1/M2); the original head-mean weight is written in each cell.'])
        for case in cases:
            print('Render prompt-only comparison:',case['case_id'],flush=True)
            captures={s:torch.load(OUT/'attention_cache'/f"M{s}_{case['case_id']}.pt",weights_only=True,map_location='cpu') for s in STAGES}
            n=case['sequence_length'];q=n-1;labels=case['axis_labels'];predictions={}
            for stage,c in captures.items():
                assert c['input_ids']==case['input_ids'] and len(c['input_ids'])==n
                probabilities=c['full_vocab_probabilities'];assert abs(float(probabilities.double().sum())-1)<1e-4
                prediction=tokenizer.decode([int(probabilities.argmax())]);predictions[stage]=prediction
                for layer in LAYERS:
                    a=c['attentions'][layer];assert a.shape==(32,n,n)
                    assert float((a.float().sum(-1)-1).abs().max())<.012
                    assert float(a.float().triu(1).abs().max())==0
                checks.append({'case':case['case_id'],'stage':stage,'model':NAMES[stage],'input_length':n,'same_prompt_ids':True,
                               'next_token':prediction,'attention_shape':[32,n,n],'answer_appended':False})
                prediction_rows.append({'case':case['case_id'],'true_stars':case['true_stars'],'reference_label':case['true_option'],'model':NAMES[stage],
                                        'predicted_token':prediction,**{f'P_{c}':float(probabilities[t]) for c,t in zip('ABCDE',ids)}})
            for layer in LAYERS:
                means={s:captures[s]['attentions'][layer].float().mean(0).numpy() for s in STAGES}
                for stage in STAGES:
                    left=4.4;bottom=4.4;size=n*.18;w=left+size+1.8;h=bottom+size+2.1
                    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,size/w,size/h])
                    im=ax.imshow(np.ma.masked_less(means[stage],1e-5),cmap='viridis',norm=NORM,interpolation='nearest',aspect='equal',rasterized=True)
                    all_tokens(ax,labels,True);ax.axhline(q-.5,color='#3775BA',linewidth=1.3)
                    ax.set_xlabel('Key token: every input token and position',fontsize=16,labelpad=14);ax.set_ylabel('Query token: every input token and position',fontsize=16,labelpad=14)
                    cax=fig.add_axes([(left+size+.35)/w,bottom/h,.3/w,size/h]);fig.colorbar(im,cax=cax,label='Attention weight; log scale')
                    fig.suptitle(f"{NAMES[stage]} | block {layer} | {case['case_id']} | reference label={case['true_option']}\nPROMPT ONLY | predicted next token={predictions[stage]} | mean of 32 heads",fontsize=20,y=1-.35/h)
                    fig.text(.5,.35/h,f'Final input position {q} is the prediction query. Reference label is metadata only; no gold/answer token is present.',ha='center',fontsize=13)
                    export(fig,f"{case['case_id']}_M{stage}_layer{layer}_prompt_full",full_pdf);count_full+=1
                    left=2.3;bottom=4.4;width=n*.18;height=7.2;w=left+width+1.8;h=bottom+height+2.1
                    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,width/w,height/h])
                    values=captures[stage]['attentions'][layer].float()[:,q,:].numpy()
                    im=ax.imshow(np.ma.masked_less(values,1e-5),cmap='viridis',norm=NORM,interpolation='nearest',aspect='auto',rasterized=True)
                    all_tokens(ax,labels);ax.set_yticks(range(32),[str(i+1) for i in range(32)],fontsize=12)
                    ax.set_ylabel('All 32 query heads',fontsize=16);ax.set_xlabel('Key token: full decoded text and position',fontsize=16,labelpad=14)
                    cax=fig.add_axes([(left+width+.35)/w,bottom/h,.3/w,height/h]);fig.colorbar(im,cax=cax)
                    fig.suptitle(f"{NAMES[stage]} | block {layer} | {case['case_id']}\nPrompt-only query position {q} | predicted next token={predictions[stage]} | reference label={case['true_option']}",fontsize=20,y=1-.35/h)
                    fig.text(.5,.35/h,'All models use identical input and query positions. Reference answer is not part of the attention keys.',ha='center',fontsize=13)
                    export(fig,f"{case['case_id']}_M{stage}_layer{layer}_prompt_heads",head_pdf);count_head+=1
                rows=np.stack([means[s][q,:] for s in STAGES]);row_labels=[f"{NAMES[s]}\nquery {q}; next={predictions[s]}" for s in STAGES]
                for start in range(0,n,32):
                    stop=min(n,start+32);width=max(5.,(stop-start)*.46);left=3.5;bottom=3.3;height=2.3;w=left+width+1.7;h=bottom+height+1.8
                    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,width/w,height/h]);values=rows[:,start:stop]
                    im=ax.imshow(np.ma.masked_less(values,1e-5),cmap='viridis',norm=NORM,aspect='auto',interpolation='nearest')
                    ax.set_xticks(range(stop-start),labels[start:stop],rotation=90,fontsize=12,ha='center',va='top');ax.set_yticks(range(3),row_labels,fontsize=12)
                    for text in ax.get_xticklabels():
                        if '[QUERY]' in text.get_text():text.set_color('#3775BA');text.set_fontweight('bold')
                    for i,j in np.ndindex(values.shape):
                        value=values[i,j];label=f'{value:.3f}' if value>=.001 else f'{value:.0e}'
                        rgba=im.cmap(im.norm(max(value,1e-5)));color='white' if .299*rgba[0]+.587*rgba[1]+.114*rgba[2]<.5 else '#162033'
                        ax.text(j,i,label,ha='center',va='center',fontsize=10,color=color)
                    cax=fig.add_axes([(left+width+.3)/w,bottom/h,.23/w,height/h]);fig.colorbar(im,cax=cax)
                    fig.suptitle(f"{case['case_id']} | block {layer} | keys {start}–{stop-1} | reference label={case['true_option']}\nM0 base / M1 after AGNEWS / M2 after YAHOO | prompt only, no answer appended",fontsize=17,y=.965)
                    fig.text(.5,.025,'Rows: each model last-prompt-query attention, averaged over 32 heads. All tokens are included across consecutive tiles.',ha='center',fontsize=11)
                    export(fig,f"{case['case_id']}_layer{layer}_prompt_zoom_{start:03d}_{stop-1:03d}",zoom_pdf);count_zoom+=1
    write_csv(OUT/'three_model_case_predictions_and_probabilities.csv',prediction_rows)
    # Aggregate probability figures use identical fixed validation examples in the existing GPU caches.
    probs={s:torch.load(meta['model_cache_paths'][s]['AMAZON'],weights_only=True,mmap=True,map_location='cpu')['log_probs'].exp().numpy() for s in STAGES}
    gold=np.array([ord(x)-65 for x in manifest['labels']]);means={s:probs[s].mean(0) for s in STAGES}
    token_ids=np.argsort(-np.stack(list(means.values())).mean(0));token_ids=[int(t) for t in token_ids if max(means[s][t] for s in STAGES)>=.005][:16]
    with PdfPages(OUT/'prompt_only_probabilities_M0_M1_M2.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(20,7))
        plot_curves(axes[0],{s:means[s][ids] for s in STAGES},[f'{c} ({i+1} star)' for i,c in enumerate('ABCDE')],'Mean A–E probability: same 320 inputs')
        plot_curves(axes[1],{s:means[s][token_ids] for s in STAGES},[f'{clean_token(tokenizer.decode([t]))}\n#{t}' for t in token_ids],'Shared tokens: mean probability >= 0.005 in any model')
        fig.suptitle('Prompt-only next-token probabilities: M0 base, M1, M2',fontsize=22)
        fig.text(.5,.02,'64 validation samples per true class. Full-vocabulary softmax; displayed tokens are not renormalized.',ha='center',fontsize=12)
        fig.tight_layout(rect=(0,.06,1,.93),pad=2);export(fig,'three_model_probability_overview',pdf)
        fig,axes=plt.subplots(2,3,figsize=(20,12))
        for k,ax in enumerate(axes.flat[:5]):plot_curves(ax,{s:probs[s][gold==k][:,ids].mean(0) for s in STAGES},list('ABCDE'),f'True {k+1} star: 64 identical inputs')
        axes.flat[5].axis('off');axes.flat[5].text(.05,.85,'M0 = base\nM1 = after AGNEWS\nM2 = after YAHOO\n\nD = 4 stars, E = 5 stars\n\nReference labels are not input.',va='top',fontsize=17)
        fig.suptitle('Probability curves by true star rating: all three models',fontsize=22)
        fig.tight_layout(rect=(0,.03,1,.93),pad=2);export(fig,'three_model_probabilities_by_true_star',pdf)
        fig,axes=plt.subplots(2,1,figsize=(20,9));order=np.lexsort((np.asarray(manifest['indices']),gold))
        for ax,letter,column in zip(axes,['D','E'],[3,4]):
            for s in STAGES:ax.plot(range(320),probs[s][order,ids[column]],color=COLORS[s],label=NAMES[s],linewidth=1,alpha=.8)
            for boundary in [64,128,192,256]:ax.axvline(boundary-.5,color='#777777',linewidth=.7)
            ax.set_xticks([31.5+64*i for i in range(5)],[f'True {i+1} star' for i in range(5)]);ax.set_ylabel(f'P({letter})');ax.set_ylim(0,1);ax.legend(frameon=False)
        fig.suptitle('Paired-input D/E curves with base included',fontsize=22)
        fig.text(.5,.015,'All three models share the same validation input order; raw unsmoothed probabilities.',ha='center',fontsize=12)
        fig.tight_layout(rect=(0,.04,1,.94),pad=2);export(fig,'three_model_paired_D_E_curves',pdf)
        for case in cases:
            fig,ax=plt.subplots(figsize=(17,8));values={s:torch.load(OUT/'attention_cache'/f"M{s}_{case['case_id']}.pt",weights_only=True,map_location='cpu')['full_vocab_probabilities'][ids].numpy() for s in STAGES}
            plot_curves(ax,values,[f'{c} | token id {t}\n{i+1} star' for i,(c,t) in enumerate(zip('ABCDE',ids))],f"True {case['true_stars']} star; reference label={case['true_option']} (not input)")
            ax.set_ylim(0,1.05);fig.suptitle(case['case_id']+' | same CPU prompt-only inputs',fontsize=22)
            fig.text(.5,.02,'Each curve uses the same CPU eager backend as its attention capture.',ha='center',fontsize=13)
            fig.tight_layout(rect=(0,.05,1,.93),pad=2);export(fig,case['case_id']+'_three_model_probability',pdf)
    assert count_full==count_head==18 and count_zoom==32
    save_json(OUT/'validation.json',{'complete':True,'input_mode':'prompt only','answer_or_gold_appended':False,'same_prompt_ids_across_models':True,
                                   'models':NAMES,'layers':LAYERS,'attention_tensor_indices':[15,31],'heads':32,'attention_device':'cpu',
                                   'full_attention_pages':count_full+1,'head_attention_pages':count_head+1,'token_zoom_pages':count_zoom+1,'probability_pages':6,
                                   'probability_examples':320,'all_axis_tokens_labelled':True,'checks':checks})
    (OUT/'README.md').write_text('''# 当前版本：仅input，加入base

M0=原始base；M1=AGNEWS训练后；M2=YAHOO训练后。三个模型使用相同的三条评论、相同prompt和相同token位置。没有拼接gold、答案或结束符；真实标签仅为标题中的参考信息。末尾[QUERY]是实际prompt的最后一个token，用于预测首个答案。

- prompt_only_token_zoom_M0_M1_M2.pdf：33页，推荐。每页最多32个key token，三行依次为base、M1、M2；所有token完整标注，并给出原始权重。
- prompt_only_full_attention_M0_M1_M2.pdf：19页。每页一张第16/32层完整attention大图，两轴全部标token。
- prompt_only_heads_M0_M1_M2.pdf：19页。每页一张全部32头的最后query注意力图。
- prompt_only_probabilities_M0_M1_M2.pdf：6页。含320条验证输入的概率总体、分星级、D/E逐样本曲线，以及三个真实attention样本的三模型概率。

M1/M2复用原始prompt-only CPU缓存；base新增CPU BF16 eager前向，不使用训练显卡。每个样本概率使用同样CPU后端；320条总体曲线复用同一批GPU验证缓存。概率是全词表softmax，不对A–E重新归一化。Attention使用统一对数色标，权重小于1e-5不着色，所有token仍完整保留。原始attention数据在attention_cache，逐token对照在prompt_tokens.csv。
''',encoding='utf-8')
    save_json(OUT/'status.json',{'complete':True,'state':'complete','input_mode':'prompt only','models':NAMES})
    with zipfile.ZipFile(OUT/'prompt_only_base_M1_M2_figures_and_data.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.pdf','.png','.csv','.json','.md','.pt'):archive.write(p,str(p.relative_to(OUT)))
    print('Complete:',OUT,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(exist_ok=True);save_json(OUT/'status.json',{'complete':False,'state':'failed','error':str(error),'traceback':traceback.format_exc()});raise

"""Large, fully token-labelled M1/M2 attention, including an actual teacher-forced gold token."""
import csv
import gc
import json
from pathlib import Path
import re
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

from plot_amazon_DE_probability_attention import OUT as ORIGINAL, RUN, MODEL_DIR, STAGES, LAYERS, NAMES, COLORS, clean_token, save_json, write_csv
from model.utils import build_tokenizer, GENERAL_TEMPLATE
from tasks import AMAZON

OUT=ORIGINAL/'large_token_labeled'
NORM=LogNorm(vmin=1e-5,vmax=1)


def export(fig,stem,pdf):
    # Text stays vector in PDF. Single plots have large physical dimensions; PNG is a high-resolution preview.
    fig.savefig(OUT/f'{stem}.pdf',dpi=140,facecolor='white')
    fig.savefig(OUT/f'{stem}.png',dpi=140,facecolor='white')
    pdf.savefig(fig,dpi=140)
    plt.close(fig)


def axes_labels(ax,labels,gold,query_axis=False):
    ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=10,ha='center',va='top')
    if query_axis:ax.set_yticks(range(len(labels)),labels,fontsize=10)
    for text in ax.get_xticklabels():
        if '[GOLD]' in text.get_text():text.set_color('#B64342');text.set_fontweight('bold')
    if query_axis:
        for text in ax.get_yticklabels():
            if '[GOLD]' in text.get_text():text.set_color('#B64342');text.set_fontweight('bold')
    ax.tick_params(length=2,pad=4)


def guide(pdf,title,lines):
    fig,ax=plt.subplots(figsize=(17,12));ax.axis('off');fig.suptitle(title,fontsize=24,y=.965)
    ax.text(.01,.96,'\n'.join(lines),va='top',transform=ax.transAxes,fontsize=14,linespacing=1.65)
    fig.subplots_adjust(left=.035,right=.98,top=.92,bottom=.035)
    pdf.savefig(fig);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);(OUT/'gold_attention_cache').mkdir(exist_ok=True)
    torch.set_num_threads(8)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})
    cases=json.loads((ORIGINAL/'selected_examples.json').read_text())['examples']
    tokenizer=build_tokenizer(type('Args',(),{})(),chat_template=GENERAL_TEMPLATE,padding_side='left',truncation_side='left')
    option_ids=[tokenizer.encode(letter,add_special_tokens=False)[0] for letter in 'ABCDE']
    for case in cases:
        full=tokenizer(AMAZON.format_prompt(case['example'],tokenizer,include_answer=True),add_special_tokens=False)['input_ids']
        length=case['sequence_length'];assert full[:length]==case['input_ids']
        gold=tokenizer.encode(case['true_option'],add_special_tokens=False);assert len(gold)==1 and full[length]==gold[0]
        case['gold_token_id']=gold[0];case['gold_position']=length;case['gold_input_ids']=case['input_ids']+gold
        case['axis_labels']=[f'{i}: {token}' for i,token in enumerate(case['tokens'])]+[f"{length}: {case['true_option']} [GOLD]"]
        case['regions_with_gold']=case['regions']+['gold answer']
        write_csv(OUT/f"{case['case_id']}_all_tokens_including_gold.csv",[{'position':i,'token_id':tid,'decoded_token':clean_token(tokenizer.decode([tid])),
                  'axis_label':case['axis_labels'][i],'region':case['regions_with_gold'][i],'is_gold':i==length,
                  'is_prediction_query':i==length-1} for i,tid in enumerate(case['gold_input_ids'])])
    save_json(OUT/'selected_examples_with_gold.json',{'input_mode':'Identical prompt plus one true answer token, teacher forced in both models. No EOS appended.','examples':cases})
    model=None;active=None;checks=[]
    try:
        for stage in STAGES:
            missing=[c for c in cases if not (OUT/'gold_attention_cache'/f"M{stage}_{c['case_id']}.pt").exists()]
            if not missing:continue
            if model is None:
                save_json(OUT/'status.json',{'complete':False,'state':'loading_CPU_model','stage':stage})
                model=AutoModelForCausalLM.from_pretrained(MODEL_DIR,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,device_map={'':'cpu'})
                model.config.use_cache=False
                path=RUN/('01_AGNEWS' if stage==1 else '02_YAHOO')/'best_model'
                model=PeftModel.from_pretrained(model,str(path),adapter_name=f'M{stage}',is_trainable=False);active=stage
            if active!=stage:
                model.load_adapter(str(RUN/'02_YAHOO/best_model'),adapter_name='M2',is_trainable=False);model.set_adapter('M2');active=2
            model.eval()
            for case in missing:
                print('Collect teacher-forced gold attention:',NAMES[stage],case['case_id'],flush=True)
                save_json(OUT/'status.json',{'complete':False,'state':'CPU_gold_attention','stage':stage,'case':case['case_id']})
                ids=torch.tensor([case['gold_input_ids']]);length=case['sequence_length']
                with torch.inference_mode():result=model(input_ids=ids,attention_mask=torch.ones_like(ids),output_attentions=True,use_cache=False,logits_to_keep=2)
                assert len(result.attentions)==32 and result.logits.shape[1]==2
                probabilities=result.logits[0,0].float().softmax(-1).cpu()
                expected=option_ids[3 if stage==1 else 4];assert int(probabilities.argmax())==expected
                attention={layer:result.attentions[layer-1][0].detach().cpu() for layer in LAYERS}
                prefix_checks=[]
                old=torch.load(ORIGINAL/'attention_cache'/f"M{stage}_{case['case_id']}.pt",weights_only=True,map_location='cpu')
                for layer,values in attention.items():
                    assert values.shape==(32,length+1,length+1)
                    assert float((values.float().sum(-1)-1).abs().max())<.012
                    assert float(values.float().triu(1).abs().max())==0
                    assert float(values[:,:length,-1].abs().max())==0
                    prefix_checks.append({'layer':layer,'max_prompt_prefix_attention_difference_vs_old_CPU_prompt_only':float((values[:,:length,:length].float()-old['attentions'][layer].float()).abs().max())})
                torch.save({'model':NAMES[stage],'input_ids':case['gold_input_ids'],'prompt_sha256':case['prompt_sha256'],
                            'gold_position':length,'gold_token_id':case['gold_token_id'],'gold_token':case['true_option'],
                            'attentions':attention,'first_answer_probabilities':probabilities,
                            'after_gold_next_token_probabilities':result.logits[0,1].float().softmax(-1).cpu(),
                            'prefix_checks':prefix_checks,'mode':'teacher-forced gold input; CPU BF16 eager; first-answer prediction is at row L-1, gold-processing row is L'},
                           OUT/'gold_attention_cache'/f"M{stage}_{case['case_id']}.pt")
                del result,attention,old;gc.collect()
    finally:
        del model;gc.collect()
    full_path=OUT/'large_full_attention_all_tokens.pdf';heads_path=OUT/'large_head_attention_all_tokens.pdf';zoom_path=OUT/'attention_token_zoom_with_gold.pdf'
    common=['Order: AGNEWS -> YAHOO -> AMAZON -> DBPEDIA. M1 = after AGNEWS; M2 = after YAHOO.',
            'Three identical examples from true 1, 3, 5 stars. Every tick gives the full decoded token and position.',
            'Spaces are shown as a middle dot; literal newline escapes are preserved. Gold labels are red.',
            'Both models receive the same prompt followed by the same true answer token (teacher forcing).',
            'The final GOLD row is a real model attention row, collected with the gold token in the input.',
            'The row immediately before GOLD predicts the first answer; causal masking prevents it from seeing GOLD.',
            'The GOLD row processes a supplied answer and predicts the following token. It is not evidence of why D/E was chosen.',
            'Block 16/32 mean attentions[15]/attentions[31]. Raw 32 query heads are retained in data files.',
            'All plots use original weights, with one common log color scale. No attention renormalization.',
            'Large full/head PDFs have one heatmap per page and vector token labels. Zoom tiles keep every token readable.',
            'CPU inference avoids the training GPU. Raw token IDs and roles are in each all_tokens_including_gold.csv.']
    full_count=head_count=zoom_count=0
    with PdfPages(full_path) as full_pdf, PdfPages(heads_path) as head_pdf, PdfPages(zoom_path) as zoom_pdf:
        guide(full_pdf,'Large full attention: every input token plus GOLD',common)
        guide(head_pdf,'Large per-head attention: prediction query and GOLD query',common)
        guide(zoom_pdf,'Readable token-by-token attention tiles',common+['Each tile has four rows: M1/M2 first-answer prediction queries, then M1/M2 gold-processing queries.',
                                                                     'Column labels include every token, split over consecutive windows of at most 32 keys.',
                                                                     'Numbers are original head-mean attention weights. The future gold cell in prediction rows is causally masked.'])
        for case in cases:
            print('Render large labelled maps:',case['case_id'],flush=True)
            captures={s:torch.load(OUT/'gold_attention_cache'/f"M{s}_{case['case_id']}.pt",weights_only=True,map_location='cpu') for s in STAGES}
            labels=case['axis_labels'];n=len(labels);gold=case['gold_position']
            for stage in STAGES:
                saved=captures[stage];assert saved['input_ids']==case['gold_input_ids'] and saved['gold_token_id']==case['gold_token_id']
                checks.append({'case':case['case_id'],'stage':stage,'gold':case['true_option'],'gold_position':gold,
                               'predicted_first_answer':tokenizer.decode([int(saved['first_answer_probabilities'].argmax())]),'prefix_checks':saved['prefix_checks']})
            for layer in LAYERS:
                for stage in STAGES:
                    attention=captures[stage]['attentions'][layer].float().numpy();mean=attention.mean(0)
                    pitch=.18;left=4.4;bottom=4.4;size=n*pitch;w=left+size+1.8;h=bottom+size+2.1
                    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,size/w,size/h])
                    im=ax.imshow(np.ma.masked_less(mean,1e-5),cmap='viridis',norm=NORM,interpolation='nearest',aspect='equal',rasterized=True)
                    axes_labels(ax,labels,gold,True)
                    ax.axhline(gold-.5,color='#B64342',linewidth=1.4);ax.axvline(gold-.5,color='#B64342',linewidth=1.4)
                    ax.set_xlabel('Key token: every decoded token and position',fontsize=16,labelpad=14);ax.set_ylabel('Query token: every decoded token and position',fontsize=16,labelpad=14)
                    cax=fig.add_axes([(left+size+.35)/w,bottom/h,.3/w,size/h]);fig.colorbar(im,cax=cax,label='Attention weight; log scale')
                    fig.suptitle(f"{NAMES[stage]} | block {layer} | true {case['true_stars']} star | GOLD={case['true_option']}\n{case['case_id']} | mean of all 32 query heads | gold is supplied, not generated",fontsize=20,y=1-.35/h)
                    fig.text(.5,.35/h,'Final row/column: actual supplied GOLD token (red). Previous row: first-answer prediction; it cannot attend to the future gold.',ha='center',fontsize=13)
                    export(fig,f"{case['case_id']}_M{stage}_layer{layer}_full_all_tokens_gold",full_pdf);full_count+=1
                    for mode,query in [('prediction',gold-1),('gold',gold)]:
                        left=2.3;bottom=4.4;width=n*.18;height=7.2;w=left+width+1.8;h=bottom+height+2.1
                        fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,width/w,height/h])
                        values=attention[:,query,:].copy()
                        if mode=='prediction':values[:,-1]=np.nan
                        im=ax.imshow(np.ma.masked_invalid(np.ma.masked_less(values,1e-5)),cmap='viridis',norm=NORM,aspect='auto',interpolation='nearest',rasterized=True)
                        axes_labels(ax,labels,gold);ax.set_yticks(range(32),[str(i+1) for i in range(32)],fontsize=12)
                        ax.set_ylabel('Query head (all 32)',fontsize=16);ax.set_xlabel('Key token: full decoded token and position',fontsize=16,labelpad=14)
                        ax.axvline(gold-.5,color='#B64342',linewidth=1.5)
                        cax=fig.add_axes([(left+width+.35)/w,bottom/h,.3/w,height/h]);fig.colorbar(im,cax=cax,label='Attention weight; log scale')
                        role='last prompt token: predicts D/E; cannot see GOLD' if mode=='prediction' else f"supplied GOLD token {case['true_option']}: predicts the following token"
                        fig.suptitle(f"{NAMES[stage]} | block {layer} | {case['case_id']} | GOLD={case['true_option']}\nQuery position {query}: {role}",fontsize=20,y=1-.35/h)
                        fig.text(.5,.35/h,'Red column is the actual GOLD token; it is a masked future key for the prediction query.',ha='center',fontsize=13)
                        export(fig,f"{case['case_id']}_M{stage}_layer{layer}_{mode}_all_heads_gold",head_pdf);head_count+=1
                query_rows=np.stack([captures[s]['attentions'][layer].float().mean(0)[q,:].numpy() for q in [gold-1,gold] for s in STAGES])
                query_rows[:2,gold]=np.nan
                row_labels=[f'M1 prompt pos {gold-1}: predicts D',f'M2 prompt pos {gold-1}: predicts E',f"M1 GOLD pos {gold}: input {case['true_option']}",f"M2 GOLD pos {gold}: input {case['true_option']}"]
                for start in range(0,n,32):
                    stop=min(start+32,n);count=stop-start;left=4.0;bottom=3.3;width=max(5.,count*.46);height=2.6;w=left+width+1.7;h=bottom+height+1.8
                    fig=plt.figure(figsize=(w,h));ax=fig.add_axes([left/w,bottom/h,width/w,height/h]);values=query_rows[:,start:stop]
                    im=ax.imshow(np.ma.masked_invalid(np.ma.masked_less(values,1e-5)),cmap='viridis',norm=NORM,aspect='auto',interpolation='nearest')
                    ax.set_xticks(range(count),labels[start:stop],rotation=90,fontsize=12,ha='center',va='top');ax.set_yticks(range(4),row_labels,fontsize=12)
                    for text in ax.get_xticklabels():
                        if '[GOLD]' in text.get_text():text.set_color('#B64342');text.set_fontweight('bold')
                    for i,j in np.ndindex(values.shape):
                        value=values[i,j]
                        if np.isnan(value):label='MASK';color='#707984'
                        else:
                            label=f'{value:.3f}' if value>=.001 else f'{value:.0e}'
                            rgba=im.cmap(im.norm(max(value,1e-5)));color='white' if .299*rgba[0]+.587*rgba[1]+.114*rgba[2]<.5 else '#162033'
                        ax.text(j,i,label,ha='center',va='center',fontsize=10,color=color)
                    cax=fig.add_axes([(left+width+.3)/w,bottom/h,.23/w,height/h]);fig.colorbar(im,cax=cax)
                    fig.suptitle(f"{case['case_id']} | block {layer} | keys {start}–{stop-1} | GOLD={case['true_option']} ({case['true_stars']} star)\nEvery input key is displayed across consecutive tiles; weights are mean over 32 heads",fontsize=17,y=.965)
                    fig.text(.5,.025,'Top two rows: prediction before GOLD. Bottom two rows: processing supplied GOLD. MASK means the future token is causally invisible.',ha='center',fontsize=11)
                    stem=f"{case['case_id']}_layer{layer}_zoom_keys_{start:03d}_{stop-1:03d}"
                    export(fig,stem,zoom_pdf);zoom_count+=1
    # Probability curves now identify the exact gold token instead of only its star count.
    with PdfPages(OUT/'large_probability_curves_with_gold.pdf') as pdf:
        for case in cases:
            fig,ax=plt.subplots(figsize=(17,8))
            for stage in STAGES:
                old=torch.load(ORIGINAL/'attention_cache'/f"M{stage}_{case['case_id']}.pt",weights_only=True,map_location='cpu')
                ps=old['full_vocab_probabilities'][option_ids].numpy()
                ax.plot(range(5),ps,marker='o',linewidth=3,markersize=10,color=COLORS[stage],label=NAMES[stage])
                for i,value in enumerate(ps):ax.annotate(f'{value:.3f}',(i,value),xytext=(0,10 if stage==1 else -18),textcoords='offset points',ha='center',fontsize=13,color=COLORS[stage])
            g=ord(case['true_option'])-65;ax.axvspan(g-.16,g+.16,color='#6EA96B',alpha=.18,label=f"GOLD = {case['true_option']} (token id {case['gold_token_id']})")
            ax.set_xticks(range(5),[f"{c} | token id {tid}\n{i+1} star"+(' [GOLD]' if c==case['true_option'] else '') for i,(c,tid) in enumerate(zip('ABCDE',option_ids))],fontsize=15)
            ax.set_ylabel('Full-vocabulary next-token probability',fontsize=16);ax.set_ylim(0,1.05);ax.legend(fontsize=14,frameon=False);ax.grid(axis='y',alpha=.2)
            fig.suptitle(f"{case['case_id']} | Gold token: {case['true_option']} | M1 predicts D; M2 predicts E",fontsize=22)
            fig.text(.5,.02,'These probabilities are predicted BEFORE the gold answer is supplied; A–E probabilities are not renormalized.',ha='center',fontsize=13)
            fig.tight_layout(rect=(0,.06,1,.92),pad=2);export(fig,f"{case['case_id']}_large_probability_GOLD",pdf)
    assert full_count==12 and head_count==24
    expected_zoom=sum((len(c['gold_input_ids'])+31)//32*2 for c in cases);assert zoom_count==expected_zoom
    validation={'complete':True,'gold_in_input':True,'gold_mode':'teacher forcing; one gold answer token appended, no EOS',
                'prediction_row_cannot_see_gold':True,'gold_input_prefix_matches_prompt':True,'all_token_ticks_shown_without_truncation':True,
                'full_heatmap_pages':full_count+1,'head_heatmap_pages':head_count+1,'zoom_pages':zoom_count+1,
                'probability_pages':3,'layers':LAYERS,'models':NAMES,'checks':checks}
    save_json(OUT/'validation.json',validation)
    text='''# 大图版 attention：全部 token 与真实 GOLD

优先阅读 `attention_token_zoom_with_gold.pdf`：每页最多32个key token，全部标位置和完整文本，并给出每格权重。最后一组页面包含红色GOLD token（A/C/E）。同一页面前两行是M1/M2预测答案前的位置，后两行是接入相同真实答案后的gold位置。

`large_full_attention_all_tokens.pdf`：每页仅一张完整attention图，M1/M2、第16/32层和三个样本全部展开。每个横纵轴刻度均标记实际token，没有省略或截短。单页物理尺寸约32–39英寸，PDF文本为矢量，可放大阅读。

`large_head_attention_all_tokens.pdf`：全部32个头，每页一图，分别显示预测前query和gold query对所有key的权重。

`large_probability_curves_with_gold.pdf`：每条样本单独一页，明确gold字母、token ID、星级，以及M1/M2的真实预测概率。

GOLD不是预测出来的D/E：一星样本gold=A，三星gold=C，五星gold=E。新图真实地将该gold token接入输入重新前向计算（teacher forcing），并标红最后一行/列。因果掩码保证前面的预测位置看不到未来gold；gold行则描述模型处理已给定正确答案后的注意力，不能用它解释此前为何选D/E。两模型接收同一条prompt和同一个gold，未训练、未更新权重，使用CPU。

空白为因果掩码或低于1e-5的权重，不是丢失token。所有token均保留，完整原始矩阵在gold_attention_cache，token表含位置、ID、实际解码文本、gold标记和预测query标记。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    save_json(OUT/'status.json',{'complete':True,'state':'complete','full_pdf':str(full_path),'zoom_pdf':str(zoom_path)})
    with zipfile.ZipFile(OUT/'large_attention_all_tokens_and_gold.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ('.pdf','.png','.csv','.json','.md','.pt'):archive.write(p,str(p.relative_to(OUT)))
    print('Complete:',OUT,flush=True)
    print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        OUT.mkdir(exist_ok=True);save_json(OUT/'status.json',{'complete':False,'state':'failed','error':str(error),'traceback':traceback.format_exc()});raise

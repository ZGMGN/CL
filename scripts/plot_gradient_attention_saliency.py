"""Frozen CPU attention-gradient analysis; adapted from Zhang et al., ICLR 2026, eqs. 3-5.

First-answer NLL is evaluated at the last PROMPT position. Neither gold nor the
predicted answer is fed back as input: an answer-query row predicts the next token,
and cannot explain the answer already generated. A second shared E-vs-D objective
retains signs to make cross-model preference changes easier to interpret.
"""
import argparse
import gc
import json
from pathlib import Path
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

from plot_amazon_DE_probability_attention import OUT as ORIGINAL, RUN, MODEL_DIR, save_json, write_csv, clean_token
from model.utils import build_tokenizer, GENERAL_TEMPLATE

OUT = ORIGINAL / 'gradient_saliency_with_base'
LAYERS = (16, 32)
NAMES = ('M0 base', 'M1 after AGNEWS', 'M2 after YAHOO', 'M3 after AMAZON')
COLORS = ('#596773', '#3775BA', '#B64342', '#42949E')
ADAPTER_FOLDERS = (None, '01_AGNEWS', '02_YAHOO', '03_AMAZON')
PAPER = 'https://arxiv.org/html/2601.20279v1#S2.SS1'


def collect(cases, tokenizer):
    torch.set_num_threads(8)
    model = None
    try:
        for stage in range(len(NAMES)):
            missing = [c for c in cases if not (OUT / 'cache' / f"M{stage}_{c['case_id']}.pt").exists()]
            if not missing:
                continue
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_DIR, dtype=torch.bfloat16, attn_implementation='eager',
                    local_files_only=True, device_map={'': 'cpu'})
                active = 0
            if stage != active:
                folder = RUN / ADAPTER_FOLDERS[stage] / 'best_model'
                if active == 0:
                    model = PeftModel.from_pretrained(model, str(folder), adapter_name=f'M{stage}', is_trainable=False)
                else:
                    model.load_adapter(str(folder), adapter_name=f'M{stage}', is_trainable=False)
                    model.set_adapter(f'M{stage}')
                active = stage
            model.eval().requires_grad_(False)
            for case in missing:
                print('CPU attention-gradient:', NAMES[stage], case['case_id'], flush=True)
                save_json(OUT/'status.json', {'complete': False, 'state': 'CPU_backward', 'model': NAMES[stage], 'case': case['case_id']})
                ids = torch.tensor([case['input_ids']])
                embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                output = model(inputs_embeds=embeds, attention_mask=torch.ones_like(ids),
                               output_attentions=True, use_cache=False, logits_to_keep=1)
                logits = output.logits[0, -1].float()
                target = int(logits.argmax())
                actual_path = ORIGINAL/'actual_output_with_base/actual_answer_cache'/f"M{stage}_{case['case_id']}.pt"
                if actual_path.exists():
                    actual = torch.load(actual_path, weights_only=True)
                else:
                    with torch.no_grad():
                        generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                                   do_sample=False, max_new_tokens=1, use_cache=False,
                                                   pad_token_id=tokenizer.pad_token_id)
                    assert torch.equal(generated[:, :-1], ids)
                    actual = {'generated_token_id': int(generated[0, -1])}
                    del generated
                assert target == actual['generated_token_id']
                attention = tuple(output.attentions[layer-1] for layer in LAYERS)
                nll = -logits.log_softmax(-1)[target]
                grads = torch.autograd.grad(nll, attention, retain_graph=True)
                d, e = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in 'DE']
                contrast = logits[e] - logits[d]  # exactly log p(E) - log p(D)
                contrast_grads = torch.autograd.grad(contrast, attention)
                captures = {}
                for layer, a, g, cg in zip(LAYERS, attention, grads, contrast_grads):
                    a, g, cg = [x[0].detach().float().cpu() for x in (a, g, cg)]
                    product = a*g
                    raw = product.abs().tril().sum(0)
                    denominator = float(torch.linalg.vector_norm(raw))
                    assert denominator > 0 and torch.isfinite(raw).all()
                    normalized = raw/denominator
                    assert abs(float(torch.linalg.vector_norm(normalized))-1) < 1e-5
                    assert float(raw.triu(1).abs().max()) == 0
                    if layer == 32:
                        # At the final block, only the last prompt query can affect its last logit.
                        assert float(g[:, :-1].abs().max()) == 0
                    if 'attentions' in actual:
                        old = actual['attentions'][layer][:, :-1, :-1].float()
                        assert float((old-a).abs().max()) == 0
                    captures[layer] = {'attention': a, 'nll_gradient': g,
                                       'saliency_sum_heads': raw, 'saliency_l2': normalized,
                                       'saliency_l2_denominator': denominator,
                                       'signed_E_minus_D': (a*cg).sum(0),
                                       'E_minus_D_gradient': cg}
                assert all(p.grad is None and not p.requires_grad for p in model.parameters())
                result = {'model': NAMES[stage], 'input_ids': case['input_ids'],
                          'prompt_sha256': case['prompt_sha256'], 'target_id': target,
                          'target': tokenizer.decode([target]), 'target_probability': float(logits.detach().softmax(-1)[target]),
                          'nll': float(nll.detach()), 'E_minus_D_log_odds': float(contrast.detach()),
                          'query_position': len(case['input_ids'])-1, 'gold_used_as_input': False,
                          'layers': captures, 'method': 'abs(A*dNLL/dA), tril, sum heads, global Frobenius L2',
                          'device': 'cpu', 'dtype': 'bfloat16', 'dropout': False}
                torch.save(result, OUT/'cache'/f"M{stage}_{case['case_id']}.pt")
                del output, embeds, logits, attention, grads, contrast_grads, nll, contrast, captures, result, actual, a, g, cg
                gc.collect()
    finally:
        del model
        gc.collect()


def finish(fig, stem, pdf, dpi=160):
    fig.savefig(OUT/f'{stem}.png', dpi=dpi, facecolor='white', bbox_inches='tight')
    pdf.savefig(fig, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def token_labels(case):
    return [f'{i}: {clean_token(t)}' for i, t in enumerate(case['tokens'])]


def review_word_positions(case, special):
    prompt = case['prompt']
    title_start = prompt.index('###Title:') + len('###Title:')
    review_header = prompt.index('###Review:')
    review_start = review_header + len('###Review:')
    question_start = prompt.index('What is the star rating')
    positions = []
    for i, (start, end) in enumerate(case['offsets']):
        in_body = title_start <= start < review_header or review_start <= start < question_start
        text = case['tokens'][i]
        for escape in ('\\n', '\\r', '\\t'):
            text = text.replace(escape, '')
        if in_body and case['input_ids'][i] not in special and any(c.isalnum() for c in text):
            positions.append(i)
    return positions


def render(cases, tokenizer):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white'})
    rows, summary, checks = [], [], []
    special = set(tokenizer.all_special_ids)
    with PdfPages(OUT/'attention_vs_saliency_token_zoom.pdf') as zoom, \
         PdfPages(OUT/'saliency_full_matrices.pdf') as full, \
         PdfPages(OUT/'saliency_top_input_tokens.pdf') as top, \
         PdfPages(OUT/'D_E_signed_preference.pdf') as contrast_pdf:
        for case in cases:
            cached = [torch.load(OUT/'cache'/f"M{s}_{case['case_id']}.pt", weights_only=True) for s in range(len(NAMES))]
            labels = token_labels(case)
            eligible = review_word_positions(case, special)
            for stage, data in enumerate(cached):
                summary.append({'case': case['case_id'], 'model': NAMES[stage], 'answer': data['target'],
                                'probability': data['target_probability'], 'nll': data['nll'],
                                'log_pE_minus_log_pD': data['E_minus_D_log_odds']})
            for layer in LAYERS:
                attention = [x['layers'][layer]['attention'].mean(0).numpy() for x in cached]
                saliency = [x['layers'][layer]['saliency_l2'].numpy() for x in cached]
                signed = [x['layers'][layer]['signed_E_minus_D'][-1].numpy() for x in cached]
                vmax = max(float(x.max()) for x in saliency)
                rawmax = max(float(x[-1].max()) for x in attention)
                saliency_norm = LogNorm(vmin=max(vmax*1e-5,1e-10), vmax=vmax)
                attention_norm = LogNorm(vmin=max(rawmax*1e-5,1e-10), vmax=rawmax)
                bound = max(float(np.abs(x).max()) for x in signed)
                for stage, data in enumerate(cached):
                    values = data['layers'][layer]
                    checks.append({'case': case['case_id'], 'model': NAMES[stage], 'layer': layer,
                                   'global_saliency_L2': float(values['saliency_l2'].norm()),
                                   'raw_global_L2': values['saliency_l2_denominator'],
                                   'nonzero_query_rows': int((values['saliency_sum_heads'].sum(1)>0).sum())})
                    for pos in range(len(labels)):
                        rows.append({'case': case['case_id'], 'model': NAMES[stage], 'answer': data['target'],
                                     'layer': layer, 'position': pos, 'token': case['tokens'][pos],
                                     'region': case['regions'][pos], 'raw_attention': float(attention[stage][-1,pos]),
                                     'saliency_l2': float(saliency[stage][-1,pos]),
                                     'saliency_raw': float(values['saliency_sum_heads'][-1,pos]),
                                     'signed_E_minus_D': float(signed[stage][pos])})
                    # One large matrix per page; every token labelled on both axes.
                    size = max(22, len(labels)*.21)
                    fig, ax = plt.subplots(figsize=(size, size))
                    matrix = np.ma.masked_where(np.triu(np.ones_like(saliency[stage]), 1)>0, saliency[stage])
                    im = ax.imshow(matrix, cmap='magma', norm=saliency_norm, interpolation='nearest', rasterized=True)
                    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=9)
                    ax.set_yticks(range(len(labels)), labels, fontsize=9)
                    ax.set_xlabel('Input key token'); ax.set_ylabel('Query token (loss is only on the first answer)')
                    ax.set_title(f"{case['case_id']} | {NAMES[stage]} | block {layer} | predicted {data['target']}\n"
                                 'Saliency = sum_heads |attention * gradient of answer NLL|; global L2 normalization', pad=18)
                    fig.colorbar(im, ax=ax, fraction=.035, pad=.02, label='Normalized saliency (log color scale shared across models)')
                    finish(fig, f"full_{case['case_id']}_M{stage}_L{layer}", full, dpi=110)
                # All tokens, 32 per page. Raw and weighted maps have independent labelled scales.
                for start in range(0,len(labels),32):
                    end = min(start+32,len(labels))
                    fig, axes = plt.subplots(2*len(NAMES),1,figsize=(max(11,(end-start)*.40),14),sharex=True)
                    fig.subplots_adjust(left=.16,right=.85,bottom=.32,top=.89,hspace=.5)
                    for stage in range(len(NAMES)):
                        for kind, mats, norm, cmap in ((0, attention, attention_norm, 'Blues'), (1, saliency, saliency_norm, 'magma')):
                            ax = axes[stage*2+kind]
                            im = ax.imshow(mats[stage][-1,start:end][None],aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
                            ax.set_yticks([0],[f"{NAMES[stage]}\n{'Attention' if kind==0 else 'Saliency'}\nanswer={cached[stage]['target']}"])
                            fig.colorbar(im,ax=ax, fraction=.025,pad=.02)
                    axes[-1].set_xticks(range(end-start),labels[start:end],rotation=90,fontsize=11)
                    fig.suptitle(f"{case['case_id']} | block {layer} | input tokens {start}–{end-1}\n"
                                 'Last prompt query: predicting the actual answer (no gold input)',fontsize=15)
                    fig.text(.16,.03,'Saliency uses abs(A × dNLL/dA), then sums heads and normalizes the full matrix.\n'
                             'Log color scales are shared across M0/M1/M2/M3 for each metric; zero cells are white.',fontsize=11)
                    finish(fig,f"zoom_{case['case_id']}_L{layer}_{start:03d}",zoom)
                # The review-only ranking is supplementary; full maps retain all tokens, including BOS.
                fig, axes = plt.subplots(len(NAMES),2,figsize=(17,19))
                for stage in range(len(NAMES)):
                    for col, mats, title in ((0, attention, 'Raw attention'), (1, saliency, 'Gradient-weighted saliency')):
                        ax = axes[stage,col]
                        selected = sorted(eligible,key=lambda p: float(mats[stage][-1,p]), reverse=True)[:12][::-1]
                        ax.barh(range(len(selected)),[mats[stage][-1,p] for p in selected],color=COLORS[stage])
                        ax.set_yticks(range(len(selected)),[labels[p] for p in selected],fontsize=11)
                        ax.set_title(f"{NAMES[stage]} | predicted {cached[stage]['target']} | {title}")
                        ax.set_xlabel('Attention probability' if col==0 else 'Saliency (global L2-normalized)')
                        ax.spines[['top','right']].set_visible(False)
                fig.suptitle(f"{case['case_id']} | block {layer} | top 12 title/review word tokens",fontsize=16)
                fig.tight_layout(rect=(0,.04,1,.96),pad=2)
                fig.text(.05,.015,'Ranking is restricted to title/review words; full matrices and CSV retain every token. Own predicted target may differ across models.',fontsize=10)
                finish(fig,f"top_{case['case_id']}_L{layer}",top)
                # Same objective at every checkpoint: avoids comparing gradients for different targets.
                fig, axes = plt.subplots(len(NAMES),1,figsize=(14,18))
                selection = sorted(eligible,key=lambda p: max(abs(s[p]) for s in signed), reverse=True)[:18]
                for stage, ax in enumerate(axes):
                    order = sorted(selection,key=lambda p: signed[stage][p])
                    ax.barh(range(len(order)),[signed[stage][p] for p in order],
                            color=['#B64342' if signed[stage][p]>0 else '#3775BA' for p in order])
                    ax.set_yticks(range(len(order)),[labels[p] for p in order],fontsize=10)
                    ax.axvline(0,color='#555555',lw=.7)
                    ax.set_title(f"{NAMES[stage]} | actual answer {cached[stage]['target']} | log p(E)/p(D) = {cached[stage]['E_minus_D_log_odds']:.3f}")
                    ax.set_xlabel('sum_heads A × gradient[log p(E) − log p(D)] (signed, unnormalized)')
                fig.suptitle(f"{case['case_id']} | block {layer} | common E vs D objective\n"
                             'Positive: increasing this attention edge locally favors E over D; negative favors D',fontsize=14)
                fig.tight_layout(rect=(0,0,1,.94),pad=2)
                finish(fig,f"contrast_{case['case_id']}_L{layer}",contrast_pdf)
    write_csv(OUT/'token_saliency.csv',rows)
    write_csv(OUT/'answers_and_objectives.csv',summary)
    save_json(OUT/'validation.json',{'complete':True,'models':len(NAMES),'examples':len(cases),'layers':list(LAYERS),
                                   'gold_used_as_input':False,'parameters_updated':False,'checks':checks})
    return summary


def report(cases, summary):
    text = '''# AMAZON：梯度加权 Attention 显著性分析

模型为 M0（base）、M1（AGNEWS 后）、M2（YAHOO 后）、M3（刚训练完 AMAZON），使用与上一版相同的三个 AMAZON 验证样本，分别分析第 16、32 个 Transformer block 的 attention。训练顺序为 AGNEWS → YAHOO → AMAZON → DBPEDIA；M3 对应 03_AMAZON/best_model。

## 计算过程

1. 输入只有原始 prompt，不拼接 gold，也不把预测答案重新输入。先核对当前最高概率 token 与上一版实际生成的答案完全一致。
2. 在最后一个 prompt 位置计算该预测答案 token 的负对数概率，也就是单 token 交叉熵损失。答案是模型自己的预测，并非正确标签。
3. 求此损失对各层、各头 softmax 后 attention 权重的梯度。逐元素计算 attention × 梯度，再取绝对值，保留下三角。
4. 将各头的绝对值矩阵求和，然后除以整个矩阵所有元素的 L2 范数，得到归一化显著性。先绝对值再汇总，避免不同头的正负梯度抵消。
5. 同时保留未归一化值、梯度和 attention 原始张量。全矩阵使用统一的跨模型色标；放大图将原始 attention 与显著性并列。

这里采用论文公式 3–5 的 Attention × Gradient 核心定义。范数明确实现为整个矩阵展平后的 L2 / Frobenius 范数。官方演示代码的绘图还使用 min–max/log 变换；本报告不将这些变换混入指标计算。

## 为什么不是拼接答案后的那一行

因果模型最后一个 prompt 位置的 logits 预测第一个答案 token。拼接答案后的 query 行用于预测答案之后的 token；它不能解释已经生成的答案。本次梯度在真正预测答案的位置计算。

第 32 层除最后一个 query 外出现全零梯度是正常现象：这一层其他位置不会再经后续 attention 传到最后位置。第 16 层较早 query 仍可能通过后续层影响答案。

## D/E 方向补充图

由于 M0、M1、M2、M3 可能输出不同字母，各自预测答案的损失目标不同。另加完全相同的目标：log p(E) − log p(D)，计算 attention × 该目标对 attention 的梯度，保留正负号，不取绝对值、不做 L2 归一化。这一图是本分析的补充，不是论文的原始绝对值 saliency。

正值表示局部增加该 attention 边（其他边保持不变）会提高 E 相对 D 的偏好，负值表示相反。它是 attention 边的局部一阶敏感性，不等于删除或替换输入词后的因果效果，也不等于词义对输出的完整贡献。

## 阅读文件

- attention_vs_saliency_token_zoom.pdf：每页最多 32 个输入 token，逐个标注位置和文本，上下比较四种模型的 attention 与 saliency。
- saliency_top_input_tokens.pdf：标题/评论中显著性最高的词 token，与原始 attention 排名并排。
- saliency_full_matrices.pdf：每页一个大矩阵，所有 query/key token 均有标注，包括特殊 token。
- D_E_signed_preference.pdf：统一 D/E 目标，观察偏好方向。
- token_saliency.csv：每个模型、样本、层、输入 token 的原始数值。
- cache/：梯度、attention、全矩阵显著性，CPU BF16 前向，float32 指标计算；未执行优化器更新。

绝对值显著性衡量局部敏感程度，不区分支持与反对预测，也不是准确率或遗忘分数。L2 归一化强调相对结构，不能据此断言某模型整体更依赖输入。三个样本是用于观察此前 D→E 现象的个案，不能推断整个任务的规律。

论文原本分析多模态模型生成过程中先前输出 token 对后续 token 的影响。本次将其核心显著性定义用于文本分类的输入 prompt → 第一个答案，是方法适配，不使用其幻觉阈值，也不将分类错误称为幻觉。

## 参考

ZHANG X, ZHU Y, GU C, et al. Hallucination Begins Where Saliency Drops[C]//The Fourteenth International Conference on Learning Representations. 2026.

论文：https://arxiv.org/abs/2601.20279

官方代码：https://github.com/zhangbaijin/LVLMs-Saliency
'''
    text += '\n## 实际答案\n\n|样本|模型|实际答案|该 token 概率|log p(E)/p(D)|\n|---|---|---|---|---|\n'
    for row in summary:
        text += f"|{row['case']}|{row['model']}|{row['answer']}|{row['probability']:.4f}|{row['log_pE_minus_log_pD']:.4f}|\n"
    text += '\n热图采用对数颜色刻度，显示同一层中强弱相差多个数量级的值；每个样本、层、指标在四个模型间共用色标。颜色下限为该组最大值的十万分之一，仅影响显示，不修改指标或 CSV。零值显示为白色。\n'
    (OUT/'README.md').write_text(text,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render_only',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True); (OUT/'cache').mkdir(exist_ok=True)
    cases = json.loads((ORIGINAL/'selected_examples.json').read_text())['examples']
    tokenizer = build_tokenizer(type('Args',(),{})(),chat_template=GENERAL_TEMPLATE,padding_side='left',truncation_side='left')
    if not args.render_only:
        collect(cases,tokenizer)
    save_json(OUT/'status.json',{'complete':False,'state':'rendering'})
    summary = render(cases,tokenizer)
    report(cases,summary)
    with zipfile.ZipFile(OUT/'gradient_saliency_figures_and_data.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix not in ('.zip','.tmp') and path.name != 'status.json':
                archive.write(path,path.relative_to(OUT))
    save_json(OUT/'status.json',{'complete':True,'state':'complete','output':str(OUT)})
    print('Complete:',OUT,flush=True)


if __name__ == '__main__':
    main()

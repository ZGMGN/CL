"""Reusable Chinese calculation notes for the sequential model/task report.

These notes describe the local numerical implementation. They do not recompute
metrics, change evaluation settings or load any model weights.
"""
import json
import re
from pathlib import Path
import unicodedata


def calculation_sections(payload):
    settings = payload['settings']
    tasks = payload['tasks']
    counts = {row['evaluated_task']: row['validation_probe_count'] for row in payload['model_task_results']}
    config_path = Path(payload['models'][0]['base_model']) / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    width, vocab = config.get('hidden_size', 'd'), config.get('vocab_size', 'V')
    layer = payload['summary_layer']
    depth = config.get('num_hidden_layers', 'L')
    count_text = '；'.join(f'{task}: N={count}' for task, count in counts.items())
    references = '；'.join(f'{task} → M{index + 1}' for index, task in enumerate(tasks))
    test_counts = []
    accuracy_source = Path(payload['accuracy_source']).parent
    for index, task in enumerate(tasks, 1):
        result_path = accuracy_source / '00_base' / f'{index:02d}_{task}.json'
        if result_path.is_file():
            total = json.loads(result_path.read_text())['metrics']['total']
            test_counts.append(f'{task} {total} 条')
    test_count_text = '本次完整测试集数量：' + '；'.join(test_counts) + '。' if test_counts else ''
    example = next(row for row in payload['task_learned_reference_results']
                   if row['evaluated_task'] == tasks[0] and row['model'] == payload['models'][-1]['model'])
    focus_layers = payload.get('selected_representation_layers', [layer])
    mean_note = ('报告先展示 last pooling，再追加第 16、32 层的 mean pooling。后者只排除 padding，平均所有有效提示 token，包括指令与类别选项；两方案使用同一批验证样本。'
                 if payload.get('mean_pooling_results_dir') else f"本报告采用 {settings['pooling']} pooling。")
    sections = []

    def section(title, blocks):
        sections.append({'title': title, 'blocks': blocks})

    section('计算说明 1：模型、数据与比较对象', [
        ('heading', '1. 固定模型身份与训练顺序'),
        ('paragraph', '本次训练顺序：' + ' → '.join(tasks) + '。M0 是未加载 adapter 的本地 base；Mj 是同一 base 加第 j 阶段保存的 best_model adapter。每个 adapter 包含此前全部训练任务的累计结果。'),
        ('heading', '2. 明确两个模型和一个评估任务'),
        ('paragraph', '记 A 为参照模型，B 为被比较模型。先固定评估任务 Ti，再让 A、B 接收该任务完全相同的输入。base 总览中 A=M0；任务专属参照总览中 A=Mi，即刚学完 Ti 的模型；两两比较图中 A、B 分别是行、列模型。'),
        ('paragraph', '本次任务专属参照：' + references + '。模型编号与训练顺序绑定，换顺序后会自动重新确定。'),
        ('heading', '3. 测试集与验证 probe 分开使用'),
        ('paragraph', '准确率使用 data/<TASK>/test.jsonl 的完整测试集。' + test_count_text + 'CKA、表征漂移、KL 使用固定的 val.jsonl 子集：' + '；'.join(f'{task} {counts[task]} 条' for task in tasks) + '。这些漂移指标不使用测试标签。'),
        ('paragraph', f"验证样本优先复用来源评估的索引、顺序及 SHA-256 内容 hash；首次抽样才按 seed+任务索引无放回抽取。本次 seed={settings['seed']}。同任务的所有模型严格复用同一批样本，不将不同任务的样本直接互算 CKA。"),
        ('heading', '4. 提示、截断与输出形状'),
        ('paragraph', f"使用任务格式化函数生成不含目标 assistant 答案的提示，保留任务输入和类别选项，附加生成提示。tokenizer 左侧 padding、左侧截断，最大输入 {settings['max_input_length']} tokens。模型设为 eval，关闭 dropout，不更新参数。"),
        ('paragraph', f"hidden_states 每层形状为 [batch, length, {width}]；pooling 后拼接为 X 或 Y，形状 [N, {width}]。首答案位置的 log probabilities 拼接为 [N, {vocab}]。batch_size={settings['batch_size']} 只控制计算分批，指标在拼接后的全部样本上汇总。"),
        ('heading', '5. 统计范围'),
        ('paragraph', f"{len(payload['models'])} 个模型 × {len(tasks)} 个任务得到 {len(payload['model_task_results'])} 组准确率；每任务比较全部不同模型对，共 {len(payload['pairwise_results'])} 组。增加模型对不会增加独立验证样本数。当前为点估计，未计算置信区间。"),
    ])

    section('计算说明 2：准确率、变化量与遗忘', [
        ('heading', '1. 逐条生成并判断答案'),
        ('paragraph', f"在完整测试集上，输入不含目标答案的提示，do_sample=False，最多生成 {settings['max_new_tokens']} 个新 token。只解码新增部分，用对应任务的 extract_answer_letter 提取类别答案，与该任务映射后的真实答案比较。解析失败也计入总样本数并判错。"),
        ('formula', r'$a(M,T)=\frac{1}{N_{\rm test}}\sum_{s=1}^{N_{\rm test}}\mathbf{1}[\hat y_s=y_s]$'),
        ('paragraph', 'JSON/CSV 的原始 accuracy 保存为 0–1；图和文字中的准确率乘 100，单位为 %。每个任务单独使用该任务全部测试样本，不把不同任务样本合并后计算一个总分。'),
        ('heading', '2. 相对 base 或上一阶段的变化'),
        ('formula', r'$\Delta_{\rm base}=100[a(M_j,T_i)-a(M_0,T_i)]$'),
        ('formula', r'$\Delta_{\rm previous}=100[a(M_j,T_i)-a(M_{j-1},T_i)]$'),
        ('paragraph', '变化量的单位为百分点（pp），正值代表改善；M0 没有上一阶段，因此 vs_previous 为 null。这里的百分点差不是相对百分比提升。'),
        ('heading', '3. 相对该任务刚训练后模型的变化与遗忘'),
        ('formula', r'$\Delta_{\rm learned}=100[a(M_j,T_i)-a(M_i,T_i)]$'),
        ('formula', r'$F_{i,j}=100[a(M_i,T_i)-a(M_j,T_i)]=-\Delta_{\rm learned},\quad j\geq i$'),
        ('paragraph', 'Mi 是刚学完任务 Ti 的模型，不是所有阶段中准确率最高的模型。j<i 的模型是训练前回溯对照，可以展示差值，但遗忘量输出 null；j=i 自身比较为 0；j>i 时正遗忘表示下降，负遗忘表示后续改善。'),
        ('paragraph', f"实际示例：在 {example['evaluated_task']} 上，参照 {example['reference_model']} 的准确率为 {100*example['reference_test_accuracy']:.2f}%，{example['model']} 为 {100*example['test_accuracy']:.2f}%，所以变化为 {example['test_accuracy_change_vs_task_learned_pp']:+.2f} pp。"),
        ('paragraph', '原 transition 表的 forgetting_pp 比较 M(j−1) 与 Mj 对旧任务的一步影响；本报告 forgetting_since_task_learned_pp 比较 Mi 与 Mj 的累计变化。两者的参照不同，应按列名区分。'),
    ])

    section('计算说明 3：提取表征与 Representation drift', [
        ('paragraph', mean_note),
        ('heading', '1. 选择同一层，并在同一输入位置提取表征'),
        ('paragraph', f"本报告汇总使用 hidden_states[{layer}]，pooling={settings['pooling']}。hidden_states[0] 是 embedding 输出，1–{depth} 对应各层输出，最后一个索引包含最终归一化。逐层图各层单独计算，汇总不是对所有层求平均。重点展示的表征层为 {focus_layers}；各层分别在同层、同任务上比较模型，层间结果对照不跨层互算 CKA。"),
        ('paragraph', 'last pooling 取提示最后一个非 padding token 的 hidden 向量。因为本实现左侧 padding，hidden[:, -1] 对应有效输入末尾；这是输入提示末尾的表示，不是生成答案之后的表示。'),
        ('formula', r'$x_s=H^A_{s,\ell,\rm last},\quad y_s=H^B_{s,\ell,\rm last}$'),
        ('paragraph', '如果运行时选择 mean pooling，则只对 attention_mask=1 的位置求均值；padding 不参与分子，分母是有效 token 数。'),
        ('formula', r'$x_s=\frac{\sum_t m_{st}H^A_{s,\ell,t}}{\max(1,\sum_t m_{st})}$'),
        ('paragraph', '图表统一使用 Representation drift（表征漂移）；具体度量为 Mean cosine distance，即逐样本余弦距离的均值。'),
        ('heading', '2. 逐样本计算 cosine，再变成距离'),
        ('formula', r'$c_s=\frac{x_s^T y_s}{\|x_s\|_2\|y_s\|_2},\qquad d_s=1-\operatorname{clip}(c_s,-1,1)$'),
        ('paragraph', '向量先转为 float64。只保留两个向量范数均非零的样本 I，逐样本计算 d_s，再对这些有效样本等权平均。全是零范数时返回 null/N/A。'),
        ('formula', r'$D_{\rm repr}=\frac{1}{|I|}\sum_{s\in I}d_s$'),
        ('heading', '3. 如何理解汇总值'),
        ('paragraph', '范围为 0–2：0 表示相同方向，1 表示正交，2 表示相反方向。使用 1−cos 是为了让数值随“漂移变大”而增大；不把所有样本展平成一个向量，以免大范数样本主导结果。'),
        ('paragraph', '对各 batch 的 pooled 向量先按样本顺序拼接，再对全部有效样本求平均；不是对各 batch 的均值再等权平均。clamp 仅抑制浮点误差导致的微小越界。本报告的 Representation drift（mean cosine distance）对幅度变化不敏感，也不等于测试准确率变化。'),
    ])

    section('计算说明 4：Linear CKA 的完整过程', [
        ('heading', '1. 组装对齐的表征矩阵'),
        ('paragraph', f"对同一任务、同一层与同一种 pooling，将 A、B 的 N 条验证样本向量按相同顺序堆叠，得到 X、Y，形状均为 [N, {width}]。它们的第 s 行必须来自同一个输入。不是把一对样本的 CKA 算完再平均。本次样本数：{count_text}。"),
        ('heading', '2. 对样本维中心化，再构造 Gram 矩阵'),
        ('formula', r'$X_c=X-\mathbf{1}\bar{x}^T,\quad Y_c=Y-\mathbf{1}\bar{y}^T,\qquad K=X_cX_c^T,\ L=Y_cY_c^T$'),
        ('paragraph', '每个特征减去其在 N 个样本上的均值。得到的 K、L 均为 N×N，描述同一个模型内部各输入样本之间的内积结构；例如 N=128 时，每个 Gram 为 128×128。'),
        ('heading', '3. 再对 Gram 做双中心化'),
        ('formula', r'$\widetilde K_{uv}=K_{uv}-\frac{1}{N}\sum_r K_{rv}-\frac{1}{N}\sum_r K_{ur}+\frac{1}{N^2}\sum_{r,q}K_{rq}$'),
        ('paragraph', 'L 使用同样操作得到双中心化矩阵 Lc。代码先做特征中心化，再做 Gram 双中心化；理想算术下后一步等价于清除剩余的行列均值，数值上也可抑制残差。'),
        ('heading', '4. Frobenius 内积除以两个 Frobenius 范数'),
        ('formula', r'$\operatorname{CKA}(X,Y)=\frac{\sum_{u,v}\widetilde K_{uv}\widetilde L_{uv}}{\|\widetilde K\|_F\|\widetilde L\|_F},\qquad \|K\|_F=\sqrt{\sum_{u,v}K_{uv}^2}$'),
        ('paragraph', '分子是两个中心化 Gram 对应元素相乘后求和，不是直接对矩阵乘积 K@L 的全部元素求和。等价地，可写为 tr(Kc^T Lc)。实现使用 float64，并将最后结果 clamp 到 0–1。'),
        ('heading', '5. 取值、异常与统计口径'),
        ('paragraph', '这里是 biased centered linear CKA，不采用 unbiased HSIC 修正，也不平均各 batch 的 CKA。任一中心化 Gram 范数为 0 时，输出 null/N/A；例如 last pooling 的 embedding 层若所有提示末 token 相同，可能零方差，不能强行填 1 或 0。'),
        ('paragraph', 'CKA 越接近 1，样本间的几何结构越相似；它对统一缩放、平移和正交特征变换不敏感。因此 CKA 高不保证逐向量 cosine 相同，也不保证输出分布或任务准确率相同。原 transition 中的 cka_drift 定义为 1−CKA。'),
    ])

    section('计算说明 5：KL 与 symmetric KL', [
        ('heading', '1. 获取首个答案位置的词表分布'),
        ('paragraph', f"A、B 输入完全相同的不含目标答案提示，只前向一次，不生成完整回答。logits_to_keep=1 保留最后一个输入位置的 logits，形状 [batch,1,{vocab}]，取 [:,-1] 后为 [batch,{vocab}]；该位置预测第一个答案 token。"),
        ('formula', r'$\ell^A_{sv}=z^A_{sv}-\log\sum_u e^{z^A_{su}},\quad p_{sv}=e^{\ell^A_{sv}};\qquad q_{sv}=e^{\ell^B_{sv}}$'),
        ('paragraph', '代码先将 logits 转 float32，再对整个词表做 log_softmax；KL 求和时将保存的 log probabilities 转 float64。没有只截取 A/B/C 等类别 token，也没有进行 top-k、temperature 调整或答案长度归一化。'),
        ('heading', '2. 每条样本分别计算正反 KL'),
        ('formula', r'$k_s^{A\to B}=\sum_{v=1}^{V}p_{sv}(\ell^A_{sv}-\ell^B_{sv})$'),
        ('formula', r'$k_s^{B\to A}=\sum_{v=1}^{V}q_{sv}(\ell^B_{sv}-\ell^A_{sv})$'),
        ('paragraph', 'exp 把 log probability 还原为 probability。每条样本在整个词表上求和；采用自然对数，单位为 nats。实现逐样本计算，降低大词表的临时内存占用；各样本的和使用 clamp_min(0) 抑制浮点误差导致的微小负值。'),
        ('heading', '3. 全部样本平均，再对两个方向平均'),
        ('formula', r'$D_{A\to B}=\frac{1}{N}\sum_s k_s^{A\to B},\quad D_{B\to A}=\frac{1}{N}\sum_s k_s^{B\to A}$'),
        ('formula', r'$D_{\rm sym}=\frac{D_{A\to B}+D_{B\to A}}{2}$'),
        ('paragraph', '原始 pair CSV 的 kl_a_to_b 和 kl_b_to_a 分别对应 model_a、model_b 的方向。报告另将 R 固定为 M0 或该任务刚训练后的 Mi，M 为被比较模型：KL(R || M) 是参照在第一项，KL(M || R) 是被比较模型在第一项。即使 M 早于 R，方向也不互换。'),
        ('paragraph', '方向矩阵单元格为 KL(行模型 P || 列模型 Q)，通常不对称；转置位置才是反向 KL。新增 ref 统计表保存两列的均值，并保留 symmetric KL。sym 对交换模型不变；原 transition 的 func_drift 是 sym。'),
        ('paragraph', 'KL 不使用表征 pooling，不区分第 16/32 层。本报告对 last/mean 运行的正反 KL 都进行了匹配校验，确认相同后只展示一份双向 KL 结果。'),
        ('heading', '4. 数值含义与范围'),
        ('paragraph', '值越大表示首答案位置的输出分布变化越大；0 表示分布相同，无固定上界。这里只平均样本，不平均词表 token 数，也不平均输入序列位置；该量不是完整生成序列的 KL，也不是 Jensen–Shannon divergence。自比较的 symmetric KL 直接记为 0。'),
    ])

    section('计算说明 6：关联 transition 的梯度 cosine', [
        ('heading', '使用范围'),
        ('paragraph', '本页解释原 sequential_drift_eval.py 的 grad_cosine；它不作为当前 5×任务单模型矩阵中的独立分数，也未在报告更新时重新计算。对旧任务 Ti 和新任务 Tj，在同一个训练前模型 M(j−1) 上比较梯度。'),
        ('heading', '1. 固定模型、输入样本与可求导参数'),
        ('paragraph', '模型使用 eval 模式关闭 dropout，仅使当前 LoRA A/B 参数 requires_grad=True；按参数名排序固定向量坐标。每任务使用原 probe 清单的前 grad_samples 条；这与表征分析使用完整 probe 清单的口径不同。'),
        ('heading', '2. 仅对答案 token 计算 loss'),
        ('paragraph', 'AnswerOnlyDataCollator 把提示和 padding 的 labels 设为 −100。因因果语言模型使用位移后的标签，单条样本的有效答案 token 数为 labels[:,1:] != −100 的数量 m_s。模型返回这些答案 token 的平均交叉熵 L_s。'),
        ('formula', r'$g_T=\frac{1}{\sum_s m_s}\sum_s\nabla_{\theta_{\rm LoRA}}(m_s L_s)$'),
        ('paragraph', '每条样本执行 (loss*tokens).backward()，累加总答案 loss 的梯度，最后除以总答案 token 数；这得到 token 加权平均梯度，不是将每条样本的平均 loss 梯度等权平均。只求梯度，不执行 optimizer.step()。'),
        ('heading', '3. 展平参数梯度并归一化内积'),
        ('formula', r'$C_{\rm grad}(T_i,T_j\mid M_{j-1})=\frac{g_{T_i}^Tg_{T_j}}{\|g_{T_i}\|_2\|g_{T_j}\|_2}$'),
        ('paragraph', '依排序把所有 LoRA 参数的梯度展平、拼接；没有梯度的参数补零。两个任务在相同模型状态、相同参数坐标上计算，最后用 float64 求 cosine 并 clamp 到 [−1,1]。零范数返回 null。'),
        ('heading', '4. 如何解释'),
        ('paragraph', '正值表示两个任务的局部梯度方向相近，负值表示相反，接近 0 表示近似正交。它描述采样点附近的局部一阶关系，不等于实际遗忘量，也不保证有限步训练后的表现。计算后清空梯度并恢复 requires_grad 与模型模式，不改模型权重。'),
    ])

    section('计算说明 7：关联 transition 的 LoRA 更新 cosine', [
        ('heading', '使用范围与有效更新'),
        ('paragraph', '本页解释原 transition 的 lora_cosine，当前报告未新增或重算该指标。它比较两个训练阶段实际有效权重增量的方向；不比较不同任务输入，也不直接对 LoRA A/B 元组做 cosine。'),
        ('formula', r'$D_s^{(m)}=\gamma_s^{(m)}B_s^{(m)}A_s^{(m)},\quad D_0^{(m)}=0,\qquad U_s^{(m)}=D_s^{(m)}-D_{s-1}^{(m)}$'),
        ('paragraph', 'm 为目标线性模块；A 的形状为 [r,in]，B 为 [out,r]。标准 LoRA 缩放 γ=alpha/r，rsLoRA 为 alpha/sqrt(r)。Ds 是第 s 阶段累计 adapter 相对 base 的有效更新；Us 才是任务 Ts 训练这一步的增量。'),
        ('heading', '1. 汇总所有模块的 Frobenius 内积'),
        ('formula', r'$C_{\rm LoRA}(U_i,U_j)=\frac{\sum_m\langle U_i^{(m)},U_j^{(m)}\rangle_F}{\sqrt{\sum_m\|U_i^{(m)}\|_F^2}\sqrt{\sum_m\|U_j^{(m)}\|_F^2}}$'),
        ('paragraph', '这是把所有模块的有效增量视作一个大向量后的全局 cosine，不是每个模块 cosine 的简单平均。零范数返回 null；数值误差导致的负平方范数压到 0，最终 cosine clamp 到 [−1,1]。'),
        ('heading', '2. 用低秩因子计算，避免构造 dense 矩阵'),
        ('formula', r'$\langle\gamma_1B_1A_1,\gamma_2B_2A_2\rangle_F=\gamma_1\gamma_2\sum_{u,v}(B_1^TB_2)_{uv}(A_1A_2^T)_{uv}$'),
        ('paragraph', 'factor_inner 用 r×r 的乘积精确计算有效更新内积。不同 LoRA 分解只要 BA 相同，得到相同有效更新，所以直接比较 BA 比拼接 A/B 更合适。'),
        ('formula', r'$\langle D_a-D_b,D_c-D_d\rangle=\langle D_a,D_c\rangle-\langle D_a,D_d\rangle-\langle D_b,D_c\rangle+\langle D_b,D_d\rangle$'),
        ('paragraph', 'update_inner 按上述四项展开；空字典代表 base 更新为 0。再分别求两份增量的自内积得到平方范数。该实现支持当前项目的标准、无 bias 线性 LoRA；不支持 DoRA、modules_to_save 和非统一 rank/alpha patterns。'),
        ('heading', '3. 如何解释'),
        ('paragraph', '正/负分别表示有效权重增量同向/反向，但它是权重空间关系，不等于梯度 cosine 或某一任务的 Representation drift。依次训练得到的累计 adapter 不能还原“只独立训练任务 Tj”的模型，也不能据此推断相反任务顺序的结果。'),
    ])
    return sections


def calculation_markdown(payload):
    lines = ['## 各指标的详细计算过程', '',
             '以下说明与本项目实际实现一致；仅补充计算口径，不改变既有评估数值。', '']
    for section in calculation_sections(payload):
        lines += ['### ' + section['title'], '']
        for kind, content in section['blocks']:
            if kind == 'heading':
                lines += ['**' + content + '**', '']
            elif kind == 'formula':
                lines += ['$$', content.strip('$'), '$$', '']
            else:
                lines += [content, '']
    return '\n'.join(lines)


def _wrap_display(text, width=148):
    """Wrap CJK paragraphs while keeping ASCII words and parameter names intact."""
    rows, current, used = [], '', 0
    tokens = re.findall(r'[A-Za-z0-9_./<>=:+\-]+|\s+|.', text)
    for token in tokens:
        size = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in token)
        if used + size > width and current:
            rows.append(current.rstrip())
            current, used = '', 0
            token = token.lstrip()
            size = sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in token)
        if size <= width:
            current += token
            used += size
        else:
            for character in token:
                character_size = 2 if unicodedata.east_asian_width(character) in ('W', 'F') else 1
                if used + character_size > width:
                    rows.append(current.rstrip())
                    current, used = '', 0
                current += character
                used += character_size
    if current:
        rows.append(current.rstrip())
    return rows


def plot_calculation_pages(payload, output_dir, font_path):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    if not font_path.is_file():
        raise FileNotFoundError(f'Chinese report font is required: {font_path}. See sequential_model_task_eval.md for setup.')
    chinese = FontProperties(fname=str(font_path))
    pages = []
    sections = calculation_sections(payload)
    for index, section in enumerate(sections, 1):
        content_rows = []
        for kind, content in section['blocks']:
            if kind == 'paragraph':
                content_rows += [(kind, row) for row in _wrap_display(content)]
                content_rows.append(('gap', ''))
            else:
                content_rows.append((kind, content))
        heights = {'paragraph': 19, 'heading': 29, 'formula': 72, 'gap': 8}
        height_points = max(760, sum(heights[kind] for kind, _ in content_rows) + 165)
        fig = plt.figure(figsize=(16, height_points / 72))
        fig.text(0.07, 1 - 40 / height_points, section['title'], fontproperties=chinese, fontsize=21, va='top', color='#0f172a')
        fig.text(0.07, 1 - 82 / height_points, '输入 → 逐步计算 → 汇总 → 结果解释；公式对应项目实际实现', fontproperties=chinese, fontsize=12, color='#475569', va='top')
        cursor = 120
        for kind, content in content_rows:
            if kind == 'gap':
                cursor += heights[kind]
                continue
            options = {'fontsize': 12, 'fontproperties': chinese, 'color': '#0f172a'}
            if kind == 'heading':
                options.update(fontsize=14, color='#1d4ed8')
            elif kind == 'formula':
                options = {'fontsize': 17, 'color': '#0f172a'}
            fig.text(0.07, 1 - cursor / height_points, content, va='top', **options)
            cursor += heights[kind]
        if cursor > height_points - 35:
            raise ValueError(f'Calculation notes overflow: {section["title"]}')
        fig.text(0.07, 24 / height_points, f'计算过程说明 {index}/{len(sections)} | 对照源码：sequential_drift_eval.py、drift_metrics.py、general_task_eval.py',
                 fontproperties=chinese, fontsize=10, color='#475569')
        fig.savefig(output_dir / f'calculation_methods_{index:02d}.png', dpi=150)
        pages.append(fig)
    return pages

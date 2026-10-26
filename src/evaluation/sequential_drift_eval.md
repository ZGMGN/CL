# 顺序训练的 CKA 与漂移评估

入口：`src/evaluation/sequential_drift_eval.py`。使用当前项目的本地 Llama 模型、
`data/<TASK>/val.jsonl` 和每阶段 `best_model/`，不更新任何模型权重。

```bash
cd /root/autodl-tmp/CL
python src/evaluation/sequential_drift_eval.py \
  --run_dir results/standard_cl_DBPEDIA_AMAZON_YAHOO_AGNEWS_seed_1_bs8_ga2 \
  --probe_samples 128 \
  --grad_samples 8 \
  --batch_size 2 \
  --eval_batch_size 16 \
  --max_input_length 1536 \
  --max_new_tokens 8 \
  --layers all
```

所有阶段 adapter 保存完成后再运行。不提供 `--accuracy_matrix` 时，会先调用
`sequential_heatmap_eval.py`，对基础模型及所有阶段 adapter 进行全部任务的完整测试集评估。
如果已经完成该评估，可避免重复生成：

```bash
python src/evaluation/sequential_drift_eval.py \
  --run_dir results/standard_cl_DBPEDIA_AMAZON_YAHOO_AGNEWS_seed_1_bs8_ga2 \
  --accuracy_matrix results/standard_cl_DBPEDIA_AMAZON_YAHOO_AGNEWS_seed_1_bs8_ga2/evaluation_heatmap/accuracy_matrix.json \
  --layers all
```

复用的矩阵必须完整，任务顺序、adapter 路径、输入长度和生成长度必须一致。
漂移 probe batch 和生成评估 batch 分别由 `--batch_size`、`--eval_batch_size` 控制。

## 测量定义

记 `M0` 为基础模型，`Mj` 为学完第 j 个任务后的最佳 adapter 模型。
每个 `Ti -> Tj`（i<j）表示训练 Tj 这一步对旧任务 Ti 的影响，比较 `M(j-1)` 与 `Mj`。
四任务顺序产生六条 transition。当前 checkpoint 不包含“只训练 Tj”的独立模型，
也不能据此生成相反任务顺序的结果。

| 字段 | 定义 |
| --- | --- |
| `grad_cosine` | 在共同的 `M(j-1)` 上，Ti 与 Tj 的答案 loss 对当前 LoRA 参数的梯度 cosine；eval 模式关闭 dropout，按答案 token 数平均，不做 optimizer step |
| `lora_cosine` | 实际有效权重增量 `Ui=Wi-W(i-1)` 与 `Uj=Wj-W(j-1)` 的全局 Frobenius cosine；W 包含 alpha/r（或 rsLoRA）缩放，使用低秩乘积精确计算，避免展开完整权重 |
| `rep_drift` | 固定 Ti 验证 probe 的逐样本 hidden-state cosine distance 均值 |
| `cka` | 相同样本、相同层的 biased centered linear CKA；1 表示高度相似 |
| `cka_drift` | `1-CKA` |
| `kl_forward` | 在相同无答案提示上，首个答案位置整个词表的 `KL(P_before || P_after)`，单位 nats |
| `func_drift` / `symmetric_kl` | 正反 KL 的算术平均；不代表整段生成序列的 KL |
| `forgetting_pp` | `100 × (acc(M(j-1),Ti)-acc(Mj,Ti))`，单位百分点，负值表示改善 |
| `cumulative_forgetting_pp` | `100 × (acc(Mi,Ti)-acc(Mj,Ti))`，相对刚学完 Ti 时的累计变化；不是历史最佳准确率减当前准确率 |

Probe 固定从验证集抽取，所有模型复用完全相同的样本与顺序；清单保存原始行索引与内容 hash。
梯度使用同一清单的前 `grad_samples` 条。准确率和遗忘量使用完整测试集，和验证 probe 区分。

`--pooling last`（默认）取无答案提示的最后一个非 padding token；`mean` 则仅平均非 padding token。
`--layers -1` 默认仅分析最后 hidden-state 层；`all` 分析所有层，包括 embedding 输出。
数字是 Hugging Face hidden_states 的索引（0 是 embedding，最后一个是最终层输出）。
分层结果保存在 `layer_metrics.csv`；transition 汇总列使用所选列表中的最后一层。

只需前向分析时可用 `--skip_grad_cosine`；此时 gradient 列为空，其他指标正常计算。
零范数 cosine、零方差 CKA 输出 null/N/A，不伪造为零。
LoRA 更新分析支持项目使用的标准线性 LoRA；DoRA、额外可训练 bias、modules_to_save、
非统一 rank/alpha 等配置会明确报错。

## 输出

默认输出到 `<run_dir>/evaluation_drift/`：

- `accuracy/`：未复用矩阵时生成的 `(n+1)×n` 全任务准确率矩阵和 heatmap。
- `model_drift_matrices.json`：相对基础模型的 `(n+1)×n` CKA、表征漂移、KL 矩阵及分层 CKA。
- `cka_vs_base_heatmap.png/.pdf`：各阶段模型与基础模型在各任务 probe 上的 CKA。
- `rep_drift_vs_base_heatmap.png/.pdf`、`symmetric_kl_vs_base_heatmap.png/.pdf`。
- `transition_metrics.csv/.json`：每条 transition 的指标、阶段编号和测量定义。
- `transition_metrics.png/.pdf`：参考图中的指标总览；各列使用独立色标。
- `layer_metrics.csv`：每条 transition 的分层 cosine drift、CKA、1-CKA。
- `probe_manifest.json`：固定验证样本的复现记录。

CKA 定义参考 [Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html)。
CKA 为几何相似性指标；请结合功能漂移和准确率判断，不能仅凭高 CKA 认定没有遗忘。

## 当前任务与紧邻上一任务的训练前后对比

每个阶段训练 Tj 后，新增 `adjacent_task_metrics.json/.csv/.png/.pdf`，分别在当前任务 Tj
和紧邻上一任务 T(j-1) 上比较同一对模型 M(j-1)、Mj。所有比较复用各任务固定的验证 probe，
不会对两个不同任务的样本直接互算 CKA。测试准确率仍来自完整 `(n+1)×n` 矩阵。

- `task_role=current`：当前任务的学习情况；`learning_gain_pp=100*(after-before)`。
- `task_role=previous`：紧邻上一任务的保持情况；`forgetting_pp=100*(before-after)`。
- `accuracy_change_pp` 对两种角色都统一定义为 `100*(after-before)`，正值代表改善。
- 两行都保存准确率前后值、表征 drift、逐层 CKA、KL。
- 梯度与 LoRA cosine 是相邻任务对的指标，两行数值相同；第一个阶段没有上一任务，记为 null。
- 第一阶段只有 current 一行，因此 n 个任务共有 `2n-1` 行；四任务共 7 行。
- `adjacent_layer_metrics.csv` 保存分层结果。原有旧任务 transition 表和 base 对比保持原定义。

已有运行可指定一个新的输出目录，并用 `--accuracy_matrix` 复用完整准确率矩阵。
当前/上一任务报告利用原本已经收集的前后表征，不增加模型前向次数。

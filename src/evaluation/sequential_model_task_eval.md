# 全部模型在全部任务上的评估

入口：`src/evaluation/sequential_model_task_eval.py`。
对于四任务顺序，评估 M0（base）和 M1–M4（每阶段累计训练后的最佳 adapter），
覆盖全部 5×4 组模型/任务，包括尚未训练过的任务。

准确率是完整测试集上的单模型结果。CKA、表征 cosine drift、KL 都需要两个模型，
因此另为每个任务比较全部 10 对模型，共 40 组模型对/任务结果。
所有比较严格限定在同一任务、相同验证样本和同一层，绝不把不同任务样本直接互算 CKA。

```bash
cd /root/autodl-tmp/CL
python src/evaluation/sequential_model_task_eval.py \
  --run_dir results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2 \
  --source_eval_dir results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2/evaluation_adjacent \
  --accuracy_matrix results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2/evaluation_drift/accuracy/accuracy_matrix.json \
  --probe_samples 128 --batch_size 2 --layers all
```

默认输出到 `<run_dir>/evaluation_all_models/`。未指定 `--source_eval_dir` 时优先查找
`evaluation_adjacent`、其次 `evaluation_drift`。有 probe manifest 就严格复用其行索引、
顺序与内容 hash；不存在时按原漂移脚本规则抽样。数量、任务顺序和预处理设置不匹配时会报错。
`--accuracy_matrix` 可自动寻找；如果没有已完成矩阵，会调用原评估流程计算。
`--pooling`、`--max_input_length`、`--max_new_tokens`、`--seed` 应与来源评估一致。

模型只加载一次，每个模型的各任务 probe 只做一次前向；不更新模型或 adapter 权重。
`probe_cache/` 缓存 CPU 表征和首答案位置的完整词表 log probabilities；同样命令重跑时
可以复用缓存。缓存绑定 adapter 权重/config、样本清单、层、pooling、长度、batch 和 seed。
如要更改这些设置，请使用新输出目录。默认 128 个 probe、全部层，四任务缓存约数 GB。

## 输出及含义

- `all_model_task_report.pdf`：四任务时共 10 页，分别是模型/adapter 对照与计算说明、
  5×4 base 指标总览、任务专属参照总览、逐任务准确率轨迹、4 页各任务 5×5 模型对比较、两种参照下的逐层 CKA。
- `all_model_task_overview.png/.pdf`：准确率、CKA vs M0、表征漂移 vs M0、KL vs M0。
  行标签明确是哪一个模型，列标签明确评估数据任务。
- `all_model_accuracy_trajectories.png/.pdf`：每个任务在五个模型上的完整测试准确率。
  竖线表示该任务训练完成的阶段，包含训练前和训练后的结果。
- `pairwise_<TASK>.png/.pdf`：在指定任务上，行模型与列模型的 CKA、cosine drift、symmetric KL。
  对角线是自身比较；零方差 CKA 为 N/A。
- `all_model_layerwise_cka.png/.pdf`：每个任务、每个模型与 M0 的逐层 CKA。
  汇总使用所选层列表中的最后一层；默认 all 为 hidden_states[32]，0 为 embedding。
- `all_model_task_metrics.json`：完整矩阵、设置、模型映射、单模型任务表及两两比较表。
- `all_model_task_metrics.csv`：20 行，含准确率、相对 base/上一阶段变化、
  相对刚学完该任务时的遗忘量，以及与 base 比较的 CKA/drift/KL。
- `all_model_pair_metrics.csv`：40 行，明确 model_a、adapter_a、model_b、adapter_b、
  evaluated_task；含两方向 KL、symmetric KL、CKA、表征漂移和两模型测试准确率。
- `all_model_pair_layer_metrics.csv`：全部模型对、全部任务、各层 CKA 和 cosine drift。
- `analysis.md`：中文逐任务结果分析、准确率表、最终模型相对 base 和刚学完该任务时的比较。
- `model_adapter_mapping.json`：模型编号、完整 adapter 路径及累计任务列表。
- `probe_manifest.json`、`probe_cache/`、`status.json`：样本复现、输出缓存和运行状态。

测试准确率保存为 0–1，图和文字显示百分比；变化量单位为百分点。
`forgetting_since_task_learned_pp` = 刚学完该任务的准确率减当前准确率，再乘 100，
未学过该任务的阶段为 null；它不是历史最好准确率减当前准确率。

CKA 高表示几何结构相似，cosine drift/KL 低表示对应变化小；它们不能替代测试准确率。
KL 只度量首个答案位置，并非完整生成序列。
这里增加的是模型与任务的覆盖面和模型对数量，验证 probe 的独立样本数仍是每任务 128，
没有因为比较变多就获得更多独立样本或置信区间。
梯度与 LoRA 更新 cosine 仍由原 transition 报告分析，不作为单个模型的任务分数。

## 以每个任务刚训练后的模型为参照

除原有与 M0 比较的图外，新增 `task_learned_reference_overview.png/.pdf`。
对第 i 个训练任务 Ti，参照模型为 Mi，即**刚学完该任务时保存的累计 adapter 模型**。
以 YAHOO→AMAZON→AGNEWS→DBPEDIA 为例，四列分别使用 M1、M2、M3、M4 作参照。
每列明确标出任务、参照模型，全部五个模型均保留在图中。

- 准确率差：`100*(acc(Mj,Ti)-acc(Mi,Ti))`，正值表示相对参照改善。
- CKA、cosine drift、symmetric KL：同一任务、相同样本上的 Mj 与 Mi 比较。
- 训练前模型 `j<i` 是回溯性的训练前对照，不解释为遗忘；`j>=i` 的遗忘量
  为 `100*(acc(Mi,Ti)-acc(Mj,Ti))`，参照模型自身为 0。
- `task_learned_reference_metrics.json/.csv`：20 行明确记录被比较模型、参照模型及双方 adapter 路径。
- `task_learned_reference_layer_metrics.csv`、`task_learned_reference_layerwise_cka.png/.pdf`：
  同一参照定义的逐层 CKA；零方差仍保留 null/N/A。
- `all_model_task_metrics.json/.csv` 同步增加上述参照与指标列，`analysis.md` 增加每任务五模型对照表。

已有全部模型对结果时，无需加载模型或缓存 tensor，可直接更新报告：

```bash
python src/evaluation/sequential_model_task_eval.py \
  --run_dir results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2 \
  --report_only
```

`--report_only` 读取完整的 `all_model_task_metrics.json`，保留原有准确率与模型对数值，
只派生任务专属参照结果并重绘报告。

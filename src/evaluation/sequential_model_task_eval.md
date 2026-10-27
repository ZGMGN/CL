# 全部模型在全部任务上的评估

入口：`src/evaluation/sequential_model_task_eval.py`。
对于四任务顺序，评估 M0（base）和 M1–M4（每阶段累计训练后的最佳 adapter），
覆盖全部 5×4 组模型/任务，包括尚未训练过的任务。

准确率是完整测试集上的单模型结果。CKA、Representation drift（表征漂移）、KL 都需要两个模型，
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
`--max_input_length`、`--max_new_tokens`、`--seed` 应与来源评估一致。
可以在新的输出目录切换 `--pooling`；来源评估的样本清单不依赖 pooling，输出缓存严格区分 pooling。

模型只加载一次，每个模型的各任务 probe 只做一次前向；不更新模型或 adapter 权重。
`probe_cache/` 缓存 CPU 表征和首答案位置的完整词表 log probabilities；同样命令重跑时
可以复用缓存。缓存绑定 adapter 权重/config、样本清单、层、pooling、长度、batch 和 seed。
如要更改这些设置，请使用新输出目录。默认 128 个 probe、全部层，四任务缓存约数 GB。

## 输出及含义

- `all_model_task_report.pdf`：四任务、同时包含第 16/32 层时共 23 页，分别是模型/adapter 对照与计算说明、
  5×4 base 指标总览、任务专属参照总览、逐任务准确率轨迹、4 页各任务 5×5 模型对比较、两种参照下的逐层 CKA，以及 7 页中文计算过程说明。
- `all_model_task_overview.png/.pdf`：准确率、CKA vs M0、表征漂移 vs M0、KL vs M0。
  行标签明确是哪一个模型，列标签明确评估数据任务。
- `all_model_accuracy_trajectories.png/.pdf`：每个任务在五个模型上的完整测试准确率。
  竖线表示该任务训练完成的阶段，包含训练前和训练后的结果。
- `pairwise_<TASK>.png/.pdf`：在指定任务上，行模型与列模型的 CKA、Representation drift、symmetric KL。
  对角线是自身比较；零方差 CKA 为 N/A。
- `all_model_layerwise_cka.png/.pdf`：每个任务、每个模型与 M0 的逐层 CKA。
  汇总使用所选层列表中的最后一层；默认 all 为 hidden_states[32]，0 为 embedding。
- `all_model_task_metrics.json`：完整矩阵、设置、模型映射、单模型任务表及两两比较表。
- `all_model_task_metrics.csv`：20 行，含准确率、相对 base/上一阶段变化、
  相对刚学完该任务时的遗忘量，以及与 base 比较的 CKA/drift/KL。
- `all_model_pair_metrics.csv`：40 行，明确 model_a、adapter_a、model_b、adapter_b、
  evaluated_task；含两方向 KL、symmetric KL、CKA、表征漂移和两模型测试准确率。
- `all_model_pair_layer_metrics.csv`：全部模型对、全部任务、各层 CKA 和 Representation drift。
- `analysis.md`：中文逐任务结果分析、准确率表、最终模型相对 base 和刚学完该任务时的比较。
- `model_adapter_mapping.json`：模型编号、完整 adapter 路径及累计任务列表。
- `probe_manifest.json`、`probe_cache/`、`status.json`：样本复现、输出缓存和运行状态。

测试准确率保存为 0–1，图和文字显示百分比；变化量单位为百分点。
`forgetting_since_task_learned_pp` = 刚学完该任务的准确率减当前准确率，再乘 100，
未学过该任务的阶段为 null；它不是历史最好准确率减当前准确率。

CKA 高表示几何结构相似，Representation drift/KL 低表示对应变化小；它们不能替代测试准确率。
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
- CKA、Representation drift、symmetric KL：同一任务、相同样本上的 Mj 与 Mi 比较。
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

## 指标计算过程说明

报告最后追加 7 页中文说明，并额外导出 `metric_calculations.pdf` 与 `metric_calculations.md`。
`analysis.md` 同步包含完整说明，内容来自 `sequential_metric_explanations.py`：

1. 模型身份、两种参照、测试/验证集、样本对齐、提示与张量形状。
2. 准确率、相对 base/上一阶段/任务刚训练后模型的变化、累计遗忘及符号。
3. last/mean pooling、逐样本 cosine distance、有效样本平均与零范数处理。
4. Linear CKA 的特征中心化、Gram 构造、双中心化、Frobenius 归一化及零方差。
5. 首答案位置的 log_softmax、全词表正反 KL、逐样本平均、symmetric KL 与 nats。
6. 原 transition 梯度 cosine 的答案 token 加权梯度与共同模型坐标；本矩阵不新增此项数值。
7. 原 transition LoRA cosine 的有效 BA 更新、阶段差分、全模块 Frobenius cosine 和低秩内积。

说明从当前 payload 读取任务顺序、参照、样本数、层及设置。追加说明不会重新推理或改变指标。
中文 PDF 使用缓存字体 `.cache/fonts/NotoSansCJKsc-Regular.otf`，附带 OFL 许可证。
迁移到新环境时，先从 Noto 官方仓库准备字体：

```bash
mkdir -p .cache/fonts
curl -L --fail https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf \
  -o .cache/fonts/NotoSansCJKsc-Regular.otf
curl -L --fail https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE \
  -o .cache/fonts/NotoSansCJK-LICENSE.txt
```

然后使用 `--report_only` 更新即可。中文计算附录的逐页预览另存为 `calculation_methods_01.png` 至 `07.png`。

## 第 16 层表征与 CKA 分析

图表主名称统一为 **Representation drift**，图注说明其度量是
**Mean per-example cosine distance（逐样本余弦距离的均值）**。数据键 `rep_drift`
及函数名保留，兼容已有结果与脚本。

当 `resolved_layers` 包含 16 时，报告自动增加：

- `layer_16_vs_base.png/.pdf`：5 个模型 × 全部任务，第 16 层 CKA 与 Representation drift，相对 M0。
- `layer_16_vs_task_learned.png/.pdf`：相同覆盖范围，各任务用刚学完该任务的 Mi 为参照。
- `layer_16_vs_summary_layer.png/.pdf`：固定最终模型，分别以 M0、任务刚学完模型为参照，对照第 16 层与汇总层（默认 32）的两项指标。
- `selected_layer_metrics.json/.csv`：两层、两种参照的完整结果；四任务共 80 行。
- `layer_16_metrics.csv`：第 16 层、两种参照共 40 行，明确任务、模型、参照和 adapter 路径。
- `layer_16_analysis.md`：中文表征分析与逐任务层间数值对照，亦并入 `analysis.md`。

第 16 层具体指 `hidden_states[16]`，不是 embedding 输出，也不是对 1–16 层求平均。
只增加中间层表征指标，准确率和首答案位置 KL 仍来自原模型输出。

已有 `all_model_pair_layer_metrics.csv` 与逐层 CKA 时，`--report_only` 直接复用原逐层数值，
不进行模型前向；若某个自比较的 CKA 为 null，会读取对应缓存表征来判定自身 cosine 距离是否定义。
缺少第 16 层时需先通过 `--layers all` 或包含 16 的层列表完成 probe 评估；不会用第 32 层数值冒充。

## 将 mean pooling 的第 16、32 层分析接在 last pooling 后面

先在独立目录补算 mean pooling，复用 last 评估的相同 probe 清单和准确率矩阵：

```bash
python src/evaluation/sequential_model_task_eval.py \
  --run_dir results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2 \
  --output_dir results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2/evaluation_all_models_mean \
  --source_eval_dir results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2/evaluation_adjacent \
  --accuracy_matrix results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2/evaluation_drift/accuracy/accuracy_matrix.json \
  --pooling mean --layers 16,32 --probe_samples 128 --batch_size 2
```

完成后更新 last 目录的报告：

```bash
python src/evaluation/sequential_model_task_eval.py \
  --run_dir results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2 \
  --report_only \
  --mean_results_dir results/standard_cl_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_bs8_ga2/evaluation_all_models_mean
```

合并报告四任务时共 30 页：1–13 页为 last pooling 分析，14–20 页为 mean pooling
及 last/mean 对照，21–23 页为双向 KL，24–30 页为共同计算说明。
mean 部分依次展示第 16 层相对 M0/任务参照、第 32 层相对 M0/任务参照、
mean 下第 16/32 层对照，最后是两个层各自的 last/mean 对照。
各层总览覆盖全部 5×4 组模型任务；pooling 对照条形图固定最终模型，完整 80 行结果保存在表格中。

- `mean_layer_16_vs_base.png/.pdf`、`mean_layer_16_vs_task_learned.png/.pdf`。
- `mean_layer_32_vs_base.png/.pdf`、`mean_layer_32_vs_task_learned.png/.pdf`。
- `mean_layer_16_vs_summary_layer.png/.pdf`。
- `last_vs_mean_layer_16.png/.pdf`、`last_vs_mean_layer_32.png/.pdf`。
- `pooling_comparison_metrics.json/.csv`：80 行，明确层、任务、模型、参照、两种 pooling 的数值及差。
- `mean_pooling_analysis.md`：mean pooling 结果与 last/mean 中文对照，亦并入主 `analysis.md`，位于 last 分析后。

合并时检查任务顺序、模型/adapter、准确率、输入长度、生成长度、seed、batch、
完整验证样本清单一致，并比较缓存中的 adapter 权重 hash、base config hash；
只有 pooling 与所选层列表允许不同。首答案位置 KL 也应保持一致。

Mean 只排除 padding，平均其余完整提示 token，包括指令、选项与模板。
最后 token 的缓存向量无法还原 mean，因此首次补算需要重新前向；后续可复用 mean 专属缓存。
合并路径会记入主结果元数据，以后 `--report_only` 会继续保留 mean 附加部分。

## 双向 KL 的明确统计与展示

当前报告保留 symmetric KL，同时新增：

- `directional_kl_vs_base.png/.pdf`：各模型在各任务上与 M0 比较的两个方向和对称均值。
- `directional_kl_vs_task_learned.png/.pdf`：各任务参照 Mi 与被比较模型 Mj 的两个方向和均值。
- `directional_kl_all_pairs.png/.pdf`：每个任务的完整 5×5 方向矩阵，单元格为 `KL(行模型 P || 列模型 Q)`，通常不对称。
- `directional_kl_metrics.json/.csv`：5 个模型 × 4 个任务 × 两种参照，共 40 行；每行包含明确的两个模型、adapter 路径、正反 KL 和 symmetric KL。
- `directional_kl_all_pairs.csv`：80 行有方向模型对，排除自身比较。
- `directional_kl_analysis.md`：中文方向说明和全部参照统计，亦并入主分析文档。

`KL(reference || model)` 始终以参照分布加权；`KL(model || reference)` 反之。
参照为 Mi 时，即使被比较模型 Mj 的 j<i，依然按实际模型身份匹配原始 pair 表的方向，
不能简单把原始 `kl_a_to_b` 当成参照到被比较模型。自身 KL 为 0。

主 `all_model_task_metrics.json/.csv` 同步增加与 base/任务参照的双向列，
`task_learned_reference_metrics.json/.csv` 增加两个方向；原始 pair 数值保持不变。

正反 KL 都是首答案位置全词表求和、固定验证样本平均，单位 nats。
该输出指标与 hidden-state pooling、层选择无关；合并报告验证 last/mean 的两个方向一致后只展示一份。
所有数据直接派生自保存的 `pairwise_results`，使用原来的 `--report_only` 即可更新，无需模型前向。

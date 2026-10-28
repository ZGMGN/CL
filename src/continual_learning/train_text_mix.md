# QA 与文本混合顺序训练

`train_text_mix.py` 保留每个任务原有的完整 QA 训练集，每阶段额外加入该任务的 192 条自然文本。
同一个累计 LoRA adapter 按 `--tasks` 的顺序持续训练。只提供一个任务时，也可以单独训练该任务。

| 任务 | 原 QA 训练集 | 新增文本 | 混合训练集 |
| --- | ---: | ---: | ---: |
| YAHOO | 10000 | 192 | 10192 |
| AMAZON | 8000 | 192 | 8192 |
| AGNEWS | 8000 | 192 | 8192 |
| DBPEDIA | 14000 | 192 | 14192 |

脚本不会把 QA 下采样成统一数量，也不会修改原始数据文件。
QA 使用 `data/TASK/train.jsonl`；文本使用 `data/TASK_text/TASK_domain_192.jsonl` 的 `text` 字段。
验证、测试分别使用原始 `data/TASK/val.jsonl` 和 `test.jsonl`，不会加入转换文本。

## 使用

在项目根目录运行。先检查数据、完整文本长度和来源记录，不加载模型权重：

```bash
python src/continual_learning/train_text_mix.py \
  --tasks YAHOO,AMAZON,AGNEWS,DBPEDIA \
  --dry_run
```

开始训练：

```bash
python src/continual_learning/train_text_mix.py \
  --tasks YAHOO,AMAZON,AGNEWS,DBPEDIA \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --max_seq_length 1536 \
  --eval_batch_size 16
```

单任务示例：

```bash
python src/continual_learning/train_text_mix.py --tasks AGNEWS
```

模型与 tokenizer 复用 `model/utils.py` 中的本地 Llama-3.1-8B-Instruct 配置。
LoRA、训练与评估参数沿用原顺序脚本。上述学习率等是原实验默认值，并非为领域文本确定的最优参数。
默认输出目录包含任务顺序、seed、额外文本数量、batch size 和梯度累积步数：

```text
results/text_mix_step_YAHOO_AMAZON_AGNEWS_DBPEDIA_seed_1_text192_bs8_ga2/
```

可以用 `--output_dir` 指定新目录。已有训练阶段的目录会被拒绝，避免覆盖已有模型；
只运行过 `--dry_run` 的目录可以继续正式训练。
`--data_dir` 可指定其他数据根目录。
`--text_samples_per_task N` 会读取 `TASK_text/TASK_domain_N.jsonl`，且要求文件恰好有 N 条；默认值为 192。

## 两种监督方式

- QA：使用原任务对话模板；输入提示、问题、选项不计算 loss，只监督 assistant 答案及结束标记。
- 文本：不添加对话模板；使用 `BOS + 完整正文 + EOS`，在下一个 token 预测中监督正文和真实 EOS。
- padding：只根据真实长度屏蔽，标签为 `-100`；即使 `pad_token_id == eos_token_id`，真实结束符仍然保留监督。
- 模型内部完成 causal label shift，不手动将 labels 移位。

默认 `--mixing_strategy step`：先分别打乱 QA 和文本，再各自按完整 optimizer step 分组，最后打乱整组顺序。
单卡 batch size=8、gradient accumulation=2 时，每组 16 条，连续两个 batch 均来自同一种数据。
192 条文本正好对应 12 个完整文本更新组。QA 数量保持原值：

| 任务 | QA 更新组 | 文本更新组 | 每 epoch 总更新数 |
| --- | ---: | ---: | ---: |
| YAHOO | 625 | 12 | 637 |
| AMAZON | 500 | 12 | 512 |
| AGNEWS | 500 | 12 | 512 |
| DBPEDIA | 875 | 12 | 887 |

采样器每个 epoch 重新打乱，且由 Trainer 实际数据加载器保留组边界。
每条 QA 和每条文本各出现一次，不丢弃、重复或跨类型补齐样本。
组大小按实际 batch size × 梯度累积步数 × 数据并行进程数计算；两种数据数量均须整除该大小。
不满足条件会在加载模型权重之前报错。

这种设计隔离的是当前 step 的数据来源。两类 step 仍更新同一 LoRA，并共用 Adam 优化器状态，
因此文本更新仍可能影响后续 QA，不能保证没有干扰或准确率一定提高。

文本保持“一条源文本对应一个训练样本”，不切成多条，也不静默截断。
超过最大长度会在模型加载之前报错，需要显式增加 `--max_seq_length` 或调整源文本。
当前四任务的 192 条文本均可完整放入默认的 1536 tokens。

使用标准因果语言模型的有效 token 交叉熵，不额外设置 QA/text 两套 loss 权重。
默认 step 模式中，每次更新只计算当前一种数据的 token 平均 loss。
若切换到旧的 sample 模式，**样本数比例不等于 loss 比例**：长文本在混合更新中可能占较多监督 token。
训练日志的 `loss` 包含 QA 与文本；它不代表某个任务的 performance。

新增文本是由相应 QA 训练集转换得到的，因此是原训练样本的另一种表述，
并非 192 个独立的新知识来源。若提供转换 manifest，脚本会核验文本及 QA 训练文件的 SHA256。

## QA performance 与最优权重

每个 epoch 只在当前任务原始 QA 验证集上生成答案并计算准确率。
`load_best_model_at_end=True`，以 `eval_accuracy` 最高的 checkpoint 作为该阶段最优模型。
保存最优 adapter 后，用它评估已学任务的原始完整 QA 测试集，并传递给下一阶段；
每阶段重新建立优化器和学习率计划，模型权重持续累积。
文本不计算准确率，也不参与最优 checkpoint 的选择。

输出包含：

- `mixing_manifest.json`：任务顺序、数据文件及 hash、QA/text 数量、更新组大小和组数、正文长度、loss 与评估规则。
- `01_YAHOO/best_model/` 等：该阶段 QA 验证准确率最优的累计 adapter。
- 每阶段的 `test_TASK.json`：只含 QA 测试结果和指标。
- `sequential_results.json`：每阶段样本数、最优 QA 验证准确率、已学任务的 QA 测试 performance。

模型目录与 `sequential_results.json` 保持原顺序训练的命名结构，
后续可继续用现有 sequential heatmap/drift 评估脚本评估 base 和各阶段 adapter。


## 与旧的 200 条按样本混合实验区分

各任务新增 `TASK_domain_192.jsonl`、TXT 和 manifest 文件，由原 200 条均衡选取，
保留 20 种叙述格式，每种格式 9 或 10 条；保留文本逐字不变。
原 200 条数据、已训练的 adapter 和评估报告均保留。

若需要原来的按样本混合方式，显式指定：

```bash
python src/continual_learning/train_text_mix.py \
  --tasks YAHOO,AMAZON,AGNEWS,DBPEDIA \
  --mixing_strategy sample --text_samples_per_task 200 \
  --output_dir results/text_mix_sample_200_new_run
```

192 条分组方案已启动三个 order 的正式 8B 训练与完整评估流水线，tmux 会话为 `text_mix_three_orders_192`。
统一状态见 `results/text_mix_step_three_orders_seed_1_text192_bs8_ga2/queue_status.json`；各实验目录分别保存训练日志、QA heatmap、漂移指标和合并 PDF 报告。

# CL

基于本地 Llama-3.1-8B-Instruct 的持续学习实验，包含分类数据准备、普通 LoRA 模型加载和生成式分类评估。

## 项目结构

```text
CL/
├── src/
│   ├── metrics.py
│   ├── model/
│   │   └── utils.py
│   ├── tasks/
│   │   └── AGNEWS.py, QQP.py, SST2.py, AMAZON.py, YELP.py, DBPEDIA.py, YAHOO.py
│   ├── evaluation/
│   │   ├── general_task_eval.py
│   │   ├── AGNEWS_eval.py, AMAZON_eval.py, YELP_eval.py
│   │   └── DBPEDIA_eval.py, YAHOO_eval.py, SST2_eval.py, QQP_eval.py
│   └── continual_learning/
│       ├── train_single_task.py
│       └── train_sequential_tasks.py
├── data/<TASK>/{train,val,test}.jsonl
├── scripts/
│   ├── run_olora_three_orders.py
│   └── plot_training_loss.py
├── results/
├── Llama-3.1-8B-Instruct/
└── tests/
```

模型权重和 tokenizer 存放在项目根目录的 `Llama-3.1-8B-Instruct/`。
代码按脚本位置计算项目路径；下面的命令均在项目根目录运行。

## 运行环境与依赖

`requirements.txt` 记录当前三顺序实验环境的直接依赖和关键 tokenizer / 权重读取库版本。
当前环境为 Python 3.12，PyTorch GPU 构建为 `2.12.1+cu130`；清单中的 `torch==2.12.1` 不限定 CUDA 构建来源。
迁移环境时先安装适配目标 GPU / CUDA 的 PyTorch，再安装其余依赖；尚未验证从全新环境安装此清单。

核心训练依赖为 torch、transformers、datasets、peft 和 accelerate；分析绘图使用 numpy、matplotlib、Pillow、tqdm。
本项目使用普通 LoRA 和本地生成评估，当前代码不依赖 O-LoRA 清单中的 DeepSpeed、FairScale、W&B、TensorBoard、NLTK 或 ROUGE。
O-LoRA 仓库的旧版本固定值与本项目当前环境不同，不应直接覆盖当前运行环境。

## 数据任务

Standard CL 五个分类任务为 AGNEWS、AMAZON、YELP、DBPEDIA 和 YAHOO；AMAZON、YELP 使用 Full 五分类版本。
QQP 和 SST2 是额外任务。

| 任务 | 类别数 | 训练集 | 验证集 | 测试集 |
| --- | ---: | ---: | ---: | ---: |
| AGNEWS | 4 | 8000 | 2000 | 2000 |
| AMAZON | 5 | 8000 | 2000 | 2000 |
| YELP | 5 | 8000 | 2000 | 2000 |
| DBPEDIA | 14 | 14000 | 1750 | 1750 |
| YAHOO | 10 | 10000 | 2000 | 2000 |
| QQP | 2 | 8000 | 1000 | 1000 |
| SST2 | 2 | 8000 | 2000 | 2000 |

这些是项目自定义分层划分和均衡采样后的规模，不代表论文原始 benchmark 的完整评估协议。
数据来源、构建接口和长度过滤见 [任务说明](src/tasks/README.md)。

评估直接读取已保存的 `data/<TASK>/test.jsonl`，不会重新下载源数据或重新划分。
数据准备脚本可以直接运行，例如：

```bash
python src/tasks/YAHOO.py
```

YAHOO、DBPEDIA 在构建时过滤完整训练对话超过 1536 token 的样本，并补采样以保持类别数量。
AGNEWS 主入口仅对已有本地数据做 sanity check；重新生成需显式调用 `build_agnews_files()`。
重新构建数据可能覆盖已有 JSONL。

## 指令与标签格式

数据 JSONL 保留原始文本和零起始整数 `label`；训练时才将编号映射为类别名称，因此切换输出格式无需重建数据文件。
共享逻辑在 `src/tasks/label_utils.py`，各任务的 `LABELS` 定义实际映射。

| 任务 | 按整数编号递增的输出标签 |
| --- | --- |
| AGNEWS | World, Sports, Business, Science or Technology |
| AMAZON / YELP | very negative, negative, neutral, positive, very positive |
| DBPEDIA | Company, Educational Institution, Artist, Athlete, Office Holder, Mean of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work |
| YAHOO | Society & Culture, Science & Mathematics, Health, Education & Reference, Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships, Politics & Government |
| SST2 | Bad, Good（0 为负面，1 为正面） |
| QQP | False, True（0 为不重复，1 为重复） |

例如 AGNEWS 的 user 指令内容为：

```text
What is the topic of the following paragraph? Choose one from the option.
Option: World, Sports, Business, Science or Technology
Text: <新闻文本>

Answer:
```

对应 assistant 训练答案直接为 `Sports`，没有字母选项。实际训练仍使用本地 chat template 包装消息。
`extract_answer_label()` 执行类别匹配；旧接口名 `extract_answer_letter()` 保留为兼容别名，返回值同样是类别名称。

## 生成式评估

七个任务脚本共用 `src/evaluation/general_task_eval.py`：

- `build_eval_model()`：加载本地基础模型，可选加载普通 LoRA adapter。
- `evaluate_qa()`：构造不含标准答案的提示，批量生成，仅对新生成的文本做归一化类别匹配并统计准确率。
- `save_json_results()`：保存汇总指标和逐条结果。

各任务保留 `evaluate_<TASK>_QA()`、`tiny_test()` 和 `run_full_evaluation()` 入口。
生成使用 `do_sample=False`，左侧 padding，进度条按 batch 更新。评估结束后恢复模型训练状态及 tokenizer 的 padding、truncation 方向。

运行前检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

模型加载器在 CUDA 可用时使用 GPU 和 BF16，否则使用 CPU 和 FP32；8B 模型在 CPU 上评估会很慢。

### 完整评估

```bash
python src/evaluation/AGNEWS_eval.py
python src/evaluation/AMAZON_eval.py
python src/evaluation/YELP_eval.py
python src/evaluation/DBPEDIA_eval.py
python src/evaluation/YAHOO_eval.py
python src/evaluation/SST2_eval.py
python src/evaluation/QQP_eval.py
```

默认参数：`--batch_size 32`、`--max_input_length 1536`、`--max_new_tokens 8`。
输入超过长度上限时从左侧截断，显存不足时可减小 batch size。
`--print_sample` 在评估完成后打印前三条结果；`--help` 查看全部参数。

### 两条样本试跑

```bash
python src/evaluation/SST2_eval.py --unit_test
```

读取测试集前两条样本，使用相同模型加载与生成评估流程。它用于检查流程，不是带预期准确率断言的单元测试。

### 加载普通 LoRA

```bash
python src/evaluation/SST2_eval.py \
  --peft_model_folder /path/to/adapter \
  --batch_size 16
```

adapter 目录应包含 `adapter_config.json` 和相应权重。未指定该参数时评估原始基础模型。
`--unit_test` 同样支持加载 LoRA。模型加载器使用固定的本地模型目录，没有 `--model_name` 参数。

## 结果文件

完整评估默认保存到 `results/<TASK>/<task小写>_qa_results.json`，例如 `results/SST2/sst2_qa_results.json`。
试跑默认保存在 `results/<TASK>/unit_test/`。

可通过 `--output_dir`、`--qa_output_json_name` 自定义完整评估输出；试跑使用
`--unit_test_output_dir`、`--qa_unit_test_json_name`。同一路径再次运行会覆盖结果，比较基础模型和 LoRA 时应指定不同输出路径。

结果格式示例：

```json
{
  "metrics": {"accuracy": 0.5, "exact_match": 50.0, "correct": 1, "total": 2},
  "results": [
    {"index": 0, "ground_truth_answer": "Sports", "predicted_answer": "Sports", "full_response": "sports.", "correct": true},
    {"index": 1, "ground_truth_answer": "World", "predicted_answer": null, "full_response": "The answer is World", "correct": false}
  ]
}
```

- `index`：本次评估数据中的顺序索引。
- `ground_truth_answer`：标准类别名称。
- `predicted_answer`：与生成文本归一化后完全匹配的类别名称；没有匹配类别时为 `null`。
- `full_response`：新生成部分解码、移除特殊 token 并去掉首尾空白后的文本。
- `correct`：预测类别是否等于标准类别；未匹配仍计入总数，并按错误计分。
- `accuracy` 为 0–1 比例；`exact_match` 为同一评分的 0–100 百分数。

归一化遵循 O-LoRA：转为小写，删除 `string.punctuation` 中的英文标点，再合并多余空白。
例如 `sports.` 可匹配 `Sports`；`B` 和 `The answer is Sports` 均不匹配。
当前本地评估输出 accuracy / exact_match，不计算 O-LoRA 仓库中的 ROUGE 指标。
旧字母提示下的基础模型分数不适用于新标签格式，因此不再列为当前基线。

## O-LoRA 的三个任务顺序

这里使用本地普通 LoRA 持续训练，采用 [O-LoRA](https://github.com/cmnfriend/O-LoRA) 的三个任务顺序与类别名称指令格式。
基础模型仍为本地 Llama-3.1-8B-Instruct，保留本地 chat template、数据划分、训练参数和验证选优方式。
当前代码未实现 O-LoRA 的正交约束算法，因此实验应标为普通 LoRA，不能视为 O-LoRA 论文结果复现。

| Order | 任务顺序 |
| --- | --- |
| 1 | DBPEDIA → AMAZON → YAHOO → AGNEWS |
| 2 | DBPEDIA → AMAZON → AGNEWS → YAHOO |
| 3 | YAHOO → AMAZON → AGNEWS → DBPEDIA |

在项目根目录启动队列：

```bash
python -u scripts/run_olora_three_orders.py
```

该脚本依次完成三个 order 的训练，再对三个 order 分别运行完整评估；任一子进程失败后队列停止，错误写入 `status.json`。
脚本没有断点恢复或跳过已完成训练的逻辑，不要对正在运行或已有结果的队列重复启动。
后台运行时可使用 tmux；当前训练使用会话 `olora_three_orders_labels`。

训练设置：seed=1，每任务 1 epoch，batch size=8，梯度累积=2，LoRA r=16、alpha=64、dropout=0.05；
学习率 2e-4、cosine 调度、warmup ratio=0.01、weight decay=1e-4；输入上限 1536，最多生成 8 token，评估 batch size=16。
每个任务按验证集 accuracy 选出 best adapter，再将其权重传递给下一任务。
训练阶段还会评估当前已学任务的完整测试集。

三个顺序训练结束后的完整评估包括：

- 基础模型 M0 和四个阶段模型 M1–M4，分别评估全部四个完整测试集，生成 5×4 准确率矩阵及逐条预测。
- 每任务固定 128 条验证 probe，比较所有模型对的各层 CKA、表征漂移和 KL 指标。
- 分别计算 last / mean pooling 的表征指标，mean 复用相同 probe 和测试准确率矩阵。
- KL 基于相同提示下首个答案位置的全词表分布；这些表征分析使用验证 probe，并非完整测试集。

结果目录：

```text
results/olora_three_orders_label_seed_1/
├── experiment_config.json
├── status.json
└── order_1/                         # order_2、order_3 同结构
    ├── train.log
    ├── sequential_results.json
    ├── 01_DBPEDIA/best_model/       # 其余阶段使用相应序号和任务名
    ├── evaluation_all_models/      # last pooling；含 accuracy/ 和分析报告
    ├── evaluation_all_models_mean/ # mean pooling
    ├── evaluation_last.log
    ├── evaluation_mean.log
    ├── training_loss.png
    ├── training_loss.svg
    └── training_loss.csv
```

### Step–loss 曲线

另一个轻量进程从训练日志提取每 25 个 optimizer step 记录的 loss，不影响训练进程。
曲线横轴为任务内 optimizer step，纵轴为该日志区间的训练 loss；四个任务分别显示，不将任务切换处连线。
每个 order 保存 PNG / SVG 总览和 CSV，每个已有日志点的阶段另存 `training_loss.png`。

生成一次：

```bash
python scripts/plot_training_loss.py --queue_dir results/olora_three_orders_label_seed_1
```

持续刷新：

```bash
python scripts/plot_training_loss.py --queue_dir results/olora_three_orders_label_seed_1 --watch
```

`--watch` 每 30 秒刷新，队列完成或失败时最终刷新并退出。当前刷新进程使用 tmux 会话 `training_loss_plots`。

## 检查

在本地 tokenizer 和已准备的数据存在时运行已适配类别名称的任务与评估检查，无需加载完整 8B 模型。其他历史测试可能仍包含字母格式断言：

```bash
python -m unittest tests.test_standard_cl_tasks tests.test_shared_evaluation
```

单独检查共享评估逻辑：

```bash
python -m unittest discover -s tests -p 'test_shared_evaluation.py' -v
```

共享评估测试使用模拟模型，覆盖七个任务的分批生成、答案解析与计分、异常后状态恢复、本地数据加载及 JSON 保存。

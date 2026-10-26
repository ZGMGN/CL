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
│   │   └── DBPEDIA_eval.py, YAHOO_eval.py, SST2_eval.py
│   └── continual_learning/
│       ├── train_single_task.py
│       └── train_sequential_tasks.py
├── data/<TASK>/{train,val,test}.jsonl
├── results/<TASK>/
├── reports/
├── Llama-3.1-8B-Instruct/
└── tests/
```

模型权重和 tokenizer 存放在项目根目录的 `Llama-3.1-8B-Instruct/`。
代码按脚本位置计算项目路径；下面的命令均在项目根目录运行。

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

## 生成式评估

六个任务脚本共用 `src/evaluation/general_task_eval.py`：

- `build_eval_model()`：加载本地基础模型，可选加载普通 LoRA adapter。
- `evaluate_qa()`：构造不含标准答案的提示，批量生成，仅解析新生成的文本并统计准确率。
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
  "metrics": {"accuracy": 0.5, "correct": 1, "total": 2},
  "results": [
    {"index": 0, "ground_truth_answer": "A", "predicted_answer": "A", "full_response": "A", "correct": true},
    {"index": 1, "ground_truth_answer": "B", "predicted_answer": null, "full_response": "Unable to answer", "correct": false}
  ]
}
```

- `index`：本次评估数据中的顺序索引，可用于查找对应的测试样本。
- `ground_truth_answer`：标准选项字母，即 gold answer。
- `predicted_answer`：从模型输出解析出的选项字母；解析失败为 `null`。
- `full_response`：新生成部分解码、移除特殊 token 并去掉首尾空白后的文本，保留字母以外的内容。
- `correct`：解析出的选项是否与标准答案相同；解析失败按错误计分，仍计入总数。

模型输出不一定只有单个字母，也可能包含标点、选项名称或解释。
解析失败可能是没有明确答案，也可能是现有解析规则未覆盖回答格式；不能简单把任意文本的首字母当作选项。

## 当前基础模型结果

以下结果来自本地 Llama-3.1-8B-Instruct，未加载 LoRA，使用上述项目测试集、batch size 32、输入上限 1536 和最多生成 8 token。
结果按当前答案解析规则计分，并非 Standard CL 训练后的最终结果。

| 任务 | 测试数 | 正确数 | 准确率 | 解析失败数 |
| --- | ---: | ---: | ---: | ---: |
| SST2 | 2000 | 1807 | 90.35% | 0 |
| AGNEWS | 2000 | 1762 | 88.10% | 0 |
| AMAZON | 2000 | 1136 | 56.80% | 1 |
| YELP | 2000 | 1260 | 63.00% | 2 |
| DBPEDIA | 1750 | 1610 | 92.00% | 6 |
| YAHOO | 2000 | 1320 | 66.00% | 2 |

## 检查

在本地 tokenizer 和已准备的数据存在时运行检查，无需加载完整 8B 模型：

```bash
python -m unittest discover -s tests
```

单独检查共享评估逻辑：

```bash
python -m unittest discover -s tests -p 'test_shared_evaluation.py' -v
```

共享评估测试使用模拟模型，覆盖六个任务的分批生成、答案解析与计分、异常后状态恢复、本地数据加载及 JSON 保存。

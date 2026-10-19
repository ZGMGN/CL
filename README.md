# CL

Continual learning experiments.

Project layout:

```text
CL/
├── src/
│   ├── model/
│   │   └── utils.py
│   ├── tasks/
│   │   └── AGNEWS.py, QQP.py, SST2.py, AMAZON.py, YELP.py, DBPEDIA.py, YAHOO.py
│   ├── train_single_task.py
│   └── train_sequential_tasks.py
├── data/
├── Llama-3.1-8B-Instruct/
└── tests/
```

Run task scripts from the project root:

```bash
python src/tasks/YAHOO.py
# Alternatively:
python -m src.tasks.YAHOO
```

Import tasks from the project root with `from src.tasks.YAHOO import load_yahoo_data`.
Model utilities are available as `src.model.utils`. Task code resolves data,
cache and tokenizer locations from the project root, regardless of the current
working directory. The model downloader saves weights to `Llama-3.1-8B-Instruct/`.

Run offline checks:

```bash
python -m unittest discover -s tests
```

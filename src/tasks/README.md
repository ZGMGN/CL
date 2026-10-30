# Classification tasks

The additional Standard CL tasks use the same custom stratified 6:2:2 split of
upstream **training data** as AGNEWS, QQP and SST2 in this project. These are
project experiment splits, not the published benchmark's exact evaluation protocol.

| Module | HF dataset | Classes | Prompt fields |
| --- | --- | --- | --- |
| AMAZON.py | [vgaraujov/amazon_review_full](https://huggingface.co/datasets/vgaraujov/amazon_review_full) | 5 ratings, 1–5 stars | title, content |
| YELP.py | [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full) | 5 ratings, 1–5 stars | text |
| DBPEDIA.py | [fancyzhx/dbpedia_14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | 14 categories | title, content |
| YAHOO.py | [community-datasets/yahoo_answers_topics](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) | 10 topics | question_title, question_content, best_answer |

Amazon uses a community-hosted copy of the five-star review dataset. Its label
values are already 0–4; do not subtract one. Yahoo's `topic` column is renamed
to `label`. All four modules export zero-based `label`, source row `idx`, and the
original text fields. At runtime, prompts list category names and assistant targets
contain the category name directly, following O-LoRA's instruction format.
AMAZON / YELP map 0–4 to `very negative`, `negative`, `neutral`, `positive`,
`very positive`. DBPEDIA and YAHOO use the names in each module's `LABELS`.
Changing this runtime mapping does not require rewriting existing JSONL data.

From the project directory, run any task separately:

```bash
python src/tasks/AMAZON.py
python src/tasks/YELP.py
python src/tasks/DBPEDIA.py
python src/tasks/YAHOO.py
```

Each task writes `train.jsonl`, `val.jsonl` and `test.jsonl` into `data/<TASK>/`,
using uppercase task names. Re-running replaces that task's existing files.
Like the existing tasks, each module sets `HF_ENDPOINT` to
`https://hf-mirror.com`. Downloads require network access and sufficient local storage.

Sample sizes are **per class**. AMAZON and YELP use 1600/400/400;
DBPEDIA uses 1000/125/125, and YAHOO uses 1000/200/200:

| Task | Train total | Validation total | Test total |
| --- | ---: | ---: | ---: |
| AMAZON | 8000 | 2000 | 2000 |
| YELP | 8000 | 2000 | 2000 |
| DBPEDIA | 14000 | 1750 | 1750 |
| YAHOO | 10000 | 2000 | 2000 |

Each module follows the AGNEWS / QQP / SST2 file structure: shared instruction construction in `label_utils.py`, `_single_example_to_message`, `example_to_messages`, `render_chat`,
`format_prompt`, answer / label extraction, accuracy, `load_<task>_data`,
`_balanced_subsample`, `build_<task>_files`, and an inline `__main__` block.
Loaders return a dictionary of full custom splits; builders call the loader
once and export balanced subsamples. All sizes and `seed` can be set in Python:

```python
from src.tasks.AMAZON import build_amazon_files

build_amazon_files('data/AMAZON', n_train=1000, n_val=200, n_test=200, seed=1)
```

`example_to_messages` accepts a single example or a column-oriented batch.
As in AGNEWS / QQP / SST2, `format_prompt` and `extract_label_from_example`
operate on single examples; pass batched messages to `render_chat` one by one.
`format_prompt` takes an explicit `include_answer` argument. Set the module's
`<TASK>_CHAT_TEMPLATE_LLAMA31` on the tokenizer when using Llama 3.1.
Pass only generated response text to `extract_answer_label`.
`extract_answer_letter` remains a compatibility alias and also returns category names.
Matching lowercases text, removes ASCII punctuation and collapses whitespace, then
requires the entire normalized response to equal a category name. Explanatory
answers and letter-only answers are not accepted.

Offline checks (synthetic data; no dataset download):

```bash
python -m unittest discover -s tests -p 'test_standard_cl_tasks.py'
```

The local tokenizer rendering check runs if `Llama-3.1-8B-Instruct` is present.

After exporting, the `__main__` block loads the three local JSONL splits once,
initializes the local Llama tokenizer, renders single and batch examples,
preview a prompt without an answer, check label extraction, and verify accuracy
is 0.5 on the tiny example. It does not load model weights or run inference.

Amazon uses `n_train=1600`, `n_val=400`, `n_test=400` per class. Its builder
calls `load_amazon_data()` once to obtain all three full splits.
The main block then loads the exported JSONL once and reuses it for every
sanity check. `load_amazon_data()` returns a dictionary with `train`, `val`
and `test` keys; it has no `split` parameter. Use `datasets['train']` to select
a split from that returned dictionary.

DBpedia follows the same reuse pattern: `load_dbpedia_data(seed=1)` returns
`{"train": train_set, "val": val_set, "test": test_set}` without a `split`
parameter. `build_dbpedia_files` calls it once, and the main block reads the
exported JSONL once for the remaining checks. Its per-class defaults remain
1000/125/125.

YAHOO and YELP also return dictionaries from `load_yahoo_data(seed=1)` and
`load_yelp_data(seed=1)`. Neither accepts a `split` parameter. Their builders
load and split the source once, and their main blocks reuse the exported JSONL
for all checks. Yahoo still renames `topic` to `label`; both tasks keep their
1000/200/200 (YAHOO) and 1600/400/400 (YELP) per-class sampling defaults.

YAHOO exports only samples whose complete Llama training conversation (template,
input fields, choices and assistant answer) is at most 1536 tokens. The builder
checks sampled candidates and draws replacements for rejected examples until
each class reaches its configured count; it does not encode all 1.4M source
rows. `build_yahoo_files(..., max_length=1536)` uses the local Llama tokenizer.
Overlong samples are dropped, not truncated.

DBPEDIA uses the same complete-chat 1536-token filter as YAHOO, including
choices and assistant answers. Rejected candidates are replaced to retain
1000/125/125 samples per class. `build_dbpedia_files(..., max_length=1536)`
loads the local Llama tokenizer and never truncates individual samples.

Accuracy is implemented once in `src/metrics.py`. Task modules import and
use `compute_accuracy` for their existing sanity checks; training code
can use `from src.metrics import compute_accuracy` from the project root.

The Llama template and `render_chat` are shared in `src/model/utils.py`. Task
modules keep their `<TASK>_CHAT_TEMPLATE_LLAMA31` names as aliases for the shared
`GENERAL_TEMPLATE`, so existing callers can keep using those names. AGNEWS's
main sanity check reads existing local JSONL files and the local tokenizer;
regenerating its data is an explicit call to `build_agnews_files`.

Full generation evaluation uses the local exported test JSONL and is separate
from these data preparation sanity checks. See the [project README](../../README.md)
for the seven evaluation scripts, ordinary LoRA loading, label mappings, output
fields, three-order training queue, complete evaluation and step/loss plots.

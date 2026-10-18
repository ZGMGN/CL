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
original text fields. Prompts map labels to A–E, A–E, A–N, and A–J respectively.

From the project directory, run any task separately:

```bash
python tasks/AMAZON.py
python tasks/YELP.py
python tasks/DBPEDIA.py
python tasks/YAHOO.py
```

Each task writes `train.jsonl`, `val.jsonl` and `test.jsonl` into `data/<TASK>/`,
using uppercase task names. Re-running replaces that task's existing files.
By default `HF_ENDPOINT` is `https://hf-mirror.com`; a preconfigured endpoint is
respected. Downloads require network access and sufficient local storage.

The `n_train=2000`, `n_val=500`, `n_test=500` defaults are **per class**:

| Task | Train total | Validation total | Test total |
| --- | ---: | ---: | ---: |
| AMAZON | 10000 | 2500 | 2500 |
| YELP | 10000 | 2500 | 2500 |
| DBPEDIA | 28000 | 7000 | 7000 |
| YAHOO | 20000 | 5000 | 5000 |

Each module provides `load_<task>_data`, `build_<task>_files` and the existing
message / formatting / accuracy helpers. `load_<task>_data` returns the full
custom split; builders export balanced subsamples. Builders load the upstream
dataset once. All sizes and `seed` can be set in Python:

```python
from tasks.AMAZON import build_amazon_files

build_amazon_files('data/AMAZON', n_train=1000, n_val=200, n_test=200, seed=1)
```

`example_to_messages`, `format_prompt`, and `extract_label_from_example` support
both single examples and column-oriented batches. `format_prompt` can omit the
answer without requiring a label. Set the module's `<TASK>_CHAT_TEMPLATE_LLAMA31`
on your tokenizer when using Llama 3.1. Pass only generated response text to
`extract_answer_letter`, without the original prompt and choices.

Offline checks (synthetic data; no dataset download):

```bash
python -m unittest discover -s tests -p 'test_standard_cl_tasks.py'
```

The local tokenizer rendering check runs if `Llama-3.1-8B-Instruct` is present.

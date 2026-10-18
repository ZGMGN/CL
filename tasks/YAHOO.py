"""YAHOO classification with the project's stratified 6:2:2 split.

n_train, n_val and n_test specify samples PER CLASS, not total samples.
Only the upstream training split is used; test.jsonl is a custom held-out split.
"""

import os
import re
import string
from collections import Counter
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np
from datasets import ClassLabel, load_dataset

ROOT_DIR = str(Path(__file__).resolve().parents[1])
DATASET_ID = 'community-datasets/yahoo_answers_topics'
CLASS_NAMES = ('Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference', 'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment & Music', 'Family & Relationships', 'Politics & Government')
TEXT_FIELDS = (('question_title', 'Question title'), ('question_content', 'Question content'), ('best_answer', 'Best answer'))
SOURCE_LABEL_COL = 'topic'
ANSWER_LETTERS = string.ascii_uppercase[:len(CLASS_NAMES)]
MessageList = List[Dict[str, str]]

__all__ = [
    'YAHOO_CHAT_TEMPLATE_LLAMA31', 'example_to_messages', 'render_chat',
    'format_prompt', 'extract_answer_letter', 'extract_label_from_example',
    'compute_accuracy', 'load_yahoo_data', 'build_yahoo_files',
]

YAHOO_CHAT_TEMPLATE_LLAMA31 = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

_PROMPT_SUFFIX = (
    'What is the topic of this question and answer?' + '\n'
    + '\n'.join(f'{letter}. {name}' for letter, name in zip(ANSWER_LETTERS, CLASS_NAMES))
    + '\nAnswer with ' + ', '.join(ANSWER_LETTERS) + ' only.\nAnswer:'
)


def _build_answer(index: int) -> str:
    if not isinstance(index, Integral) or not 0 <= index < len(CLASS_NAMES):
        raise ValueError(f'YAHOO label must be in [0, {len(CLASS_NAMES) - 1}], got {index}')
    return ANSWER_LETTERS[index]


def _user_message(example: Dict[str, Any]) -> Dict[str, str]:
    parts = []
    for field, heading in TEXT_FIELDS:
        value = example[field]
        if value is None:
            value = ''
        if not isinstance(value, str):
            raise TypeError(f'{field} must be a string or None')
        parts.append(f'###{heading}: {value.strip()}')
    return {'role': 'user', 'content': '\n'.join(parts) + '\n' + _PROMPT_SUFFIX}


def _is_batched(example: Dict[str, Any]) -> bool:
    return not isinstance(example[TEXT_FIELDS[0][0]], (str, type(None)))


def _batch_rows(example: Dict[str, Any]):
    fields = [field for field, _ in TEXT_FIELDS]
    if 'label' in example:
        fields.append('label')
    size = len(example[fields[0]])
    if any(len(example[field]) != size for field in fields):
        raise ValueError('Batch columns must have the same length')
    return [{field: example[field][i] for field in fields} for i in range(size)]


def _single_example_to_message(example: Dict[str, Any]) -> MessageList:
    return [_user_message(example), {'role': 'assistant', 'content': _build_answer(example['label'])}]


def example_to_messages(example: Dict[str, Any]) -> Union[MessageList, List[MessageList]]:
    if _is_batched(example):
        return [_single_example_to_message(row) for row in _batch_rows(example)]
    return _single_example_to_message(example)


def render_chat(messages: MessageList, tokenizer, add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
    )


def format_prompt(example: Dict[str, Any], tokenizer, include_answer: bool = False) -> Union[str, List[str]]:
    if _is_batched(example):
        return [format_prompt(row, tokenizer, include_answer) for row in _batch_rows(example)]
    messages = _single_example_to_message(example) if include_answer else [_user_message(example)]
    return render_chat(messages, tokenizer, add_generation_prompt=not include_answer)


def extract_answer_letter(text: str) -> Optional[str]:
    """Parse only generated answer text, without the original prompt/options."""
    text_u = text.strip().upper()
    letters = re.escape(ANSWER_LETTERS)
    m = re.fullmatch(rf'\(?([{letters}])\)?[.。]?', text_u)
    if m:
        return m.group(1)
    m = re.search(
        rf'\b(?:(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s*(?:IS\b|:|：|-)?'
        rf'\s*(?:OPTION\b|CHOICE\b)?|OPTION|CHOICE)\s*[:：]?\s*\(?([{letters}])\b',
        text_u,
    )
    if m:
        return m.group(1)
    m = re.match(rf'^([{letters}])[.)、]\s*', text_u)
    return m.group(1) if m else None


def extract_label_from_example(example: Dict[str, Any]) -> Union[str, List[str]]:
    if _is_batched(example):
        return [_build_answer(row['label']) for row in _batch_rows(example)]
    return _build_answer(example['label'])


def compute_accuracy(preds, golds) -> float:
    if len(preds) != len(golds):
        raise ValueError('Prediction and label lengths must match')
    if len(golds) == 0:
        return 0.0
    return sum(p == gold for p, gold in zip(preds, golds)) / len(golds)


def _load_source_data(cache_dir, token):
    ds = load_dataset(DATASET_ID, split='train', cache_dir=cache_dir, token=token)
    if SOURCE_LABEL_COL != 'label':
        ds = ds.rename_column(SOURCE_LABEL_COL, 'label')
    # Amazon's CSV labels are integers; stratify_by_column requires ClassLabel.
    ds = ds.cast_column('label', ClassLabel(names=list(CLASS_NAMES)))
    if 'idx' not in ds.column_names:
        ds = ds.add_column('idx', list(range(len(ds))))
    return ds


def _split_data(ds, seed):
    rng_seed = 1 if seed is None else seed
    first = ds.train_test_split(test_size=0.2, seed=rng_seed, stratify_by_column='label')
    second = first['train'].train_test_split(test_size=0.25, seed=rng_seed, stratify_by_column='label')
    return {'train': second['train'], 'val': second['test'], 'test': first['test']}


def load_yahoo_data(
    split: str = 'train', cache_dir: str = f'{ROOT_DIR}/.cache/huggingface_models/',
    token: Optional[str] = None, seed: Optional[int] = 1,
):
    if split not in ('train', 'val', 'test'):
        raise ValueError(f'split must be train, val or test, got {split}')
    return _split_data(_load_source_data(cache_dir, token), seed)[split]


def _balanced_subsample(ds, n_per_class: int, seed: int = 1, label_col: str = 'label'):
    if not isinstance(n_per_class, Integral) or n_per_class <= 0:
        raise ValueError('n_per_class must be a positive integer')
    rng = np.random.default_rng(1 if seed is None else seed)
    labels = np.asarray(ds[label_col])
    if len(labels) == 0:
        raise ValueError('Cannot sample an empty dataset')
    picked = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) < n_per_class:
            raise ValueError(f'class {lab} has only {len(idx)} examples, need {n_per_class}')
        picked.append(rng.choice(idx, size=n_per_class, replace=False))
    return ds.select(rng.permutation(np.concatenate(picked)).tolist())


def build_yahoo_files(
    out_dir, n_train: int = 2000, n_val: int = 500, n_test: int = 500, seed: int = 1,
    cache_dir: str = f'{ROOT_DIR}/.cache/huggingface_models/', token: Optional[str] = None,
):
    """Export raw fields + label + idx; sample sizes are per class."""
    parts = _split_data(_load_source_data(cache_dir, token), seed)
    sizes = {'train': n_train, 'val': n_val, 'test': n_test}
    subsets = {}
    for split, n in sizes.items():
        sub = _balanced_subsample(parts[split], n, seed=seed)
        counts = Counter(sub['label'])
        if counts != {lab: n for lab in range(len(CLASS_NAMES))}:
            raise ValueError(f'{split} has missing or unbalanced classes: {counts}')
        subsets[split] = sub
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(subsets[left]['idx']) & set(subsets[right]['idx']):
            raise ValueError(f'{left} and {right} share source rows')
    os.makedirs(out_dir, exist_ok=True)
    for split, sub in subsets.items():
        sub.to_json(os.path.join(out_dir, f'{split}.jsonl'), force_ascii=False)
        print(split, len(sub), dict(Counter(sub['label'])))


if __name__ == '__main__':
    build_yahoo_files(os.path.join(ROOT_DIR, 'data', 'YAHOO'))

import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from collections import Counter

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])
import sys
import_dir = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, import_dir)

__all__ = [
    "SST2_CHAT_TEMPLATE_LLAMA31",
    "example_to_messages",
    "render_chat",
    "format_prompt",
    "extract_answer_letter",
    "extract_label_from_example",
    "compute_accuracy"
]

SST2_CHAT_TEMPLATE_LLAMA31 = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

_prompt_sentence = '###Sentence: '
_prompt_format_after = (
    'What is the sentiment of this sentence?\n'
    'A. Negative\n'
    'B. Positive\n'
    'Answer with A or B only.\n'
    'Answer:'
)

def _build_answer(index: int) -> str:
    if index not in (0, 1):
        raise ValueError(f'SST-2 label must be 0 or 1, got {index}')
    return 'AB'[index]

def _single_example_to_message(example: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {
            'role': 'user',
            'content': _prompt_sentence + example['sentence'].strip() + '\n' + _prompt_format_after,
        },
        {
            'role': 'assistant',
            'content': _build_answer(example['label']),
        },
    ]

def example_to_messages(example: Dict[str, Any]) -> List[Dict[str,str]]:
    if isinstance(example['sentence'],str):
        return _single_example_to_message(example)
    messages = []
    for i in range(len(example['sentence'])):
        messages.append(
            _single_example_to_message(
                {
                    'sentence':example['sentence'][i],
                    'label':example['label'][i]
                }
            )
        )
    return messages


def render_chat(messages: List[Dict[str,str]],tokenizer, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

def format_prompt(example:Dict[str,Any],tokenizer,include_answer):
    messages = example_to_messages(example)
    if not include_answer:
        messages = [msg for msg in messages if msg['role'] != 'assistant']
    return render_chat(messages, tokenizer, add_generation_prompt= not include_answer)

def extract_answer_letter(text: str) -> Optional[str]:
    text_u = text.strip().upper()

    # 例如：B、B.、(B)、B)
    m = re.fullmatch(r"\(?([AB])\)?[.。]?\)?", text_u)
    if m:
        return m.group(1)

    # 例如：The correct answer is B、Answer: A、Option B
    m = re.search(
        r"\b(?:"
        r"(?:THE\s+)?(?:CORRECT\s+)?ANSWER"
        r"\s*(?:IS\b|:|：|-)?\s*(?:OPTION\b|CHOICE\b)?"
        r"|OPTION|CHOICE"
        r")\s*[:：]?\s*\(?([AB])\b",
        text_u,
    )
    if m:
        return m.group(1)

    # 例如：A. Negative、B) Positive
    m = re.match(r"^([AB])[.)、]\s*", text_u)
    if m:
        return m.group(1)

    return None


def extract_label_from_example(example):
    messages = example_to_messages(example)
    for msg in reversed(messages):
        if msg['role'] == 'assistant':
            return extract_answer_letter(msg['content'])
    return None

def compute_accuracy(preds, golds):
    if len(preds) != len(golds):
        raise ValueError('长度不一致')
    if not preds:
        return 0.0
    return sum(p == l for p,l in zip(preds,golds))/len(golds)

def load_sst2_data(
    split:str = 'train',
    cache_dir = f'{ROOT_DIR}/.cache/huggingface_models/',
    token = None,
    seed = 1
):
    assert split in ['train', 'test', 'val'], f"split must be 'train','test',or 'val' but got {split}."
    ds = load_dataset(
        "nyu-mll/glue", 
        "sst2",
        split='train',
        cache_dir=cache_dir,
        token=token
    )
    rng_seed = 1 if seed is None else seed
    stratify_col = 'label'
    train_val_and_test = ds.train_test_split(
        test_size=0.2, seed=rng_seed, stratify_by_column=stratify_col
    )
    train_val_set = train_val_and_test["train"]
    test_set = train_val_and_test["test"]

    if split == "test":
        return test_set

    train_and_val = train_val_set.train_test_split(
        test_size=0.25, seed=rng_seed, stratify_by_column=stratify_col
    )
    train_set = train_and_val["train"]
    val_set = train_and_val["test"]

    if split == "train":
        return train_set
    return val_set

def _balanced_subsample(ds, n_per_class:int , seed:int =1, label_col:str = 'label'):
    rng = np.random.default_rng(seed)
    labels = np.asarray(ds[label_col])
    picked = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx)<n_per_class:
            raise ValueError(f'class {lab} has only {len(idx)} examples, need {n_per_class}')
        picked.append(rng.choice(idx, size=n_per_class,replace=False))
    picked = rng.permutation(np.concatenate(picked))
    return ds.select(picked.tolist())

def build_sst2_files(out_dir, n_train=4000, n_val=1000, n_test = 1000, seed=1):
    os.makedirs(out_dir, exist_ok=True)
    sizes = {
        'train':n_train,
        'test':n_test,
        'val':n_val
    }
    for split, n in sizes.items():
        ds = load_sst2_data(split, seed=seed)
        sub = _balanced_subsample(ds, n,seed=seed)
        sub.to_json(os.path.join(out_dir,f'{split}.jsonl'), force_ascii=False)
        counts = Counter(sub['label'])
        assert len(set(counts.values())) == 1, f'{split} not balanced: {counts}'
        print(split, len(sub),dict(counts))


if __name__ == "__main__":
    out_dir = os.path.join(ROOT_DIR, "data", "SST2")
    build_sst2_files(out_dir)

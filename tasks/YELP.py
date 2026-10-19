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
    "YELP_CHAT_TEMPLATE_LLAMA31",
    "example_to_messages",
    "render_chat",
    "format_prompt",
    "extract_answer_letter",
    "extract_label_from_example",
    "compute_accuracy"
]

YELP_CHAT_TEMPLATE_LLAMA31 = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)

def _build_answer(index:int)->str:
    letters = string.ascii_uppercase[:5]
    if index < 0 or index > len(letters)-1:
        raise ValueError(f'Option must be in [0, {len(letters)-1}],but got {index}.')
    return letters[index]

_prompt_text = '###Review: '
_prompt_format_after = (
    'What is the star rating of this business review?\n'
    'A. 1 star\n'
    'B. 2 stars\n'
    'C. 3 stars\n'
    'D. 4 stars\n'
    'E. 5 stars\n'
    'Answer with A, B, C, D or E only.\n'
    'Answer:'
)

def _single_example_to_message(example:Dict[str, Any])->List[Dict[str,str]]:
    return [
        {
            'role':'user',
            'content':_prompt_text+(example['text'] or '').strip()+ '\n'+_prompt_format_after
        },
        {
            'role':'assistant',
            'content':_build_answer(example['label'])
        },
    ]

def example_to_messages(example: Dict[str, Any]) -> List[Dict[str,str]]:
    if isinstance(example['text'],str) or example['text'] is None:
        return _single_example_to_message(example)
    messages = []
    for i in range(len(example['text'])):
        messages.append(
            _single_example_to_message(
                {
                    'text':example['text'][i],
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
    m = re.fullmatch(r"\(?([ABCDE])\)?[.。]?\)?", text_u)
    if m:
        return m.group(1)

    # 例如：The correct answer is B、Answer: C、Option D
    m = re.search(
        r"\b(?:"
        r"(?:THE\s+)?(?:CORRECT\s+)?ANSWER"
        r"\s*(?:IS\b|:|：|-)?\s*(?:OPTION\b|CHOICE\b)?"
        r"|OPTION|CHOICE"
        r")\s*[:：]?\s*\(?([ABCDE])\b",
        text_u,
    )
    if m:
        return m.group(1)

    # 例如：B. 后面跟选项说明
    m = re.match(r"^([ABCDE])[.)、]\s*", text_u)
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

def load_yelp_data(
    cache_dir = f'{ROOT_DIR}/.cache/huggingface_models/',
    token = None,
    seed = 1
):
    # 一次加载和划分，返回三个集合供后续复用。
    ds = load_dataset(
        "Yelp/yelp_review_full",
        split='train',
        cache_dir=cache_dir,
        token=token
    )
    # 在划分前添加原始行号，用来检查集合之间是否交叉。
    ds = ds.add_column('idx', np.arange(len(ds), dtype=np.int64))
    rng_seed = 1 if seed is None else seed
    stratify_col = 'label'
    train_val_and_test = ds.train_test_split(
        test_size=0.2, seed=rng_seed, stratify_by_column=stratify_col
    )
    train_val_set = train_val_and_test["train"]
    test_set = train_val_and_test["test"]

    train_and_val = train_val_set.train_test_split(
        test_size=0.25, seed=rng_seed, stratify_by_column=stratify_col
    )
    train_set = train_and_val["train"]
    val_set = train_and_val["test"]

    return {"train": train_set, "val": val_set, "test": test_set}

def _balanced_subsample(ds, n_per_class:int , seed:int =1, label_col:str = 'label'):
    if not isinstance(n_per_class, (int, np.integer)) or n_per_class <= 0:
        raise ValueError('n_per_class must be a positive integer')
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

def build_yelp_files(out_dir, n_train=2000, n_val=400, n_test = 400, seed=1):
    os.makedirs(out_dir, exist_ok=True)
    sizes = {
        'train':n_train,
        'test':n_test,
        'val':n_val
    }
    # 完整数据只加载和划分一次。
    datasets = load_yelp_data(seed=seed)
    seen_idx = {}
    for split, n in sizes.items():
        ds = datasets[split]
        sub = _balanced_subsample(ds, n,seed=seed)
        sub.to_json(os.path.join(out_dir,f'{split}.jsonl'), force_ascii=False)
        seen_idx[split] = set(sub['idx'])
        counts = Counter(sub['label'])
        assert len(set(counts.values())) == 1, f'{split} not balanced: {counts}'
        print(split, len(sub),dict(counts))
    assert not (seen_idx['train'] & seen_idx['val'])
    assert not (seen_idx['train'] & seen_idx['test'])
    assert not (seen_idx['val'] & seen_idx['test'])

if __name__ == '__main__':
    data_path = os.path.join(ROOT_DIR, "data", "YELP")
    os.makedirs(data_path,exist_ok=True)
    build_yelp_files(data_path)
    # 导出之后只读取抽样后的 JSONL，不再处理完整数据。
    datasets = load_dataset(
        'json',
        data_files={
            split: os.path.join(data_path, f'{split}.jsonl')
            for split in ['train', 'val', 'test']
        },
        cache_dir=f'{ROOT_DIR}/.cache/huggingface_models/',
    )
    dataset_train = datasets['train']
    dataset_val = datasets['val']
    dataset_test = datasets['test']

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_DIR, r"Llama-3.1-8B-Instruct"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = 1500
    tokenizer.chat_template = YELP_CHAT_TEMPLATE_LLAMA31

    print(
        f"\nLength of train set: {len(dataset_train)}, val set: {len(dataset_val)}, "
        f"test set: {len(dataset_test)}"
    )
    for split in ['train','val','test']:
        print(f'\n\nRendering some {split} examples...')
        ds = datasets[split]
        for i in range(2):
            example = ds[i]
            messages = example_to_messages(example)
            rendered = render_chat(messages, tokenizer, add_generation_prompt=False)
            print(f"\n--- Example {i} ---\n")
            print(rendered)

        k = min(2, len(ds))
        batched = ds[:k]
        batched_messages = example_to_messages(batched)
        assert len(batched_messages) == k
        for j in range(k):
            single = {col:batched[col][j] for col in batched}
            assert example_to_messages(single) == batched_messages[j]
        for j, messages in enumerate(batched_messages):
            rendered = render_chat(messages, tokenizer, add_generation_prompt=False)
            print(f"\n--- Batched example {j} (split={split}) ---\n")
            print(rendered)

    # Sanity check: format prompt without the answer.
    sample = dataset_train[0]
    prompt = format_prompt(sample, tokenizer, include_answer=False)
    print("\n--- Prompt Preview ---\n")
    print(prompt)

    # Sanity check: label extraction.
    gold_label = extract_label_from_example(sample)
    parsed_label = extract_answer_letter(example_to_messages(sample)[-1]["content"])
    assert gold_label in 'ABCDE'
    assert gold_label == parsed_label
    print(f"\nGold label: {gold_label}, Parsed label: {parsed_label}")

    # Sanity check: tiny accuracy check.
    predictions = [gold_label, gold_label]
    labels = [gold_label, "Z"]
    print(f"\nTiny accuracy check (expect 0.5): {compute_accuracy(predictions, labels)}")
    assert compute_accuracy(predictions, labels) == 0.5
    print("YELP sanity checks: PASS")

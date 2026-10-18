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
    "AGNEWS_CHAT_TEMPLATE_LLAMA31",
    "example_to_messages",
    "render_chat",
    "format_prompt",
    "extract_answer_letter",
    "extract_label_from_example",
    "compute_accuracy"
]

AGNEWS_CHAT_TEMPLATE_LLAMA31 = (
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
    letters = string.ascii_uppercase
    if index < 0 or index > len(letters)-1:
        raise ValueError(f'Option must be in [0, {len(letters)-1}],but got {index}.')
    return letters[index]

_prompt_news = '###News: '
_prompt_format_after = (
    'What is the topic of this news article?\n'
    'A. World\n'
    'B. Sports\n'
    'C. Business\n'
    'D. Science/Technology\n'
    'Answer with A, B, C or D only.\n'
    'Answer:'
)

def _single_example_to_message(example:Dict[str, Any])->List[Dict[str,str]]:
    return [
        {
            'role':'user',
            'content':_prompt_news+example['text'].strip()+ '\n'+_prompt_format_after
        },
        {
            'role':'assistant',
            'content':_build_answer(example['label'])
        },
    ]

def example_to_messages(example: Dict[str, Any]) -> List[Dict[str,str]]:
    if isinstance(example['text'],str):
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
    m = re.fullmatch(r"\(?([ABCD])\)?[.。]?\)?", text_u)
    if m:
        return m.group(1)

    # 例如：The correct answer is B、Answer: C、Option D
    m = re.search(
        r"\b(?:"
        r"(?:THE\s+)?(?:CORRECT\s+)?ANSWER"
        r"\s*(?:IS\b|:|：|-)?\s*(?:OPTION\b|CHOICE\b)?"
        r"|OPTION|CHOICE"
        r")\s*[:：]?\s*\(?([ABCD])\b",
        text_u,
    )
    if m:
        return m.group(1)

    # 例如：B. Sports、C) Business
    m = re.match(r"^([ABCD])[.)、]\s*", text_u)
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

def load_agnews_data(
    split:str = 'train',
    cache_dir = f'{ROOT_DIR}/.cache/huggingface_models/',
    token = None,
    seed = 1
):
    assert split in ['train', 'test', 'val'], f"split must be 'train','test',or 'val' but got {split}."
    ds = load_dataset(
        "fancyzhx/ag_news",
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

def build_agnews_files(out_dir, n_train=2000, n_val=500, n_test = 500, seed=1):
    os.makedirs(out_dir, exist_ok=True)
    sizes = {
        'train':n_train,
        'test':n_test,
        'val':n_val
    }
    seen_idx = {}
    for split, n in sizes.items():
        ds = load_agnews_data(split, seed=seed)
        sub = _balanced_subsample(ds, n,seed=seed)
        sub.to_json(os.path.join(out_dir,f'{split}.jsonl'), force_ascii=False)
        counts = Counter(sub['label'])
        assert len(set(counts.values())) == 1, f'{split} not balanced: {counts}'
        print(split, len(sub),dict(counts))

if __name__ == '__main__':
    data_path = os.path.join(ROOT_DIR, "data", "AGNEWS")
    os.makedirs(data_path,exist_ok=True)
    build_agnews_files(data_path)
    dataset_train = load_agnews_data(split='train')
    dataset_val = load_agnews_data(split="val")
    dataset_test = load_agnews_data(split="test")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_DIR, r"Llama-3.1-8B-Instruct"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = 1500
    tokenizer.chat_template = AGNEWS_CHAT_TEMPLATE_LLAMA31

    print(
        f"\nLength of train set: {len(dataset_train)}, val set: {len(dataset_val)}, "
        f"test set: {len(dataset_test)}"
    )  
    for split in ['train','val','test']:
        print(f'\n\nRendering some {split} examples...')
        ds = load_agnews_data(split=split)
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
    print(f"\nGold label: {gold_label}, Parsed label: {parsed_label}")

    # Sanity check: tiny accuracy check.
    predictions = [gold_label, gold_label]
    labels = [gold_label, "Z"]
    print(f"\nTiny accuracy check (expect 0.5): {compute_accuracy(predictions, labels)}")   
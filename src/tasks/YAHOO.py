import os
import re
import string
from typing import Any, Dict, List, Optional, Tuple

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from collections import Counter

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])
import sys
import_dir = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, import_dir)

__all__ = [
    "YAHOO_CHAT_TEMPLATE_LLAMA31",
    "example_to_messages",
    "render_chat",
    "format_prompt",
    "extract_answer_letter",
    "extract_label_from_example",
    "compute_accuracy"
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

def _build_answer(index:int)->str:
    letters = string.ascii_uppercase[:10]
    if index < 0 or index > len(letters)-1:
        raise ValueError(f'Option must be in [0, {len(letters)-1}],but got {index}.')
    return letters[index]

_prompt_question_title = '###Question title: '
_prompt_question_content = '###Question content: '
_prompt_best_answer = '###Best answer: '
_prompt_format_after = (
    'What is the topic of this question and answer?\n'
    'A. Society & Culture\n'
    'B. Science & Mathematics\n'
    'C. Health\n'
    'D. Education & Reference\n'
    'E. Computers & Internet\n'
    'F. Sports\n'
    'G. Business & Finance\n'
    'H. Entertainment & Music\n'
    'I. Family & Relationships\n'
    'J. Politics & Government\n'
    'Answer with A, B, C, D, E, F, G, H, I or J only.\n'
    'Answer:'
)

def _single_example_to_message(example:Dict[str, Any])->List[Dict[str,str]]:
    return [
        {
            'role':'user',
            'content':_prompt_question_title+(example['question_title'] or '').strip()+ '\n'
            +_prompt_question_content+(example['question_content'] or '').strip()+ '\n'
            +_prompt_best_answer+(example['best_answer'] or '').strip()+ '\n'+_prompt_format_after
        },
        {
            'role':'assistant',
            'content':_build_answer(example['label'])
        },
    ]

def example_to_messages(example: Dict[str, Any]) -> List[Dict[str,str]]:
    if isinstance(example['question_title'],str) or example['question_title'] is None:
        return _single_example_to_message(example)
    messages = []
    for i in range(len(example['question_title'])):
        messages.append(
            _single_example_to_message(
                {
                    'question_title':example['question_title'][i],
                    'question_content':example['question_content'][i],
                    'best_answer':example['best_answer'][i],
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
    m = re.fullmatch(r"\(?([ABCDEFGHIJ])\)?[.。]?\)?", text_u)
    if m:
        return m.group(1)

    # 例如：The correct answer is B、Answer: C、Option D
    m = re.search(
        r"\b(?:"
        r"(?:THE\s+)?(?:CORRECT\s+)?ANSWER"
        r"\s*(?:IS\b|:|：|-)?\s*(?:OPTION\b|CHOICE\b)?"
        r"|OPTION|CHOICE"
        r")\s*[:：]?\s*\(?([ABCDEFGHIJ])\b",
        text_u,
    )
    if m:
        return m.group(1)

    # 例如：B. 后面跟选项说明
    m = re.match(r"^([ABCDEFGHIJ])[.)、]\s*", text_u)
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

def load_yahoo_data(
    cache_dir = f'{ROOT_DIR}/.cache/huggingface_models/',
    token = None,
    seed = 1
):
    # 一次加载和划分，返回三个集合供后续复用。
    ds = load_dataset(
        "community-datasets/yahoo_answers_topics",
        split='train',
        cache_dir=cache_dir,
        token=token
    )
    # Yahoo 原始标签列名为 topic，统一为 label。
    ds = ds.rename_column('topic', 'label')
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

def _filter_by_length(ds, tokenizer, max_length=1536):
    # 统计完整训练对话，包括模板、选项和 assistant 答案，不做截断。
    if not isinstance(max_length, (int, np.integer)) or max_length <= 0:
        raise ValueError('max_length must be a positive integer')
    kept = []
    for start in range(0, len(ds), 128):
        batch = ds.select(range(start, min(start + 128, len(ds))))
        texts = [format_prompt(example, tokenizer, include_answer=True) for example in batch]
        encoded = tokenizer(
            texts, add_special_tokens=False, truncation=False,
            padding=False, return_attention_mask=False, verbose=False,
        )
        for i, ids in enumerate(encoded['input_ids']):
            if len(ids) <= max_length:
                kept.append(start + i)
    return ds.select(kept)


def _balanced_subsample(ds, n_per_class:int , seed:int =1, label_col:str = 'label',
                        tokenizer=None, max_length=1536):
    if not isinstance(n_per_class, (int, np.integer)) or n_per_class <= 0:
        raise ValueError('n_per_class must be a positive integer')
    rng = np.random.default_rng(seed)
    labels = np.asarray(ds[label_col])
    picked = []
    checked = 0
    rejected = 0
    for lab in np.unique(labels):
        remaining = np.flatnonzero(labels == lab)
        selected = []
        while len(selected) < n_per_class:
            need = n_per_class - len(selected)
            if len(remaining) < need:
                raise ValueError(
                    f'class {lab} has too few eligible examples after length filtering; '
                    f'need {n_per_class}, selected {len(selected)}'
                )
            candidates = rng.choice(remaining, size=need, replace=False)
            if tokenizer is None:
                selected.extend(candidates.tolist())
            else:
                batch = ds.select(candidates.tolist())
                filtered = _filter_by_length(batch, tokenizer, max_length)
                valid_idx = set(filtered['idx'])
                selected.extend(int(i) for i in candidates if ds[int(i)]['idx'] in valid_idx)
                checked += len(batch)
                rejected += len(batch) - len(filtered)
            remaining = np.setdiff1d(remaining, candidates, assume_unique=True)
        picked.append(np.asarray(selected, dtype=np.int64))
    picked = rng.permutation(np.concatenate(picked))
    if tokenizer is not None:
        print(f'Length filter (<= {max_length} tokens): checked {checked}, removed {rejected}')
    return ds.select(picked.tolist())


def build_yahoo_files(out_dir, n_train=1000, n_val=200, n_test = 200, seed=1,
                      max_length=1536, tokenizer=None):
    os.makedirs(out_dir, exist_ok=True)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(ROOT_DIR, 'Llama-3.1-8B-Instruct'), local_files_only=True,
        )
    tokenizer.chat_template = YAHOO_CHAT_TEMPLATE_LLAMA31
    tokenizer.model_max_length = max_length
    sizes = {
        'train':n_train,
        'test':n_test,
        'val':n_val
    }
    # 完整数据只加载和划分一次；仅检查抽样候选，避免编码全部 140 万条。
    datasets = load_yahoo_data(seed=seed)
    seen_idx = {}
    subsets = {}
    for split, n in sizes.items():
        ds = datasets[split]
        # 超长候选被排除后继续抽取，直到每类数量补齐。
        sub = _balanced_subsample(
            ds, n, seed=seed, tokenizer=tokenizer, max_length=max_length,
        )
        seen_idx[split] = set(sub['idx'])
        counts = Counter(sub['label'])
        assert counts == {label: n for label in range(10)}, f'{split} not balanced: {counts}'
        assert len(seen_idx[split]) == len(sub), f'{split} has repeated source rows'
        subsets[split] = sub
    assert not (seen_idx['train'] & seen_idx['val'])
    assert not (seen_idx['train'] & seen_idx['test'])
    assert not (seen_idx['val'] & seen_idx['test'])
    # 所有集合检查通过后再写出，避免过滤后数量不足时只更新部分文件。
    for split, sub in subsets.items():
        sub.to_json(os.path.join(out_dir,f'{split}.jsonl'), force_ascii=False)
        print(split, len(sub),dict(Counter(sub['label'])))

if __name__ == '__main__':
    data_path = os.path.join(ROOT_DIR, "data", "YAHOO")
    os.makedirs(data_path,exist_ok=True)
    build_yahoo_files(data_path)
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
    tokenizer.model_max_length = 1536
    tokenizer.chat_template = YAHOO_CHAT_TEMPLATE_LLAMA31

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
    assert gold_label in 'ABCDEFGHIJ'
    assert gold_label == parsed_label
    print(f"\nGold label: {gold_label}, Parsed label: {parsed_label}")

    # Sanity check: tiny accuracy check.
    predictions = [gold_label, gold_label]
    labels = [gold_label, "Z"]
    print(f"\nTiny accuracy check (expect 0.5): {compute_accuracy(predictions, labels)}")
    assert compute_accuracy(predictions, labels) == 0.5
    print("YAHOO sanity checks: PASS")

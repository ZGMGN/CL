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
    "QQP_CHAT_TEMPLATE_LLAMA31",
    "example_to_messages",
    "render_chat",
    "format_prompt",
    "extract_answer_letter",
    "extract_label_from_example",
    "compute_accuracy"
]

QQP_CHAT_TEMPLATE_LLAMA31 = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
    "{% endif %}"
)
_prompt_question1 = (
    '###Question1: '
)
_prompt_question2 = (
    '###Question2: '
)
_prompt_format_after = (
    'Are these two questions duplicates?\n'
    'A. No\n'
    'B. Yes\n'
    'Answer with A or B only.\n'
    'Answer:'
)

def _build_answer(index:int)->str:
    letters = string.ascii_uppercase
    if index <0 or index >= len(letters):
        raise ValueError(f'Option index must be in [0, {len(letters)-1}], but got {index}.')
    return letters[index]

def _single_example_to_message(example:Dict[str, Any])->List[Dict[str, str]]:
    return [
        {
            'role':'user',
            'content':_prompt_question1+example['question1']+'\n'
            +_prompt_question2+example['question2']+'\n'
            +_prompt_format_after
        },
        {
            'role':'assistant',
            'content':_build_answer(example['label'])
        },
    ]

def example_to_messages(example: Dict[str, Any])-> List[Dict[str, str]]:
    if isinstance(example['question1'], str):
        return _single_example_to_message(example)
    messages = []
    for i in range(len(example['question1'])):
        messages.append(
            _single_example_to_message(
                {
                    'question1':example['question1'][i],
                    'question2':example['question2'][i],
                    'label':example['label'][i],
                }
            )
        )
    return messages

def render_chat(messages:List[Dict[str,str]], tokenizer, add_generation_prompt:bool = False)->str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

def format_prompt(example:Dict[str,Any],tokenizer,include_answer:bool=False)->str:
    messages = example_to_messages(example)
    if not include_answer:
        messages = [msg for msg in messages if msg['role']!='assistant']
    return render_chat(messages, tokenizer, add_generation_prompt=not include_answer)

def extract_answer_letter(text: str) -> Optional[str]:
    text_u = text.strip().upper()

    # A、B、(A)、B.、B)
    m = re.fullmatch(r"\(?([AB])\)?[.。]?\)?", text_u)
    if m:
        return m.group(1)

    # Answer: B、The correct answer is B、Option A
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

    # A. No、B) Yes
    m = re.match(r"^([AB])[.)、]\s*", text_u)
    if m:
        return m.group(1)

    return None

def extract_label_from_example(example:Dict[str,Any])->Optional[str]:
    messages = example_to_messages(example)
    for msg in reversed(messages):
        if msg['role'] == 'assistant':
            return extract_answer_letter(msg['content'])
    return None


def compute_accuracy(preds: List[str], labels: List[str]) -> float:
    if len(preds) != len(labels):
        raise ValueError("预测数量和标签数量必须一致")
    if not labels:
        return 0.0

    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels)

def _load_QQP_data(
    split:str = 'train',
    cache_dir: str = f'{ROOT_DIR}/.cache/huggingface_models/',
    token: Optional[str] = None,
    seed: Optional[int] = 1,
):
    assert split in ['train','val','test'], f"split must be 'train','val','test',but got {split}"
    dataset = load_dataset(
        "nyu-mll/glue",
        "qqp",
        split = 'train',
        cache_dir = cache_dir,
        token = token,
    )
    rng_seed = 1 if seed is None else seed

    stratify_col = 'label'
    # 6:2:2 split
    train_val_and_test = dataset.train_test_split(
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

def _balanced_subsample(ds, n_per_class:int, seed:int=1, label_col:str = 'label'):
    rng = np.random.default_rng(seed)
    #获取一个随机数生成器
    labels = np.asarray(ds[label_col])
    #转换成np数组
    picked = []
    #用来保存每个类别中抽中的样本索引
    for lab in np.unique(labels):
    #这里lab是单个类别，不是单个类别的索引列表
        idx = np.flatnonzero(labels == lab)
        #这里idx才是该标签在的位置
        if len(idx) < n_per_class:
            raise ValueError(f'class {lab} has only {len(idx)} examples, need {n_per_class}')
        picked.append(rng.choice(idx, size=n_per_class, replace=False))
        #表示从 idx 中随机抽取 n_per_class 个元素，且不重复抽取(replace参数控制)。
    picked = rng.permutation(np.concatenate(picked))
    #np.concatenate是合并数组，permutation是随机打乱顺序
    return ds.select(picked.tolist())
    #把numpy数组转为list，然后从ds中挑选这些索引

def build_qqp_files(out_dir, n_train=4000, n_val=500,n_test=500,seed=1):
    os.makedirs(out_dir, exist_ok=True)
    sizes = {
        'train':n_train,
        'val':n_val,
        'test':n_test
    }
    seen_idx = {}
    for split,n in sizes.items():
        ds = _load_QQP_data(split, seed=seed)
        sub = _balanced_subsample(ds, n, seed=seed)
        sub.to_json(os.path.join(out_dir, f'{split}.jsonl'),force_ascii=False)
        seen_idx[split] = set(sub['idx'])
        counts = Counter(sub['label'])
        assert len(set(counts.values())) == 1, f'{split} not balanced: {counts}'
        print(split, len(sub),dict(counts))
    assert not (seen_idx['train'] & seen_idx['val'])
    assert not (seen_idx['train'] & seen_idx['test'])
    assert not (seen_idx['val'] & seen_idx['test'])

if __name__ == '__main__':
    data_path = os.path.join(ROOT_DIR, "data",'QQP')
    os.makedirs(data_path,exist_ok=True)
    #build_qqp_files(data_path)
    dataset_train = _load_QQP_data(split='train')
    dataset_val = _load_QQP_data(split="val")
    dataset_test = _load_QQP_data(split="test")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_DIR, r"Llama-3.1-8B-Instruct"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = 1500
    tokenizer.chat_template = QQP_CHAT_TEMPLATE_LLAMA31

    print(
        f"\nLength of train set: {len(dataset_train)}, val set: {len(dataset_val)}, "
        f"test set: {len(dataset_test)}"
    )  
    for split in ['train','val','test']:
        print(f'\n\nRendering some {split} examples...')
        ds = _load_QQP_data(split=split)
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
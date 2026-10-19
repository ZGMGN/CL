import argparse
import json
import os
import re
from typing import Callable, Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])
import sys
import_dir = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, import_dir)

from model.utils import build_tokenizer, MODEL_DIR
from peft import PeftModel

def save_json_results(output_json_path: str, results: List[Dict], metrics: Dict)->None:
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump({"metrics":metrics, "results":results}, f_out, ensure_ascii=False, indent=2)

def build_eval_model(args, chat_template):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = build_tokenizer(args, chat_template=chat_template, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    if args.peft_model_folder:
        folder = os.path.abspath(args.peft_model_folder)
        model = PeftModel.from_pretrained(
            model,
            folder,
            is_trainable=False
        )
        model_tag = "finetuned"
    else:
        model_tag = "pretrained_baseline"
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, device, model_tag


@torch.no_grad()
def evaluate_qa(
    model,
    tokenizer,
    test_dataset,
    device,
    batch_size=1,
    max_input_length=1536,
    max_new_tokens=8,
    print_sample=True,
    *,
    task_name: str,
    format_prompt: Callable,
    extract_label_from_example: Callable,
    extract_answer_letter: Callable,
)-> Tuple[List[Dict],Dict]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    was_training = model.training

    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    model.eval()

    results = []
    correct=0
    try:
        for start in tqdm(
            range(0, len(test_dataset), batch_size),
            desc=f"Evaluating {task_name}",
            unit="batch",
        ):
            end = min(start+batch_size, len(test_dataset))
            entries = [test_dataset[i] for i in range(start,end)]

            prompts = [
                format_prompt(
                    entry,tokenizer, include_answer=False
                )
                for entry in entries
            ]
            labels = [
                extract_label_from_example(entry)
                for entry in entries
            ]

            if any(label is None for label in labels):
                raise ValueError(f"样本{start}:{end}中存在无效标签")
            
            inputs = tokenizer(
                prompts,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=max_input_length,
                add_special_tokens = False,
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample = False,
                pad_token_id = tokenizer.pad_token_id,
                use_cache=True
            )
            prompt_length = inputs["input_ids"].shape[1]
            responses = tokenizer.batch_decode(
                outputs[:, prompt_length:],
                skip_special_tokens=True,
            )
            for i, (response, label) in enumerate((zip(responses, labels))):
                prediction = extract_answer_letter(
                    response.strip()
                )
                is_correct = (
                    prediction is not None and prediction == label
                )
                correct += int(is_correct)
                results.append({
                    "index":start+i,
                    "ground_truth_answer": label,
                    "predicted_answer": prediction,
                    "full_response": response.strip(),
                    "correct": is_correct,
                })
        total = len(results)
        metrics = {
            "accuracy": correct/total if total else 0.0,
            "correct": correct,
            "total":total
        }
        if print_sample:
            for result in results[:3]:
                print(result)
        return results, metrics
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side
        model.train(was_training)


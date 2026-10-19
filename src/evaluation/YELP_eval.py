import os
import argparse
from datasets import load_dataset
from typing import Dict, List, Tuple

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])
DATA_DIR = os.path.join(ROOT_DIR,'data','YELP')
import sys
import_dir = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, import_dir)

from tasks.YELP import(
    YELP_CHAT_TEMPLATE_LLAMA31,
    format_prompt,
    extract_answer_letter,
    extract_label_from_example,
)

from evaluation.general_task_eval import (
    build_eval_model,
    evaluate_qa,
    save_json_results as save_qa_results,
)

__all__ = [
    "evaluate_YELP_QA",
    "save_qa_results",
]

def evaluate_YELP_QA(
    model,
    tokenizer,
    test_dataset,
    device,
    batch_size=1,
    max_input_length=1536,
    max_new_tokens=8,
    print_sample=True
)-> Tuple[List[Dict],Dict]:
    return evaluate_qa(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=device,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        print_sample=print_sample,
        task_name="YELP",
        format_prompt=format_prompt,
        extract_label_from_example=extract_label_from_example,
        extract_answer_letter=extract_answer_letter,
    )

def tiny_test(args):
    model, tokenizer, device, _ = build_eval_model(args, YELP_CHAT_TEMPLATE_LLAMA31)

    tiny_data = load_dataset(
        'json',
        data_files = os.path.join(DATA_DIR, "test.jsonl"),
        split="train",
    )
    tiny_data = tiny_data.select(range(min(2, len(tiny_data))))
    results, metrics = evaluate_YELP_QA(
        model = model,
        tokenizer = tokenizer,
        test_dataset=tiny_data,
        device=device,
        batch_size=2,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        print_sample=True,
    )
    print(f"YELP tiny test. Accuracy: {metrics['accuracy']:.3f}")
    qa_output_path = os.path.join(args.unit_test_output_dir, args.qa_unit_test_json_name)
    save_qa_results(qa_output_path, results, metrics)

def run_full_evaluation(args)->None:
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    test_set = load_dataset(
        'json',
        data_files = os.path.join(DATA_DIR, "test.jsonl"),
        split="train",
    )
    model, tokenizer, device, _ = build_eval_model(args, YELP_CHAT_TEMPLATE_LLAMA31)

    qa_results, qa_metrics = evaluate_YELP_QA(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_set,
        device=device,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        print_sample=args.print_sample,
    )
    save_qa_results(os.path.join(args.output_dir, args.qa_output_json_name), qa_results, qa_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YELP evaluator (local Llama model with optional LoRA)")
    parser.add_argument("--peft_model_folder", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_input_length", type=int, default=1536)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--print_sample", action="store_true")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "results", "YELP"))
    parser.add_argument("--qa_output_json_name", type=str, default="yelp_qa_results.json")
    parser.add_argument("--unit_test_output_dir", type=str, default=os.path.join(ROOT_DIR, "results", "YELP", "unit_test"))
    parser.add_argument("--qa_unit_test_json_name", type=str, default="yelp_qa_results.json")
    parser.add_argument("--unit_test", action="store_true")
    args = parser.parse_args()

    if args.unit_test:
        tiny_test(args)
    else:
        run_full_evaluation(args)

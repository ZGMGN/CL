"""Sequential LoRA training; carry each stage's validation-best adapter forward."""
import argparse
import gc
import json
import os
import sys
from functools import partial

import torch
from datasets import load_dataset
from peft import load_peft_weights, set_peft_model_state_dict
from transformers import TrainingArguments

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from model.utils import GENERAL_TEMPLATE, build_base_peft_model, build_tokenizer, set_seed
from evaluation.general_task_eval import save_json_results
from evaluation.AGNEWS_eval import evaluate_AGNEWS_QA
from evaluation.AMAZON_eval import evaluate_AMAZON_QA
from evaluation.YELP_eval import evaluate_YELP_QA
from evaluation.DBPEDIA_eval import evaluate_DBPEDIA_QA
from evaluation.YAHOO_eval import evaluate_YAHOO_QA
from evaluation.SST2_eval import evaluate_SST2_QA
from evaluation.QQP_eval import evaluate_QQP_QA

from continual_learning.train_single_task import (
    AnswerOnlyDataCollator,
    GenerationEvalTrainer,
    agnews_formatting_prompts_func,
    amazon_formatting_prompts_func,
    yelp_formatting_prompts_func,
    dbpedia_formatting_prompts_func,
    yahoo_formatting_prompts_func,
    sst2_formatting_prompts_func,
    qqp_formatting_prompts_func,
    agnews_answer_only_prompts_func,
    amazon_answer_only_prompts_func,
    yelp_answer_only_prompts_func,
    dbpedia_answer_only_prompts_func,
    yahoo_answer_only_prompts_func,
    sst2_answer_only_prompts_func,
    qqp_answer_only_prompts_func,
)

TASK_CONFIGS = {
    'AGNEWS': (agnews_formatting_prompts_func, agnews_answer_only_prompts_func, evaluate_AGNEWS_QA),
    'AMAZON': (amazon_formatting_prompts_func, amazon_answer_only_prompts_func, evaluate_AMAZON_QA),
    'YELP': (yelp_formatting_prompts_func, yelp_answer_only_prompts_func, evaluate_YELP_QA),
    'DBPEDIA': (dbpedia_formatting_prompts_func, dbpedia_answer_only_prompts_func, evaluate_DBPEDIA_QA),
    'YAHOO': (yahoo_formatting_prompts_func, yahoo_answer_only_prompts_func, evaluate_YAHOO_QA),
    'SST2': (sst2_formatting_prompts_func, sst2_answer_only_prompts_func, evaluate_SST2_QA),
    'QQP': (qqp_formatting_prompts_func, qqp_answer_only_prompts_func, evaluate_QQP_QA),
}


def reload_peft_adapter_weights(model, adapter_dir: str):
    """Restore saved adapter weights without replacing the trainable adapter."""
    adapter_state = load_peft_weights(adapter_dir, device='cpu')
    if not adapter_state:
        raise ValueError(f'No adapter weights found in {adapter_dir}')
    result = set_peft_model_state_dict(model, adapter_state)
    if result.unexpected_keys:
        raise ValueError(f'Unexpected adapter keys: {result.unexpected_keys}')


def parse_tasks(value):
    tasks = [task.strip().upper() for task in value.split(',') if task.strip()]
    if not tasks:
        raise ValueError('No tasks provided')
    unsupported = [task for task in tasks if task not in TASK_CONFIGS]
    if unsupported:
        raise ValueError(f'Unsupported tasks: {unsupported}. Choose from {", ".join(TASK_CONFIGS)}')
    return tasks


def load_task_data(task):
    data_files = {
        split: os.path.join(ROOT_DIR, 'data', task, f'{split}.jsonl')
        for split in ('train', 'val', 'test')
    }
    for path in data_files.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Dataset file does not exist: {path}')
    datasets = load_dataset('json', data_files=data_files)
    if any(len(datasets[split]) == 0 for split in data_files):
        raise ValueError(f'{task}: train, validation and test datasets must all be nonempty')
    return datasets


def build_training_args(args, output_dir):
    return TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        output_dir=output_dir,
        remove_unused_columns=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_num_workers=args.num_workers,
        max_grad_norm=0.3,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        **({'warmup_ratio': args.warmup_ratio}
           if 'warmup_ratio' in TrainingArguments.__dataclass_fields__
           else {'warmup_steps': args.warmup_ratio}),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        optim='adamw_torch',
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,
        report_to='none',
    )


def run_sequential(args):
    tasks = parse_tasks(args.tasks)
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError('warmup_ratio must be in [0, 1)')
    if args.max_seq_length < 2:
        raise ValueError('max_seq_length must be at least 2')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    set_seed(args.seed)
    # Validate all data before allocating the model or starting the first stage.
    datasets = {task: load_task_data(task) for task in dict.fromkeys(tasks)}
    tokenizer = build_tokenizer(args, chat_template=GENERAL_TEMPLATE)
    tokenizer.model_max_length = args.max_seq_length
    model = build_base_peft_model(args)
    output_dir = args.output_dir or os.path.join(ROOT_DIR, 'results', 'sequential')
    eval_kwargs = {
        'batch_size': args.per_device_eval_batch_size,
        'max_input_length': args.max_seq_length,
        'max_new_tokens': args.max_new_tokens,
        'print_sample': False,
    }
    summary = {'tasks': tasks, 'stages': []}
    seen_tasks = []
    for index, task in enumerate(tasks, start=1):
        _, formatting_pair_func, eval_fn = TASK_CONFIGS[task]
        # Stage numbers keep repeated tasks and different orders distinguishable.
        task_output_dir = os.path.join(output_dir, f'{index:02d}_{task}')
        model.config.use_cache = False
        trainer = GenerationEvalTrainer(
            model=model,
            args=build_training_args(args, task_output_dir),
            processing_class=tokenizer,
            data_collator=AnswerOnlyDataCollator(
                tokenizer,
                max_length=args.max_seq_length,
                formatting_pair_func=partial(formatting_pair_func, tokenizer=tokenizer),
            ),
            train_dataset=datasets[task]['train'],
            eval_dataset=datasets[task]['val'],
            generation_eval_fn=eval_fn,
            generation_eval_kwargs=eval_kwargs,
        )
        # A new optimizer/scheduler is created for each task; model weights carry forward.
        trainer.train()
        best_model_dir = os.path.join(task_output_dir, 'best_model')
        trainer.save_model(best_model_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(best_model_dir)
        trainer.accelerator.wait_for_everyone()
        model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        reload_peft_adapter_weights(model, best_model_dir)
        if task not in seen_tasks:
            seen_tasks.append(task)
        stage = {
            'stage': index,
            'task': task,
            'best_model_dir': best_model_dir,
            'best_validation_accuracy': trainer.state.best_metric,
            'test_metrics': {},
        }
        for eval_task in seen_tasks:
            results, metrics = TASK_CONFIGS[eval_task][2](
                model=model,
                tokenizer=tokenizer,
                test_dataset=datasets[eval_task]['test'],
                device=trainer.accelerator.device,
                **eval_kwargs,
            )
            stage['test_metrics'][eval_task] = metrics
            if trainer.is_world_process_zero():
                save_json_results(
                    os.path.join(task_output_dir, f'test_{eval_task}.json'), results, metrics
                )
        summary['stages'].append(stage)
        if trainer.is_world_process_zero():
            with open(os.path.join(output_dir, 'sequential_results.json'), 'w', encoding='utf-8') as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2)
            print(f'[{index}/{len(tasks)} {task}] Best adapter: {best_model_dir}')
        trainer.accelerator.wait_for_everyone()
        # Drop the previous optimizer before allocating the next stage's optimizer.
        model.zero_grad(set_to_none=True)
        del trainer
        gc.collect()
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Sequential naive training")
    parser.add_argument(
        "--tasks", type=str, required=True,
        help="Comma-separated task order, e.g. AGNEWS,AMAZON,YELP,DBPEDIA,YAHOO,SST2,QQP.",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(
            ROOT_DIR, "output_models", "llama-3.1-8b-instruct", "sequential_naive_seed_1"
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    # LoRA params
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Training params
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Eval params; retain aliases used by the shared training pipeline.
    parser.add_argument(
        "--eval_batch_size", "--per_device_eval_batch_size",
        dest="per_device_eval_batch_size", type=int, default=16,
    )
    parser.add_argument(
        "--max_input_length", "--max_seq_length",
        dest="max_seq_length", type=int, default=1536,
    )
    parser.add_argument("--max_new_tokens", type=int, default=8)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_sequential(args)

"""Train an answer-only LoRA adapter and select it by validation accuracy."""
import argparse
import os
import sys

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from evaluation.AGNEWS_eval import evaluate_AGNEWS_QA, save_qa_results
from model.utils import GENERAL_TEMPLATE, build_base_peft_model, build_tokenizer, render_chat, set_seed
from tasks.AGNEWS import example_to_messages as agnews_example_to_messages
from evaluation.AMAZON_eval import evaluate_AMAZON_QA
from tasks.AMAZON import example_to_messages as amazon_example_to_messages
from evaluation.YELP_eval import evaluate_YELP_QA
from tasks.YELP import example_to_messages as yelp_example_to_messages
from evaluation.DBPEDIA_eval import evaluate_DBPEDIA_QA
from tasks.DBPEDIA import example_to_messages as dbpedia_example_to_messages
from evaluation.YAHOO_eval import evaluate_YAHOO_QA
from tasks.YAHOO import example_to_messages as yahoo_example_to_messages
from evaluation.SST2_eval import evaluate_SST2_QA
from tasks.SST2 import example_to_messages as sst2_example_to_messages
from evaluation.QQP_eval import evaluate_QQP_QA
from tasks.QQP import example_to_messages as qqp_example_to_messages


def _example_to_batch(examples):
    keys = set().union(*(example.keys() for example in examples))
    return {key: [example.get(key) for example in examples] for key in keys}


def _answer_only_pairs_from_messages(messages_batch, tokenizer):
    prompt_without_answer, prompt_with_answer = [], []
    for messages in messages_batch:
        if not messages or messages[-1].get('role') != 'assistant':
            raise ValueError('Each conversation must end with an assistant answer')
        prompt_without_answer.append(
            render_chat(messages[:-1], tokenizer, add_generation_prompt=True)
        )
        prompt_with_answer.append(
            render_chat(messages, tokenizer, add_generation_prompt=False)
        )
    return {
        'prompt_without_answer': prompt_without_answer,
        'prompt_with_answer': prompt_with_answer,
    }


def _format_answer_only_pairs(examples, tokenizer, example_to_messages_fn):
    messages = example_to_messages_fn(examples)
    if messages and isinstance(messages[0], dict):
        messages = [messages]
    return _answer_only_pairs_from_messages(messages, tokenizer)


class AnswerOnlyDataCollator:
    def __init__(self, tokenizer, max_length, formatting_pair_func=None, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else tokenizer.model_max_length
        if self.max_length < 2:
            raise ValueError('max_length must be at least 2 for causal language modeling')
        self.formatting_pair_func = formatting_pair_func
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, examples):
        if not examples:
            raise ValueError('Cannot collate an empty batch')
        if 'prompt_without_answer' not in examples[0]:
            if self.formatting_pair_func is None:
                raise ValueError('AnswerOnlyDataCollator needs prompt fields or a formatting_pair_func')
            pair_batch = self.formatting_pair_func(_example_to_batch(examples))
            if any(len(pair_batch[key]) != len(examples) for key in (
                'prompt_without_answer', 'prompt_with_answer'
            )):
                raise ValueError('Formatter must return one prompt/answer pair per example')
            examples = [
                {key: pair_batch[key][i] for key in ('prompt_without_answer', 'prompt_with_answer')}
                for i in range(len(examples))
            ]

        input_ids, labels, attention_mask = [], [], []
        for example in examples:
            # The chat template already includes special tokens and the assistant header.
            prompt_ids = self.tokenizer(
                example['prompt_without_answer'], add_special_tokens=False
            )['input_ids']
            full_ids = self.tokenizer(
                example['prompt_with_answer'], add_special_tokens=False
            )['input_ids']
            if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError('The answer prompt must be a nonempty token prefix of the full conversation')
            answer_ids = full_ids[len(prompt_ids):]
            if not answer_ids:
                raise ValueError('Each example must contain answer tokens')
            # Keep at least one context token so the first answer token has a causal target.
            answer_ids = answer_ids[:self.max_length - 1]
            prompt_budget = self.max_length - len(answer_ids)
            prompt_ids = prompt_ids[-prompt_budget:]
            input_ids.append(prompt_ids + answer_ids)
            labels.append([self.label_pad_token_id] * len(prompt_ids) + answer_ids)

        max_len = max(map(len, input_ids))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError('tokenizer.pad_token_id must be set')
        for i in range(len(input_ids)):
            row_len = len(input_ids[i])
            pad_len = max_len - row_len
            if self.tokenizer.padding_side == 'left':
                input_ids[i] = [pad_id] * pad_len + input_ids[i]
                labels[i] = [self.label_pad_token_id] * pad_len + labels[i]
                attention_mask.append([0] * pad_len + [1] * row_len)
            else:
                input_ids[i] += [pad_id] * pad_len
                labels[i] += [self.label_pad_token_id] * pad_len
                attention_mask.append([1] * row_len + [0] * pad_len)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def _format_prompts(examples, tokenizer, example_to_messages_fn):
    messages = example_to_messages_fn(examples)
    if messages and isinstance(messages[0], dict):
        messages = [messages]
    return [render_chat(message, tokenizer, add_generation_prompt=False) for message in messages]


def agnews_formatting_prompts_func(examples, tokenizer):
    return _format_prompts(examples, tokenizer, agnews_example_to_messages)


def agnews_answer_only_prompts_func(examples, tokenizer):
    return _format_answer_only_pairs(examples, tokenizer, agnews_example_to_messages)


def amazon_formatting_prompts_func(examples, tokenizer):
    return _format_prompts(examples, tokenizer, amazon_example_to_messages)


def amazon_answer_only_prompts_func(examples, tokenizer):
    return _format_answer_only_pairs(examples, tokenizer, amazon_example_to_messages)


def yelp_formatting_prompts_func(examples, tokenizer):
    return _format_prompts(examples, tokenizer, yelp_example_to_messages)


def yelp_answer_only_prompts_func(examples, tokenizer):
    return _format_answer_only_pairs(examples, tokenizer, yelp_example_to_messages)


def dbpedia_formatting_prompts_func(examples, tokenizer):
    return _format_prompts(examples, tokenizer, dbpedia_example_to_messages)


def dbpedia_answer_only_prompts_func(examples, tokenizer):
    return _format_answer_only_pairs(examples, tokenizer, dbpedia_example_to_messages)


def yahoo_formatting_prompts_func(examples, tokenizer):
    return _format_prompts(examples, tokenizer, yahoo_example_to_messages)


def yahoo_answer_only_prompts_func(examples, tokenizer):
    return _format_answer_only_pairs(examples, tokenizer, yahoo_example_to_messages)


def sst2_formatting_prompts_func(examples, tokenizer):
    return _format_prompts(examples, tokenizer, sst2_example_to_messages)


def sst2_answer_only_prompts_func(examples, tokenizer):
    return _format_answer_only_pairs(examples, tokenizer, sst2_example_to_messages)


def qqp_formatting_prompts_func(examples, tokenizer):
    return _format_prompts(examples, tokenizer, qqp_example_to_messages)


def qqp_answer_only_prompts_func(examples, tokenizer):
    return _format_answer_only_pairs(examples, tokenizer, qqp_example_to_messages)


class GenerationEvalTrainer(Trainer):
    """Use the existing generation evaluator for checkpoint selection."""

    def __init__(self, *args, generation_eval_fn, generation_eval_kwargs, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_eval_fn = generation_eval_fn
        self.generation_eval_kwargs = generation_eval_kwargs

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        model = self.accelerator.unwrap_model(self.model_wrapped)
        with self.compute_loss_context_manager():
            _, values = self.generation_eval_fn(
                model=model,
                tokenizer=self.processing_class,
                test_dataset=dataset,
                device=self.accelerator.device,
                **self.generation_eval_kwargs,
            )
        metrics = {f'{metric_key_prefix}_{key}': value for key, value in values.items()}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


def run_general_task(
    args,
    task_name,
    chat_template,
    formatting_prompts_func,
    formatting_pair_func,
    eval_fn,
    metric_name,
    is_higher_better,
):
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError('warmup_ratio must be in [0, 1)')
    output_dir = args.output_dir or os.path.join(ROOT_DIR, 'results', task_name, 'single_task')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    set_seed(args.seed)
    data_dir = os.path.join(ROOT_DIR, 'data', task_name)
    data_files = {split: os.path.join(data_dir, f'{split}.jsonl') for split in ('train', 'val', 'test')}
    for path in data_files.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Dataset file does not exist: {path}')
    ds = load_dataset('json', data_files=data_files)
    if any(len(ds[split]) == 0 for split in data_files):
        raise ValueError('Train, validation and test datasets must all be nonempty')

    tokenizer = build_tokenizer(args, chat_template=chat_template)
    tokenizer.model_max_length = args.max_seq_length
    base_model = build_base_peft_model(args)
    training_args = TrainingArguments(
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
        # Transformers 5 accepts fractional warmup through warmup_steps.
        **({'warmup_ratio': args.warmup_ratio}
           if 'warmup_ratio' in TrainingArguments.__dataclass_fields__
           else {'warmup_steps': args.warmup_ratio}),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=is_higher_better,
        optim='adamw_torch',
        weight_decay=args.weight_decay,
        ddp_find_unused_parameters=False,
        report_to='none',
    )
    data_collator = AnswerOnlyDataCollator(
        tokenizer,
        max_length=args.max_seq_length,
        formatting_pair_func=lambda examples: formatting_pair_func(examples, tokenizer),
    )
    # Raw dataset rows go directly to the collator; no second formatting/tokenization pass.
    trainer = GenerationEvalTrainer(
        model=base_model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=ds['train'],
        eval_dataset=ds['val'],
        generation_eval_fn=eval_fn,
        generation_eval_kwargs={
            'batch_size': args.per_device_eval_batch_size,
            'max_input_length': args.max_seq_length,
            'max_new_tokens': args.max_new_tokens,
            'print_sample': False,
        },
    )
    trainer.train()
    best_model_dir = os.path.join(output_dir, 'best_model')
    trainer.save_model(best_model_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(best_model_dir)
    trainer.accelerator.wait_for_everyone()
    # train() has restored the checkpoint chosen using validation data only.
    results, metrics = eval_fn(
        model=trainer.accelerator.unwrap_model(trainer.model_wrapped),
        tokenizer=tokenizer,
        test_dataset=ds['test'],
        device=trainer.accelerator.device,
        batch_size=args.per_device_eval_batch_size,
        max_input_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        print_sample=False,
    )
    if trainer.is_world_process_zero():
        save_qa_results(os.path.join(output_dir, 'test_results.json'), results, metrics)
        print(f'Best adapter saved to {best_model_dir}; test metrics: {metrics}')
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task_name', type=str.upper, choices=TASK_RUNNERS, default='AGNEWS')
    parser.add_argument('--output_dir', default=None, help='Default: results/<TASK>/single_task')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=float, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--lr_scheduler_type', default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--max_seq_length', type=int, default=1536)
    parser.add_argument('--max_new_tokens', type=int, default=8)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    return parser


def run_agnews(args):
    """Train AGNEWS using the shared answer-only LoRA pipeline."""
    return run_general_task(
        args=args,
        task_name='AGNEWS',
        chat_template=GENERAL_TEMPLATE,
        formatting_prompts_func=agnews_formatting_prompts_func,
        formatting_pair_func=agnews_answer_only_prompts_func,
        eval_fn=evaluate_AGNEWS_QA,
        metric_name='accuracy',
        is_higher_better=True,
    )


def run_amazon(args):
    """Train AMAZON using the shared answer-only LoRA pipeline."""
    return run_general_task(
        args=args,
        task_name='AMAZON',
        chat_template=GENERAL_TEMPLATE,
        formatting_prompts_func=amazon_formatting_prompts_func,
        formatting_pair_func=amazon_answer_only_prompts_func,
        eval_fn=evaluate_AMAZON_QA,
        metric_name='accuracy',
        is_higher_better=True,
    )


def run_yelp(args):
    """Train YELP using the shared answer-only LoRA pipeline."""
    return run_general_task(
        args=args,
        task_name='YELP',
        chat_template=GENERAL_TEMPLATE,
        formatting_prompts_func=yelp_formatting_prompts_func,
        formatting_pair_func=yelp_answer_only_prompts_func,
        eval_fn=evaluate_YELP_QA,
        metric_name='accuracy',
        is_higher_better=True,
    )


def run_dbpedia(args):
    """Train DBPEDIA using the shared answer-only LoRA pipeline."""
    return run_general_task(
        args=args,
        task_name='DBPEDIA',
        chat_template=GENERAL_TEMPLATE,
        formatting_prompts_func=dbpedia_formatting_prompts_func,
        formatting_pair_func=dbpedia_answer_only_prompts_func,
        eval_fn=evaluate_DBPEDIA_QA,
        metric_name='accuracy',
        is_higher_better=True,
    )


def run_yahoo(args):
    """Train YAHOO using the shared answer-only LoRA pipeline."""
    return run_general_task(
        args=args,
        task_name='YAHOO',
        chat_template=GENERAL_TEMPLATE,
        formatting_prompts_func=yahoo_formatting_prompts_func,
        formatting_pair_func=yahoo_answer_only_prompts_func,
        eval_fn=evaluate_YAHOO_QA,
        metric_name='accuracy',
        is_higher_better=True,
    )


def run_sst2(args):
    """Train SST2 using the shared answer-only LoRA pipeline."""
    return run_general_task(
        args=args,
        task_name='SST2',
        chat_template=GENERAL_TEMPLATE,
        formatting_prompts_func=sst2_formatting_prompts_func,
        formatting_pair_func=sst2_answer_only_prompts_func,
        eval_fn=evaluate_SST2_QA,
        metric_name='accuracy',
        is_higher_better=True,
    )


def run_qqp(args):
    """Train QQP using the shared answer-only LoRA pipeline."""
    return run_general_task(
        args=args,
        task_name='QQP',
        chat_template=GENERAL_TEMPLATE,
        formatting_prompts_func=qqp_formatting_prompts_func,
        formatting_pair_func=qqp_answer_only_prompts_func,
        eval_fn=evaluate_QQP_QA,
        metric_name='accuracy',
        is_higher_better=True,
    )


TASK_RUNNERS = {
    'AGNEWS': run_agnews,
    'AMAZON': run_amazon,
    'YELP': run_yelp,
    'DBPEDIA': run_dbpedia,
    'YAHOO': run_yahoo,
    'SST2': run_sst2,
    'QQP': run_qqp,
}


if __name__ == '__main__':
    args = build_parser().parse_args()
    TASK_RUNNERS[args.task_name](args)

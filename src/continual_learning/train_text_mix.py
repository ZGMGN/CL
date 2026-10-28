"""Sequential LoRA on the original QA training sets plus 192 step-grouped text examples per task.

QA targets are answer-only; raw text uses causal next-token targets. Validation,
checkpoint selection and test performance use only the original QA datasets.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, Sampler

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / 'src'))

from continual_learning import train_sequential_tasks as sequential
from continual_learning.train_single_task import AnswerOnlyDataCollator, GenerationEvalTrainer
from evaluation.general_task_eval import save_json_results
from model.utils import GENERAL_TEMPLATE, MODEL_DIR, build_base_peft_model, build_tokenizer, set_seed

TASK_CONFIGS = sequential.TASK_CONFIGS


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_identity(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def encode_text(text, tokenizer, max_length):
    """One document per sample; keep its real EOS target even when pad == EOS."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Text examples must contain a nonempty text string')
    if tokenizer.eos_token_id is None:
        raise ValueError('Text training requires tokenizer.eos_token_id')
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if not ids:
        raise ValueError('Text example has no tokens')
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    ids = ids + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        raise ValueError(
            f'Text needs {len(ids)} tokens including BOS/EOS, exceeding --max_seq_length={max_length}; '
            'increase the length or shorten the source explicitly. Text is never silently truncated.'
        )
    if len(ids) < 2:
        raise ValueError('Text must have at least two tokens for next-token prediction')
    return ids


class MixedTaskDataset(Dataset):
    """Keep every QA row and add each encoded text document exactly once."""

    def __init__(self, qa_dataset, text_input_ids):
        self.qa_dataset = qa_dataset
        self.text_input_ids = text_input_ids

    def __len__(self):
        return len(self.qa_dataset) + len(self.text_input_ids)

    def __getitem__(self, index):
        if index < len(self.qa_dataset):
            return {'sample_type': 'qa', 'example': self.qa_dataset[index]}
        return {'sample_type': 'text', 'input_ids': self.text_input_ids[index - len(self.qa_dataset)]}


class HomogeneousStepSampler(Sampler):
    """Shuffle whole optimizer groups; never mix QA/text within an update.

    The order is global. Accelerate shards microbatches across ranks, so a group
    contains local_batch_size * accumulation_steps * world_size examples.
    """

    def __init__(self, qa_count, text_count, group_size, seed=1):
        if group_size < 1 or qa_count < 1 or text_count < 1:
            raise ValueError('Group size and both source counts must be positive')
        for name, count in (('QA', qa_count), ('text', text_count)):
            if count % group_size:
                raise ValueError(
                    f'{name} count {count} must be divisible by optimizer group size {group_size}; '
                    'samples are never dropped, repeated or mixed across group boundaries.'
                )
        self.qa_count = qa_count
        self.text_count = text_count
        self.group_size = group_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.qa_count + self.text_count

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        qa = torch.randperm(self.qa_count, generator=generator).tolist()
        text = (torch.randperm(self.text_count, generator=generator) + self.qa_count).tolist()
        groups = [source[start:start + self.group_size]
                  for source in (qa, text) for start in range(0, len(source), self.group_size)]
        order = torch.randperm(len(groups), generator=generator).tolist()
        return iter(index for group_index in order for index in groups[group_index])


def resolve_data_seed(training_args):
    """Match Trainer's fallback while preserving an explicitly configured seed of 0."""
    return training_args.seed if training_args.data_seed is None else training_args.data_seed


class StepGroupedTrainer(GenerationEvalTrainer):
    """Keep the QA-only evaluator and replace only the training sampler."""

    def __init__(self, *args, mixing_strategy='step', **kwargs):
        if mixing_strategy not in ('step', 'sample'):
            raise ValueError('mixing_strategy must be step or sample')
        self.mixing_strategy = mixing_strategy
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if self.mixing_strategy == 'sample':
            return super()._get_train_sampler(train_dataset)
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if not isinstance(dataset, MixedTaskDataset):
            raise ValueError('Step-grouped training requires MixedTaskDataset')
        group_size = self._train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        return HomogeneousStepSampler(len(dataset.qa_dataset), len(dataset.text_input_ids),
                                      group_size, resolve_data_seed(self.args))


class MixedDataCollator:
    """Explicit sample types prevent AGNEWS QA text from being treated as raw text."""

    def __init__(self, tokenizer, max_length, formatting_pair_func):
        if max_length < 2 or tokenizer.pad_token_id is None:
            raise ValueError('Need max_length >= 2 and a configured padding token')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qa_collator = AnswerOnlyDataCollator(tokenizer, max_length, formatting_pair_func)

    def __call__(self, examples):
        if not examples:
            raise ValueError('Cannot collate an empty batch')
        if any(row.get('sample_type') not in ('qa', 'text') for row in examples):
            raise ValueError('Each training example needs sample_type=qa or text')
        sequences = [None] * len(examples)
        qa_positions = [i for i, row in enumerate(examples) if row['sample_type'] == 'qa']
        if qa_positions:
            qa_batch = self.qa_collator([examples[i]['example'] for i in qa_positions])
            for row_index, position in enumerate(qa_positions):
                valid = qa_batch['attention_mask'][row_index].bool()
                sequences[position] = (
                    qa_batch['input_ids'][row_index][valid].tolist(),
                    qa_batch['labels'][row_index][valid].tolist(),
                )
        for i, row in enumerate(examples):
            if row['sample_type'] == 'text':
                ids = list(row['input_ids'])
                if not 2 <= len(ids) <= self.max_length:
                    raise ValueError('Text sequence length is outside the configured limits')
                labels = ids.copy()
                # There is no preceding real token for the first token. In left-padded
                # batches this also prevents teaching a padding-to-BOS transition.
                labels[0] = -100
                sequences[i] = (ids, labels)
        length = max(len(ids) for ids, _ in sequences)
        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for ids, labels in sequences:
            missing = length - len(ids)
            if self.tokenizer.padding_side == 'left':
                batch['input_ids'].append([self.tokenizer.pad_token_id] * missing + ids)
                batch['attention_mask'].append([0] * missing + [1] * len(ids))
                batch['labels'].append([-100] * missing + labels)
            else:
                batch['input_ids'].append(ids + [self.tokenizer.pad_token_id] * missing)
                batch['attention_mask'].append([1] * len(ids) + [0] * missing)
                batch['labels'].append(labels + [-100] * missing)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def prepare_task_data(task, args, tokenizer):
    data_dir = Path(args.data_dir).expanduser().resolve()
    qa_paths = {split: data_dir / task / f'{split}.jsonl' for split in ('train', 'val', 'test')}
    text_path = data_dir / f'{task}_text' / f'{task}_domain_{args.text_samples_per_task}.jsonl'
    for path in [*qa_paths.values(), text_path]:
        if not path.is_file():
            raise FileNotFoundError(f'{task}: data file does not exist: {path}')
    qa = load_dataset('json', data_files={split: str(path) for split, path in qa_paths.items()})
    if any(not len(qa[split]) for split in qa_paths):
        raise ValueError(f'{task}: QA train, val and test must all be nonempty')
    with text_path.open(encoding='utf-8') as stream:
        texts = [json.loads(line) for line in stream if line.strip()]
    if len(texts) != args.text_samples_per_task:
        raise ValueError(f'{task}: expected exactly {args.text_samples_per_task} text rows, got {len(texts)}')
    text_ids = [encode_text(row.get('text'), tokenizer, args.max_seq_length) for row in texts]
    qa_identities = {split: file_identity(path) for split, path in qa_paths.items()}
    text_identity = file_identity(text_path)
    provenance_path = text_path.with_name(text_path.stem + '_manifest.json')
    provenance_identity = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        if provenance.get('output_sha256') != text_identity['sha256']:
            raise ValueError(f'{task}: converted text differs from its source manifest')
        if provenance.get('source_sha256') != qa_identities['train']['sha256']:
            raise ValueError(f'{task}: text provenance does not match the current QA training file')
        provenance_identity = file_identity(provenance_path)
    mixed = MixedTaskDataset(qa['train'], text_ids)
    metadata = {
        'qa_files': qa_identities, 'text_file': text_identity,
        'text_provenance': provenance_identity,
        'qa_selection': 'Entire original train split, without subsampling, replacement or duplication.',
        'qa_train_count': len(qa['train']), 'text_train_count': len(texts),
        'mixed_train_count': len(mixed), 'qa_validation_count': len(qa['val']),
        'qa_test_count': len(qa['test']),
        'max_text_tokens_including_bos_eos': max(map(len, text_ids)),
        'text_token_count_including_bos_eos': sum(map(len, text_ids)),
        'text_truncation': 'error; preserve one full document per sample',
    }
    return {'qa': qa, 'mixed_train': mixed, 'metadata': metadata}


def output_path(args, tasks):
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    prefix = 'text_mix_step' if args.mixing_strategy == 'step' else 'text_mix'
    name = (f'{prefix}_{"_".join(tasks)}_seed_{args.seed}_text{args.text_samples_per_task}'
            f'_bs{args.per_device_train_batch_size}_ga{args.gradient_accumulation_steps}')
    return ROOT_DIR / 'results' / name


def run_text_mix(args):
    tasks = sequential.parse_tasks(args.tasks)
    if (not 0 <= args.warmup_ratio < 1 or args.max_seq_length < 2 or args.text_samples_per_task <= 0
            or args.num_train_epochs <= 0 or args.per_device_train_batch_size <= 0
            or args.per_device_eval_batch_size <= 0 or args.gradient_accumulation_steps <= 0
            or args.max_new_tokens <= 0 or args.num_workers < 0):
        raise ValueError('Invalid training limits, text count, length or warmup ratio')
    destination = output_path(args, tasks)
    if (destination / 'sequential_results.json').exists() or any(destination.glob('[0-9][0-9]_*')):
        raise FileExistsError(f'Run already contains training stages: {destination}; choose a new --output_dir')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    set_seed(args.seed)
    tokenizer = build_tokenizer(args, chat_template=GENERAL_TEMPLATE)
    tokenizer.model_max_length = args.max_seq_length
    # Check every task and every full text before allocating the base model.
    prepared = {task: prepare_task_data(task, args, tokenizer) for task in dict.fromkeys(tasks)}
    preflight_args = sequential.build_training_args(args, str(destination))
    group_size = preflight_args.train_batch_size * args.gradient_accumulation_steps * preflight_args.world_size
    if args.mixing_strategy == 'step':
        for entry in prepared.values():
            info = entry['metadata']
            sampler = HomogeneousStepSampler(info['qa_train_count'], info['text_train_count'],
                                             group_size, resolve_data_seed(preflight_args))
            # Exercise iteration too: constructor-only checks miss invalid RNG seeds.
            next(iter(sampler))
            info.update(qa_update_groups=info['qa_train_count'] // group_size,
                        text_update_groups=info['text_train_count'] // group_size)
    mixing = ('Shuffle QA and text separately, form homogeneous optimizer groups, then shuffle the whole groups each epoch.'
              if args.mixing_strategy == 'step' else 'Shuffle individual QA/text examples each epoch.')
    plan = {
        'tasks': tasks, 'training_mode': 'sequential_qa_plus_text', 'base_model': str(Path(MODEL_DIR).resolve()),
        'output_dir': str(destination), 'settings': vars(args).copy(),
        'mixing_strategy': args.mixing_strategy, 'optimizer_group_size': group_size,
        'world_size': preflight_args.world_size,
        'loss': {'qa': 'Answer-only targets, including the assistant end marker; prompt tokens are masked.',
                 'text': 'Next-token targets on the full raw document and EOS; no chat template; no padding targets.',
                 'reduction': 'Standard causal-LM cross-entropy over supervised tokens; no separate task loss weighting.',
                 'mixing': mixing},
        'evaluation': {'data': 'QA only', 'checkpoint_metric': 'validation QA accuracy',
                       'test_tasks_after_each_stage': 'All tasks seen so far; original full QA test splits.'},
        'data': {task: value['metadata'] for task, value in prepared.items()},
    }
    is_main = int(os.environ.get('RANK', '0')) == 0
    if is_main:
        save_json(destination / 'mixing_manifest.json', plan)
        for task, entry in prepared.items():
            info = entry['metadata']
            print(f'{task}: {info["qa_train_count"]} QA + {info["text_train_count"]} text = '
                  f'{info["mixed_train_count"]} training rows; mixing={args.mixing_strategy}; validation/test: QA only', flush=True)
            if args.mixing_strategy == 'step':
                print(f'  {info["qa_update_groups"]} QA updates + {info["text_update_groups"]} text updates; '
                      f'{group_size} examples per global optimizer group', flush=True)
    if args.dry_run:
        if is_main:
            print(f'Data check complete; no model weights loaded. Manifest: {destination / "mixing_manifest.json"}', flush=True)
        return plan

    model = build_base_peft_model(args)
    eval_kwargs = {'batch_size': args.per_device_eval_batch_size, 'max_input_length': args.max_seq_length,
                   'max_new_tokens': args.max_new_tokens, 'print_sample': False}
    summary = {'tasks': tasks, 'training_mode': 'sequential_qa_plus_text',
               'mixing_manifest': str(destination / 'mixing_manifest.json'),
               'mixing_strategy': args.mixing_strategy, 'optimizer_group_size': group_size, 'stages': []}
    seen_tasks = []
    for index, task in enumerate(tasks, 1):
        _, pair_formatter, evaluator = TASK_CONFIGS[task]
        entry = prepared[task]
        stage_dir = destination / f'{index:02d}_{task}'
        model.config.use_cache = False
        trainer = StepGroupedTrainer(
            mixing_strategy=args.mixing_strategy,
            model=model, args=sequential.build_training_args(args, str(stage_dir)), processing_class=tokenizer,
            data_collator=MixedDataCollator(tokenizer, args.max_seq_length, partial(pair_formatter, tokenizer=tokenizer)),
            train_dataset=entry['mixed_train'], eval_dataset=entry['qa']['val'],
            generation_eval_fn=evaluator, generation_eval_kwargs=eval_kwargs,
        )
        # The Trainer reloads the checkpoint with highest QA validation accuracy.
        # A fresh optimizer is used per task; the cumulative best adapter carries on.
        trainer.train()
        best_dir = stage_dir / 'best_model'
        trainer.save_model(str(best_dir))
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(best_dir)
        trainer.accelerator.wait_for_everyone()
        model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        sequential.reload_peft_adapter_weights(model, str(best_dir))
        if task not in seen_tasks:
            seen_tasks.append(task)
        stage = {'stage': index, 'task': task, 'best_model_dir': str(best_dir),
                 'best_validation_accuracy': trainer.state.best_metric,
                 'qa_train_count': entry['metadata']['qa_train_count'],
                 'text_train_count': entry['metadata']['text_train_count'],
                 'mixed_train_count': len(entry['mixed_train']), 'test_metrics': {}}
        for eval_task in seen_tasks:
            results, metrics = TASK_CONFIGS[eval_task][2](
                model=model, tokenizer=tokenizer, test_dataset=prepared[eval_task]['qa']['test'],
                device=trainer.accelerator.device, **eval_kwargs,
            )
            stage['test_metrics'][eval_task] = metrics
            if trainer.is_world_process_zero():
                save_json_results(str(stage_dir / f'test_{eval_task}.json'), results, metrics)
        summary['stages'].append(stage)
        if trainer.is_world_process_zero():
            save_json(destination / 'sequential_results.json', summary)
            print(f'[{index}/{len(tasks)} {task}] QA validation-best adapter: {best_dir}', flush=True)
        trainer.accelerator.wait_for_everyone()
        model.zero_grad(set_to_none=True)
        del trainer
        gc.collect()
    return summary


def build_parser():
    parser = sequential.build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir=None)
    parser.add_argument('--data_dir', default=str(ROOT_DIR / 'data'),
                        help='QA: TASK/{train,val,test}.jsonl; text: TASK_text/TASK_domain_<N>.jsonl')
    parser.add_argument('--text_samples_per_task', type=int, default=192,
                        help='Number of added text rows; reads TASK_text/TASK_domain_<N>.jsonl (default: 192)')
    parser.add_argument('--mixing_strategy', choices=('step', 'sample'), default='step',
                        help='step: shuffle homogeneous optimizer groups; sample: reproduce individual-example mixing')
    parser.add_argument('--dry_run', action='store_true',
                        help='Check full datasets, text lengths and provenance; write the plan without loading model weights')
    return parser


if __name__ == '__main__':
    run_text_mix(build_parser().parse_args())

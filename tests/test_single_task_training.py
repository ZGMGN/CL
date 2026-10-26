"""Regression checks for answer-only labels and validation checkpoint selection."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from continual_learning import train_single_task as training


class AnswerOnlyTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(ROOT / 'Llama-3.1-8B-Instruct'), local_files_only=True
        )
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.rows = [
            {'text': 'A short sports report.', 'label': 1},
            {'text': 'A much longer news story about international politics and world leaders.', 'label': 0},
        ]

    def collator(self, max_length=1536):
        return training.AnswerOnlyDataCollator(
            self.tokenizer, max_length,
            lambda examples: training.agnews_answer_only_prompts_func(examples, self.tokenizer),
        )

    def test_pairs_keep_every_example_and_single_input(self):
        pairs = training.agnews_answer_only_prompts_func(training._example_to_batch(self.rows), self.tokenizer)
        self.assertEqual([len(v) for v in pairs.values()], [2, 2])
        for i, row in enumerate(self.rows):
            single = training.agnews_answer_only_prompts_func(row, self.tokenizer)
            for key in pairs:
                self.assertEqual(single[key][0], pairs[key][i])

    def test_labels_and_attention_for_both_padding_sides(self):
        for side in ('left', 'right'):
            self.tokenizer.padding_side = side
            batch = self.collator()(self.rows)
            self.assertEqual(batch['input_ids'].shape, batch['labels'].shape)
            self.assertEqual(batch['labels'].shape, batch['attention_mask'].shape)
            self.assertTrue((batch['attention_mask'] == 0).any())
            for i, letter in enumerate(('B', 'A')):
                labels = batch['labels'][i]
                target = labels[labels != -100]
                self.assertEqual(self.tokenizer.decode(target), letter + '<|eot_id|>')
                self.assertTrue(torch.equal(batch['input_ids'][i][labels != -100], target))
                self.assertTrue((labels[batch['attention_mask'][i] == 0] == -100).all())
                self.assertTrue((labels == -100).any())

    def test_truncation_keeps_answer_and_causal_context(self):
        self.tokenizer.padding_side = 'right'
        for limit in (2, 8):
            batch = self.collator(limit)(self.rows)
            self.assertEqual(batch['input_ids'].shape[1], limit)
            self.assertTrue((batch['labels'][:, 0] == -100).all())
            self.assertTrue((batch['labels'][:, 1:] != -100).any(dim=1).all())
            if limit == 8:
                for i, letter in enumerate(('B', 'A')):
                    target = batch['labels'][i]
                    self.assertEqual(self.tokenizer.decode(target[target != -100]), letter + '<|eot_id|>')

    def test_invalid_batches_fail_clearly(self):
        with self.assertRaises(ValueError):
            self.collator()([])
        with self.assertRaises(ValueError):
            self.collator(1)
        with self.assertRaises(ValueError):
            self.collator()([{'prompt_without_answer': 'hello', 'prompt_with_answer': 'different'}])
        with self.assertRaises(ValueError):
            self.collator()([{'prompt_without_answer': 'hello', 'prompt_with_answer': 'hello'}])

    def test_all_task_runners_and_real_data_labels(self):
        for task, runner in training.TASK_RUNNERS.items():
            with self.subTest(task=task):
                args = training.build_parser().parse_args(['--task_name', task.lower()])
                self.assertEqual(args.task_name, task)
                self.assertIsNone(args.output_dir)
                module = importlib.import_module(f'tasks.{task}')
                with (ROOT / 'data' / task / 'train.jsonl').open() as stream:
                    rows = [json.loads(next(stream)) for _ in range(2)]
                with patch.object(training, 'run_general_task', return_value={'accuracy': 0.5}) as run:
                    self.assertEqual(runner(args), {'accuracy': 0.5})
                kwargs = run.call_args.kwargs
                self.assertEqual(kwargs['task_name'], task)
                self.assertEqual(kwargs['eval_fn'].__name__, f'evaluate_{task}_QA')
                self.assertIs(kwargs['args'], args)
                self.tokenizer.padding_side = 'right'
                collator = training.AnswerOnlyDataCollator(
                    self.tokenizer, 1536,
                    lambda examples: kwargs['formatting_pair_func'](examples, self.tokenizer),
                )
                batch = collator(rows)
                formatted = kwargs['formatting_prompts_func'](training._example_to_batch(rows), self.tokenizer)
                self.assertEqual(len(formatted), 2)
                for i, row in enumerate(rows):
                    labels = batch['labels'][i]
                    self.assertEqual(
                        self.tokenizer.decode(labels[labels != -100]),
                        module.extract_label_from_example(row) + '<|eot_id|>',
                    )

    def test_tiny_lora_training_selects_validation_best_and_tests_afterward(self):
        torch.set_num_threads(1)
        tokenizer = self.tokenizer
        tokenizer.padding_side = 'right'
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=256, pad_token_id=tokenizer.pad_token_id,
        ))
        model = get_peft_model(model, LoraConfig(
            r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'
        ))
        model.config.use_cache = False
        dataset = Dataset.from_list(self.rows)
        splits = {'train': dataset, 'val': dataset, 'test': dataset.select([0])}
        calls, snapshots = [], []

        def evaluate(**kwargs):
            calls.append(len(kwargs['test_dataset']))
            snapshots.append({k: v.detach().cpu().clone() for k, v in kwargs['model'].named_parameters() if v.requires_grad})
            return [], {'accuracy': [0.8, 0.4, 0.7][len(calls) - 1], 'correct': 1, 'total': calls[-1]}

        with tempfile.TemporaryDirectory() as tmp:
            args = training.build_parser().parse_args([
                '--output_dir', tmp, '--num_train_epochs', '2',
                '--per_device_train_batch_size', '2', '--gradient_accumulation_steps', '1',
                '--max_seq_length', '128', '--logging_steps', '1',
            ])
            with patch.object(training, 'build_tokenizer', return_value=tokenizer), \
                 patch.object(training, 'build_base_peft_model', return_value=model), \
                 patch.object(training, 'load_dataset', return_value=splits):
                metrics = training.run_general_task(
                    args, 'AGNEWS', training.GENERAL_TEMPLATE,
                    training.agnews_formatting_prompts_func,
                    training.agnews_answer_only_prompts_func, evaluate, 'accuracy', True,
                )
            self.assertEqual(calls, [2, 2, 1])
            self.assertEqual(metrics['accuracy'], 0.7)
            self.assertTrue((Path(tmp) / 'best_model' / 'adapter_config.json').is_file())
            self.assertTrue((Path(tmp) / 'test_results.json').is_file())
            self.assertTrue(all(torch.equal(snapshots[0][k], snapshots[2][k]) for k in snapshots[0]))
            self.assertTrue(any(not torch.equal(snapshots[0][k], snapshots[1][k]) for k in snapshots[0]))


if __name__ == '__main__':
    unittest.main()

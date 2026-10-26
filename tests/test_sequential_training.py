"""Regression checks for validation-best adapter handoff between tasks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from continual_learning import train_sequential_tasks as sequential


def snapshot(model):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}


def same(first, second):
    return first.keys() == second.keys() and all(torch.equal(first[key], second[key]) for key in first)


class SequentialTrainingTest(unittest.TestCase):
    def test_task_order_and_invalid_tasks(self):
        self.assertEqual(sequential.parse_tasks(' agnews, SST2,qqp '), ['AGNEWS', 'SST2', 'QQP'])
        self.assertEqual(sequential.parse_tasks('AGNEWS,AGNEWS'), ['AGNEWS', 'AGNEWS'])
        for task, (formatter, pair_formatter, evaluator) in sequential.TASK_CONFIGS.items():
            self.assertEqual(formatter.__name__, f'{task.lower()}_formatting_prompts_func')
            self.assertEqual(pair_formatter.__name__, f'{task.lower()}_answer_only_prompts_func')
            self.assertEqual(evaluator.__name__, f'evaluate_{task}_QA')
        for tasks in (' , ', 'AGNEWS,UNKNOWN'):
            with self.subTest(tasks=tasks), patch.object(sequential, 'build_base_peft_model') as build:
                with self.assertRaises(ValueError):
                    sequential.run_sequential(sequential.build_parser().parse_args(['--tasks', tasks]))
                build.assert_not_called()

    def test_missing_later_data_fails_before_loading_model(self):
        args = sequential.build_parser().parse_args(['--tasks', 'AGNEWS,SST2'])
        with patch.object(sequential, 'load_task_data', side_effect=[{}, FileNotFoundError('missing SST2')]), patch.object(sequential, 'build_base_peft_model') as build:
            with self.assertRaisesRegex(FileNotFoundError, 'SST2'):
                sequential.run_sequential(args)
            build.assert_not_called()

    def test_two_stage_training_carries_best_adapter_forward(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'Llama-3.1-8B-Instruct'), local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = get_peft_model(LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=256, pad_token_id=tokenizer.pad_token_id,
        )), LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        datasets = {}
        for task in ('AGNEWS', 'SST2'):
            with (ROOT / 'data' / task / 'train.jsonl').open() as stream:
                rows = [json.loads(next(stream)) for _ in range(2)]
            datasets[task] = {split: Dataset.from_list(rows) for split in ('train', 'val', 'test')}
        initial = snapshot(model)
        starts, validation, tests = [], {'AGNEWS': [], 'SST2': []}, []

        class RecordingTrainer(sequential.GenerationEvalTrainer):
            def __init__(self, *args, **kwargs):
                starts.append(snapshot(kwargs['model']))
                super().__init__(*args, **kwargs)

        def evaluator(task):
            def evaluate(**kwargs):
                dataset = kwargs['test_dataset']
                weights = snapshot(kwargs['model'])
                if dataset is datasets[task]['val']:
                    validation[task].append(weights)
                    accuracy = 0.9 if len(validation[task]) == 1 else 0.3
                else:
                    self.assertIs(dataset, datasets[task]['test'])
                    tests.append((task, weights))
                    accuracy = 0.5
                return [], {'accuracy': accuracy, 'correct': 1, 'total': len(dataset)}
            return evaluate

        configs = {task: (*sequential.TASK_CONFIGS[task][:2], evaluator(task)) for task in datasets}
        with tempfile.TemporaryDirectory() as tmp:
            args = sequential.build_parser().parse_args([
                '--tasks', 'agnews,sst2', '--output_dir', tmp,
                '--num_train_epochs', '2', '--per_device_train_batch_size', '2',
                '--gradient_accumulation_steps', '1', '--max_seq_length', '128',
                '--warmup_ratio', '0', '--logging_steps', '1', '--num_workers', '0',
            ])
            with patch.object(sequential, 'build_tokenizer', return_value=tokenizer), \
                 patch.object(sequential, 'build_base_peft_model', return_value=model) as build, \
                 patch.object(sequential, 'load_task_data', side_effect=lambda task: datasets[task]), \
                 patch.object(sequential, 'TASK_CONFIGS', configs), \
                 patch.object(sequential, 'GenerationEvalTrainer', RecordingTrainer), \
                 patch.object(sequential, 'reload_peft_adapter_weights', wraps=sequential.reload_peft_adapter_weights) as reload:
                summary = sequential.run_sequential(args)
            build.assert_called_once()
            self.assertEqual(reload.call_count, 2)
            self.assertEqual([task for task, _ in tests], ['AGNEWS', 'AGNEWS', 'SST2'])
            self.assertTrue(same(initial, starts[0]))
            self.assertFalse(same(initial, validation['AGNEWS'][0]))
            self.assertFalse(same(validation['AGNEWS'][0], validation['AGNEWS'][1]))
            self.assertTrue(same(starts[1], validation['AGNEWS'][0]))
            self.assertTrue(same(tests[0][1], validation['AGNEWS'][0]))
            self.assertTrue(same(tests[1][1], validation['SST2'][0]))
            self.assertTrue(same(tests[2][1], validation['SST2'][0]))
            self.assertEqual([stage['best_validation_accuracy'] for stage in summary['stages']], [0.9, 0.9])
            self.assertEqual(set(summary['stages'][1]['test_metrics']), {'AGNEWS', 'SST2'})
            for stage_name in ('01_AGNEWS', '02_SST2'):
                self.assertTrue((Path(tmp) / stage_name / 'best_model' / 'adapter_config.json').is_file())
            with (Path(tmp) / 'sequential_results.json').open() as stream:
                self.assertEqual(json.load(stream), summary)
            self.assertTrue((Path(tmp) / '02_SST2' / 'test_AGNEWS.json').is_file())


if __name__ == '__main__':
    unittest.main()

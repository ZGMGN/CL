"""Mixed targets, unchanged QA splits, QA-only performance and best-stage handoff."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from continual_learning import train_text_mix as mixed


def snapshot(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def same(first, second):
    return first.keys() == second.keys() and all(torch.equal(first[key], second[key]) for key in first)


def write_task(root, task, count=2):
    if task == 'AGNEWS':
        rows = [{'text': f'News story number {i} about sports.', 'label': i % 4} for i in range(count)]
    else:
        rows = [{'title': f'Review {i}', 'content': f'This product is useful. Example {i}.', 'label': i % 5} for i in range(count)]
    folder = root / task
    folder.mkdir()
    for split, entries in (('train', rows), ('val', rows[:1]), ('test', rows[-1:])):
        (folder / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in entries))
    text_dir = root / f'{task}_text'
    text_dir.mkdir()
    texts = [{'text': f'A natural domain text for {task}.'}, {'text': 'Another short domain text.'}]
    (text_dir / f'{task}_domain_2.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in texts))
    return rows


class TextMixTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'Llama-3.1-8B-Instruct'), local_files_only=True)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def test_mixed_labels_distinguish_qa_text_and_retain_real_eos(self):
        tokenizer = copy.deepcopy(self.tokenizer)
        qa = [{'text': 'A sports report.', 'label': 1},
              {'text': 'A much longer news report about international affairs and leaders.', 'label': 0}]
        text_ids = mixed.encode_text('A short domain paragraph.', tokenizer, 256)
        examples = [{'sample_type': 'text', 'input_ids': text_ids},
                    {'sample_type': 'qa', 'example': qa[0]}, {'sample_type': 'qa', 'example': qa[1]}]
        for side in ('left', 'right'):
            tokenizer.padding_side = side
            collator = mixed.MixedDataCollator(tokenizer, 256, lambda rows: mixed.TASK_CONFIGS['AGNEWS'][1](rows, tokenizer))
            batch = collator(examples)
            self.assertEqual(set(batch), {'input_ids', 'attention_mask', 'labels'})
            valid = batch['attention_mask'].bool()
            self.assertTrue((~valid).any())
            self.assertTrue((batch['labels'][~valid] == -100).all())
            raw_labels = batch['labels'][0][valid[0]]
            self.assertEqual(raw_labels[0], -100)
            self.assertEqual(raw_labels[1:].tolist(), text_ids[1:])
            # EOS and padding have the same token ID but only padding is ignored.
            self.assertEqual(raw_labels[-1], tokenizer.eos_token_id)
            for i, letter in ((1, 'B'), (2, 'A')):
                labels = batch['labels'][i]
                self.assertEqual(tokenizer.decode(labels[labels != -100]), letter + '<|eot_id|>')
            qa_only = collator(examples[1:])
            text_only = collator(examples[:1])
            self.assertEqual(len(qa_only['labels']), 2)
            self.assertEqual(len(text_only['labels']), 1)
            with self.assertRaises(ValueError):
                collator([{'text': 'Unmarked, ambiguous data'}])
        with self.assertRaisesRegex(ValueError, 'never silently truncated'):
            mixed.encode_text('This is a long text that exceeds two tokens.', tokenizer, 2)

    def test_dry_run_keeps_all_qa_rows_and_checks_provenance_before_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_task(root, 'AGNEWS', count=5)
            args = mixed.build_parser().parse_args([
                '--tasks', 'AGNEWS', '--data_dir', str(root), '--output_dir', str(root / 'output'),
                '--text_samples_per_task', '2', '--dry_run'])
            with patch.object(mixed, 'build_base_peft_model', side_effect=AssertionError('Dry run cannot load weights')):
                plan = mixed.run_text_mix(args)
            info = plan['data']['AGNEWS']
            self.assertEqual((info['qa_train_count'], info['text_train_count'], info['mixed_train_count']), (5, 2, 7))
            self.assertEqual((info['qa_validation_count'], info['qa_test_count']), (1, 1))
            self.assertEqual(plan['evaluation']['data'], 'QA only')
            self.assertTrue((root / 'output/mixing_manifest.json').is_file())
            text_file = root / 'AGNEWS_text/AGNEWS_domain_2.jsonl'
            manifest_file = root / 'AGNEWS_text/AGNEWS_domain_2_manifest.json'
            manifest_file.write_text(json.dumps({
                'output_sha256': hashlib.sha256(text_file.read_bytes()).hexdigest(),
                'source_sha256': 'incorrect source hash',
            }))
            with patch.object(mixed, 'build_base_peft_model') as build:
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    mixed.run_text_mix(args)
                build.assert_not_called()
            manifest_file.unlink()
            text_file.unlink()
            with patch.object(mixed, 'build_base_peft_model') as build:
                with self.assertRaises(FileNotFoundError):
                    mixed.run_text_mix(args)
                build.assert_not_called()

    def test_two_stage_training_uses_only_qa_metrics_and_carries_best_adapter(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        tokenizer = copy.deepcopy(self.tokenizer)
        torch.manual_seed(7)
        model = get_peft_model(LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=256, pad_token_id=tokenizer.pad_token_id,
        )), LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        starts, validation, tests, prepared = [], {'AGNEWS': [], 'AMAZON': []}, [], {}
        testcase = self

        class RecordingTrainer(mixed.GenerationEvalTrainer):
            def __init__(self, *args, **kwargs):
                starts.append(snapshot(kwargs['model']))
                dataset = kwargs['train_dataset']
                testcase.assertIsInstance(dataset, mixed.MixedTaskDataset)
                testcase.assertEqual(len(dataset), 4)
                testcase.assertEqual([dataset[i]['sample_type'] for i in range(4)], ['qa', 'qa', 'text', 'text'])
                testcase.assertIs(kwargs['eval_dataset'], prepared[tasks[len(starts) - 1]]['qa']['val'])
                super().__init__(*args, **kwargs)

        def evaluator(task):
            def evaluate(**kwargs):
                dataset = kwargs['test_dataset']
                weights = snapshot(kwargs['model'])
                if dataset is prepared[task]['qa']['val']:
                    validation[task].append(weights)
                    accuracy = .9 if len(validation[task]) == 1 else .2
                else:
                    testcase.assertIs(dataset, prepared[task]['qa']['test'])
                    tests.append((task, weights))
                    accuracy = .5
                testcase.assertNotIsInstance(dataset, mixed.MixedTaskDataset)
                testcase.assertIn('label', dataset.column_names)
                return [], {'accuracy': accuracy, 'correct': 1, 'total': len(dataset)}
            return evaluate

        original_prepare = mixed.prepare_task_data
        def prepare(task, args, tok):
            result = original_prepare(task, args, tok)
            prepared[task] = result
            return result

        tasks = ['AGNEWS', 'AMAZON']
        configs = {task: (*mixed.TASK_CONFIGS[task][:2], evaluator(task)) for task in tasks}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in tasks:
                write_task(root, task)
            args = mixed.build_parser().parse_args([
                '--tasks', ','.join(tasks), '--data_dir', str(root), '--output_dir', str(root / 'output'),
                '--text_samples_per_task', '2', '--num_train_epochs', '2', '--num_workers', '0',
                '--per_device_train_batch_size', '4', '--gradient_accumulation_steps', '1',
                '--max_seq_length', '128', '--logging_steps', '1', '--warmup_ratio', '0',
            ])
            with patch.object(mixed, 'build_tokenizer', return_value=tokenizer), \
                 patch.object(mixed, 'build_base_peft_model', return_value=model) as build, \
                 patch.object(mixed, 'prepare_task_data', side_effect=prepare), \
                 patch.object(mixed, 'TASK_CONFIGS', configs), \
                 patch.object(mixed, 'GenerationEvalTrainer', RecordingTrainer):
                summary = mixed.run_text_mix(args)
            build.assert_called_once()
            self.assertEqual([s['best_validation_accuracy'] for s in summary['stages']], [.9, .9])
            self.assertEqual([task for task, _ in tests], ['AGNEWS', 'AGNEWS', 'AMAZON'])
            self.assertTrue(same(starts[1], validation['AGNEWS'][0]))
            self.assertTrue(same(tests[0][1], validation['AGNEWS'][0]))
            self.assertTrue(same(tests[1][1], validation['AMAZON'][0]))
            self.assertTrue(same(tests[2][1], validation['AMAZON'][0]))
            self.assertFalse(same(validation['AGNEWS'][0], validation['AGNEWS'][1]))
            self.assertFalse(same(validation['AMAZON'][0], validation['AMAZON'][1]))
            self.assertEqual(set(summary['stages'][1]['test_metrics']), {'AGNEWS', 'AMAZON'})
            for stage in ('01_AGNEWS', '02_AMAZON'):
                self.assertTrue((root / 'output' / stage / 'best_model/adapter_config.json').is_file())
            self.assertEqual(json.loads((root / 'output/sequential_results.json').read_text()), summary)
            with patch.object(mixed, 'build_base_peft_model') as build:
                with self.assertRaises(FileExistsError):
                    mixed.run_text_mix(args)
                build.assert_not_called()


if __name__ == '__main__':
    unittest.main()

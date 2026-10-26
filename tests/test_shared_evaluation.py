"""Regression checks for shared generation evaluation and task entry points."""
import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('SST2', 'AGNEWS', 'AMAZON', 'YELP', 'DBPEDIA', 'YAHOO', 'QQP')


class FakeModel(torch.nn.Module):
    def __init__(self, tokenizer, replies=(), fail=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.replies = replies
        self.offset = 0
        self.fail = fail

    def generate(self, input_ids, attention_mask, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        assert self.tokenizer.padding_side == 'left'
        if self.fail:
            raise RuntimeError('generation failed')
        replies = self.replies[self.offset:self.offset + len(input_ids)]
        self.offset += len(input_ids)
        tokens = self.tokenizer(
            replies, add_special_tokens=False, padding=True, return_tensors='pt'
        )['input_ids']
        return torch.cat([input_ids, tokens], dim=1)


class SharedEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(ROOT / 'Llama-3.1-8B-Instruct'), local_files_only=True
        )
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.tokenizer.clean_up_tokenization_spaces = False

    def setUp(self):
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'

    def test_task_evaluators_count_every_sample_and_restore_state(self):
        for task in TASKS:
            with self.subTest(task=task), contextlib.redirect_stderr(io.StringIO()):
                module = importlib.import_module(f'src.evaluation.{task}_eval')
                evaluate = getattr(module, f'evaluate_{task}_QA')
                with open(Path(module.DATA_DIR) / 'test.jsonl') as stream:
                    rows = [json.loads(line) for line in stream if line.strip()]
                samples = [max(rows, key=lambda row: row['label']), rows[0], rows[-1]]
                labels = [module.extract_label_from_example(row) for row in samples]
                wrong = 'B' if labels[1] == 'A' else 'A'
                model = FakeModel(self.tokenizer, [labels[0], wrong, 'unknown'])
                results, metrics = evaluate(
                    model, self.tokenizer, samples, 'cpu',
                    batch_size=2, print_sample=False,
                )
                self.assertEqual(metrics, {'accuracy': 1 / 3, 'correct': 1, 'total': 3})
                self.assertEqual([r['index'] for r in results], [0, 1, 2])
                self.assertEqual(results[0]['predicted_answer'], labels[0])
                self.assertIsNone(results[2]['predicted_answer'])
                self.assertTrue(model.training)
                self.assertEqual(self.tokenizer.padding_side, 'right')
                self.assertEqual(self.tokenizer.truncation_side, 'right')

                failing = FakeModel(self.tokenizer, fail=True)
                with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                    evaluate(failing, self.tokenizer, samples, 'cpu', print_sample=False)
                self.assertTrue(failing.training)
                self.assertEqual(self.tokenizer.padding_side, 'right')
                self.assertEqual(self.tokenizer.truncation_side, 'right')

                model.eval()
                self.assertEqual(evaluate(model, self.tokenizer, [], 'cpu', print_sample=False)[1]['total'], 0)
                self.assertFalse(model.training)
                with self.assertRaises(ValueError):
                    evaluate(model, self.tokenizer, samples, 'cpu', batch_size=0)

    def test_task_entry_points_read_local_data_and_save_results(self):
        for task in TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                module = importlib.import_module(f'src.evaluation.{task}_eval')
                args = SimpleNamespace(
                    peft_model_folder='/tmp/mock-adapter', batch_size=32,
                    max_input_length=1536, max_new_tokens=8, print_sample=False,
                    output_dir=tmp, qa_output_json_name='full.json',
                    unit_test_output_dir=tmp, qa_unit_test_json_name='tiny.json',
                )

                def fake_evaluate(**kwargs):
                    self.assertEqual(kwargs['max_input_length'], 1536)
                    self.assertEqual(kwargs['max_new_tokens'], 8)
                    return [], {'accuracy': 0.0, 'total': len(kwargs['test_dataset'])}

                with patch.object(module, 'build_eval_model', return_value=(object(), self.tokenizer, 'cpu', 'finetuned')) as build, patch.object(module, f'evaluate_{task}_QA', side_effect=fake_evaluate), contextlib.redirect_stdout(io.StringIO()):
                    module.tiny_test(args)
                    module.run_full_evaluation(args)
                self.assertEqual(build.call_count, 2)
                with open(Path(tmp) / 'tiny.json') as stream:
                    self.assertEqual(json.load(stream)['metrics']['total'], 2)
                with open(Path(module.DATA_DIR) / 'test.jsonl') as stream:
                    expected = sum(bool(line.strip()) for line in stream)
                with open(Path(tmp) / 'full.json') as stream:
                    self.assertEqual(json.load(stream)['metrics']['total'], expected)


if __name__ == '__main__':
    unittest.main()

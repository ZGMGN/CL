"""Offline regression checks for the four added classification tasks.

Run: python -m unittest discover -s tests -p 'test_standard_cl_tasks.py'
"""
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from datasets import ClassLabel, Dataset, disable_progress_bars
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('AMAZON', 'YELP', 'DBPEDIA', 'YAHOO')
disable_progress_bars()


def load_module(task):
    spec = importlib.util.spec_from_file_location(task, ROOT / 'tasks' / f'{task}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_source(module):
    size = len(module.CLASS_NAMES) * 100
    data = {
        field: [f'{field} row {i}' for i in range(size)]
        for field, _ in module.TEXT_FIELDS
    }
    # Exercise Amazon's plain integer labels and Yahoo's topic column.
    data[module.SOURCE_LABEL_COL] = [i % len(module.CLASS_NAMES) for i in range(size)]
    if module.DATASET_ID.endswith('yahoo_answers_topics'):
        data['id'] = list(range(size))
    return Dataset.from_dict(data)


class StandardCLTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = {task: load_module(task) for task in TASKS}

    def test_source_normalization_and_splits(self):
        for task, module in self.modules.items():
            with self.subTest(task=task), patch.object(module, 'load_dataset', return_value=make_source(module)) as loader:
                source = module._load_source_data(None, None)
                self.assertIsInstance(source.features['label'], ClassLabel)
                self.assertEqual(source.features['label'].names, list(module.CLASS_NAMES))
                self.assertIn('idx', source.column_names)
                if module.SOURCE_LABEL_COL != 'label':
                    self.assertNotIn(module.SOURCE_LABEL_COL, source.column_names)
                loader.assert_called_once_with(module.DATASET_ID, split='train', cache_dir=None, token=None)
                parts = module._split_data(source, 1)
                size = len(source)
                self.assertEqual([len(parts[s]) for s in ('train', 'val', 'test')], [size * 6 // 10, size // 5, size // 5])
                for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                    self.assertFalse(set(parts[a]['idx']) & set(parts[b]['idx']))
                load = getattr(module, f'load_{task.lower()}_data')
                for split in parts:
                    self.assertEqual(load(split)['idx'], parts[split]['idx'])
                with self.assertRaises(ValueError):
                    load('invalid')

    def test_messages_and_answer_parsing(self):
        for task, module in self.modules.items():
            with self.subTest(task=task):
                rows = []
                for label, letter in enumerate(module.ANSWER_LETTERS):
                    row = {field: f'{field} content' for field, _ in module.TEXT_FIELDS}
                    row['label'] = label
                    rows.append(row)
                    messages = module.example_to_messages(row)
                    self.assertEqual(messages[-1]['content'], letter)
                    self.assertEqual(module.extract_label_from_example(row), letter)
                    for field, _ in module.TEXT_FIELDS:
                        self.assertIn(row[field], messages[0]['content'])
                    self.assertEqual(module.extract_answer_letter(f'The correct answer is {letter}.'), letter)
                    self.assertEqual(module.extract_answer_letter(f'({letter})'), letter)
                batch = {field: [row[field] for row in rows] for field in rows[0]}
                self.assertEqual(module.example_to_messages(batch), [module.example_to_messages(row) for row in rows])
                self.assertEqual(module.extract_label_from_example(batch), list(module.ANSWER_LETTERS))
                self.assertEqual(module.extract_answer_letter('I think the answer is B'), 'B')
                self.assertIsNone(module.extract_answer_letter('Z'))
                self.assertIsNone(module.extract_answer_letter('I think so'))
                with self.assertRaises(ValueError):
                    module._build_answer(-1)
                with self.assertRaises(ValueError):
                    module._build_answer(len(module.CLASS_NAMES))
                batch['label'] = []
                with self.assertRaises(ValueError):
                    module.example_to_messages(batch)
                null_row = dict(rows[0])
                null_row[module.TEXT_FIELDS[0][0]] = None
                self.assertEqual(module.example_to_messages(null_row)[-1]['content'], 'A')
                self.assertEqual(module.compute_accuracy(['A', 'B'], ['A', 'A']), 0.5)
                self.assertEqual(module.compute_accuracy([], []), 0.0)
                with self.assertRaises(ValueError):
                    module.compute_accuracy(['A'], [])

    def test_jsonl_exports_and_sampling(self):
        for task, module in self.modules.items():
            with self.subTest(task=task), patch.object(module, 'load_dataset', return_value=make_source(module)) as loader:
                with tempfile.TemporaryDirectory() as tmp:
                    build = getattr(module, f'build_{task.lower()}_files')
                    build(tmp, n_train=3, n_val=2, n_test=2)
                    self.assertEqual(loader.call_count, 1)
                    outputs = {}
                    for split, count in (('train', 3), ('val', 2), ('test', 2)):
                        rows = [json.loads(line) for line in (Path(tmp) / f'{split}.jsonl').read_text().splitlines()]
                        self.assertEqual(Counter(row['label'] for row in rows), {i: count for i in range(len(module.CLASS_NAMES))})
                        outputs[split] = {row['idx'] for row in rows}
                        for row in rows:
                            module.example_to_messages(row)
                    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                        self.assertFalse(outputs[a] & outputs[b])
                source = module._load_source_data(None, None)
                self.assertEqual(module._balanced_subsample(source, 3)['idx'], module._balanced_subsample(source, 3)['idx'])
                for n in (0, -1, 1000):
                    with self.assertRaises(ValueError):
                        module._balanced_subsample(source, n)

    def test_local_llama_rendering(self):
        path = ROOT / 'Llama-3.1-8B-Instruct'
        if not (path / 'tokenizer_config.json').exists():
            self.skipTest('Local Llama tokenizer is not available')
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        for task, module in self.modules.items():
            with self.subTest(task=task):
                tokenizer.chat_template = getattr(module, f'{task}_CHAT_TEMPLATE_LLAMA31')
                row = {field: 'example text' for field, _ in module.TEXT_FIELDS}
                row['label'] = 0
                rendered = module.format_prompt(row, tokenizer, include_answer=True)
                self.assertTrue(rendered.endswith('<|start_header_id|>assistant<|end_header_id|>\n\nA<|eot_id|>'))
                unlabeled = {key: val for key, val in row.items() if key != 'label'}
                prompt = module.format_prompt(unlabeled, tokenizer)
                self.assertTrue(prompt.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'))
                batch = {field: [value, value] for field, value in row.items()}
                self.assertEqual(module.format_prompt(batch, tokenizer), [prompt, prompt])


if __name__ == '__main__':
    unittest.main()

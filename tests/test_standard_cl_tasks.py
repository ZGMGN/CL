"""Offline checks for tasks following the AGNEWS / QQP / SST2 layout."""
import contextlib
import importlib.util
import io
import inspect
import json
import runpy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from datasets import ClassLabel, Dataset, Features, Value, disable_progress_bars, load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'AMAZON': (5, ('title', 'content'), 'label'),
    'YELP': (5, ('text',), 'label'),
    'DBPEDIA': (14, ('title', 'content'), 'label'),
    'YAHOO': (10, ('question_title', 'question_content', 'best_answer'), 'topic'),
}
disable_progress_bars()


def make_source(task, per_class=100):
    count, fields, label_col = CONFIGS[task]
    size = count * per_class
    data = {field: [f'{field} row {i}' for i in range(size)] for field in fields}
    data[label_col] = [i % count for i in range(size)]
    features = Features({field: Value('string') for field in fields})
    features[label_col] = Value('int64') if task == 'AMAZON' else ClassLabel(num_classes=count)
    return Dataset.from_dict(data, features=features)


class StandardCLTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = {}
        for task in CONFIGS:
            spec = importlib.util.spec_from_file_location(task, ROOT / 'src' / 'tasks' / f'{task}.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.modules[task] = module
        path = ROOT / 'Llama-3.1-8B-Instruct'
        cls.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True) if path.exists() else None

    def test_messages_and_answers(self):
        for task, module in self.modules.items():
            count, fields, _ = CONFIGS[task]
            with self.subTest(task=task):
                rows = [{**{field: f'{field} content' for field in fields}, 'label': label} for label in range(count)]
                for label, row in enumerate(rows):
                    letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[label]
                    messages = module.example_to_messages(row)
                    self.assertEqual(messages[-1]['content'], letter)
                    self.assertEqual(module.extract_label_from_example(row), letter)
                    for field in fields:
                        self.assertIn(row[field], messages[0]['content'])
                    self.assertEqual(module.extract_answer_letter(f'The correct answer is {letter}.'), letter)
                batch = {key: [row[key] for row in rows] for key in rows[0]}
                self.assertEqual(module.example_to_messages(batch), [module.example_to_messages(row) for row in rows])
                self.assertEqual(module.extract_answer_letter('I think the answer is B'), 'B')
                self.assertIsNone(module.extract_answer_letter('I think so'))
                self.assertIsNone(module.extract_answer_letter('Z'))
                for label in (-1, count):
                    with self.assertRaises(ValueError):
                        module._build_answer(label)
                self.assertEqual(module.compute_accuracy(['A', 'B'], ['A', 'A']), 0.5)
                self.assertEqual(module.compute_accuracy([], []), 0.0)
                with self.assertRaises(ValueError):
                    module.compute_accuracy(['A'], [])

    def test_load_and_export(self):
        for task, module in self.modules.items():
            count, _, _ = CONFIGS[task]
            with self.subTest(task=task), patch.object(module, 'load_dataset', return_value=make_source(task)) as upstream:
                load = getattr(module, f'load_{task.lower()}_data')
                parts = load()
                self.assertEqual([len(parts[s]) for s in parts], [count * 60, count * 20, count * 20])
                self.assertIsInstance(parts['train'].features['label'], ClassLabel)
                if task == 'YAHOO':
                    self.assertNotIn('topic', parts['train'].column_names)
                repeated_train = load()['train']
                self.assertEqual(repeated_train['idx'], parts['train']['idx'])
                with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                    upstream.reset_mock()
                    getattr(module, f'build_{task.lower()}_files')(tmp, n_train=3, n_val=2, n_test=2)
                    self.assertEqual(upstream.call_count, 1)
                    indices = {}
                    for split, n in (('train', 3), ('val', 2), ('test', 2)):
                        rows = [json.loads(line) for line in (Path(tmp) / f'{split}.jsonl').read_text().splitlines()]
                        self.assertEqual(Counter(row['label'] for row in rows), {i: n for i in range(count)})
                        indices[split] = {row['idx'] for row in rows}
                        for row in rows:
                            module.example_to_messages(row)
                    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                        self.assertFalse(indices[a] & indices[b])
                self.assertEqual(module._balanced_subsample(parts['train'], 3)['idx'], module._balanced_subsample(parts['train'], 3)['idx'])
                for n in (0, -1, 1000):
                    with self.assertRaises(ValueError):
                        module._balanced_subsample(parts['train'], n)

    def test_local_rendering(self):
        if self.tokenizer is None:
            self.skipTest('Local Llama tokenizer is not available')
        for task, module in self.modules.items():
            with self.subTest(task=task):
                self.tokenizer.chat_template = getattr(module, f'{task}_CHAT_TEMPLATE_LLAMA31')
                row = {field: 'example text' for field in CONFIGS[task][1]}
                row['label'] = 0
                prompt = module.format_prompt(row, self.tokenizer, include_answer=False)
                self.assertTrue(prompt.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'))
                full = module.format_prompt(row, self.tokenizer, include_answer=True)
                self.assertTrue(full.endswith('<|start_header_id|>assistant<|end_header_id|>\n\nA<|eot_id|>'))

    def test_amazon_main_loads_source_once_and_reads_jsonl(self):
        self._check_main_loads_source_once('AMAZON', 'vgaraujov/amazon_review_full', 3000)

    def test_dbpedia_main_loads_source_once_and_reads_jsonl(self):
        self._check_main_loads_source_once('DBPEDIA', 'fancyzhx/dbpedia_14', 4000)

    def test_yahoo_main_loads_source_once_and_reads_jsonl(self):
        self._check_main_loads_source_once('YAHOO', 'community-datasets/yahoo_answers_topics', 4000)

    def test_yelp_main_loads_source_once_and_reads_jsonl(self):
        self._check_main_loads_source_once('YELP', 'Yelp/yelp_review_full', 4000)

    def _check_main_loads_source_once(self, task, dataset_id, per_class):
        if self.tokenizer is None:
            self.skipTest('Local Llama tokenizer is not available')
        source = make_source(task, per_class=per_class)
        count = CONFIGS[task][0]
        defaults = inspect.signature(getattr(self.modules[task], f'build_{task.lower()}_files')).parameters
        n_train = defaults['n_train'].default
        n_val = defaults['n_val'].default
        n_test = defaults['n_test'].default
        original_export = Dataset.to_json
        original_split = Dataset.train_test_split
        split_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            def export_to_temp(ds, path, **kwargs):
                self.assertEqual(Path(path).parent, ROOT / 'data' / task)
                return original_export(ds, str(Path(tmp) / Path(path).name), **kwargs)

            def load_from_source_or_json(path, *args, **kwargs):
                if path == dataset_id:
                    return source
                self.assertEqual(path, 'json')
                for filename in kwargs['data_files'].values():
                    self.assertEqual(Path(filename).parent, ROOT / 'data' / task)
                kwargs['data_files'] = {
                    split: str(Path(tmp) / Path(filename).name)
                    for split, filename in kwargs['data_files'].items()
                }
                kwargs['cache_dir'] = str(Path(tmp) / 'cache')
                return load_dataset(path, *args, **kwargs)

            def count_split(ds, *args, **kwargs):
                split_calls.append(len(ds))
                return original_split(ds, *args, **kwargs)

            with patch('datasets.load_dataset', side_effect=load_from_source_or_json) as loader, \
                 patch.object(Dataset, 'to_json', new=export_to_temp), \
                 patch.object(Dataset, 'train_test_split', new=count_split), \
                 patch.object(AutoTokenizer, 'from_pretrained', return_value=self.tokenizer), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                runpy.run_path(str(ROOT / 'src' / 'tasks' / f'{task}.py'), run_name='__main__')
            self.assertEqual([call.args[0] for call in loader.call_args_list],
                             [dataset_id, 'json'])
            self.assertEqual(split_calls, [len(source), len(source) * 4 // 5])
            self.assertIn(f'Length of train set: {count * n_train}, val set: {count * n_val}, test set: {count * n_test}', output.getvalue())
            self.assertIn(f'{task} sanity checks: PASS', output.getvalue())
            self.assertIn('Tiny accuracy check (expect 0.5): 0.5', output.getvalue())
            indices = {}
            for split, n in (('train', n_train), ('val', n_val), ('test', n_test)):
                rows = [json.loads(line) for line in (Path(tmp) / f'{split}.jsonl').read_text().splitlines()]
                self.assertEqual(Counter(row['label'] for row in rows), {i: n for i in range(count)})
                indices[split] = {row['idx'] for row in rows}
            for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
                self.assertFalse(indices[left] & indices[right])



if __name__ == '__main__':
    unittest.main()

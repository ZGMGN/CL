"""Offline checks for SST2 text deduplication before stratified splitting."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from datasets import ClassLabel, Dataset, Features, Value, disable_progress_bars

ROOT = Path(__file__).resolve().parents[1]
disable_progress_bars()


def make_source():
    rows = [{'idx':i, 'sentence':f'unique text {i}', 'label':i % 2} for i in range(100)]
    rows += [
        {'idx':100, 'sentence':' Shared  TEXT ', 'label':0},
        {'idx':101, 'sentence':'shared text', 'label':0},
        {'idx':102, 'sentence':'AMBIGUOUS', 'label':0},
        {'idx':103, 'sentence':' ambiguous ', 'label':1},
        {'idx':104, 'sentence':' \t ', 'label':0},
    ]
    return Dataset.from_list(rows, features=Features({'idx':Value('int64'), 'sentence':Value('string'), 'label':ClassLabel(names=['negative','positive'])}))


class SST2DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('sst2_data_test', ROOT/'src'/'tasks'/'SST2.py')
        cls.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_dedup_and_conflict_exclusion_before_split(self):
        module=self.module
        with patch.object(module,'load_dataset',return_value=make_source()), contextlib.redirect_stdout(io.StringIO()) as output:
            parts=module.load_sst2_data(split=None)
            repeated=module.load_sst2_data(split=None)
            self.assertEqual(module.load_sst2_data('train')['idx'],parts['train']['idx'])
        rows=[row for ds in parts.values() for row in ds]
        self.assertEqual(len(rows),101)
        self.assertEqual(len({module._sentence_key(row['sentence']) for row in rows}),101)
        self.assertIn(100,{row['idx'] for row in rows})
        self.assertFalse({101,102,103,104} & {row['idx'] for row in rows})
        for split in parts:
            self.assertEqual(parts[split]['idx'],repeated[split]['idx'])
            self.assertEqual(set(parts[split]['label']),{0,1})
        for a,b in [('train','val'),('train','test'),('val','test')]:
            self.assertFalse(set(parts[a]['idx']) & set(parts[b]['idx']))
        self.assertIn('conflicting groups: 1 (2 rows)',output.getvalue())

    def test_balanced_export_and_text_isolation(self):
        module=self.module
        with patch.object(module,'load_dataset',return_value=make_source()) as source, tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            module.build_sst2_files(tmp,n_train=3,n_val=2,n_test=2)
            self.assertEqual(source.call_count,1)
            keys={}
            for split,n in [('train',3),('val',2),('test',2)]:
                rows=[json.loads(line) for line in (Path(tmp)/f'{split}.jsonl').read_text().splitlines()]
                self.assertEqual(Counter(row['label'] for row in rows),{0:n,1:n})
                keys[split]={module._sentence_key(row['sentence']) for row in rows}
                self.assertEqual(len(keys[split]),len(rows))
                for row in rows:
                    self.assertEqual(module.extract_label_from_example(row),'AB'[row['label']])
            for a,b in [('train','val'),('train','test'),('val','test')]:
                self.assertFalse(keys[a] & keys[b])

    def test_failed_sampling_keeps_existing_files(self):
        module=self.module
        with patch.object(module,'load_dataset',return_value=make_source()), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            files=[Path(tmp)/f'{split}.jsonl' for split in ('train','val','test')]
            for file in files:
                file.write_text('original data\n')
            with self.assertRaises(ValueError):
                module.build_sst2_files(tmp,n_train=3,n_val=999,n_test=2)
            for file in files:
                self.assertEqual(file.read_text(),'original data\n')


if __name__=='__main__':
    unittest.main()

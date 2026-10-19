"""Offline checks for Yahoo complete-chat length filtering and replenishment."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from datasets import Dataset, disable_progress_bars
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
disable_progress_bars()


class YahooLengthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('yahoo_length_test',ROOT/'src'/'tasks'/'YAHOO.py')
        cls.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'Llama-3.1-8B-Instruct'),local_files_only=True)
        cls.tokenizer.chat_template=cls.module.YAHOO_CHAT_TEMPLATE_LLAMA31

    def make_pool(self, offset=0, short_per_class=2):
        rows=[]
        for label in range(10):
            for i in range(6):
                rows.append({'idx':offset+label*6+i,'label':label,'question_title':f'Question {offset} {label} {i}',
                             'question_content':'short body','best_answer':'short answer' if i<short_per_class else ' long'*1000})
        return Dataset.from_list(rows)

    def test_full_chat_boundary(self):
        module=self.module
        row={'idx':0,'label':0,'question_title':'Short question','question_content':'','best_answer':'Short answer'}
        text=module.format_prompt(row,self.tokenizer,include_answer=True)
        length=len(self.tokenizer.encode(text,add_special_tokens=False))
        ds=Dataset.from_list([row])
        self.assertEqual(len(module._filter_by_length(ds,self.tokenizer,length)),1)
        self.assertEqual(len(module._filter_by_length(ds,self.tokenizer,length-1)),0)
        for limit in (0,-1):
            with self.assertRaises(ValueError):
                module._filter_by_length(ds,self.tokenizer,limit)

    def test_replenishment_preserves_class_counts_and_seed(self):
        module=self.module
        source=self.make_pool()
        with contextlib.redirect_stdout(io.StringIO()):
            result=module._balanced_subsample(source,2,tokenizer=self.tokenizer,max_length=256)
            repeated=module._balanced_subsample(source,2,tokenizer=self.tokenizer,max_length=256)
        self.assertEqual(result['idx'],repeated['idx'])
        self.assertEqual(Counter(result['label']),{i:2 for i in range(10)})
        self.assertEqual(len(set(result['idx'])),20)
        self.assertEqual(len(module._filter_by_length(result,self.tokenizer,256)),20)
        for row in result:
            self.assertEqual(row['best_answer'],'short answer')

    def test_export_only_eligible_rows(self):
        module=self.module
        parts={split:self.make_pool(offset) for split,offset in [('train',0),('val',100),('test',200)]}
        with patch.object(module,'load_yahoo_data',return_value=parts) as load, tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            module.build_yahoo_files(tmp,n_train=2,n_val=2,n_test=2,max_length=256,tokenizer=self.tokenizer)
            self.assertEqual(load.call_count,1)
            ids={}
            for split in parts:
                rows=[json.loads(line) for line in (Path(tmp)/f'{split}.jsonl').read_text().splitlines()]
                self.assertEqual(Counter(row['label'] for row in rows),{i:2 for i in range(10)})
                self.assertEqual(len(module._filter_by_length(Dataset.from_list(rows),self.tokenizer,256)),20)
                ids[split]={row['idx'] for row in rows}
            for a,b in [('train','val'),('train','test'),('val','test')]:
                self.assertFalse(ids[a]&ids[b])

    def test_insufficient_eligible_samples_keep_old_files(self):
        module=self.module
        parts={split:self.make_pool(offset,short_per_class=1) for split,offset in [('train',0),('val',100),('test',200)]}
        with patch.object(module,'load_yahoo_data',return_value=parts), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            paths=[Path(tmp)/f'{split}.jsonl' for split in parts]
            for path in paths:path.write_text('original\n')
            with self.assertRaises(ValueError):
                module.build_yahoo_files(tmp,n_train=2,n_val=2,n_test=2,max_length=256,tokenizer=self.tokenizer)
            for path in paths:self.assertEqual(path.read_text(),'original\n')


if __name__=='__main__':
    unittest.main()

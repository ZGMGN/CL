"""Shared template/rendering and offline AGNEWS entry-point regression checks."""
import ast
import contextlib
import importlib
import io
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch
from datasets import load_dataset
from transformers import AutoTokenizer
from src.model.utils import GENERAL_TEMPLATE, render_chat

ROOT=Path(__file__).resolve().parents[1]
TASKS=('AGNEWS','QQP','SST2','AMAZON','YELP','DBPEDIA','YAHOO')


class SharedChatTest(unittest.TestCase):
    def test_tasks_use_shared_template_and_renderer(self):
        tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'Llama-3.1-8B-Instruct'),local_files_only=True)
        tokenizer.chat_template=GENERAL_TEMPLATE
        messages=[{'role':'user','content':'Test question'},{'role':'assistant','content':'A'}]
        expected=render_chat(messages,tokenizer)
        self.assertTrue(expected.endswith('<|start_header_id|>assistant<|end_header_id|>\n\nA<|eot_id|>'))
        for task in TASKS:
            with self.subTest(task=task):
                module=importlib.import_module(f'src.tasks.{task}')
                self.assertEqual(getattr(module,f'{task}_CHAT_TEMPLATE_LLAMA31'),GENERAL_TEMPLATE)
                self.assertEqual(module.render_chat.__code__.co_filename,render_chat.__code__.co_filename)
                self.assertEqual(module.render_chat(messages,tokenizer),expected)
                tree=ast.parse((ROOT/'src'/'tasks'/f'{task}.py').read_text())
                self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='render_chat' for n in tree.body))

    def test_agnews_main_reads_only_local_jsonl(self):
        def local_dataset(path,*args,**kwargs):
            self.assertEqual(path,'json','AGNEWS main must not request its upstream HF dataset')
            self.assertEqual(set(kwargs['data_files']),{'train','val','test'})
            for filename in kwargs['data_files'].values():
                self.assertTrue(Path(filename).is_file())
            return load_dataset(path,*args,**kwargs)
        with patch('datasets.load_dataset',side_effect=local_dataset) as loader, contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(ROOT/'src'/'tasks'/'AGNEWS.py'),run_name='__main__')
        self.assertEqual(loader.call_count,1)
        self.assertIn('Length of train set: 8000, val set: 2000, test set: 2000',output.getvalue())
        self.assertIn('Tiny accuracy check (expect 0.5): 0.5',output.getvalue())


if __name__=='__main__':
    unittest.main()

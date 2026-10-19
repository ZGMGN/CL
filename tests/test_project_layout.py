"""Check src package imports and resource paths after moving model/tasks."""
import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASKS=('AGNEWS','QQP','SST2','AMAZON','YELP','DBPEDIA','YAHOO')


class ProjectLayoutTest(unittest.TestCase):
    def test_task_roots_and_existing_resources(self):
        for task in TASKS:
            with self.subTest(task=task):
                module=importlib.import_module(f'src.tasks.{task}')
                self.assertEqual(Path(module.ROOT_DIR),ROOT)
                self.assertEqual(Path(module.import_dir),ROOT/'src')
                loader_name='_load_QQP_data' if task=='QQP' else f'load_{task.lower()}_data'
                loader=getattr(module,loader_name)
                import inspect
                cache=inspect.signature(loader).parameters['cache_dir'].default
                self.assertEqual(Path(cache),ROOT/'.cache'/'huggingface_models')
                for split in ('train','val','test'):
                    path=Path(module.ROOT_DIR)/'data'/task/f'{split}.jsonl'
                    with path.open() as handle:
                        row=json.loads(next(handle))
                    self.assertEqual(module.extract_label_from_example(row), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[row['label']])
        self.assertTrue((ROOT/'Llama-3.1-8B-Instruct'/'tokenizer_config.json').is_file())
        self.assertTrue((ROOT/'src'/'model'/'utils.py').is_file())
        self.assertFalse((ROOT/'model').exists())
        self.assertFalse((ROOT/'tasks').exists())

    def test_package_imports_from_another_directory(self):
        env=dict(os.environ,PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        code=f'''import importlib
from pathlib import Path
import src.model.utils
for name in {TASKS!r}:
    module=importlib.import_module('src.tasks.'+name)
    assert Path(module.ROOT_DIR)==Path({str(ROOT)!r}), (name,module.ROOT_DIR)
'''
        result=subprocess.run([sys.executable,'-c',code],cwd='/tmp',env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_script_imports_from_another_directory(self):
        # Running by path must compute roots from __file__, not from cwd.
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        code=f'''import runpy
from pathlib import Path
import model.utils
root=Path({str(ROOT)!r})
for name in {TASKS!r}:
    namespace=runpy.run_path(str(root/'src'/'tasks'/(name+'.py')),run_name='layout_check')
    assert Path(namespace['ROOT_DIR'])==root, (name,namespace['ROOT_DIR'])
'''
        result=subprocess.run([sys.executable,'-c',code],cwd='/tmp',env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':
    unittest.main()

"""Verify every model/task pair and independent real PEFT adapter switching."""
import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation import sequential_heatmap_eval as evaluation


class HeatmapEvaluationTest(unittest.TestCase):
    def test_task_order_metadata_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'launch.json').write_text(json.dumps({'command': ['python', '--tasks', 'DBPEDIA,AMAZON']}))
            self.assertEqual(evaluation.resolve_tasks(root), ['DBPEDIA', 'AMAZON'])
            (root / 'sequential_results.json').write_text(json.dumps({'tasks': ['YAHOO', 'AGNEWS']}))
            self.assertEqual(evaluation.resolve_tasks(root), ['YAHOO', 'AGNEWS'])
            self.assertEqual(evaluation.resolve_tasks(root, 'qqp,sst2'), ['QQP', 'SST2'])
            for value in ('', 'AGNEWS,UNKNOWN'):
                with self.assertRaises(ValueError):
                    evaluation.resolve_tasks(root, value)

    def test_unfinished_training_fails_before_model_allocation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(evaluation, 'build_eval_model') as build:
            args = evaluation.build_parser().parse_args(['--run_dir', tmp, '--tasks', 'AGNEWS,SST2'])
            with self.assertRaisesRegex(FileNotFoundError, 'Finish training first'):
                evaluation.run_evaluation(args)
            build.assert_not_called()

    def test_full_matrix_and_artifacts_with_independent_adapters(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        ))
        untouched = {key: value.clone() for key, value in base.state_dict().items()}
        adapter = get_peft_model(copy.deepcopy(base), LoraConfig(
            r=2, lora_alpha=4, target_modules=['q_proj'], task_type='CAUSAL_LM'
        ))
        calls, adapters_seen = [], []
        expected = [[0.2, 0.3], [0.6, 0.7], [0.8, 0.9]]

        def evaluator(**kwargs):
            model = kwargs['model']
            task = kwargs['test_dataset'][0]['task']
            values = [param for name, param in model.named_parameters() if 'lora_B' in name]
            if not values:
                row = 0
                underlying = model
            else:
                self.assertEqual(len(values), 1, 'Previous adapter should be deleted')
                value = values[0].detach().mean().item()
                row = 1 if abs(value - 0.1) < 1e-5 else 2
                self.assertAlmostEqual(value, 0.1 * row, places=5)
                underlying = model.get_base_model()
                adapters_seen.append(model.active_adapter)
            # Loading an adapter must not modify/merge any base weight.
            for name, value in underlying.state_dict().items():
                if 'lora_' not in name:
                    self.assertTrue(torch.equal(value, untouched[name.replace('.base_layer', '')]))
            col = ['AGNEWS', 'SST2'].index(task)
            calls.append((row, col))
            return [{'index': 0, 'correct': True}], {'accuracy': expected[row][col], 'total': 1, 'correct': 1}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, task in enumerate(('AGNEWS', 'SST2'), 1):
                for name, param in adapter.named_parameters():
                    if 'lora_B' in name:
                        param.data.fill_(index * 0.1)
                adapter.save_pretrained(root / f'{index:02d}_{task}' / 'best_model')
            args = evaluation.build_parser().parse_args(['--run_dir', tmp, '--tasks', 'AGNEWS,SST2'])
            modules = {f'evaluation.{task}_eval': SimpleNamespace(**{f'evaluate_{task}_QA': evaluator}) for task in ('AGNEWS', 'SST2')}
            real_import = evaluation.importlib.import_module
            with patch.object(evaluation, 'build_eval_model', return_value=(base, object(), 'cpu', 'base')) as build, \
                 patch.object(evaluation, 'load_dataset', side_effect=lambda *a, **kw: [{'task': Path(kw['data_files']).parent.name}]), \
                 patch.object(evaluation.importlib, 'import_module', side_effect=lambda name, *a, **kw: modules[name] if name in modules else real_import(name, *a, **kw)):
                payload = evaluation.run_evaluation(args)
            build.assert_called_once()
            self.assertEqual(calls, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
            self.assertEqual(adapters_seen, ['stage_1', 'stage_1', 'stage_2', 'stage_2'])
            self.assertEqual(payload['accuracy'], expected)
            self.assertTrue(payload['complete'])
            output = root / 'evaluation_heatmap'
            self.assertEqual(json.loads((output / 'accuracy_matrix.json').read_text()), payload)
            with (output / 'accuracy_matrix.csv').open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ['model', 'AGNEWS', 'SST2'])
            self.assertEqual(len(rows), 4)
            self.assertEqual((output / 'accuracy_heatmap.png').read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertTrue((output / 'accuracy_heatmap.pdf').read_bytes().startswith(b'%PDF'))
            self.assertEqual(len(list(output.glob('[0-9]*/*.json'))), 6)


if __name__ == '__main__':
    unittest.main()

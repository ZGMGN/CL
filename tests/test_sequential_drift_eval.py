"""Numerical and end-to-end tests for sequential representation/functional drift."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation import drift_metrics as metrics
from evaluation import sequential_drift_eval as drift


class DriftMetricsTest(unittest.TestCase):
    def test_cka_identity_orthogonal_scale_translation_and_degenerate(self):
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(12, 5, generator=generator, dtype=torch.float64)
        q = torch.linalg.qr(torch.randn(5, 5, generator=generator, dtype=torch.float64)).Q
        self.assertAlmostEqual(metrics.linear_cka(x, x), 1.0)
        self.assertAlmostEqual(metrics.linear_cka(x, 3 * x @ q + 7), 1.0)
        xc, yc = x - x.mean(0), (x * torch.arange(1, 6))
        yc = yc - yc.mean(0)
        expected = (xc.T @ yc).square().sum() / ((xc.T @ xc).norm() * (yc.T @ yc).norm())
        self.assertAlmostEqual(metrics.linear_cka(xc, yc), float(expected))
        self.assertIsNone(metrics.linear_cka(torch.ones(4, 3), torch.ones(4, 3)))
        with self.assertRaises(ValueError):
            metrics.linear_cka(x[:1], x[:1])

    def test_cosine_and_kl(self):
        x = torch.tensor([[1., 0.], [0., 1.]])
        self.assertEqual(metrics.representation_cosine_drift(x, x), 0)
        self.assertEqual(metrics.representation_cosine_drift(x, -x), 2)
        self.assertIsNone(metrics.cosine(torch.zeros(2), torch.ones(2)))
        p, q = torch.tensor([[0.8, 0.2]]).double(), torch.tensor([[0.4, 0.6]]).double()
        values = metrics.distribution_drift(p.log(), q.log())
        self.assertAlmostEqual(values['kl_forward'], float((p * (p.log() - q.log())).sum()))
        self.assertAlmostEqual(values['symmetric_kl'], (values['kl_forward'] + values['kl_reverse']) / 2)
        self.assertEqual(metrics.distribution_drift(p.log(), p.log())['symmetric_kl'], 0)

    def test_effective_lora_increment_cosine_matches_dense_and_factor_basis_change(self):
        torch.manual_seed(2)
        stages = [{}]
        for _ in range(3):
            stages.append({'module': (torch.randn(6, 2), torch.randn(2, 7), 2.0)})
        def dense(stage):
            if not stage:
                return torch.zeros(6, 7, dtype=torch.float64)
            b, a, scale = stage['module']
            return b.double() @ a.double() * scale
        first, second = dense(stages[1]), dense(stages[3]) - dense(stages[2])
        actual = metrics.lora_update_cosine(stages[1], {}, stages[3], stages[2])
        self.assertAlmostEqual(actual, metrics.cosine(first, second), places=10)
        self.assertIsNone(metrics.lora_update_cosine({}, {}, stages[1], {}))
        b, a, scale = stages[1]['module']
        swapped = {'module': (b[:, [1, 0]], a[[1, 0]], scale)}
        self.assertAlmostEqual(metrics.lora_update_cosine(stages[1], {}, swapped, {}), 1)

    def test_adjacent_report_excludes_older_nonadjacent_tasks(self):
        accuracy = np.array([[0.2, 0.3, 0.4], [0.8, 0.3, 0.4], [0.7, 0.9, 0.4], [0.6, 0.8, 0.9]])
        layers = {'1': {'cosine_drift': 0.2, 'cka': 0.8, 'cka_drift': 0.2}}
        functional = {'symmetric_kl': 0.1, 'kl_forward': 0.09, 'kl_reverse': 0.11}
        tasks = ['AGNEWS', 'SST2', 'QQP']
        with self.assertRaises(ValueError):
            drift.build_adjacent_comparison(3, tasks, 0, accuracy, layers, functional, 1, 0.3, 0.4, 8)
        previous = drift.build_adjacent_comparison(3, tasks, 1, accuracy, layers, functional, 1, 0.3, 0.4, 8)
        current = drift.build_adjacent_comparison(3, tasks, 2, accuracy, layers, functional, 1, 0.3, 0.4, 8)
        self.assertEqual(previous['evaluated_task'], 'SST2')
        self.assertEqual(current['evaluated_task'], 'QQP')
        self.assertAlmostEqual(previous['forgetting_pp'], 10)
        self.assertAlmostEqual(current['learning_gain_pp'], 50)

    def test_layer_selection(self):
        self.assertEqual(drift.resolve_layers('-1', 4), [3])
        self.assertEqual(drift.resolve_layers('all', 4), [0, 1, 2, 3])
        for value in ('4', '-5', '0,0'):
            with self.assertRaises(ValueError):
                drift.resolve_layers(value, 4)


class DriftPipelineTest(unittest.TestCase):
    def test_real_tiny_model_two_stage_diagnostics(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'Llama-3.1-8B-Instruct'), local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        torch.manual_seed(7)
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            pad_token_id=tokenizer.pad_token_id, max_position_embeddings=256,
        ))
        adapter = get_peft_model(copy.deepcopy(base), LoraConfig(
            r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'
        ))
        probes = {
            'AGNEWS': [{'text': text, 'label': i % 4} for i, text in enumerate(['A world news story.', 'Sports team wins.', 'A business report.'])],
            'SST2': [{'sentence': text, 'label': i % 2} for i, text in enumerate(['A good movie.', 'Terrible acting.', 'An interesting film.'])],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, task in enumerate(('AGNEWS', 'SST2'), 1):
                for name, parameter in adapter.named_parameters():
                    if 'lora_B' in name:
                        parameter.data.fill_(index * 0.03)
                adapter.save_pretrained(root / f'{index:02d}_{task}' / 'best_model')
            cached = root / 'accuracy_matrix.json'
            cached.write_text(json.dumps({
                'complete': True, 'tasks': ['AGNEWS', 'SST2'],
                'accuracy': [[0.4, 0.5], [0.8, 0.5], [0.6, 0.9]],
                'adapter_dirs': [None, str(root / '01_AGNEWS/best_model'), str(root / '02_SST2/best_model')],
                'evaluation': {'max_input_length': 128, 'max_new_tokens': 8},
            }))
            args = drift.build_parser().parse_args([
                '--run_dir', tmp, '--tasks', 'AGNEWS,SST2', '--accuracy_matrix', str(cached),
                '--probe_samples', '3', '--grad_samples', '2', '--batch_size', '2',
                '--layers', 'all', '--max_input_length', '128',
            ])
            real_gradient = drift.mean_lora_gradient
            def checked_gradient(model, *a, **kw):
                weights = {name: p.detach().clone() for name, p in model.named_parameters()}
                flags = {name: p.requires_grad for name, p in model.named_parameters()}
                vector = real_gradient(model, *a, **kw)
                for name, p in model.named_parameters():
                    self.assertTrue(torch.equal(weights[name], p))
                    self.assertEqual(flags[name], p.requires_grad)
                    self.assertIsNone(p.grad)
                return vector
            with patch.object(drift, 'build_eval_model', return_value=(base, tokenizer, 'cpu', 'base')) as build, \
                 patch.object(drift.heatmap, 'load_dataset', side_effect=lambda *a, **kw: probes[Path(kw['data_files']).parent.name]), \
                 patch.object(drift, 'mean_lora_gradient', side_effect=checked_gradient) as gradient:
                payload = drift.run_evaluation(args)
            build.assert_called_once()
            self.assertEqual(gradient.call_count, 2)
            self.assertTrue(payload['complete'])
            self.assertEqual(len(payload['transitions']), 1)
            row = payload['transitions'][0]
            self.assertEqual(row['transition'], 'AGNEWS -> SST2')
            self.assertEqual((row['before_stage'], row['after_stage']), (1, 2))
            self.assertAlmostEqual(row['forgetting_pp'], 20.0)
            self.assertEqual(set(row['layers']), {'0', '1'})
            self.assertTrue(-1 <= row['grad_cosine'] <= 1)
            self.assertTrue(-1 <= row['lora_cosine'] <= 1)
            self.assertTrue(0 <= row['cka'] <= 1)
            self.assertGreaterEqual(row['func_drift'], 0)
            output = root / 'evaluation_drift'
            for name in ('transition_metrics.json', 'transition_metrics.csv', 'layer_metrics.csv', 'probe_manifest.json', 'transition_metrics.png', 'transition_metrics.pdf'):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(json.loads((output / 'transition_metrics.json').read_text()), payload)
            matrices = json.loads((output / 'model_drift_matrices.json').read_text())
            self.assertEqual(np.array(matrices['cka_vs_base']).shape, (3, 2))
            for value in matrices['cka_vs_base'][0]:
                self.assertAlmostEqual(value, 1.0)
            self.assertEqual(matrices['symmetric_kl_vs_base'][0], [0.0, 0.0])
            self.assertTrue(matrices['complete'])
            self.assertTrue((output / 'cka_vs_base_heatmap.png').is_file())
            adjacent = json.loads((output / 'adjacent_task_metrics.json').read_text())
            self.assertTrue(adjacent['complete'])
            comparisons = adjacent['comparisons']
            self.assertEqual([(row['after_stage'], row['evaluated_task'], row['task_role']) for row in comparisons],
                             [(1, 'AGNEWS', 'current'), (2, 'AGNEWS', 'previous'), (2, 'SST2', 'current')])
            first, previous, current = comparisons
            self.assertAlmostEqual(first['learning_gain_pp'], 40)
            self.assertIsNone(first['grad_cosine'])
            self.assertIsNone(first['lora_cosine'])
            self.assertIsNone(first['forgetting_pp'])
            self.assertEqual((previous['before_stage'], previous['after_stage']), (1, 2))
            self.assertEqual((current['before_stage'], current['after_stage']), (1, 2))
            self.assertAlmostEqual(previous['accuracy_change_pp'], -20)
            self.assertAlmostEqual(previous['forgetting_pp'], 20)
            self.assertIsNone(previous['learning_gain_pp'])
            self.assertAlmostEqual(current['accuracy_change_pp'], 40)
            self.assertAlmostEqual(current['learning_gain_pp'], 40)
            self.assertIsNone(current['forgetting_pp'])
            self.assertEqual(previous['grad_cosine'], current['grad_cosine'])
            self.assertEqual(previous['lora_cosine'], current['lora_cosine'])
            self.assertEqual(previous['cka'], payload['transitions'][0]['cka'])
            for name in ('adjacent_task_metrics.csv', 'adjacent_layer_metrics.csv', 'adjacent_task_metrics.png', 'adjacent_task_metrics.pdf'):
                self.assertTrue((output / name).is_file(), name)
            manifest = json.loads((output / 'probe_manifest.json').read_text())
            self.assertEqual(manifest['AGNEWS']['indices'], [0, 1, 2])

    def test_reject_incomplete_or_incompatible_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'matrix.json'
            args = drift.build_parser().parse_args(['--run_dir', tmp, '--accuracy_matrix', str(path)])
            for value in (
                {'complete': False, 'tasks': ['AGNEWS', 'SST2']},
                {'complete': True, 'tasks': ['AGNEWS', 'SST2'], 'accuracy': [[0.2, 0.3]]},
                {'complete': True, 'tasks': ['AGNEWS', 'SST2'], 'accuracy': [[0.2, 0.3]] * 3, 'evaluation': {'max_input_length': 20, 'max_new_tokens': 8}},
            ):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    drift.load_accuracy(args, ['AGNEWS', 'SST2'], root)


if __name__ == '__main__':
    unittest.main()

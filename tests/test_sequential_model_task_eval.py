"""Check full model/task coverage, pair semantics and resumable probe inference."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation import sequential_model_task_eval as full
from evaluation.drift_metrics import distribution_drift, linear_cka, representation_cosine_drift


class AllModelMetricsTest(unittest.TestCase):
    def test_all_pairs_include_unseen_tasks_and_match_direct_metrics(self):
        generator = torch.Generator().manual_seed(11)
        tasks = ['AGNEWS', 'SST2']
        models = full.model_mapping(Path('/tmp/run'), tasks, [Path('/tmp/a'), Path('/tmp/b')])
        outputs = [{task: {'features': {0: torch.randn(5, 4, generator=generator), 1: torch.randn(5, 4, generator=generator)},
                           'log_probs': torch.randn(5, 7, generator=generator).log_softmax(-1)}
                    for task in tasks} for _ in models]
        # Degenerate layers must stay null, including their diagonal.
        for model in outputs:
            model['AGNEWS']['features'][0] = torch.ones(5, 4)
        accuracy = np.array([[0.3, 0.5], [0.8, 0.2], [0.6, 0.9]])
        matrices, layer_matrices, model_rows, pair_rows, layer_rows = full.compare_all_models(outputs, tasks, [0, 1], accuracy, models)
        self.assertEqual(len(model_rows), 6)
        self.assertEqual(len(pair_rows), 6)
        self.assertEqual(len(layer_rows), 12)
        self.assertEqual([(row['model_a'], row['model_b'], row['evaluated_task']) for row in pair_rows],
                         [(a, b, task) for task in tasks for a, b in (('M0', 'M1'), ('M0', 'M2'), ('M1', 'M2'))])
        unseen = next(row for row in model_rows if row['model'] == 'M1' and row['evaluated_task'] == 'SST2')
        self.assertFalse(unseen['task_seen_in_training'])
        self.assertIsNone(unseen['forgetting_since_task_learned_pp'])
        self.assertAlmostEqual(unseen['test_accuracy_change_vs_previous_pp'], -30)
        learned = next(row for row in model_rows if row['model'] == 'M2' and row['evaluated_task'] == 'AGNEWS')
        self.assertAlmostEqual(learned['forgetting_since_task_learned_pp'], 20)
        for task in tasks:
            for key in full.METRICS:
                self.assertTrue(np.allclose(matrices[key][task], np.array(matrices[key][task]).T))
            self.assertTrue(np.allclose(np.diag(matrices['cka'][task]), 1))
            self.assertTrue(np.allclose(np.diag(matrices['rep_drift'][task]), 0, atol=1e-12))
            self.assertTrue(np.allclose(np.diag(matrices['symmetric_kl'][task]), 0))
        self.assertTrue(all(value is None for row in layer_matrices['0']['AGNEWS'] for value in row))
        row = next(row for row in pair_rows if (row['model_a'], row['model_b'], row['evaluated_task']) == ('M1', 'M2', 'SST2'))
        first, second = outputs[1]['SST2'], outputs[2]['SST2']
        kl = distribution_drift(first['log_probs'], second['log_probs'])
        self.assertAlmostEqual(row['cka'], linear_cka(first['features'][1], second['features'][1]))
        self.assertAlmostEqual(row['rep_drift'], representation_cosine_drift(first['features'][1], second['features'][1]))
        self.assertAlmostEqual(row['kl_a_to_b'], kl['kl_forward'])
        self.assertAlmostEqual(row['kl_b_to_a'], kl['kl_reverse'])

    def test_task_reference_uses_the_model_just_trained_on_each_task(self):
        tasks = ['AGNEWS', 'SST2']
        models = full.model_mapping(Path('/tmp/run'), tasks, [Path('/tmp/a'), Path('/tmp/b')])
        matrix = [[1., .2, .3], [.2, 1., .8], [.3, .8, 1.]]
        payload = {
            'tasks': tasks, 'models': models, 'summary_layer': 1, 'resolved_layers': [0, 1],
            'accuracy': [[.3, .5], [.8, .2], [.6, .9]],
            'pairwise_matrices': {key: {task: matrix for task in tasks} for key in full.METRICS},
            'layer_cka_matrices': {'0': {task: [[None] * 3 for _ in models] for task in tasks},
                                   '1': {task: matrix for task in tasks}},
            'model_task_results': [{'model': model['model'], 'evaluated_task': task} for model in models for task in tasks],
        }
        rows, layers = full.add_task_learned_reference(payload)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(layers), 12)
        lookup = {(r['model'], r['evaluated_task']): r for r in rows}
        self.assertEqual({r['evaluated_task']: r['reference_model'] for r in payload['task_learned_references']},
                         {'AGNEWS': 'M1', 'SST2': 'M2'})
        agnews = lookup['M2', 'AGNEWS']
        self.assertEqual(agnews['reference_adapter_dir'], str(Path('/tmp/a').resolve()))
        self.assertEqual(agnews['cka_vs_task_learned'], .8)  # M2 vs M1, rather than M2 vs M0 (.3).
        self.assertAlmostEqual(agnews['test_accuracy_change_vs_task_learned_pp'], -20)
        self.assertAlmostEqual(agnews['forgetting_since_task_learned_pp'], 20)
        before = lookup['M1', 'SST2']
        self.assertEqual(before['comparison_phase'], 'before_task_training')
        self.assertIsNone(before['forgetting_since_task_learned_pp'])
        self.assertAlmostEqual(before['test_accuracy_change_vs_task_learned_pp'], -70)
        for task, reference in (('AGNEWS', 'M1'), ('SST2', 'M2')):
            self.assertEqual(lookup[reference, task]['comparison_phase'], 'task_just_learned')
            self.assertEqual(lookup[reference, task]['test_accuracy_change_vs_task_learned_pp'], 0)
            self.assertEqual(lookup[reference, task]['cka_vs_task_learned'], 1)
        self.assertTrue(all(row['cka_vs_task_learned'] is None for row in layers if row['layer'] == 0))
        self.assertEqual(next(r for r in payload['model_task_results'] if (r['model'], r['evaluated_task']) == ('M2', 'AGNEWS'))['cka_vs_task_learned'], .8)

    def test_tiny_pipeline_and_resume_without_loading_model(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'Llama-3.1-8B-Instruct'), local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        torch.manual_seed(7)
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            pad_token_id=tokenizer.pad_token_id, max_position_embeddings=128,
        ))
        adapter = get_peft_model(copy.deepcopy(base), LoraConfig(
            r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        probes = {
            'AGNEWS': [{'text': text, 'label': i % 4} for i, text in enumerate(['A world news story.', 'Sports team wins.', 'A business report.'])],
            'SST2': [{'sentence': text, 'label': i % 2} for i, text in enumerate(['A good movie.', 'Terrible acting.', 'An interesting film.'])],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = ['AGNEWS', 'SST2']
            (root / 'sequential_results.json').write_text(json.dumps({'tasks': tasks}))
            for index, task in enumerate(tasks, 1):
                for name, parameter in adapter.named_parameters():
                    if 'lora_B' in name:
                        parameter.data.fill_(index * 0.03)
                adapter.save_pretrained(root / f'{index:02d}_{task}' / 'best_model')
            accuracy_path = root / 'accuracy_matrix.json'
            accuracy_path.write_text(json.dumps({
                'complete': True, 'tasks': tasks, 'accuracy': [[0.4, 0.5], [0.8, 0.5], [0.6, 0.9]],
                'adapter_dirs': [None, str(root / '01_AGNEWS/best_model'), str(root / '02_SST2/best_model')],
                'evaluation': {'max_input_length': 128, 'max_new_tokens': 8},
            }))
            args = full.build_parser().parse_args([
                '--run_dir', str(root), '--accuracy_matrix', str(accuracy_path), '--probe_samples', '3',
                '--batch_size', '2', '--layers', '0,1', '--max_input_length', '128', '--cpu_threads', '1'])
            with patch.object(full, 'build_eval_model', return_value=(base, tokenizer, 'cpu', 'base')) as build, \
                 patch.object(full.heatmap, 'load_dataset', side_effect=lambda *a, **kw: probes[Path(kw['data_files']).parent.name]), \
                 patch.object(full.drift, 'collect_outputs', wraps=full.drift.collect_outputs) as collect:
                payload = full.run_evaluation(args)
            build.assert_called_once()
            self.assertEqual(collect.call_count, 6)
            self.assertTrue(payload['complete'])
            out = root / 'evaluation_all_models'
            self.assertEqual(len(list((out / 'probe_cache').glob('*.pt'))), 6)
            for name in ('all_model_task_metrics.json', 'all_model_task_metrics.csv', 'all_model_pair_metrics.csv',
                         'all_model_pair_layer_metrics.csv', 'all_model_task_report.pdf', 'all_model_task_overview.png',
                         'pairwise_AGNEWS.png', 'pairwise_SST2.png', 'all_model_layerwise_cka.png', 'analysis.md',
                         'task_learned_reference_overview.png', 'task_learned_reference_metrics.csv',
                         'task_learned_reference_layerwise_cka.png'):
                self.assertTrue((out / name).is_file(), name)
            with patch.object(full, 'build_eval_model', side_effect=AssertionError('Resume must not reload GPU model')), \
                 patch.object(full.heatmap, 'load_dataset', side_effect=lambda *a, **kw: probes[Path(kw['data_files']).parent.name]), \
                 patch.object(full, 'plot_report'):
                resumed = full.run_evaluation(args)
            self.assertEqual(resumed['pairwise_matrices'], payload['pairwise_matrices'])
            report_args = copy.deepcopy(args)
            report_args.report_only = True
            with patch.object(full, 'build_eval_model', side_effect=AssertionError('Reports must not load models')), \
                 patch.object(full, 'collect_cached_outputs', side_effect=AssertionError('Reports must use existing metrics')), \
                 patch.object(full, 'plot_report'):
                report = full.run_evaluation(report_args)
            self.assertEqual(report['pairwise_matrices'], payload['pairwise_matrices'])
            self.assertEqual(len(report['task_learned_reference_results']), 6)

            # A preprocessing change must never silently reuse incompatible tensors.
            args.pooling = 'mean'
            with patch.object(full.heatmap, 'load_dataset', side_effect=lambda *a, **kw: probes[Path(kw['data_files']).parent.name]):
                with self.assertRaisesRegex(ValueError, 'cache metadata differs'):
                    full.run_evaluation(args)


if __name__ == '__main__':
    unittest.main()

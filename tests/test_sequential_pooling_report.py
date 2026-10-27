"""A pooling comparison must hold probes/model identity and references fixed."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from transformers import BatchEncoding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.sequential_drift_eval import collect_outputs
from evaluation.sequential_pooling_report import validate_pooling_inputs, join_pooling_results


class PoolingReportTest(unittest.TestCase):
    def test_masked_mean_excludes_padding_and_does_not_change_logits(self):
        class Toy(torch.nn.Module):
            def forward(self, input_ids, attention_mask, **kwargs):
                hidden = torch.stack((input_ids.float(), input_ids.float().square()), -1)
                return SimpleNamespace(hidden_states=tuple(hidden * (layer + 1) for layer in range(33)),
                                       logits=torch.zeros(len(input_ids), 1, 8))
        class Tokenizer:
            def __call__(self, prompts, **kwargs):
                return BatchEncoding({'input_ids': torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]]),
                                      'attention_mask': torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])})
        args = SimpleNamespace(pooling='mean', batch_size=2, max_input_length=128)
        task = SimpleNamespace(format_prompt=lambda row, tokenizer, include_answer: row)
        mean = collect_outputs(Toy(), Tokenizer(), ['short', 'long'], task, [16, 32], args, 'cpu')
        args.pooling = 'last'
        last = collect_outputs(Toy(), Tokenizer(), ['short', 'long'], task, [16, 32], args, 'cpu')
        for layer in (16, 32):
            self.assertTrue(torch.equal(mean['features'][layer], torch.tensor([[1.5, 2.5], [2.5, 7.5]]) * (layer + 1)))
            self.assertTrue(torch.equal(last['features'][layer], torch.tensor([[2., 4.], [4., 16.]]) * (layer + 1)))
        self.assertTrue(torch.equal(mean['log_probs'], last['log_probs']))

    def test_alignment_and_reference_aware_join(self):
        last = {'complete': True, 'run_dir': '/tmp/run', 'tasks': ['A', 'B'], 'models': [{'model': 'M0'}],
                'accuracy': [[.4, .5]], 'settings': {'pooling': 'last', 'max_input_length': 128,
                                                   'max_new_tokens': 8, 'seed': 1, 'batch_size': 2},
                'selected_representation_layers': [16, 32], 'selected_layer_results': []}
        for layer in (16, 32):
            for task, reference in (('A', 'M1'), ('B', 'M2')):
                last['selected_layer_results'].append({'layer': layer, 'evaluated_task': task, 'model': 'M4',
                    'reference_kind': 'task_learned', 'reference_model': reference, 'adapter_dir': '/tmp/m4',
                    'reference_adapter_dir': '/tmp/' + reference, 'validation_probe_count': 128,
                    'cka': .6 if layer == 16 else .8, 'representation_drift': .2})
        mean = copy.deepcopy(last)
        mean['settings']['pooling'] = 'mean'
        for row in mean['selected_layer_results']:
            row['cka'] += .1
            row['representation_drift'] = .1
        mean['selected_layer_results'].reverse()
        probes = {'A': {'indices': [1, 4, 6]}, 'B': {'indices': [2, 3, 7]}}
        validate_pooling_inputs(last, mean, probes, probes)
        joined = join_pooling_results(last, mean)
        self.assertEqual(len(joined), 4)
        for row in joined:
            self.assertAlmostEqual(row['cka_change_mean_minus_last'], .1)
            self.assertAlmostEqual(row['representation_drift_change_mean_minus_last'], -.1)
            self.assertEqual(row['reference_model'], 'M1' if row['evaluated_task'] == 'A' else 'M2')
        with self.assertRaisesRegex(ValueError, 'identical validation'):
            validate_pooling_inputs(last, mean, probes, {'A': {'indices': [1, 4, 8]}})
        altered = copy.deepcopy(mean)
        altered['models'][0]['model'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'models'):
            validate_pooling_inputs(last, altered, probes, probes)
        altered = copy.deepcopy(mean)
        altered['selected_layer_results'][0]['reference_model'] = 'M0'
        with self.assertRaisesRegex(ValueError, 'cover the same'):
            join_pooling_results(last, altered)
        altered = copy.deepcopy(mean)
        altered['selected_representation_layers'] = [32]
        with self.assertRaisesRegex(ValueError, 'layers 16 and 32'):
            validate_pooling_inputs(last, altered, probes, probes)


if __name__ == '__main__':
    unittest.main()

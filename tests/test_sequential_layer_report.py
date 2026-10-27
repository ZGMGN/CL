"""Layer selection, explicit references and degenerate self-distances."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation import sequential_model_task_eval as full
from evaluation import sequential_layer_report as report
from evaluation.drift_metrics import linear_cka, representation_cosine_drift


class LayerReportTest(unittest.TestCase):
    def test_saved_layer_rows_match_direct_metrics_for_both_references(self):
        generator = torch.Generator().manual_seed(29)
        tasks = ['AGNEWS', 'SST2']
        models = full.model_mapping(Path('/tmp/run'), tasks, [Path('/tmp/a'), Path('/tmp/b')])
        outputs = [{task: {'features': {layer: torch.randn(6, 5, generator=generator) for layer in (16, 32)},
                           'log_probs': torch.randn(6, 8, generator=generator).log_softmax(-1)}
                    for task in tasks} for _ in models]
        outputs[0]['AGNEWS']['features'][16].zero_()
        accuracy = np.array([[.3, .4], [.8, .4], [.7, .9]])
        pairs, cka, model_rows, pair_rows, layer_rows = full.compare_all_models(outputs, tasks, [16, 32], accuracy, models)
        payload = {'tasks': tasks, 'models': models, 'resolved_layers': [16, 32], 'summary_layer': 32,
                   'settings': {'pooling': 'last'}, 'pairwise_matrices': pairs, 'layer_cka_matrices': cka,
                   'model_task_results': model_rows}
        def self_distance(stage, task, layer):
            features = outputs[stage][task]['features'][layer]
            return representation_cosine_drift(features, features)
        matrices, rows = report.derive_layer_results(payload, layer_rows, self_distance)
        self.assertEqual(set(matrices), {'16', '32'})
        self.assertEqual(len(rows), 24)
        self.assertIsNone(matrices['16']['rep_drift']['AGNEWS'][0][0])
        for row in rows:
            first, second = int(row['model'][1:]), int(row['reference_model'][1:])
            task, layer = row['evaluated_task'], row['layer']
            x, y = outputs[first][task]['features'][layer], outputs[second][task]['features'][layer]
            expected_cka, expected_drift = linear_cka(x, y), representation_cosine_drift(x, y)
            if expected_cka is None:
                self.assertIsNone(row['cka'])
            else:
                self.assertAlmostEqual(row['cka'], expected_cka)
            if expected_drift is None:
                self.assertIsNone(row['representation_drift'])
            else:
                self.assertAlmostEqual(row['representation_drift'], expected_drift)
            expected_reference = 'M0' if row['reference_kind'] == 'base' else f'M{tasks.index(task) + 1}'
            self.assertEqual(row['reference_model'], expected_reference)
        # An incomplete CSV must not silently turn missing results into N/A.
        with self.assertRaisesRegex(ValueError, 'missing model-pair'):
            report.derive_layer_results(payload, layer_rows[1:], self_distance)
        with self.assertRaisesRegex(ValueError, 'cached features'):
            report.derive_layer_results(payload, layer_rows)
        payload['resolved_layers'] = [32]
        self.assertEqual(report.derive_layer_results(payload, layer_rows), ({}, []))


if __name__ == '__main__':
    unittest.main()

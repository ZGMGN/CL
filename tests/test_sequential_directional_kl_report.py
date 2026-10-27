"""Directional KL must preserve P/Q order, including pre-reference models."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluation.sequential_directional_kl_report import directional_kl_matrices, add_directional_kl_results


class DirectionalKLTest(unittest.TestCase):
    def payload(self):
        tasks = ['A', 'B']
        models = [{'model': f'M{i}', 'stage': i, 'adapter_dir': None if i == 0 else f'/tmp/m{i}'} for i in range(3)]
        values = [(0, 1, .2, .8), (0, 2, .4, 1.2), (1, 2, .6, 1.8)]
        pairs = []
        symmetric = {task: [[0.] * 3 for _ in models] for task in tasks}
        for offset, task in enumerate(tasks, 1):
            for a, b, forward, reverse in values:
                forward, reverse = forward * offset, reverse * offset
                symmetric[task][a][b] = symmetric[task][b][a] = (forward + reverse) / 2
                pairs.append({'evaluated_task': task, 'model_a': f'M{a}', 'model_b': f'M{b}',
                              'kl_a_to_b': forward, 'kl_b_to_a': reverse})
        return {'tasks': tasks, 'models': models, 'pairwise_results': pairs,
                'pairwise_matrices': {'symmetric_kl': symmetric},
                'model_task_results': [{'model': m['model'], 'evaluated_task': task, 'validation_probe_count': 128} for m in models for task in tasks],
                'task_learned_reference_results': [{'model': m['model'], 'evaluated_task': task, 'reference_stage': index + 1} for m in models for index, task in enumerate(tasks)]}

    def test_asymmetric_matrix_and_earlier_model_reference_direction(self):
        payload = self.payload()
        rows = add_directional_kl_results(payload)
        matrix = payload['directional_kl_matrices']
        self.assertEqual(matrix['A'][0][1], .2)
        self.assertEqual(matrix['A'][1][0], .8)
        self.assertEqual([matrix['A'][i][i] for i in range(3)], [0., 0., 0.])
        self.assertEqual(len(rows), 12)
        lookup = {(r['evaluated_task'], r['reference_kind'], r['model']): r for r in rows}
        # Task B's reference is M2, but compared M1 is earlier: use saved B->A direction.
        before = lookup['B', 'task_learned', 'M1']
        self.assertEqual(before['reference_model'], 'M2')
        self.assertEqual(before['kl_reference_to_model'], 3.6)
        self.assertEqual(before['kl_model_to_reference'], 1.2)
        self.assertEqual(before['model_relative_to_reference'], 'earlier_model')
        after = lookup['A', 'task_learned', 'M2']
        self.assertEqual(after['kl_reference_to_model'], .6)
        self.assertEqual(after['kl_model_to_reference'], 1.8)
        for row in rows:
            self.assertAlmostEqual((row['kl_reference_to_model'] + row['kl_model_to_reference']) / 2, row['symmetric_kl'])
        model_row = next(r for r in payload['model_task_results'] if (r['model'], r['evaluated_task']) == ('M1', 'B'))
        self.assertEqual(model_row['kl_task_learned_to_model'], 3.6)
        self.assertEqual(model_row['kl_model_to_task_learned'], 1.2)

    def test_missing_duplicate_or_inconsistent_saved_values_are_rejected(self):
        payload = self.payload()
        broken = copy.deepcopy(payload); broken['pairwise_results'].pop()
        with self.assertRaisesRegex(ValueError, 'missing'):
            directional_kl_matrices(broken)
        broken = copy.deepcopy(payload); broken['pairwise_results'].append(broken['pairwise_results'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            directional_kl_matrices(broken)
        broken = copy.deepcopy(payload); broken['pairwise_results'][0]['kl_a_to_b'] = .9
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            directional_kl_matrices(broken)
        broken = copy.deepcopy(payload); broken['pairwise_results'][0]['kl_a_to_b'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            directional_kl_matrices(broken)


if __name__ == '__main__':
    unittest.main()

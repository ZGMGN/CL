"""Shared accuracy and task re-export checks."""
import importlib
import unittest
import numpy as np
from src.metrics import compute_accuracy


class MetricsTest(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(compute_accuracy(['A','B'],['A','A']),0.5)
        self.assertEqual(compute_accuracy(['A','B'],['A','B']),1.0)
        self.assertEqual(compute_accuracy(['B'],['A']),0.0)

    def test_empty_inputs(self):
        self.assertEqual(compute_accuracy([],[]),0.0)
        self.assertEqual(compute_accuracy(np.array([]),np.array([])),0.0)

    def test_mismatched_lengths(self):
        for preds,labels in ((['A'],[]),([],['A']),(['A','B'],['A'])):
            with self.subTest(preds=preds,labels=labels), self.assertRaises(ValueError):
                compute_accuracy(preds,labels)

    def test_tasks_reexport_shared_implementation(self):
        for task in ('AGNEWS','QQP','SST2','AMAZON','YELP','DBPEDIA','YAHOO'):
            with self.subTest(task=task):
                module=importlib.import_module(f'src.tasks.{task}')
                self.assertEqual(module.compute_accuracy.__code__.co_filename, compute_accuracy.__code__.co_filename)


if __name__=='__main__':
    unittest.main()

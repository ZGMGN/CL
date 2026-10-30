"""Check class conditioning, KL directions and combined pooling collection."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evaluation.sequential_classwise_eval import balanced_probes, collect_outputs, per_example_kl
from evaluation.drift_metrics import distribution_drift, linear_cka


def test_balanced_probes_use_identical_indices_and_all_classes():
    first, _, manifest = balanced_probes(64, 1)
    second, _, repeated = balanced_probes(64, 1)
    assert manifest == repeated and first == second
    assert sum(len(rows) for rows in first.values()) == 64 * 33
    for task, item in manifest.items():
        assert set(item['class_counts'].values()) == {64}
        assert len(item['indices']) == len(set(item['indices']))


def test_per_example_kl_matches_known_distribution_and_class_partition():
    before = torch.tensor([[.8, .2], [.3, .7], [.6, .4], [.5, .5]], dtype=torch.double).log()
    after = torch.tensor([[.4, .6], [.8, .2], [.2, .8], [.5, .5]], dtype=torch.double).log()
    forward, reverse = per_example_kl(before, after)
    expected = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=before), torch.distributions.Categorical(logits=after))
    assert torch.allclose(forward, expected)
    assert torch.allclose(per_example_kl(after, before)[0], reverse)
    whole = distribution_drift(before, after)
    assert abs(float(forward.mean()) - whole['kl_forward']) < 1e-12
    assert abs(float((forward[:2].mean() + forward[2:].mean()) / 2) - whole['kl_forward']) < 1e-12


def test_class_cka_uses_class_samples_and_invariant_geometry():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(64, 12, generator=generator, dtype=torch.double)
    q, _ = torch.linalg.qr(torch.randn(12, 12, generator=generator, dtype=torch.double))
    y = 3 * x @ q + 4
    assert abs(linear_cka(x, y) - 1) < 1e-12
    shuffled = y[torch.randperm(64, generator=generator)]
    assert linear_cka(x, shuffled) < .5


def test_single_forward_collects_masked_mean_and_last_at_both_layers():
    class Inputs(dict):
        def to(self, device):
            return self
    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            return Inputs(input_ids=torch.zeros(2, 3, dtype=torch.long), attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]))
    class Model:
        def eval(self):
            return self
        def __call__(self, **kwargs):
            assert kwargs['logits_to_keep'] == 1 and not kwargs['use_cache']
            hidden = torch.tensor([[[999., 999.], [2., 4.], [4., 8.]], [[1., 2.], [3., 6.], [5., 10.]]])
            states = tuple(hidden + index for index in range(33))
            return SimpleNamespace(logits=torch.tensor([[[1., 2.]], [[3., 1.]]]), hidden_states=states)
    module = SimpleNamespace(format_prompt=lambda row, tokenizer, include_answer: 'question')
    result = collect_outputs(Model(), Tokenizer(), [{}, {}], module, 2, 1536, 'cpu')
    assert result['features']['last'][16].shape == (2, 2)
    assert torch.equal(result['features']['last'][16], torch.tensor([[20., 24.], [21., 26.]]))
    assert torch.equal(result['features']['mean'][32], torch.tensor([[35., 38.], [35., 38.]]))
    assert torch.allclose(result['log_probs'].exp().sum(1), torch.ones(2))


def test_pair_table_reverses_kl_for_earlier_model_controls_and_masks_forgetting():
    import contextlib
    import io
    import tempfile
    from evaluation.sequential_classwise_eval import compare_order
    tasks = ['AGNEWS', 'AMAZON', 'DBPEDIA', 'YAHOO']
    classes = dict(zip(tasks, [4, 5, 14, 10]))
    manifest = {task: {'labels': [chr(65 + label) for label in range(count) for _ in range(64)],
                       'samples_per_class': 64} for task, count in classes.items()}
    scores = {task: [{chr(65 + label): 20 + 10 * stage for label in range(count)} for stage in range(5)]
              for task, count in classes.items()}
    with tempfile.TemporaryDirectory() as temporary:
        paths = []
        for stage in range(5):
            stage_paths = {}
            for task in tasks:
                n = len(manifest[task]['labels'])
                generator = torch.Generator().manual_seed(10)
                x = torch.randn(n, 8, generator=generator)
                p = (stage + 1) / 7
                probabilities = torch.tensor([p, 1 - p]).log().repeat(n, 1)
                result = {'features': {pool: {layer: x + stage for layer in (16, 32)} for pool in ('last', 'mean')},
                          'log_probs': probabilities}
                path = Path(temporary) / f'{stage}_{task}.pt'
                torch.save(result, path)
                stage_paths[task] = str(path)
            paths.append(stage_paths)
        with contextlib.redirect_stdout(io.StringIO()):
            pairs, references, adjacent = compare_order({'tasks': tasks}, paths, manifest, scores)
        assert len(pairs) == 1320 and len(references) == 660
        control = next(row for row in references if row['evaluated_task'] == 'YAHOO' and row['current_model'] == 'M0')
        expected = distribution_drift(torch.tensor([[5 / 7, 2 / 7]]).log(), torch.tensor([[1 / 7, 6 / 7]]).log())
        assert abs(control['kl_reference_to_current'] - expected['kl_forward']) < 1e-6
        assert abs(control['kl_current_to_reference'] - expected['kl_reverse']) < 1e-6
        assert control['test_forgetting_pp'] is None
        retained = next(row for row in references if row['evaluated_task'] == 'AGNEWS' and row['current_model'] == 'M4')
        assert retained['test_forgetting_pp'] == 30
        assert all(int(row['model_a'][1:]) >= tasks.index(row['evaluated_task']) + 1 for row in adjacent)

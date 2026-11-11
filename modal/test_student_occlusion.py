"""Small synthetic checks of mask provenance and the censored CTC objective."""
import ast
import itertools
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from student_occlusion import censored_ctc_loss, imitation_loss, output_masks, spec_augment


def collapse(path):
    return tuple(p for i, p in enumerate(path) if p != 0 and (i == 0 or p != path[i-1]))


class ObservationCensoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_augmentation_preserves_original_draws_and_output(self):
        # The original recipe fixes the draw order, including zero-width masks.
        def original(feats, rng):
            feats = feats.clone()
            T = len(feats)
            for _ in range(2):
                f = int(rng.integers(0, 28))
                f0 = int(rng.integers(0, 80-f+1))
                feats[:, f0:f0+f] = 0
            for _ in range(int(rng.integers(0, 11))):
                t = int(rng.integers(0, max(1, int(T*.05))+1))
                t0 = int(rng.integers(0, max(1, T-t)))
                feats[t0:t0+t] = 0
            return feats

        for length in (1, 45, 1000):
            x = torch.arange(length*80).reshape(length, 80).float()+1
            for seed in range(10):
                a, b = np.random.default_rng(seed), np.random.default_rng(seed)
                expected = original(x, a)
                actual, mask = spec_augment(x, b, return_mask=True)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(a.bit_generator.state, b.bit_generator.state)
                self.assertTrue((actual[mask] == 0).all())
                self.assertTrue((x > 0).all())

    def test_mask_union_is_provenance_not_inferred_silence(self):
        class Draws:
            values = iter((3, 4, 2, 4, 4, 0, 12))
            def integers(self, *bounds):
                value = next(self.values)
                return value

        x = torch.zeros((20, 80))  # genuine all-zero observations carry no mask
        _, mask = spec_augment(x, Draws(), n_freq=0, n_time=3, max_t_frac=.3, return_mask=True)
        self.assertEqual(mask.nonzero().flatten().tolist(), list(range(2, 8)))
        _, frequency = spec_augment(x, np.random.default_rng(2), n_time=0, return_mask=True)
        self.assertFalse(frequency.any())

    def test_output_cell_requires_full_mask_and_never_extends_into_padding(self):
        source = torch.zeros(14, dtype=torch.bool)
        source[2:6] = True
        source[7] = True
        source[10:] = True
        hidden, touched = output_masks(source, 4)
        self.assertEqual(hidden.tolist(), [True, False, True, False])
        self.assertEqual(touched.tolist(), [True, True, True, False])
        with self.assertRaises(ValueError):
            output_masks(source.float(), 3)

    def test_censored_ctc_matches_exhaustive_paths_including_repeated_phones(self):
        probabilities = torch.tensor([[.2, .6, .2], [.4, .1, .5], [.3, .4, .3], [.2, .3, .5]], dtype=torch.float64)
        hidden = torch.tensor([False, True, True, False])
        marginal = probabilities.clone()
        marginal[hidden] = 1/3
        for target in ((1,), (1, 2), (1, 1), (2, 1, 2)):
            total = sum(math.prod(float(marginal[t, p]) for t, p in enumerate(path))
                        for path in itertools.product(range(3), repeat=4) if collapse(path) == target)
            loss = censored_ctc_loss(probabilities.log(), torch.tensor(target), hidden)
            self.assertAlmostEqual(float(loss), -math.log(total), places=10)

    def test_hidden_rows_have_no_gradient_but_progress_is_not_blank_forced(self):
        logits = torch.tensor([[0., 2., -1.], [-2., 0., 3.], [0., -1., 2.]], requires_grad=True)
        hidden = torch.tensor([False, True, False])
        loss = censored_ctc_loss(logits.log_softmax(-1), torch.tensor([1, 2]), hidden)
        loss.backward()
        torch.testing.assert_close(logits.grad[1], torch.zeros(3), rtol=0, atol=0)
        self.assertGreater(float(logits.grad[[0, 2]].abs().sum()), 0)
        # A repeated identity requires the ordinary separating-blank topology.
        impossible = censored_ctc_loss(torch.zeros((2, 3)).log_softmax(-1), torch.tensor([1, 1]), torch.ones(2, dtype=torch.bool))
        self.assertTrue(torch.isinf(impossible))
        # A one-frame masked observation can consume a transcript phone.
        one = censored_ctc_loss(torch.zeros((1, 3)).log_softmax(-1), torch.tensor([2]), torch.ones(1, dtype=torch.bool))
        self.assertAlmostEqual(float(one), math.log(3), places=6)

    def test_no_mask_is_exact_ordinary_ctc_and_impossible_targets_fail(self):
        lp = torch.randn((8, 4), generator=torch.Generator().manual_seed(7)).log_softmax(-1)
        target = torch.tensor([1, 2, 2, 3])
        ordinary = torch.nn.functional.ctc_loss(lp[:, None], target, torch.tensor([8]), torch.tensor([4]), reduction='sum')
        actual = censored_ctc_loss(lp, target, torch.zeros(8, dtype=torch.bool))
        torch.testing.assert_close(actual, ordinary, rtol=0, atol=0)
        for target in (torch.tensor([0]), torch.tensor([4])):
            with self.assertRaises(ValueError):
                censored_ctc_loss(lp, target, torch.zeros(8, dtype=torch.bool))

    def test_imitation_excludes_hidden_rows_and_retains_real_pauses(self):
        logits = torch.zeros((3, 3), requires_grad=True)
        teacher = torch.tensor([[.98, .01, .01], [.01, .98, .01], [.01, .01, .98]]).log()
        imitation_loss(logits.log_softmax(-1), teacher, torch.tensor([False, True, False])).backward()
        torch.testing.assert_close(logits.grad[1], torch.zeros(3), rtol=0, atol=0)
        self.assertLess(float(logits.grad[0, 0]), 0)  # genuine blank/pause target
        self.assertLess(float(logits.grad[2, 2]), 0)
        logits.grad = None
        imitation_loss(logits.log_softmax(-1), teacher, torch.ones(3, dtype=torch.bool)).backward()
        self.assertFalse(logits.grad.any())

    def test_collation_carries_mask_with_false_padding(self):
        # Loading the Modal app would need credentials/image setup; isolate its
        # actual pure collation function from the module-level deployment code.
        module = ast.parse((Path(__file__).parent/'student_train.py').read_text())
        function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'collate')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'student_train.py', 'exec'), namespace)
        items = [dict(i=i, feats=torch.zeros((length, 80)), time_mask=torch.ones(length, dtype=torch.bool),
                      target=torch.tensor([1]), top_ids=torch.ones((length, 5), dtype=torch.long),
                      top_p=torch.ones((length, 5))) for i, length in enumerate((2, 4))]
        batch = namespace['collate'](items)
        self.assertEqual(batch['time_mask'].tolist(), [[True, True, False, False], [True]*4])
        self.assertEqual(batch['feat_lens'].tolist(), [2, 4])


if __name__ == '__main__':
    unittest.main()

"""Training constraints and complete-CTC semantics, independent of practice data."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import torch
import student_fixed_effect_phone as training


class FixedEffectTrainingTest(unittest.TestCase):
    def test_double_centering_preserves_gradients_and_has_no_blank_effect(self):
        raw = torch.randn(5, 4, requires_grad=True)
        effect = training.recording_offsets(raw)
        torch.testing.assert_close(effect[:, 0], torch.zeros(5))
        torch.testing.assert_close(effect[:, 1:].sum(1), torch.zeros(5), atol=1e-6, rtol=0)
        torch.testing.assert_close(effect.mean(0), torch.zeros(5), atol=1e-6, rtol=0)
        (effect[0, 1] + effect.square().mean()).backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad.abs().sum()), 0)

    def test_centering_removes_nonidentifiable_recording_and_phone_intercepts(self):
        raw = torch.randn(5, 4)
        shifted = raw + torch.randn(5, 1) + torch.randn(1, 4)
        torch.testing.assert_close(training.recording_offsets(raw), training.recording_offsets(shifted),
                                   atol=1e-6, rtol=0)

    def test_ctc_matches_enumerated_complete_paths_and_latent_gradients(self):
        logits = torch.tensor([[.2, -.3, .5], [.1, .6, -.2], [-.4, .7, .1]], requires_grad=True)
        for ids in ([1], [1, 2], [1, 1]):
            probability = 0.
            lp = logits.detach().log_softmax(-1)
            for path in itertools.product(range(3), repeat=3):
                collapsed = [p for i, p in enumerate(path) if p and (i == 0 or p != path[i-1])]
                if collapsed == ids:
                    probability += float(torch.exp(sum(lp[t, p] for t, p in enumerate(path))))
            self.assertAlmostEqual(training.sequence_score(lp, ids), np.log(probability), places=5)
        loss = training.transcript_loss(logits, [1, 2])
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum(1) > 0).all())
        # At zero effects, pooled and treatment likelihoods are exactly identical.
        effect = training.recording_offsets(torch.zeros(2, 2))[0]
        torch.testing.assert_close(training.transcript_loss(logits+effect, [1, 2]), loss)

    def test_speaker_groups_and_release_exclusion(self):
        rows = [dict(speaker=s) for s in ('a', 'a', 'b', 'c')]
        folds = training.speaker_folds(rows, ['a', 'b', 'c', 'd'])
        self.assertEqual(set(folds.values()), {0, 1, 2})
        r = dict(source='help-train', split='train', excluded=False, speaker='heldout', id='take',
                 kind='none', features=np.zeros((4, 32)).tolist(),
                 logProbs=np.log(np.full((4, 3), 1/3)).tolist(), pronunciation=[1, 2])
        with self.assertRaisesRegex(ValueError, 'provenance'):
            training.prepare([r], {'heldout'}, 3)
        for field, value in [('split', 'test'), ('source', 'practice'), ('excluded', True)]:
            with self.assertRaises(ValueError):
                training.prepare([dict(r, **{field: value})], set(), 3)

    def test_matched_control_and_export_discard_recording_offsets(self):
        rng = np.random.default_rng(1)
        rows = [dict(features=rng.normal(size=(6, 32)).astype('float32'), pronunciation=[1, 2],
                     logProbs=np.log(np.full((6, 3), 1/3)).astype('float32')) for _ in range(3)]
        with patch.object(training, 'CHECKPOINTS', (2,)), patch.object(training, 'WARMUP_EPOCHS', 1):
            first, center, scale = training.fit(rows, 3, .01)
            second, _, _ = training.fit(rows, 3, 1.)
        for a, b in zip(first[2][0].parameters(), second[2][0].parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertGreater(first[2][2]['rms'], 0)
        exported = training.serialize(first[2][1])
        self.assertEqual(set(exported), {'hidden', 'hiddenBias', 'output', 'outputBias'})
        # Shared inference works on a new recording without an identity or offset.
        self.assertEqual(tuple(training.outputs(first[2][1], rows[0]['features'], center, scale).shape), (6, 3))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

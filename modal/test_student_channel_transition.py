"""Small deterministic tests; run on the supervisor alongside SDK tests."""
import unittest
import numpy as np
import torch
from student_channel_transition import (channel_views, training_take, speaker_folds,
    occupancies, temporal_features, alternatives, pronunciation_consistency, ctc_nll)


class ChannelTransitionTests(unittest.TestCase):
    def test_crossfade_keeps_clock_and_fixed_endpoint_acoustics(self):
        sr = 16000
        x = .2*np.sin(np.arange(3*sr)*2*np.pi*200/sr)+.1*np.sin(np.arange(3*sr)*2*np.pi*4200/sr)
        (a,t,b), geometry = channel_views(x, np.random.default_rng(1729))
        self.assertEqual(a.shape, x.shape)
        self.assertEqual(t.shape, x.shape)
        start = int((geometry['centerSeconds']-geometry['widthSeconds']/2)*sr)
        end = int((geometry['centerSeconds']+geometry['widthSeconds']/2)*sr)+1
        np.testing.assert_array_equal(t[:start], a[:start])
        np.testing.assert_array_equal(t[end:], b[end:])
        self.assertGreater(np.max(np.abs(t-a)), .001)
        self.assertGreater(np.max(np.abs(t-b)), .001)
        again, g = channel_views(x, np.random.default_rng(1729))
        self.assertEqual(g, geometry)
        for left,right in zip((a,t,b), again):
            np.testing.assert_array_equal(left, right)

    def test_training_excludes_recordings_and_held_speakers(self):
        r = dict(split='train', speaker='one', kind='none', events=[], non_errors=[])
        self.assertTrue(training_take(r, set()))
        self.assertFalse(training_take(r, {'one'}))
        self.assertFalse(training_take(dict(r, split='eval'), set()))
        self.assertFalse(training_take(dict(r, excluded='unscorable'), set()))
        self.assertFalse(training_take(dict(r, kind='substituted_word'), set()))
        with self.assertRaises(ValueError):
            training_take(dict(r, non_errors=[dict(kind='repeat')]), set())
        folds = speaker_folds([dict(speaker=s) for s in ['a','a','b','c','d']])
        self.assertEqual(set(folds.values()), {0,1,2})

    def test_temporal_windows_share_real_neighbors_and_replicate_only_record_edges(self):
        a = np.arange(4*32, dtype='float32').reshape(4,32)
        windows = temporal_features(a, np.zeros(32, dtype='float32'), np.ones(32, dtype='float32'))
        self.assertEqual(windows.dtype, np.float32)
        np.testing.assert_array_equal(windows[2], np.concatenate([a[1],a[2],a[3]]))
        np.testing.assert_array_equal(windows[0], np.concatenate([a[0],a[0],a[1]]))

    def test_complete_pronunciation_consistency_is_symmetric_and_backpropagates(self):
        torch.manual_seed(1729)
        x = torch.randn(10,4, requires_grad=True)
        y = torch.randn(10,4, requires_grad=True)
        candidates = [[1,2], [1,3], [2,2], [3,1]]
        a, b = x.log_softmax(-1), y.log_softmax(-1)
        loss = pronunciation_consistency(a,b,candidates)
        self.assertAlmostEqual(loss.item(), pronunciation_consistency(b,a,candidates).item(), places=6)
        self.assertLess(pronunciation_consistency(a,a,candidates).abs().item(), 1e-7)
        loss.backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertGreater(y.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(ctc_nll(a, candidates)).all())

    def test_soft_transcript_occupancies_preserve_blank_and_repeated_phones(self):
        lp = torch.tensor([[2.,0.,-1.],[0.,2.,-1.],[2.,0.,-1.],[0.,2.,-1.],[2.,0.,-1.]]).log_softmax(-1).numpy()
        targets = occupancies(lp,[1,1])
        np.testing.assert_allclose(targets.sum(1), 1., atol=1e-6)
        self.assertGreater(targets[:,0].sum(), 0)
        self.assertIsNone(occupancies(lp[:2],[1,1]))
        candidates = alternatives([1,1],3,np.random.default_rng(1729))
        self.assertEqual(candidates[0], [1,1])
        self.assertTrue(all(len(c)==2 for c in candidates))
        self.assertTrue(all(sum(a!=b for a,b in zip(c,[1,1]))==1 for c in candidates[1:]))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

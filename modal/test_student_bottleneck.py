"""Readout mathematics and data isolation; run with the Modal supervisor."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from student_bottleneck import cache_rows, fit, gaussian_kl, grouped_split, quadrature


class BottleneckTest(unittest.TestCase):
    def test_shared_prior_kl(self):
        mu = torch.tensor([[0., 0.], [2., -1.]], requires_grad=True)
        logvar = torch.zeros_like(mu, requires_grad=True)
        kl = gaussian_kl(mu, logvar)
        torch.testing.assert_close(kl, torch.tensor([0., 2.5]))
        kl.sum().backward()
        torch.testing.assert_close(mu.grad, mu.detach())
        torch.testing.assert_close(logvar.grad, torch.zeros_like(logvar))
        self.assertGreater(float(gaussian_kl(torch.zeros(2), torch.ones(2))), 0)

    def test_gaussian_quadrature(self):
        mu, std = torch.tensor([2., -3.]), torch.tensor([.5, 2.])
        points = quadrature(mu, std)
        torch.testing.assert_close(points.mean(0), mu)
        residual = points-mu
        torch.testing.assert_close(residual.T @ residual/len(points), torch.diag(std.square()))
        torch.testing.assert_close(points, quadrature(mu, std))

    def test_whole_speakers_are_partitioned(self):
        rows = [{'speaker':str(i//2)} for i in range(12)]
        for fold in range(3):
            train, held = grouped_split(rows, fold)
            self.assertFalse({r['speaker'] for r in train} & {r['speaker'] for r in held})
            self.assertEqual(len(train)+len(held), len(rows))
        with self.assertRaises(ValueError):
            grouped_split([{'speaker':'one'}], 0)

    def test_canonical_labels_exclude_edits_and_held_out_data(self):
        row = dict(source='help-train', split='train', excluded=False, speaker='training',
                   kind='none', pronunciation='ab', features=[[0., 1.], [1., 0.]],
                   controlPronunciations=['aa'])
        meta = dict(width=2, tokens='ab')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cache.jsonl'
            for override in ({'split':'eval'}, {'events':[{}]}, {'non_errors':[{}]},
                             {'kind':'wrong_word'}, {'excluded':True}, {'speaker':''}):
                path.write_text(json.dumps({**row, **override})+'\n')
                with self.assertRaises(ValueError):
                    cache_rows(path, meta)
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual(cache_rows(path, meta)[0]['pronunciation'], 'ab')

    def test_fit_normalizes_only_fitting_speakers_and_matches_capacity(self):
        torch.set_num_threads(1)
        rows = [dict(speaker=f'train{i}', pronunciation='ab', x=np.tile(np.array([1., 2.], dtype=np.float32), (5, 1)),
                     controlPronunciations=['aa']) for i in range(2)]
        held = [dict(speaker='held', pronunciation='ab', x=np.tile(np.array([50., 100.], dtype=np.float32), (5, 1)),
                     controlPronunciations=['aa'])]
        args = SimpleNamespace(latent=2, seed=1729, recording_batch=2, checkpoints=[1])
        meta = dict(schema=1, width=2, inventory='test', tokens='ab', featureOutput='features', primarySha256='parent')
        model, report = fit(rows, meta, .01, args, lambda: None, held)
        self.assertEqual(model['center'], [1., 2.])
        self.assertEqual(report['validationSpeakers'], ['held'])
        counts = [sum(np.asarray(v).size for v in model[k].values())
                  for k in ('bottleneck','deterministic','fixedNoise')]
        self.assertEqual(len(set(counts)), 1)
        self.assertEqual(counts[0], report['parametersPerArm'])


if __name__ == '__main__':
    unittest.main()

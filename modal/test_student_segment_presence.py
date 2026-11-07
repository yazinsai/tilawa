"""Exhaustive boundary and weak-label normalization checks."""
import math
import unittest
import torch
from student_segment_presence import advance, partition, SegmentScorer
from student_segment_features import eligible


class SegmentTests(unittest.TestCase):
    def test_only_training_none_and_single_scripted_omissions(self):
        row = dict(split='train', scored=True, kind='none', ayahs=['example'], events=[], non_errors=[], unsupported=[])
        self.assertTrue(eligible(row))
        for change in [dict(split='eval'), dict(scored=False), dict(excluded='unusable'), dict(kind='repeat'), dict(non_errors=[{}])]:
            self.assertFalse(eligible({**row, **change}))
        omission = {**row, 'kind':'skip_word', 'events':[dict(errorClass='omitted_word',words=1)]}
        self.assertTrue(eligible(omission))
        self.assertFalse(eligible({**omission,'events':[dict(errorClass='omitted_word',words=2)]}))

    def test_every_boundary_once(self):
        emissions = torch.tensor([[.2, -.1, .5, .3], [-.3, .1, .4, -.2]], dtype=torch.float64)
        stay = torch.log(torch.tensor([.6, .7], dtype=torch.float64))
        exit = torch.log(torch.tensor([.4, .3], dtype=torch.float64))
        paths = []
        for boundary in range(1, 4):
            paths.append(emissions[0,:boundary].sum() + emissions[1,boundary:].sum() +
                         (boundary-1)*stay[0] + (3-boundary)*stay[1] + exit.sum())
        self.assertAlmostEqual(float(partition(emissions, stay, exit, torch.tensor(-1.))),
                               float(torch.logsumexp(torch.stack(paths), 0)), places=10)

    def test_skip_consumes_no_frames_and_union_has_one_skip(self):
        e = torch.tensor([[.1,.5,.2],[-.2,.3,-.1]], dtype=torch.float64)
        stay, exit = torch.log(torch.tensor([.6,.7], dtype=torch.float64)), torch.log(torch.tensor([.4,.3], dtype=torch.float64))
        skip = torch.tensor(-.8, dtype=torch.float64)
        for w in range(2):
            other = 1-w
            expected = e[other].sum() + 2*stay[other] + exit[other] + skip - math.log(2)
            self.assertAlmostEqual(float(partition(e,stay,exit,skip,omitted=w)),float(expected),places=10)
        scores = [partition(e,stay,exit,skip,omitted=w) for w in [None,0,1]]
        self.assertAlmostEqual(float(partition(e,stay,exit,skip,union=True)),float(torch.logsumexp(torch.stack(scores),0)),places=10)

    def test_weak_loss_has_finite_gradients(self):
        model = SegmentScorer(5,7)
        x,q,lens = torch.randn(12,5),torch.randn(3,7),torch.tensor([2.,3.,2.])
        loss = model.loss(x,q,lens,1)
        loss.backward()
        self.assertGreaterEqual(float(loss),0)
        for p in model.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_more_words_than_frames_cannot_read(self):
        score = partition(torch.zeros(3,2),torch.full((3,),-.5),torch.full((3,),-.9),torch.tensor(0.))
        self.assertLess(float(score),-1e20)


if __name__ == '__main__':
    unittest.main()

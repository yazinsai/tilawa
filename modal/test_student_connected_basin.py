import copy
import unittest

from student_connected_basin import quadratic, make_head, speaker_partitions, RECIPE
from student_basin_features import has_clean_transcript, select_original_records


class AuthorizationTests(unittest.TestCase):
    def test_repeat_and_planted_scripts_never_get_canonical_ctc(self):
        self.assertTrue(has_clean_transcript(dict(kind='none', split='train')))
        for fields in (dict(kind='substituted_word'), dict(kind='none', non_errors=[{}]),
                       dict(kind='none', events=[{}]), dict(kind='none', scored=False)):
            self.assertFalse(has_clean_transcript(fields))

    def test_original_exclusion_removes_whole_speaker(self):
        rows = [dict(shard='a', row=i, split='train', cluster=c, speaker=str(c), bucket='clean-match',
                     n_samples=16000, key=str(i)) for i, c in enumerate([1, 1, 2, -1])]
        kept, banned = select_original_records(rows, {('a', '0')}, 100, 30)
        self.assertEqual([r['cluster'] for r in kept], [2])
        self.assertIn('1', banned)

    def test_speaker_partition_never_splits_views_or_sources(self):
        rows = [dict(source=s, speaker=f'{s}/{i}') for s in ('help', 'original') for i in range(10)]
        mapping = speaker_partitions(rows+rows)
        self.assertEqual(len(mapping), 20)
        self.assertEqual(sum(p == 'validation' for p in mapping.values()), 4)


class ConnectionTests(unittest.TestCase):
    def test_bending_curve_learns_without_moving_endpoints(self):
        import torch
        left = {'x': torch.tensor([1., -2.], requires_grad=True)}
        right = {'x': torch.tensor([3., 4.], requires_grad=True)}
        ctrl = torch.nn.Parameter(torch.tensor([7., -3.]))
        a, b = left['x'].clone(), right['x'].clone()
        torch.testing.assert_close(quadratic(left, {'x': ctrl}, right, 0)['x'], a)
        torch.testing.assert_close(quadratic(left, {'x': ctrl}, right, 1)['x'], b)
        midpoint = quadratic(left, {'x': ctrl}, right, .5)['x']
        self.assertFalse(torch.allclose(midpoint, (a+b)/2))
        opt = torch.optim.SGD([ctrl], lr=.1)
        opt.zero_grad(); midpoint.square().sum().backward(); opt.step()
        self.assertIsNone(left['x'].grad); self.assertIsNone(right['x'].grad)
        torch.testing.assert_close(left['x'], a); torch.testing.assert_close(right['x'], b)
        self.assertLess(quadratic(left, {'x': ctrl}, right, .5)['x'].square().sum(), midpoint.square().sum())

    def test_independent_nonlinear_endpoints_and_functional_curve(self):
        import torch
        from torch.func import functional_call
        torch.set_num_threads(1)
        a = make_head(3, 4, RECIPE['seed']); b = make_head(3, 4, RECIPE['seed']+1)
        control = copy.deepcopy(a)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())))
        x = torch.tensor([[1., 2., -1.], [-1., 0., 2.]])
        for t, head in ((0., a), (1., b)):
            p = quadratic(a.state_dict(), dict(control.named_parameters()), b.state_dict(), t)
            torch.testing.assert_close(functional_call(control, p, x), head(x))


if __name__ == '__main__':
    unittest.main()

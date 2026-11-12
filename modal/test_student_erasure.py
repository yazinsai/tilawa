"""Exact tiny-path/gradient checks for emission erasure; no acoustic datasets."""
import itertools
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student_erasure import (ARMS, head_objective, neutralize, select_exclusion,
                             sequence_nll, viterbi_states)
from student_erasure_train import RECIPE, choose_checkpoint, fit, read_plan
from student_head_refit import ROOT, has_clean_transcript, write_json, digest


def collapse(path):
    return [v for i, v in enumerate(path) if v and (i == 0 or v != path[i-1])]


def exhaustive_nll(lp, target):
    scores = [sum(lp[t, label] for t, label in enumerate(path))
              for path in itertools.product(range(lp.shape[1]), repeat=len(lp))
              if collapse(path) == list(target)]
    return -torch.logsumexp(torch.stack(scores), 0) / len(lp)


class Erasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_forward_sum_and_logit_gradient_match_every_ctc_path(self):
        for target in ([1], [1, 2], [1, 1], []):
            logits = torch.randn(5, 3, generator=torch.Generator().manual_seed(71),
                                 dtype=torch.float64, requires_grad=True)
            for frame in (None, 0, 2, 4):
                lp = logits.log_softmax(-1)
                if frame is not None:
                    lp = neutralize(lp, frame)
                exact = exhaustive_nll(lp, target)
                actual = sequence_nll(lp, torch.tensor(target, dtype=torch.long))
                torch.testing.assert_close(actual, exact, atol=1e-12, rtol=1e-12)
                a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
                b = torch.autograd.grad(exact, logits, retain_graph=True)[0]
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
                if frame is not None:
                    self.assertTrue(torch.equal(a[frame], torch.zeros_like(a[frame])))

    def test_neutral_row_preserves_surrounding_rows_and_time(self):
        lp = torch.randn(6, 4).log_softmax(-1).requires_grad_()
        altered = neutralize(lp, 2)
        self.assertEqual(altered.shape, lp.shape)
        self.assertTrue(torch.equal(altered[[0, 1, 3, 4, 5]], lp[[0, 1, 3, 4, 5]]))
        torch.testing.assert_close(altered[2], torch.full((4,), -math.log(4)))
        altered.sum().backward()
        self.assertTrue(torch.equal(lp.grad[2], torch.zeros(4)))
        self.assertTrue(torch.equal(lp.grad[3], torch.ones(4)))

    def test_viterbi_path_is_legal_and_has_maximum_joint_score(self):
        rng = np.random.default_rng(56)
        for target in ([1], [1, 2], [1, 1], []):
            lp = rng.normal(size=(5, 3))
            states = viterbi_states(lp, target)
            labels = [0]
            for phone in target:
                labels.extend([phone, 0])
            path = [labels[s] for s in states]
            self.assertEqual(collapse(path), list(target))
            best = max(sum(lp[t, label] for t, label in enumerate(p))
                       for p in itertools.product(range(3), repeat=5) if collapse(p) == list(target))
            self.assertAlmostEqual(sum(lp[t, label] for t, label in enumerate(path)), best)

    def test_repeated_consonants_choose_their_own_peak(self):
        lp = torch.tensor([[.01, .90, .09], [.02, .80, .18], [.98, .01, .01],
                           [.02, .70, .28], [.01, .95, .04]]).log()
        y = torch.tensor([1, 1])
        seen = set()
        for seed in range(30):
            altered, report = select_exclusion(lp, y, 'با', np.random.default_rng(seed), 'strongest')
            self.assertEqual(report['frame'], [0, 4][report['occurrence']])
            self.assertEqual(report['skipped'], None)
            self.assertEqual(int((altered != lp).any(-1).sum()), 1)
            seen.add(report['occurrence'])
        self.assertEqual(seen, {0, 1})

    def test_controls_share_occurrence_draw_and_random_is_not_peak_selection(self):
        lp = torch.tensor([[.1, .7, .2], [.8, .1, .1], [.1, .2, .7]]).log()
        y = torch.tensor([1, 2])
        frames = set()
        for seed in range(20):
            reports = {}
            for arm in ARMS:
                got, reports[arm] = select_exclusion(lp, y, 'بت', np.random.default_rng(seed), arm)
                again, report = select_exclusion(lp, y, 'بت', np.random.default_rng(seed), arm)
                self.assertEqual(report, reports[arm])
                self.assertTrue(torch.equal(got, again))
                if arm == 'ordinary':
                    self.assertTrue(torch.equal(got, lp))
            self.assertEqual(len({r['occurrence'] for r in reports.values()}), 1)
            frames.add(reports['random']['frame'])
        self.assertEqual(frames, {0, 1, 2})

    def test_infeasible_transcripts_are_rejected_without_zero_infinity(self):
        self.assertIsNone(viterbi_states(np.zeros((2, 3)), [1, 1]))
        lp = torch.tensor([[0., -torch.inf, -torch.inf]] * 4)
        self.assertIsNone(viterbi_states(lp.numpy(), [1]))
        with self.assertRaisesRegex(ValueError, 'mathematically infeasible'):
            select_exclusion(lp, torch.tensor([1]), 'بت', np.random.default_rng(0), 'strongest')
        # A neutral row keeps a timestep. It must not insert the blank separator
        # between identical consonants or treat erasure as deletion of time.
        minimum = torch.tensor([[.1, .8, .1], [.8, .1, .1], [.1, .8, .1]]).log()
        self.assertIsNotNone(viterbi_states(neutralize(minimum, 0).numpy(), [1, 1]))
        self.assertTrue(torch.isfinite(sequence_nll(neutralize(minimum, 0), torch.tensor([1, 1]))))

    def test_no_consonant_retains_ordinary_loss(self):
        lp = torch.randn(5, 3).log_softmax(-1)
        for arm in ARMS:
            loss, report = head_objective(lp, torch.tensor([2]), lp, 'با', np.random.default_rng(0), arm)
            torch.testing.assert_close(loss, 2 * sequence_nll(lp, torch.tensor([2])))
            self.assertEqual(report['skipped'], 'no consonant')

    def test_every_arm_computes_two_complete_sequence_likelihoods(self):
        lp = torch.randn(5, 3).log_softmax(-1)
        for arm in ARMS:
            with patch('student_erasure.sequence_nll', wraps=sequence_nll) as likelihood:
                head_objective(lp, torch.tensor([1]), lp, 'با', np.random.default_rng(0), arm)
                self.assertEqual(likelihood.call_count, 2)

    def test_erasure_trains_surviving_acoustic_support(self):
        logits = torch.tensor([[0., 5., -2.], [2., -1., -2.], [2., -1., -2.]], requires_grad=True)
        erased = neutralize(logits.log_softmax(-1), 0)
        loss = sequence_nll(erased, torch.tensor([1]))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0], torch.zeros(3)))
        # Descent raises consonant support in the unmasked acoustic frames.
        self.assertLess(float(logits.grad[1, 1]), 0)
        self.assertLess(float(logits.grad[2, 1]), 0)


class TrainingContract(unittest.TestCase):
    def test_mistakes_repeats_and_unscored_takes_cannot_supply_canonical_text(self):
        self.assertTrue(has_clean_transcript(dict(kind='none', events=[], non_errors=[])))
        for r in (dict(kind='skipped_word'), dict(kind='none', events=[{}]),
                  dict(kind='none', non_errors=[dict(kind='repeat')]), dict(kind='none', scored=False),
                  dict(kind='none', unsupported=['unknown'])):
            self.assertFalse(has_clean_transcript(r))

    def test_checkpoint_ignores_control_scores_and_pairs_epoch(self):
        history = [dict(epoch=0, validation=dict(strongest=2., ordinary=0., random=0.)),
                   dict(epoch=1, validation=dict(strongest=1., ordinary=4., random=4.)),
                   dict(epoch=2, validation=dict(strongest=1., ordinary=0., random=0.))]
        self.assertEqual(choose_checkpoint(history)['epoch'], 1)
        with self.assertRaises(ValueError):
            choose_checkpoint([dict(epoch=0, validation=dict(strongest=float('nan')))])

    def test_cache_provenance_and_speaker_partitions_are_required(self):
        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            out = Path(tmp)
            for file in ('initial-head.npz', 'encoder-features.onnx'):
                (out/file).write_bytes(b'fixture identity')
            records, views = [], []
            for partition in ('train', 'validation'):
                file = out/f'{partition}.npz'
                file.write_bytes(b'fixture cached features')
                records.append(dict(key=partition, speaker=partition, partition=partition))
                views.append(dict(file=str(file), sha256=digest(file), recording=partition,
                                  speaker=partition, partition=partition))
            plan = dict(recipe=RECIPE, records=records, views=views, excluded_speakers=[],
                        initial_head_sha256=digest(out/'initial-head.npz'),
                        feature_graph_sha256=digest(out/'encoder-features.onnx'))
            write_json(out/'plan.json', plan)
            self.assertEqual(read_plan(out)['views'], views)
            views[1]['speaker'] = 'train'
            write_json(out/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                read_plan(out)
            records[1]['speaker'] = 'train'
            write_json(out/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'separation'):
                read_plan(out)
            views[1]['speaker'] = records[1]['speaker'] = 'validation'
            plan['excluded_speakers'] = ['train']
            write_json(out/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'separation'):
                read_plan(out)

    def test_paired_refit_uses_equal_updates_and_freezes_input_features(self):
        # Tiny complete runner exercise, not a mirrored mock of its loop.
        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            out = Path(tmp)
            plan = dict(quantized=False, views=[])
            for partition in ('train', 'validation'):
                for view in (0, 1):
                    file = out/f'{partition}-{view}.npz'
                    x = np.asarray([[1., 0.], [0., 1.], [1., 1.]], dtype='float32')
                    baseline = np.full((3, 43), -math.log(43), dtype='float32')
                    np.savez(file, features=x, baseline=baseline, target=np.asarray([2], dtype='int64'))
                    plan['views'].append(dict(file=str(file), partition=partition, view=view,
                                              speaker=partition, recording=partition))
            np.savez(out/'initial-head.npz', weight=np.zeros((43, 2), dtype='float32'),
                     bias=np.zeros(43, dtype='float32'))
            write_json(out/'plan.json', plan)
            before = {v['file']: digest(v['file']) for v in plan['views']}
            with patch('student_erasure_train.read_plan', return_value=plan), patch.dict(RECIPE, epochs=2, batch=1):
                fit(out)
            import json
            selection = json.loads((out/'selection.json').read_text())
            self.assertEqual(set(selection['arms']), set(ARMS))
            self.assertEqual({a['evaluations'] for a in selection['audit'].values()}, {8})
            self.assertEqual([h['updates'] for h in selection['history']], [0, 2, 4])
            self.assertEqual(before, {file: digest(file) for file in before})
            for name, checkpoint in selection['arms'].items():
                self.assertEqual(Path(checkpoint['path']).name, f"{name}-{selection['selected']['epoch']}.npz")


if __name__ == '__main__':
    unittest.main()

"""Synthetic retention, provenance and paired-training checks; no practice data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from student_early_learning import (PredictionBank, ROOT, RECIPE, EXPERIMENT,
    compare, fit, read_plan, speaker_partitions, transcript_loss, preservation_loss)
from student_head_refit import digest, has_clean_transcript, select_original_records, write_json

TORCH = importlib.util.find_spec('torch') is not None


class Provenance(unittest.TestCase):
    def test_speakers_and_all_views_stay_in_one_partition(self):
        rows = [dict(source=source, speaker=source+'/'+str(i))
                for source in ('help', 'original') for i in range(10)]
        result = speaker_partitions(rows)
        self.assertEqual(result, speaker_partitions(list(reversed(rows))))
        for source in ('help', 'original'):
            self.assertEqual(sum(v == 'validation' for k, v in result.items() if k.startswith(source+'/')), 2)
        with self.assertRaises(ValueError):
            speaker_partitions(rows[:2])

    def test_scripts_and_repeats_never_become_performed_transcripts(self):
        self.assertTrue(has_clean_transcript(dict(kind='none', events=[], non_errors=[])))
        for label in (dict(kind='substituted_word'), dict(kind='none', events=[{}]),
                      dict(kind='none', non_errors=[{}]), dict(kind='none', unsupported=[{}])):
            self.assertFalse(has_clean_transcript(label))

    def test_one_reserved_original_take_excludes_its_entire_speaker(self):
        rows = [dict(key=str(i), speaker='s'+str(i//2), split='train', shard='a', row=i,
                     cluster=i//2, bucket='clean-match', n_samples=16000) for i in range(6)]
        rows[4]['split'] = 'validation'
        admitted, banned = select_original_records(rows, {('a', '0')}, 10, 30)
        self.assertEqual({r['speaker'] for r in admitted}, {'s1'})
        self.assertEqual(banned, {'s0', 's2'})


@unittest.skipUnless(TORCH, 'Torch numerical checks run on Modal')
class Retention(unittest.TestCase):
    def test_history_is_detached_separate_per_view_and_cloned_per_arm(self):
        import torch

        bank = PredictionBank(.9)
        p = torch.tensor([[.2, .8]], requires_grad=True)
        bank.observe('recording-view-0', p)
        bank.observe('recording-view-1', torch.tensor([[.9, .1]]))
        p.data.fill_(.5)
        torch.testing.assert_close(bank.rows['recording-view-0'], torch.tensor([[.2, .8]]))
        self.assertFalse(bank.rows['recording-view-0'].requires_grad)
        second = copy.deepcopy(bank)
        second.observe('recording-view-0', torch.tensor([[.8, .2]]))
        torch.testing.assert_close(second.rows['recording-view-0'], torch.tensor([[.26, .74]]))
        self.assertEqual(bank.counts['recording-view-0'], 1)
        with self.assertRaises(ValueError):
            bank.observe('recording-view-0', torch.tensor([[.2, .8], [.2, .8]]))
        with self.assertRaises(ValueError):
            bank.penalty('unseen-validation-view', p.log())

    def test_regularizer_pushes_toward_past_prediction_without_target_gradient(self):
        import torch

        bank = PredictionBank(.9)
        history = torch.tensor([[.05, .9, .05]], requires_grad=True)
        bank.observe('example', history)
        logits = torch.tensor([[0., -2., 2.]], requires_grad=True)
        loss = bank.penalty('example', logits.log_softmax(-1))
        loss.backward()
        self.assertLess(float(logits.grad[0, 1]), 0.)
        self.assertGreater(float(logits.grad[0, 2]), 0.)
        self.assertIsNone(history.grad)
        step = logits.detach()-.2*logits.grad
        self.assertLess(float(bank.penalty('example', step.log_softmax(-1))), float(loss))

    def test_complete_ctc_keeps_repeated_articulations_and_mistake_preservation(self):
        import torch

        lp = torch.tensor([[.01, .99], [.99, .01], [.01, .99]]).log()
        target = torch.tensor([1, 1])
        self.assertLess(float(transcript_loss(lp, target)), .02)
        with self.assertRaises(ValueError):
            transcript_loss(lp[:2], target)
        with self.assertRaises(ValueError):
            transcript_loss(lp, torch.tensor([], dtype=torch.long))
        for invalid in (torch.tensor([0]), torch.tensor([2]), torch.tensor([-1]), torch.tensor([1.])):
            with self.assertRaises(ValueError):
                transcript_loss(lp, invalid)
        self.assertAlmostEqual(float(preservation_loss(lp, lp)), 0., places=6)
        self.assertGreater(float(preservation_loss(lp.flip(-1), lp)), 0.)

    def test_zero_regularization_reproduces_control_and_target_probe_is_separate(self):
        import numpy as np
        import torch

        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            out = Path(tmp)
            torch.manual_seed(31)
            head = torch.nn.Linear(2, 3)
            np.savez(out/'initial-head.npz', weight=head.weight.detach().numpy(), bias=head.bias.detach().numpy())
            (out/'encoder-features.onnx').write_bytes(b'synthetic encoder identity')
            views = []
            for i, (partition, supervised) in enumerate((('train', True), ('train', False), ('validation', True))):
                path = out/f'data-{i}.npz'
                np.savez(path, features=np.asarray([[1., 0.], [0., 1.], [1., 0.]], dtype='float32'),
                    baseline=np.log(np.full((3, 3), 1/3, dtype='float32')),
                    target=np.asarray([1], dtype='int64') if supervised else np.asarray([], dtype='int64'))
                views.append(dict(file=str(path), sha256=digest(path), partition=partition,
                    supervised=supervised, speaker=str(i), recording=str(i)))
            recipe = dict(RECIPE, epochs=3, warmups=[1], strengths=[0., .3], batch=1, preservation_batch=1)
            write_json(out/'plan.json', dict(id=EXPERIMENT, recipe=recipe, views=views,
                excluded_speakers=[], initial_head_sha256=digest(out/'initial-head.npz'),
                feature_graph_sha256=digest(out/'encoder-features.onnx')))
            with patch('student_early_learning.RECIPE', recipe):
                fit(out)
                # The no-retention arm must exactly reproduce both parameters
                # and its held-speaker score at every matched checkpoint.
                selected = json.loads((out/'selection.json').read_text())
                for row in selected['histories']['1'][1:]:
                    zero = next(r for r in row['retention'] if r['weight'] == 0.)
                    self.assertEqual(row['control_validation'], zero['validation'])
                    with np.load(row['control']['path']) as a, np.load(zero['checkpoint']['path']) as b:
                        for key in ('weight', 'bias'):
                            np.testing.assert_array_equal(a[key], b[key])
                before = digest(out/'selection.json')
                windows = out/'windows.json'
                write_json(windows, [dict(key='synthetic', features=views[0]['file'], from_frame=0, to_frame=3,
                    feature_graph_sha256=digest(out/'encoder-features.onnx'), written=[1], accused=[2])])
                compare(out, windows)
                self.assertEqual(before, digest(out/'selection.json'))
                self.assertIn('not engine-warning wins', json.loads((out/'premise.json').read_text())['limitation'])
                # Tampering cannot silently alter the frozen feature plan.
                (out/'data-0.npz').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    read_plan(out)


if __name__ == '__main__':
    unittest.main()

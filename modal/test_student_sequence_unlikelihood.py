"""Sequence-level loss, training provenance, and matched-arm controls."""
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from student_head_refit import ROOT, digest, reserved_speakers, write_json
from student_sequence_unlikelihood import (RECIPE, fit, read_mining,
                                          sequence_log_probability, sequence_unlikelihood)


def collapse(path):
    return tuple(p for i, p in enumerate(path) if p != 0 and (i == 0 or p != path[i-1]))


class SequenceLoss(unittest.TestCase):
    def test_ctc_matches_exhaustive_complete_paths_including_repetitions(self):
        import torch

        logits = torch.tensor([[.2, .9, -.1], [-.5, .7, .1], [.4, -.3, .8], [.1, .6, .3]], dtype=torch.float64)
        lp = logits.log_softmax(-1)
        for target in [[], [1], [1, 2], [2, 1], [1, 1]]:
            mass = sum(math.exp(sum(float(lp[i, p]) for i, p in enumerate(path)))
                for path in itertools.product(range(3), repeat=4) if collapse(path) == tuple(target))
            self.assertAlmostEqual(float(sequence_log_probability(lp, target).exp()), mass, places=12)
            self.assertAlmostEqual(float(sequence_unlikelihood(lp, [target])), -math.log1p(-mass), places=12)

    def test_duplicate_negatives_do_not_duplicate_penalties_and_impossible_paths_have_zero_gradient(self):
        import torch

        logits = torch.tensor([[.5, .2, .3], [.1, .6, .3]], requires_grad=True)
        lp = logits.log_softmax(-1)
        self.assertEqual(float(sequence_unlikelihood(lp, [[1, 2], [1, 2]])),
                         float(sequence_unlikelihood(lp, [[1, 2]])))
        loss = sequence_unlikelihood(lp, [[1, 1]])  # needs a separating blank: three frames
        self.assertEqual(float(loss), 0.)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_order_changes_loss_and_gradient_with_the_same_phone_classes(self):
        import torch

        logits = torch.tensor([[0., 3., -2.], [0., -2., 3.]], dtype=torch.float64, requires_grad=True)
        lp = logits.log_softmax(-1)
        first = sequence_unlikelihood(lp, [[1, 2]])
        second = sequence_unlikelihood(lp, [[2, 1]])
        self.assertGreater(float(first), float(second))
        a = torch.autograd.grad(first, logits, retain_graph=True)[0]
        b = torch.autograd.grad(second, logits)[0]
        self.assertFalse(torch.allclose(a, b))
        # Negative supervision lowers these classes only in their offending sequence positions.
        self.assertGreater(float(a[0, 1]), 0.)
        self.assertLess(float(a[1, 1]), 0.)

    def test_extreme_probabilities_are_stable_and_invalid_transcripts_fail(self):
        import torch

        for logits in [torch.tensor([[0., -1000.]], dtype=torch.float64),
                       torch.tensor([[-25., 0.]], dtype=torch.float64)]:
            logits.requires_grad_()
            loss = sequence_unlikelihood(logits.log_softmax(-1), [[1]])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())
        lp = torch.tensor([[.4, .6]], dtype=torch.float64).log()
        for target in [[0], [2], [-1]]:
            with self.assertRaises(ValueError):
                sequence_log_probability(lp, target)


class Provenance(unittest.TestCase):
    def test_source_and_whole_speaker_partitions_are_rechecked(self):
        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            root = Path(tmp)
            source = root/'help-train'
            source.mkdir()
            rows = [dict(id=k, speaker=k, split='train', kind='none', wav=k+'.wav', ayahs=['synthetic'])
                    for k in ('a', 'b', 'c')]
            rows += [dict(rows[0], id='outside', split='eval')]
            write_json(source/'labels.jsonl', rows[0])
            (source/'labels.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            reference = root/'packages/core/assets/ref-v2.json'
            reference.parent.mkdir(parents=True)
            reference.write_text('synthetic reference')
            admitted = rows[1:3]
            reserved = reserved_speakers(admitted)
            records = []
            for row in admitted:
                file = source/row['wav']
                file.write_bytes(b'synthetic '+row['id'].encode())
                feature = root/(row['id']+'.fbank')
                feature.write_bytes(b'synthetic features')
                records.append(dict(row, file=str(file), source_sha256=digest(file),
                    feature_file=str(feature), feature_sha256=digest(feature), supervised=True,
                    partition='validation' if row['speaker'] in reserved else 'train', negatives=[]))
            mining = root/'mining.json'
            value = dict(labels_root=str(source), labels_sha256=digest(source/'labels.jsonl'),
                         reference_sha256=digest(reference), reserved=sorted(reserved), excluded_speakers=['a'], records=records)
            write_json(mining, value)
            with patch.dict(os.environ, LAB_DATA=str(root)), patch('student_sequence_unlikelihood.ROOT', root):
                self.assertEqual(len(read_mining(mining)['records']), 2)
                records[0]['partition'] = 'outside'
                write_json(mining, value)
                with self.assertRaisesRegex(ValueError, 'partition'):
                    read_mining(mining)
                records[0]['partition'] = 'validation' if records[0]['speaker'] in reserved else 'train'
                records.append(dict(rows[0], partition='train'))
                write_json(mining, value)
                with self.assertRaisesRegex(ValueError, 'admitted takes'):
                    read_mining(mining)


class MatchedArms(unittest.TestCase):
    def test_positive_control_uses_the_same_records_and_updates_with_preservation(self):
        import numpy as np
        import torch
        import torch.nn.functional as F

        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            out = Path(tmp)
            recipe = dict(RECIPE, epochs=2, batch=2, lr=.01)
            weight = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype='float32')
            bias = np.zeros(3, dtype='float32')
            np.savez(out/'initial-head.npz', weight=weight, bias=bias)
            (out/'encoder-features.onnx').write_bytes(b'synthetic frozen features')
            mining = out/'mining.json'
            mining.write_text('{}')
            views, arrays = [], []
            for i, (partition, supervised) in enumerate([('train', True), ('train', False), ('validation', True)]):
                features = np.array([[2., 0.], [0., 2.], [0., 1.]], dtype='float32')
                baseline = torch.from_numpy(features@weight.T).log_softmax(-1).numpy()
                target = np.array([2, 1] if supervised else [], dtype='int64')
                file = out/f'{i}.npz'
                np.savez(file, features=features, baseline=baseline, target=target)
                views.append(dict(recording=str(i), speaker=partition, partition=partition, supervised=supervised,
                    file=str(file), sha256=digest(file),
                    negatives=[dict(from_frame=0, to_frame=3, target=[1, 2])] if supervised else []))
                arrays.append((torch.from_numpy(features), torch.from_numpy(baseline), torch.from_numpy(target)))
            write_json(out/'plan.json', dict(recipe=recipe, views=views, initial_head_sha256=digest(out/'initial-head.npz'),
                feature_graph_sha256=digest(out/'encoder-features.onnx'), mining_path=str(mining), mining_sha256=digest(mining)))
            with patch('student_sequence_unlikelihood.RECIPE', recipe):
                fit(out)
            selection = json.loads((out/'selection.json').read_text())
            self.assertTrue(selection['identical_updates'])
            self.assertTrue(selection['identical_recordings'])
            self.assertEqual([h['updates'] for h in selection['history']], [1, 2])
            # Independently reproduce the positive-only arm, including shipped
            # posterior preservation on the mistake take, without negative loss.
            h = torch.nn.Linear(2, 3)
            with torch.no_grad():
                h.weight.copy_(torch.from_numpy(weight))
                h.bias.copy_(torch.from_numpy(bias))
            for epoch in range(selection['selected']['epoch']):
                h.zero_grad()
                losses = []
                for features, baseline, target in arrays[:2]:
                    lp = h(features).log_softmax(-1)
                    loss = recipe['anchor']*F.kl_div(lp, baseline.exp(), reduction='sum')/len(lp)
                    if len(target):
                        loss -= sequence_log_probability(lp, target)/len(lp)
                    losses.append(loss)
                (sum(losses)/2).backward()
                with torch.no_grad():
                    for p in h.parameters():
                        p.add_(p.grad, alpha=-recipe['lr'])
            control = np.load(selection['arms']['positive']['path'])
            candidate = np.load(selection['arms']['unlikelihood']['path'])
            np.testing.assert_allclose(control['weight'], h.weight.detach().numpy(), atol=1e-7)
            self.assertFalse(np.allclose(control['weight'], candidate['weight']))


if __name__ == '__main__':
    unittest.main()

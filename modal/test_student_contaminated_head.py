"""Synthetic probability, supervision and matched-refit checks; no Quran audio."""
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import student_contaminated_head as experiment
from student_head_refit import ROOT, digest, has_clean_transcript, select_original_records, write_json


class Contamination(unittest.TestCase):
    def test_transcript_distribution_normalizes_over_finite_strings(self):
        # Unlike a length-normalized score, this distribution includes its
        # length/EOS mass. Sum all binary strings through length 12 and account
        # for the exact residual live-state probability.
        q = experiment.transcript_background([[1, 2], [2], [1]], 3)
        total = sum(math.exp(experiment.transcript_log_probability(q, y))
            for n in range(13) for y in itertools.product((1, 2), repeat=n))
        live = q[0, 1:].copy()
        for _ in range(12):
            live = live@q[1:, 1:]
        self.assertAlmostEqual(total+live.sum(), 1., places=12)
        self.assertTrue((q[:, 0] > 0).all())
        self.assertLess(total, 1.)
        with self.assertRaises(ValueError):
            experiment.transcript_background([[0]], 3)

    def test_contamination_mixture_has_recording_level_gradient(self):
        import torch

        for log_a in (-2., -30.):
            nll = torch.tensor(-log_a, dtype=torch.float64, requires_grad=True)
            log_q, pi = -10., .2
            loss = experiment.mixture_nll(nll, log_q, pi)
            probability = (1-pi)*math.exp(log_a)+pi*math.exp(log_q)
            self.assertAlmostEqual(float(loss.detach()), -math.log(probability), places=12)
            loss.backward()
            expected = (1-pi)*math.exp(log_a)/probability
            self.assertAlmostEqual(float(nll.grad), expected, places=12)
        self.assertGreater(float(experiment.acoustic_responsibility(-2., -10., .2)), .99)
        self.assertLess(float(experiment.acoustic_responsibility(-30., -10., .2)), 1e-7)

    def test_zero_contamination_is_exact_ordinary_ctc(self):
        import torch

        nll = torch.tensor(17., requires_grad=True)
        self.assertIs(experiment.mixture_nll(nll, -20., 0.), nll)
        experiment.mixture_nll(nll, -20., 0.).backward()
        self.assertEqual(float(nll.grad), 1.)

    def test_calibration_estimates_incompatibility_without_audio_labels(self):
        clean = experiment.estimate_contamination([-1.]*20, [-20.]*20)
        bad = experiment.estimate_contamination([-40.]*20, [-20.]*20)
        self.assertAlmostEqual(clean, 1/22, places=7)
        self.assertAlmostEqual(bad, 21/22, places=7)
        self.assertLess(clean, bad)
        with self.assertRaises(ValueError):
            experiment.estimate_contamination([], [])
        with self.assertRaises(ValueError):
            experiment.acoustic_responsibility(float('nan'), -2., .5)

    def test_partition_keeps_all_takes_and_sources_separate(self):
        rows = [dict(source=source, speaker=source+str(i))
            for source in ('original', 'help') for i in range(10) for _ in range(2)]
        partitions = experiment.speaker_partitions(rows)
        for source in ('original', 'help'):
            self.assertEqual({partitions[r['speaker']] for r in rows if r['source'] == source},
                             {'train', 'calibration', 'validation'})
        self.assertEqual(len(partitions), 20)

    def test_scripted_takes_never_admit_canonical_supervision(self):
        self.assertTrue(has_clean_transcript(dict(kind='none', split='train')))
        for change in ({'kind': 'substituted_word'}, {'events': [{'class': 'wrong_word'}]},
                       {'non_errors': [{'kind': 'repeat'}]}, {'unsupported': ['tajweed']}):
            self.assertFalse(has_clean_transcript(dict({'kind': 'none'}, **change)))

    def test_original_holdouts_exclude_entire_speaker_cluster_before_sampling(self):
        def recording(row, cluster, **change):
            return dict(dict(row=row, shard='training-shard', cluster=cluster,
                key='original/'+str(row), speaker='original/'+str(cluster), split='train',
                bucket='clean-match', n_samples=16000), **change)

        rows = [recording(0, 'gold'), recording(1, 'gold'),
                recording(2, 'reserved', split='val'), recording(3, 'reserved'),
                recording(4, 'allowed'), recording(5, -1), recording(6, None),
                recording(7, 'deviant', bucket='deviant'), recording(8, 'long', n_samples=32*16000)]
        selected, excluded = select_original_records(rows, {('training-shard', '0')}, 100, 30.)
        self.assertEqual([r['row'] for r in selected], [4])
        self.assertEqual(excluded, {'original/gold', 'original/reserved'})

    def test_matched_refit_counts_and_frozen_validation_selection(self):
        import torch

        recipe = dict(experiment.RECIPE, epochs=2, batch=1, preservation_batch=1)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as temp, patch.object(experiment, 'RECIPE', recipe):
            out = Path(temp)
            initial = out/'initial-head.npz'
            np.savez(initial, weight=np.array([[0., 0.], [1., 0.], [0., 1.]], dtype='float32'),
                     bias=np.zeros(3, dtype='float32'))
            (out/'encoder-features.onnx').write_bytes(b'frozen synthetic encoder')
            records, views = [], []
            for i, (partition, supervised) in enumerate([
                ('train', True), ('train', True), ('train', False),
                ('calibration', True), ('validation', True), ('validation', False)]):
                key, speaker = 'recording'+str(i), 'speaker'+str(i)
                target = [1, 2] if supervised else []
                x = np.array([[2., 0.], [0., 0.], [0., 2.], [0., 0.]], dtype='float32')
                baseline = torch.nn.functional.linear(torch.from_numpy(x),
                    torch.tensor([[0., 0.], [1., 0.], [0., 1.]])).log_softmax(-1).numpy()
                file = out/(key+'.npz')
                np.savez(file, features=x, baseline=baseline, target=np.array(target, dtype=np.int64))
                records.append(dict(key=key, speaker=speaker, partition=partition, target=target,
                                    source='help' if supervised else 'preservation'))
                views.append(dict(recording=key, speaker=speaker, partition=partition, supervised=supervised,
                                  view=0, file=str(file), sha256=digest(file)))
            q = experiment.transcript_background([[1, 2], [1, 2]], 3)
            plan = dict(schema=1, recipe=recipe, records=records, views=views,
                excluded_speakers=[], background=q.tolist(), primary_sha256='synthetic',
                initial_head_sha256=digest(initial), feature_graph_sha256=digest(out/'encoder-features.onnx'))
            write_json(out/'plan.json', plan)
            experiment.fit(out)
            selected = json.loads((out/'selection.json').read_text())
            self.assertTrue(selected['identical_updates'])
            self.assertTrue(selected['identical_minibatches'])
            self.assertEqual([r['updates'] for r in selected['history']], [2, 4])
            self.assertEqual(selected['selected'], min(selected['history'],
                key=lambda r: (r['validation']['mixture'], r['epoch'])))
            a, b = [np.load(selected['arms'][name]['path']) for name in ('ordinary', 'mixture')]
            self.assertFalse(np.array_equal(a['weight'], b['weight']))
            np.testing.assert_array_equal(np.load(initial)['weight'], [[0., 0.], [1., 0.], [0., 1.]])
            # Leak a fit speaker into calibration; auditing must reject it.
            plan['views'][3]['speaker'] = plan['views'][0]['speaker']
            write_json(out/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'Whole-speaker'):
                experiment.read_plan(out)


if __name__ == '__main__':
    unittest.main()

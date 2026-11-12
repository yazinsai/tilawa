"""Synthetic checks for background identity and marginalized phone occurrences."""
import itertools
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from student_local_noise_head import (RECIPE, crossed_contrast, crossed_view,
    eligible, fit, occurrence_occupancy, ordinary_loss, pairing, partitions, read_plan, recorded_background)
from student_noise_features import ROOT, digest, write_json


class Background(unittest.TestCase):
    def test_controls_recompose_and_donor_energy_matches(self):
        import numpy as np
        rng = np.random.default_rng(7)
        own = rng.normal(0, .001, 8000).astype('float32')
        donor = rng.normal(0, .003, 7000).astype('float32')
        wave = (np.sin(np.arange(16000)*.15)*.1+np.resize(own, 16000)).astype('float32')
        swap, speech, noise, scaled = crossed_view(wave, own, donor)
        np.testing.assert_allclose(speech+noise, wave, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(swap, speech+scaled)
        self.assertAlmostEqual(float(np.mean(noise.astype('float64')**2)),
                               float(np.mean(scaled.astype('float64')**2)), places=12)
        self.assertEqual(len(wave), len(swap))
        self.assertFalse(np.array_equal(swap, wave))

    def test_requires_contiguous_recorded_quiet_audio(self):
        import numpy as np
        rng = np.random.default_rng(9)
        quiet = rng.normal(0, .0001, 5000).astype('float32')
        speech = rng.normal(0, .1, 20000).astype('float32')
        found = recorded_background(np.r_[quiet, speech])
        self.assertIsNotNone(found)
        noise, provenance = found
        self.assertLessEqual(provenance['start_sample']+len(noise), len(quiet))
        self.assertIsNone(recorded_background(np.zeros(16000, dtype='float32')))
        self.assertIsNone(recorded_background(speech))

    def test_none_only_transcripts_and_whole_speaker_exclusion(self):
        self.assertTrue(eligible(dict(kind='none', events=[])))
        self.assertFalse(eligible(dict(kind='none', non_errors=[dict(kind='repeat')])))
        self.assertFalse(eligible(dict(kind='substituted_word')))
        labels = [dict(speaker=str(i), split='train') for i in range(8)]
        labels += [dict(speaker='0', split='eval')]
        mapping, banned = partitions(labels)
        self.assertNotIn('0', mapping)
        self.assertEqual(banned, ['0'])
        self.assertIn('train', mapping.values())
        self.assertIn('validation', mapping.values())

    def test_shared_donors_never_leak_speaker_partitions(self):
        records = [dict(key=f'{p}-{s}-{n}', speaker=f'{p}-{s}', partition=p)
                   for p in ('train', 'validation') for s in range(3) for n in range(3)]
        pairs = pairing(records)
        self.assertEqual(pairs, pairing(list(reversed(records))))
        lookup = {r['key']: r for r in records}
        for pair in pairs:
            self.assertNotIn(pair['donor'], pair['speech'])
            for key in pair['speech']+[pair['donor']]:
                self.assertEqual(lookup[key]['partition'], pair['partition'])

    def test_frozen_plan_rejects_changed_background_identity(self):
        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            out = Path(tmp)
            for name in ('initial-head.npz', 'encoder-features.onnx'):
                (out/name).write_bytes(name.encode())
            views = []
            paired = []
            for key in ('first', 'second'):
                files = []
                for view in ('control', 'swap'):
                    file = out/(key+'-'+view)
                    file.write_bytes(file.name.encode())
                    files.append(str(file))
                    views.append(dict(file=str(file), sha256=digest(file), recording=key,
                        speaker=key, partition='train', view=view,
                        donor='background' if view == 'swap' else None, supervised=True))
                paired.append(files)
            plan = dict(id='listener-local-noise-contrast-training', recipe=RECIPE,
                initial_head_sha256=digest(out/'initial-head.npz'),
                feature_graph_sha256=digest(out/'encoder-features.onnx'), views=views,
                records=[dict(key=k, partition='train') for k in ('first', 'second', 'background')],
                excluded_speakers=[], pairs=[dict(speech=['first', 'second'], donor='background',
                    partition='train', views=paired)])
            write_json(out/'plan.json', plan)
            self.assertEqual(read_plan(out), plan)
            plan['views'][1]['donor'] = 'second'
            write_json(out/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                read_plan(out)


class Marginalization(unittest.TestCase):
    def test_occurrences_match_exhaustive_ctc_paths_including_repeated_labels(self):
        import numpy as np
        rng = np.random.default_rng(1)
        probs = rng.uniform(.1, 1, (5, 3))
        probs /= probs.sum(1, keepdims=True)
        for target in ([1, 2], [1, 1]):
            states = [0]
            for phone in target:
                states += [phone, 0]
            weighted = np.zeros((len(target), len(probs)))
            total = 0.
            for path in itertools.product(range(len(states)), repeat=len(probs)):
                if path[0] not in (0, 1) or path[-1] not in (len(states)-2, len(states)-1):
                    continue
                valid = all(b == a or b == a+1 or
                    (b == a+2 and states[b] != 0 and states[b] != states[a]) for a, b in zip(path, path[1:]))
                if not valid:
                    continue
                mass = float(np.prod([probs[t, states[s]] for t, s in enumerate(path)]))
                total += mass
                for t, s in enumerate(path):
                    if s % 2:
                        weighted[s//2, t] += mass
            expected = weighted/weighted.sum(1, keepdims=True)
            got = occurrence_occupancy(np.log(probs), target)
            np.testing.assert_allclose(got, expected, atol=1e-6)
            np.testing.assert_allclose(got.sum(1), 1.)
            self.assertGreater(total, 0.)

    def test_impossible_complete_transcript_is_rejected(self):
        import numpy as np
        with self.assertRaisesRegex(ValueError, 'complete CTC'):
            occurrence_occupancy(np.log(np.full((2, 3), 1/3)), [1, 1])

    def test_invalid_occurrence_inputs_are_rejected(self):
        import numpy as np
        frames = np.log(np.full((5, 3), 1/3))
        for target in ([], [0], [3]):
            with self.assertRaisesRegex(ValueError, 'valid phone transcript'):
                occurrence_occupancy(frames, target)
        with self.assertRaisesRegex(ValueError, 'frame matrix'):
            occurrence_occupancy(frames[:0], [1])

    def test_contrast_rejects_collapsed_predictions_and_same_background_shortcut(self):
        import torch
        def take(phone, feature):
            return dict(features=torch.tensor([feature], dtype=torch.float32),
                        target=torch.tensor([phone]), occupancy=torch.ones(1, 1))
        pair = [[take(1, [1., 0.]), take(1, [1., 0.])],
                [take(2, [0., 1.]), take(2, [0., 1.])]]
        head = torch.nn.Linear(2, 3)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        collapsed = crossed_contrast(head, pair)
        self.assertAlmostEqual(float(collapsed.detach()), float(torch.log(torch.tensor(2.))), places=5)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[0., 0.], [8., -8.], [-8., 8.]]))
        distinct = crossed_contrast(head, pair)
        self.assertLess(float(distinct.detach()), .01)
        distinct.backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        # Exchanging speech identities while retaining the background must hurt.
        pair[0][1], pair[1][1] = pair[1][1], pair[0][1]
        pair[0][1]['target'] = torch.tensor([1])
        pair[1][1]['target'] = torch.tensor([2])
        self.assertGreater(float(crossed_contrast(head, pair).detach()), float(distinct.detach())+.5)


class Training(unittest.TestCase):
    def test_head_refit_learns_and_persists_training_only_selection(self):
        import numpy as np
        import torch

        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            out = Path(tmp)
            recipe = dict(RECIPE, epochs=2)
            np.savez(out/'initial-head.npz', weight=np.zeros((3, 2), dtype='float32'),
                     bias=np.zeros(3, dtype='float32'))
            (out/'encoder-features.onnx').write_bytes(b'fixture encoder identity')
            views, records, pairs, validation_data = [], [], [], []
            def capture(key, partition, phone, view, donor=None):
                features = np.tile([float(phone == 1), float(phone == 2)], (6, 1)).astype('float32')
                baseline = np.full((6, 3), -np.log(3), dtype='float32')
                target = np.asarray([phone] if phone else [], dtype='int64')
                occupancy = occurrence_occupancy(baseline, target) if phone else np.empty((0, 6), dtype='float32')
                data = dict(features=features, baseline=baseline, target=target, occupancy=occupancy)
                file = out/(key+'-'+view+'.npz')
                np.savez(file, **data)
                views.append(dict(file=str(file), sha256=digest(file), recording=key, speaker=key,
                    partition=partition, view=view, donor=donor, supervised=bool(phone)))
                if partition == 'validation':
                    validation_data.append({k: torch.from_numpy(v) for k, v in data.items()})
                return str(file)
            for partition in ('train', 'validation'):
                speech = [partition+'-first', partition+'-second']
                donor = partition+'-background'
                records.extend(dict(key=k, partition=partition) for k in speech+[donor])
                paired = [[capture(key, partition, i+1, 'control'),
                           capture(key, partition, i+1, 'swap', donor)] for i, key in enumerate(speech)]
                pairs.append(dict(speech=speech, donor=donor, partition=partition, views=paired))
            records.append(dict(key='preservation', partition='train'))
            capture('preservation', 'train', 0, 'control')
            write_json(out/'plan.json', dict(id='listener-local-noise-contrast-training', recipe=recipe,
                initial_head_sha256=digest(out/'initial-head.npz'), feature_graph_sha256=digest(out/'encoder-features.onnx'),
                views=views, records=records, pairs=pairs, excluded_speakers=[]))
            before = torch.nn.Linear(2, 3)
            with torch.no_grad():
                before.weight.zero_()
                before.bias.zero_()
            initial_loss = float(ordinary_loss(before, validation_data[0]).detach())
            with patch('student_local_noise_head.RECIPE', recipe), contextlib.redirect_stdout(io.StringIO()):
                fit(out)
            selection = json.loads((out/'selection.json').read_text())
            self.assertEqual(selection['plan_sha256'], digest(out/'plan.json'))
            self.assertLess(selection['arms']['ctc']['loss'], initial_loss)
            self.assertIn(selection['selected'], ('contrast-0.1', 'contrast-0.3'))
            for arm in selection['arms'].values():
                self.assertEqual(arm['sha256'], digest(arm['path']))
                with np.load(arm['path']) as weights:
                    self.assertEqual(weights['weight'].shape, (3, 2))
                    self.assertTrue(np.isfinite(weights['weight']).all())


if __name__ == '__main__':
    unittest.main()

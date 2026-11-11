"""Small numerical tests; no corpus audio or practice labels."""
import itertools
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from student_envelope import features, fit, forward_backward, prediction, read_training, score, transitions


class EnvelopeTest(unittest.TestCase):
    def test_frequency_prediction(self):
        r = -20 / 30.03
        s = -(11 + r * 20) / (30.03 * (1 - r * r))
        np.testing.assert_allclose(prediction([1, 2, 3, 4], 2, .001), [math.atanh(r), math.atanh(s)])

    def test_forward_backward_matches_enumeration(self):
        e = np.array([[-1., -.1, -3], [-.3, -.2, -2], [-1, -3, -.1]])
        states, size = [0, 1, 0, 2, 0], 5
        a = np.exp(transitions(size))
        mass, prior = 0, 0
        expected = np.zeros_like(e)
        paths = []
        for path in itertools.product(range(size), repeat=len(e)):
            if path[0] not in (0, 1) or path[-1] not in (size - 2, size - 1):
                continue
            weight = .5 * math.prod(a[s, t] for s, t in zip(path, path[1:]))
            likelihood = math.exp(sum(e[t, states[s]] for t, s in enumerate(path)))
            prior += weight
            mass += weight * likelihood
            paths.append((path, weight * likelihood))
        for path, value in paths:
            for t, s in enumerate(path):
                expected[t, states[s]] += value / mass
        value, occupancy = forward_backward(e, [1, 2], True)
        self.assertAlmostEqual(value, math.log(mass / prior), places=12)
        np.testing.assert_allclose(occupancy, expected, atol=1e-12)
        np.testing.assert_allclose(occupancy.sum(1), 1)

    def test_real_fitting_uses_soft_occupancies_and_exports_finite_parameters(self):
        rows = []
        for i in range(3):
            x = np.vstack([np.zeros((2, 19)), np.ones((4, 19)) * (1 + i / 100), -np.ones((4, 19)), np.zeros((2, 19))])
            rows.append(dict(speaker=str(i), pronunciation='بن', envelope=x, control=x))
        model, history = fit(rows, 'بن', 'toy', 2, .001, 8, epochs=2)
        self.assertEqual(len(history['envelope']), 3)
        self.assertIsNotNone(score(model, rows[0]['envelope'], 'بن', 'envelope'))
        for arm in ['control', 'envelope']:
            self.assertTrue(np.isfinite([e['mean'] for e in model[arm]]).all())
            self.assertTrue((np.array([e['variance'] for e in model[arm]]) > 0).all())

    def test_equal_capacity_and_deterministic_audio_features(self):
        wave = np.sin(np.arange(6400) * .19).astype(np.float32)
        for order in [2, 4]:
            f = features(wave, order, .001)
            self.assertEqual(f['control'].shape, (10, 1 + 6 * (order + 1)))
            self.assertEqual(f['envelope'].shape, f['control'].shape)
            self.assertTrue(np.isfinite(f['envelope']).all())
            np.testing.assert_array_equal(f['envelope'], features(wave, order, .001)['envelope'])

    def test_excludes_whole_speakers_and_non_none_takes_before_audio_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'help-train'
            root.mkdir()
            labels = []
            for i in range(3):
                path = root / f'{i}.wav'
                with wave.open(str(path), 'wb') as w:
                    w.setparams((1, 2, 16000, 6400, 'NONE', 'not compressed'))
                    w.writeframes((np.sin(np.arange(6400) * .19) * 12000).astype('<i2').tobytes())
                labels.append(dict(speaker=str(i), kind='none', split='train', wav=path.name, ayahs=['1:1']))
            labels.extend([
                dict(speaker='held-out', kind='none', split='train', wav='must-not-open.wav', ayahs=['1:1']),
                dict(speaker='held-out', kind='none', split='eval', wav='must-not-open.wav', ayahs=['1:1']),
                dict(speaker='explicit', kind='none', split='train', wav='must-not-open.wav', ayahs=['1:1']),
                dict(speaker='0', kind='wrong_word', split='train', wav='must-not-open.wav', ayahs=['1:1']),
                dict(speaker='0', kind='none', split='train', excluded='label audit', wav='must-not-open.wav', ayahs=['1:1']),
                dict(speaker='0', kind='none', split='train', scored=False, wav='must-not-open.wav', ayahs=['1:1']),
            ])
            (root / 'labels.jsonl').write_text('\n'.join(json.dumps(l) for l in labels))
            reference = dict(inventory=dict(tokens='بن'), ayahs=[dict(s=1, a=1, p='بن', o=[0], iso=['بن'])])
            rows, excluded = read_training(root, reference, {'explicit'})
            self.assertEqual(len(rows), 3)
            self.assertEqual(excluded, {'explicit', 'held-out'})

    def test_rejects_missing_training_split_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'labels.jsonl').write_text(json.dumps(dict(speaker='unknown', kind='none')))
            with self.assertRaisesRegex(ValueError, 'metadata required'):
                read_training(root, {}, set())


if __name__ == '__main__':
    unittest.main()

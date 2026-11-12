"""Small numerical checks; run on Modal with NumPy, never on practice data."""
import itertools
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from student_response_fingerprint import boundary, fit, topology, train


class FingerprintTrainingTests(unittest.TestCase):
    def test_marginalization_and_occupancies_match_complete_path_enumeration(self):
        values = np.array([[.1, .3, .2], [.5, .4, .1], [.2, .8, .4], [.4, .2, .7]])
        for ids in ([1], [2], [1, 2], [1, 1]):
            labels, edges = topology(ids)
            mass, timing = 0., 0.
            counts = np.zeros_like(values)
            for path in itertools.product(range(len(labels)), repeat=len(values)):
                if path[0] not in (0, 1) or path[-1] not in (len(labels)-2, len(labels)-1):
                    continue
                if any(b not in edges[a] for a, b in zip(path, path[1:])):
                    continue
                prior = np.prod([1/len(edges[s]) for s in path[:-1]])
                weight = prior*np.exp(sum(values[t, labels[s]] for t, s in enumerate(path)))
                timing += prior
                mass += weight
                for t, s in enumerate(path):
                    counts[t, labels[s]] += weight
            score, occupancy = boundary(values, ids, True)
            self.assertAlmostEqual(score, np.log(mass/timing), places=12)
            np.testing.assert_allclose(occupancy, counts/mass, atol=1e-12)
            np.testing.assert_allclose(occupancy.sum(1), 1., atol=1e-12)

    def test_incomplete_pronunciations_and_repeat_separation(self):
        self.assertEqual(boundary(np.zeros((1, 3)), [1, 2]), -np.inf)
        self.assertEqual(boundary(np.zeros((2, 3)), [1, 1]), -np.inf)
        self.assertAlmostEqual(boundary(np.zeros((3, 3)), [1, 1]), 0.)

    def test_matching_arms_have_equal_capacity_and_finite_regularized_emissions(self):
        rng = np.random.default_rng(17)
        rows = []
        for phone in (1, 2):
            observations = rng.normal(size=(8, 6))*.05
            observations[:, 0] += phone
            log_probs = np.full((8, 3), -8.)
            log_probs[:, phone] = -.1
            rows.append(dict(ids=[phone], response=observations, control=observations.copy(), log_probs=log_probs))
        response, control = (fit(rows, key, 3, 16.) for key in ('response', 'control'))
        self.assertEqual(response, control)
        for emission in response['phones']:
            precision = np.asarray(emission['precision'])
            self.assertTrue(np.isfinite(precision).all())
            self.assertTrue((np.diag(precision) > 0).all())
            np.testing.assert_array_equal(precision, np.tril(precision))

    def test_grouped_fit_exports_unchanged_primary_and_checks_cache_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            primary = root/'primary'
            primary.mkdir()
            # Export integrity is tested separately from executing an actual ONNX graph.
            graph = b'fixture graph bytes'
            digest = hashlib.sha256(graph).hexdigest()
            (primary/'model.onnx').write_bytes(graph)
            original_manifest = json.dumps(dict(file='model.onnx', sha256=digest))
            (primary/'manifest.json').write_text(original_manifest)
            (primary/'teacher.data').write_bytes(b'unchanged auxiliary data')
            basis = []
            for center in (.28, .72):
                distances = [abs(i/79-center)/.2 for i in range(80)]
                basis.append([(1+math.cos(math.pi*d))/2 if d < 1 else 0. for d in distances])
            metadata = dict(schema=1, tokens='ab', inventory='test', dimensions=6,
                primarySha256=digest, basis=basis, epsilon=.05, radius=4)
            (root/'metadata.json').write_text(json.dumps(metadata))
            (root/'excluded.json').write_text('[]')
            deadline = (time.time()+60)*1000
            rng = np.random.default_rng(9)
            rows = []
            for speaker in ('one', 'two', 'three'):
                for phone in (1, 2):
                    x = rng.normal(size=(8, 6))*.1
                    x[:, 0] += phone
                    lp = np.full((8, 3), -8.)
                    lp[:, phone] = -.1
                    rows.append(dict(source='help-train', kind='none', split='train', excluded=False,
                        speaker=speaker, pronunciation='ab'[phone-1], primarySha256=digest,
                        deadlineMs=deadline, response=x.tolist(), control=x.tolist(), log_probs=lp.tolist()))
            path = root/'rows.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            args = SimpleNamespace(metadata=str(root/'metadata.json'), excluded_speakers=str(root/'excluded.json'),
                input=str(path), primary_bundle=str(primary), out=str(root/'bundle'), deadline=deadline)
            with contextlib.redirect_stdout(io.StringIO()):
                train(args)
            out = root/'bundle'
            self.assertEqual((out/'model.onnx').read_bytes(), graph)
            self.assertEqual((out/'manifest.json').read_text(), original_manifest)
            self.assertEqual((out/'teacher.data').read_bytes(), b'unchanged auxiliary data')
            asset = json.loads((out/'response-fingerprint-asset.json').read_text())['files'][0]
            self.assertEqual(asset['sha256'], hashlib.sha256((out/asset['file']).read_bytes()).hexdigest())
            report = json.loads((out/'response-fingerprint-training-report.json').read_text())
            self.assertEqual(set(report['speakerFolds'].values()), {0, 1, 2})
            rows[0]['primarySha256'] = 'wrong'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, 'different primary graph'):
                train(args)
            rows[0]['primarySha256'] = digest
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            (root/'excluded.json').write_text('["one"]')
            with self.assertRaisesRegex(ValueError, 'release-held-out'):
                train(args)


if __name__ == '__main__':
    unittest.main()

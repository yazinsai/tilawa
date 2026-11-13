"""Training objective, complete soft occupancies, and bundle invariants."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from student_word_scrub import eligible_labels, occupancies, partition, shuffle_words
from student_word_scrub_bundle import assemble


class ConditionalWordScrubTests(unittest.TestCase):
    def test_independent_attestation_and_speaker_exclusion(self):
        rows = [dict(id=str(i), speaker=str(i//2), source='help-train', split='train', kind='none',
                     ids=[1, 2], wordLabels=['a' if i%2 else 'b', 'only']) for i in range(8)]
        self.assertEqual(eligible_labels(rows), {1: ['a', 'b']})
        a, b = partition(rows, set())
        self.assertFalse({r['speaker'] for r in a} & {r['speaker'] for r in b})
        with self.assertRaises(ValueError): partition(rows, {'0'})
        rows[0]['kind'] = 'wrong_word'
        with self.assertRaises(ValueError): partition(rows, set())

    def test_shuffling_conditions_on_phone_and_speaker(self):
        rows = [dict(phone=p, speaker=s, word=w) for p in (1, 2) for s in ('a', 'b') for w in range(16)]
        shuffled = shuffle_words(rows)
        self.assertNotEqual(rows, shuffled)
        for p in (1, 2):
            for s in ('a', 'b'):
                self.assertEqual(sorted(r['word'] for r in rows if r['phone']==p and r['speaker']==s),
                                 sorted(r['word'] for r in shuffled if r['phone']==p and r['speaker']==s))
        self.assertEqual([(r['phone'], r['speaker']) for r in rows], [(r['phone'], r['speaker']) for r in shuffled])

    def test_soft_ctc_chain_matches_exhaustive_paths(self):
        import itertools
        p = np.array([[.2, .7, .1], [.4, .4, .2], [.2, .1, .7]])
        q, score = occupancies(np.log(p), [1, 2])
        mass = 0
        expected = np.zeros_like(p)
        for path in itertools.product(range(3), repeat=3):
            collapsed = [v for i, v in enumerate(path) if (i==0 or path[i-1]!=v) and v]
            if collapsed != [1, 2]: continue
            value = np.prod([p[t, v] for t, v in enumerate(path)]); mass += value
            for t, v in enumerate(path): expected[t, v] += value
        self.assertAlmostEqual(score, np.log(mass))
        np.testing.assert_allclose(q.sum(1), 1)
        labels = [0, 1, 0, 2, 0]; observed = np.zeros_like(p)
        for state, label in enumerate(labels): observed[:, label] += q[:, state]
        np.testing.assert_allclose(observed, expected/mass)
        with self.assertRaises(ValueError): occupancies(np.log(p[:2]), [1, 1])

    def test_bundle_keeps_primary_and_auxiliary_bytes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); model = root/'model'; fitted=root/'fit'; model.mkdir(); fitted.mkdir()
            (model/'model.onnx').write_bytes(b'primary')
            sha=hashlib.sha256(b'primary').hexdigest()
            manifest=dict(file='model.onnx', sha256=sha, tokens='ab', inventory='test')
            (model/'manifest.json').write_text(json.dumps(manifest)); (model/'teacher.onnx.data').write_bytes(b'inherited teacher')
            metadata=dict(primarySha256=sha, tokens='ab', inventory='test')
            for file in ('word-scrub.json','word-scrub-features-metadata.json'): (fitted/file).write_text(json.dumps(metadata))
            (fitted/'word-scrub-features.onnx').write_bytes(b'features')
            assets={}
            for file in ('word-scrub.json','word-scrub-features.onnx'):
                data=(fitted/file).read_bytes(); assets[file]=dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            (fitted/'word-scrub-assets.json').write_text(json.dumps(assets))
            assemble(model, fitted, root/'out')
            self.assertEqual((root/'out/teacher.onnx.data').read_bytes(), b'inherited teacher')
            (fitted/'word-scrub-features.onnx').unlink()
            with self.assertRaises(FileNotFoundError): assemble(model, fitted, root/'missing')


if __name__ == '__main__': unittest.main()

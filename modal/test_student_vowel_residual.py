"""Small CPU tests for training exclusion, residual/control isolation and assets."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from student_vowel_residual import nuisance_fit, partition, sequence_log_probability, shuffled_context
from student_vowel_residual_bundle import ASSETS, DESCRIPTOR, assemble, fingerprint, resolve_listener


class TrainingTests(unittest.TestCase):
    def rows(self):
        return [dict(id=f'{s}-{i}', speaker=s, source='help-train', split='train', kind='none',
                     context=np.eye(10).tolist()) for s in ('a', 'b', 'c', 'd', 'e') for i in range(2)]

    def test_partition_keeps_complete_speakers_apart_and_refuses_heldout(self):
        rows = self.rows()
        train, valid = partition(rows, set())
        self.assertFalse({r['speaker'] for r in train} & {r['speaker'] for r in valid})
        self.assertEqual(train+valid, partition(rows, set())[0]+partition(rows, set())[1])
        for key, value in [('split', 'test'), ('kind', 'wrong_word'), ('source', 'practice'), ('non_errors', ['repeat'])]:
            changed = self.rows()
            changed[0][key] = value
            with self.assertRaises(ValueError):
                partition(changed, set())
        with self.assertRaises(ValueError):
            partition(rows, {'a'})

    def test_real_context_removes_predictable_vowels_and_shuffle_does_not(self):
        rng = np.random.default_rng(9)
        phi = np.repeat(np.eye(10), 30, axis=0)
        nuisance = rng.normal(size=(32, 10))
        consonant = rng.normal(size=(len(phi), 32)) * .01
        x = phi @ nuisance.T + consonant
        real = nuisance_fit([x], [phi])
        rows = [dict(speaker='s', context=phi.tolist())]
        shuffled = shuffled_context(rows)[0]
        control = nuisance_fit([x], [shuffled])
        self.assertLess(np.mean((x-phi@real.T)**2), .01)
        self.assertGreater(np.mean((x-phi@control.T)**2), .1)
        # Shuffling preserves the context population and is reproducible.
        np.testing.assert_array_equal(np.sort(shuffled, axis=0), np.sort(phi, axis=0))
        np.testing.assert_array_equal(shuffled, shuffled_context(rows)[0])

    def test_complete_ctc_cannot_skip_an_unsupported_phone(self):
        lp = np.log(np.array([[.1, .8, .1], [.8, .1, .1], [.1, .1, .8]]))
        self.assertTrue(np.isfinite(sequence_log_probability(lp, [1, 2])))
        self.assertEqual(sequence_log_probability(lp[:2], [1, 1]), -np.inf)
        self.assertEqual(sequence_log_probability(lp, [1, 2, 1, 2]), -np.inf)


class BundleTests(unittest.TestCase):
    def test_preserves_all_inherited_files_and_rejects_missing_declared_assets(self):
        scratch = Path(__file__).resolve().parents[1]/'.scratch'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            source, fitted = root/'models'/'parent', root/'fit'
            source.mkdir(parents=True)
            fitted.mkdir()
            (source/'model.onnx').write_bytes(b'primary-immutable')
            sha = hashlib.sha256((source/'model.onnx').read_bytes()).hexdigest()
            manifest = dict(file='model.onnx', sha256=sha, tokens='test', inventory='test')
            (source/'manifest.json').write_text(json.dumps(manifest))
            (source/'teacher.onnx').write_bytes(b'teacher')
            (source/'teacher.onnx.data').write_bytes(b'teacher-external-data')
            card = root/'card.json'
            card.write_text(json.dumps(dict(listener='parent')))
            self.assertEqual(resolve_listener(card, root/'models'), source)
            metadata = dict(primarySha256=sha, tokens='test', inventory='test')
            (fitted/ASSETS[0]).write_text(json.dumps(metadata))
            (fitted/'vowel-residual-features-metadata.json').write_text(json.dumps(metadata))
            (fitted/ASSETS[1]).write_bytes(b'extractor')
            descriptor = {name: fingerprint(fitted/name) for name in ASSETS}
            (fitted/DESCRIPTOR).write_text(json.dumps(descriptor))
            report = assemble(source, fitted, root/'bundle')
            self.assertEqual((root/'bundle'/'manifest.json').read_bytes(), (source/'manifest.json').read_bytes())
            self.assertEqual((root/'bundle'/'teacher.onnx.data').read_bytes(), b'teacher-external-data')
            self.assertEqual(report['primarySha256'], sha)
            (fitted/ASSETS[1]).unlink()
            with self.assertRaises(FileNotFoundError):
                assemble(source, fitted, root/'missing')
            self.assertFalse((root/'missing').exists())


if __name__ == '__main__':
    unittest.main()

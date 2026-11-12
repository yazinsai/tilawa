import argparse
import json
import tempfile
import unittest
from pathlib import Path

from student_sparse_factor import cache_rows, fit, grouped_split


class DataIsolation(unittest.TestCase):
    def test_speakers_are_disjoint_before_any_fit(self):
        rows = [{'speaker': s, 'id': i} for s in 'abcdef' for i in range(4)]
        held_all = []
        for fold in range(3):
            training, held = grouped_split(rows, fold)
            self.assertFalse({r['speaker'] for r in training} & {r['speaker'] for r in held})
            held_all.extend(r['speaker'] for r in held)
        self.assertEqual(sorted(held_all), sorted(r['speaker'] for r in rows))

    def test_mistake_scripts_and_heldout_features_are_rejected(self):
        row = dict(source='help-train', split='train', excluded=False, speaker='a',
            features=[[1., 2.]], pronunciation='a', kind='none')
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / 'cache.jsonl'
            for change in ({'split': 'eval'}, {'kind': 'wrong_word'}, {'events': [{}]}, {'excluded': True}):
                path.write_text(json.dumps({**row, **change}) + '\n')
                with self.assertRaises(ValueError):
                    cache_rows(path, {'width': 2, 'tokens': 'ab'})
            path.write_text(json.dumps({**row, 'kind': 'wrong_word', 'pronunciation': None}) + '\n')
            self.assertIsNone(cache_rows(path, {'width': 2, 'tokens': 'ab'})[0]['pronunciation'])


class MatchedTraining(unittest.TestCase):
    def test_dictionary_is_fixed_and_controls_have_identical_inference_capacity(self):
        import numpy as np
        args = argparse.Namespace(seed=1729, ae_epochs=1, ctc_epochs=1, frame_batch=8, recording_batch=2)
        rows = [dict(speaker=s, pronunciation='ab', x=np.array([[1,0],[1,.1],[0,1],[.1,1]], dtype=np.float32))
            for s in 'abcd']
        meta = dict(schema=1, inventory='test', tokens='ab', width=2, featureOutput='features', primarySha256='parent')
        model, report = fit(rows, meta, .1, args, lambda: None)
        self.assertTrue(report['frozenEncodersUnchanged'])
        self.assertEqual(model['hidden'], 4)
        counts = []
        for arm in ('sparseFactor', 'dense', 'control'):
            counts.append(sum(np.asarray(v).size for v in model[arm].values()))
        self.assertEqual(counts, [report['inferenceParameters']] * 3)
        for dictionary in report['dictionaries']:
            np.testing.assert_allclose(np.linalg.norm(dictionary, axis=0), 1, atol=1e-6)
        self.assertFalse(np.array_equal(model['sparseFactor']['encoder'], model['dense']['encoder']))

    def test_budget_aborts_before_publishing_weights(self):
        def expired():
            raise TimeoutError('budget')
        args = argparse.Namespace()
        with self.assertRaises(TimeoutError):
            fit([], {}, .1, args, expired)


if __name__ == '__main__':
    unittest.main()

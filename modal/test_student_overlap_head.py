"""Independent CTC enumeration and population-weighting fixtures."""
import itertools
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from student_head_refit import ROOT, digest, has_clean_transcript, write_json
from student_overlap_head import (domain_mass, domain_probability, phone_scales,
                                 soft_occupancies, speaker_folds, fit, read_plan, RECIPE)


class Overlap(unittest.TestCase):
    def test_soft_alignment_matches_enumeration_including_repeated_labels(self):
        p = np.asarray([[.5, .3, .2], [.2, .5, .3], [.4, .2, .4], [.1, .6, .3]])
        for target in ([1], [1, 2], [1, 1]):
            expected, total = np.zeros_like(p), 0.
            for path in itertools.product(range(3), repeat=len(p)):
                collapsed = [x for i, x in enumerate(path) if x and (i == 0 or path[i-1] != x)]
                if collapsed != target:
                    continue
                mass = np.prod([p[t, x] for t, x in enumerate(path)])
                total += mass
                for t, x in enumerate(path):
                    expected[t, x] += mass
            np.testing.assert_allclose(soft_occupancies(np.log(p), target), expected/total, atol=1e-7)
        with self.assertRaisesRegex(ValueError, 'no complete CTC path'):
            soft_occupancies(np.log(p[:1]), [1, 1])

    def test_overlap_suppresses_domain_extremes_without_changing_phone_mass(self):
        q = [np.asarray([[.8, .2, 0], [.1, .9, 0]]),
             np.asarray([[.2, .7, .1], [.1, .8, .1]])]
        p = [np.asarray([[.5, .5, .5], [.01, .01, .5]]),
             np.asarray([[.5, .5, .5], [.99, .99, .5]])]
        sources = ['original', 'help']
        scales = phone_scales(q, p, sources)
        original = q[0].sum(0)
        for name, scale in scales.items():
            weighted = [a*scale*(b if s == 'original' else 1-b) if name == 'weighted' else a*scale
                        for a, b, s in zip(q, p, sources)]
            np.testing.assert_allclose(sum(a.sum(0) for a in weighted), original, atol=1e-7)
            self.assertEqual(sum(a[:, 2].sum() for a in weighted), 0)
            if name == 'weighted':
                self.assertLess(weighted[0][1, 1]/weighted[0][0, 1], q[0][1, 1]/q[0][0, 1])

    def test_indistinguishable_domains_reproduce_uniform_arm(self):
        q = [np.asarray([[.6, .4]]), np.asarray([[.1, .9]])]
        p = [np.full_like(a, .5) for a in q]
        scales = phone_scales(q, p, ['original', 'help'])
        for a in q:
            np.testing.assert_allclose(a*.5*scales['weighted'], a*scales['uniform'])

    def test_domain_priors_condition_on_phone_not_transcript_frequencies(self):
        q = np.asarray([[.9, .1, 0], [.8, .2, 0], [.1, .8, .1]], dtype='float32')
        y = np.asarray([0, 0, 1])
        mass, present = domain_mass(q, y)
        np.testing.assert_array_equal(present, [True, True, False])
        np.testing.assert_allclose(mass[y == 0].sum(0), [.25, .25, 0], atol=1e-7)
        np.testing.assert_allclose(mass[y == 1].sum(0), [.25, .25, 0], atol=1e-7)
        np.testing.assert_allclose(mass.sum(), 1.)

    def test_domain_without_shared_phone_uses_neutral_odds(self):
        model = dict(mean=np.zeros(2), std=np.ones(2), weight=np.ones((2, 3)),
                     bias=np.zeros(3), present=np.asarray([True, True, False]))
        p = domain_probability(model, np.asarray([[0., 0.], [1., 1.]]))
        np.testing.assert_array_equal(p[:, 2], [.5, .5])
        self.assertGreater(p[1, 1], p[0, 1])

    def test_all_takes_and_views_of_a_speaker_stay_in_one_domain_fold(self):
        views = [dict(source=source, speaker=f'{source}/{speaker}')
                 for source in ('original', 'help') for speaker in range(6) for _ in range(3)]
        folds = speaker_folds(views, 3)
        for fold in range(3):
            fitted = {v['speaker'] for v, f in zip(views, folds) if f != fold}
            predicted = {v['speaker'] for v, f in zip(views, folds) if f == fold}
            self.assertFalse(fitted & predicted)
            self.assertEqual({v['source'] for v, f in zip(views, folds) if f == fold}, {'original', 'help'})

    def test_scripted_mistakes_repeats_and_unscored_takes_are_not_transcripts(self):
        self.assertTrue(has_clean_transcript(dict(kind='none')))
        for change in (dict(kind='wrong_word'), dict(events=[{}]), dict(non_errors=[{}]),
                       dict(scored=False), dict(unsupported=True)):
            self.assertFalse(has_clean_transcript(dict(dict(kind='none'), **change)))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is supplied in the supervisor train image')
    def test_matched_refits_keep_speaker_separation_and_frozen_selection(self):
        recipe = dict(RECIPE, epochs=2, domain_steps=10, domain_l2=[.1])
        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp, patch('student_overlap_head.RECIPE', recipe):
            out = Path(tmp)
            initial = dict(weight=np.asarray([[.1, 0], [0, .2], [0, -.2]], dtype='float32'),
                           bias=np.zeros(3, dtype='float32'))
            np.savez(out/'initial-head.npz', **initial)
            (out/'encoder-features.onnx').write_bytes(b'fixture feature provenance')
            views = []
            for source in ('original', 'help'):
                for speaker in range(4):
                    key = f'{source}-{speaker}'
                    rng = np.random.default_rng(speaker)
                    x = rng.normal(size=(12, 2)).astype('float32')
                    x[:, 0] += -1 if source == 'original' else 1
                    logits = x@initial['weight'].T
                    lp = logits-np.log(np.exp(logits).sum(1, keepdims=True))
                    q = soft_occupancies(lp, [1, 2])
                    file = out/(key+'.npz')
                    np.savez(file, features=x, baseline=lp, occupancy=q)
                    views.append(dict(file=str(file), sha256=digest(file), source=source,
                        recording=key, speaker=key, partition='validation' if speaker == 3 else 'train'))
            write_json(out/'plan.json', dict(recipe=recipe, views=views, excluded_speakers=[],
                primary_sha256='fixture', initial_head_sha256=digest(out/'initial-head.npz'),
                feature_graph_sha256=digest(out/'encoder-features.onnx')))
            fit(out)
            selection = json.loads((out/'selection.json').read_text())
            domain = json.loads((out/'domain-selection.json').read_text())
            self.assertTrue(selection['identical_updates'])
            self.assertTrue(selection['identical_phone_mass'])
            self.assertEqual(selection['selected']['updates'], selection['selected']['epoch'])
            self.assertEqual(selection['domain_selection_sha256'], digest(out/'domain-selection.json'))
            for fold in domain['folds']:
                self.assertFalse(set(fold['fit_speakers']) & set(fold['predicted_speakers']))
                self.assertFalse(any(s.endswith('-3') for s in fold['fit_speakers']))
            for checkpoint in selection['arms'].values():
                self.assertEqual(checkpoint['sha256'], digest(checkpoint['path']))
            read_plan(out)
            Path(domain['weights'][0]['path']).write_bytes(b'changed after selection')
            with self.assertRaisesRegex(ValueError, 'Frozen domain model or overlap weights changed'):
                read_plan(out)


if __name__ == '__main__':
    unittest.main()

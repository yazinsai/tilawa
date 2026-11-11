"""Bilevel chain-rule, speaker separation and matched-continuation checks."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from student_head_refit import ROOT, authorized_records, digest, write_json
from student_transfer_head import (RECIPE, ctc_loss, episodes, fit,
                                   help_preservation, preservation_indices, preservation_loss, read_plan, transfer_weights)


class Admission(unittest.TestCase):
    def test_excluded_original_and_help_speakers_are_removed_in_full(self):
        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as directory:
            root = Path(directory)
            original = root/'student'
            original.mkdir()
            (original/'tiny.parquet').touch()
            exclusion = root/'docs/benchmarks/gold-per-v1-exclude.txt'
            exclusion.parent.mkdir(parents=True)
            exclusion.write_text('source:1\n')
            metadata = [dict(shard='source', row=i, cluster=i//2, bucket='clean-match',
                             split='test' if i == 3 else 'train', n_samples=16000)
                        for i in range(12)]
            help_root = root/'data/help-train'
            help_root.mkdir(parents=True)
            labels = []
            for i in range(6):
                (help_root/f'{i}.wav').touch()
                labels.append(dict(id=f'none-{i}', speaker=str(i), split='train', kind='none',
                                   wav=f'{i}.wav', ayahs=['1:1']))
            # One excluded take excludes all other takes of the same speaker.
            labels.append(dict(id='release-take', speaker='0', split='eval', kind='none', wav='0.wav'))
            labels.append(dict(id='script', speaker='1', split='train', kind='wrong_word',
                               wav='1.wav', scripted_transcript='never a performed transcript'))
            labels.append(dict(id='repeat', speaker='2', split='train', kind='none', wav='2.wav',
                               non_errors=[dict(kind='repeat')]))
            write_json_lines = '\n'.join(json.dumps(l) for l in labels)
            (help_root/'labels.jsonl').write_text(write_json_lines)
            parquet = SimpleNamespace(read_schema=lambda _: SimpleNamespace(names=list(metadata[0])),
                                      read_table=lambda *a, **k: SimpleNamespace(to_pylist=lambda: metadata))
            modules = {'pyarrow': SimpleNamespace(parquet=parquet), 'pyarrow.parquet': parquet,
                       'soundfile': SimpleNamespace(info=lambda _: SimpleNamespace(duration=1.))}

            def source_path(value):
                return original if str(value) == '/student/s1' else Path(value)

            with patch.dict('sys.modules', modules), patch.dict(os.environ, LAB_DATA=str(root/'data')), \
                    patch('student_head_refit.ROOT', root), patch('student_head_refit.Path', source_path):
                records, excluded, _ = authorized_records(512, 30)
                self.assertFalse({'original/0', 'original/1', 'help/0'} & {r['speaker'] for r in records})
                self.assertTrue({'original/0', 'original/1', 'help/0'} <= set(excluded))
                self.assertNotIn('help/script', {r['key'] for r in records})
                self.assertNotIn('help/repeat', {r['key'] for r in records})
                controls = help_preservation(30)
                self.assertEqual([r['key'] for r in controls], ['help/script', 'help/repeat'])
                self.assertTrue(all(r['target'] == [] for r in controls))
                for source in ('original', 'help'):
                    train = {r['speaker'] for r in records if r['source'] == source and r['partition'] == 'train'}
                    validation = {r['speaker'] for r in records if r['source'] == source and r['partition'] == 'validation'}
                    self.assertTrue(train and validation)
                    self.assertFalse(train & validation)


class SpeakerEpisodes(unittest.TestCase):
    def test_preservation_updates_cannot_use_query_speakers(self):
        bank = [dict(speaker='query'), dict(speaker='support'), dict(speaker='control')]
        for update in range(4):
            indices = preservation_indices(bank, {'query'}, update, 3)
            self.assertEqual(len(indices), 3)
            self.assertFalse({bank[i]['speaker'] for i in indices} & {'query'})
        with self.assertRaisesRegex(ValueError, 'outside the query speakers'):
            preservation_indices(bank, {'query', 'support', 'control'}, 0, 3)

    def test_whole_speakers_are_disjoint_and_all_views_support_each_epoch(self):
        views = [dict(speaker=f's{i}', recording=f's{i}-v{j}') for i in range(7) for j in range(i+1)]
        for epoch in range(4):
            plan = list(episodes(views, epoch, 5, 4, 831))
            self.assertEqual(sorted(i for support, _ in plan for i in support), list(range(len(views))))
            for support, query in plan:
                s = {views[i]['speaker'] for i in support}
                q = {views[i]['speaker'] for i in query}
                self.assertFalse(s & q)
                self.assertEqual(len(q), len(query))
            self.assertEqual(plan, list(episodes(views, epoch, 5, 4, 831)))
        self.assertNotEqual(list(episodes(views, 0, 5, 4, 831)), list(episodes(views, 1, 5, 4, 831)))

    def test_insufficient_speakers_cannot_masquerade_as_cross_speaker_training(self):
        with self.assertRaisesRegex(ValueError, 'four supervised training speakers'):
            list(episodes([dict(speaker='same')]*8, 0, 4, 2, 1))


class MetaGradient(unittest.TestCase):
    def data(self, target):
        import torch

        return dict(features=torch.tensor([[1., .5], [1., .4], [1., .3]], dtype=torch.float64),
                    baseline=torch.full((3, 3), -__import__('math').log(3), dtype=torch.float64),
                    target=torch.tensor([target], dtype=torch.long))

    def parameters(self):
        import torch

        return (torch.zeros((3, 2), dtype=torch.float64, requires_grad=True),
                torch.zeros(3, dtype=torch.float64, requires_grad=True))

    def test_chain_rule_matches_finite_difference_through_virtual_update(self):
        import torch

        params = self.parameters()
        support = [self.data(1), self.data(2)]
        query = [self.data(1)]
        anchors = support
        lr = .1
        weights, benefits = transfer_weights(params, support, query, anchors, lr)
        anchor = RECIPE['anchor']*sum(preservation_loss(params, d) for d in anchors)/len(anchors)
        ga = torch.autograd.grad(anchor, params)
        for i, d in enumerate(support):
            gs = torch.autograd.grad(ctc_loss(params, d), params)

            def virtual_nll(epsilon):
                virtual = tuple(p-lr*(a+epsilon*s) for p, a, s in zip(params, ga, gs))
                return float(ctc_loss(virtual, query[0]).detach())

            epsilon = 1e-5
            derivative = (virtual_nll(epsilon)-virtual_nll(-epsilon))/(2*epsilon)
            self.assertAlmostEqual(float(benefits[i]), -derivative, places=8)
        self.assertGreater(float(benefits[0]), 0)
        self.assertLess(float(benefits[1]), 0)
        self.assertEqual(weights.tolist(), [1., 0.])

    def test_better_self_fit_does_not_justify_a_harmful_transfer(self):
        import torch

        params = self.parameters()
        bad, good = self.data(2), self.data(1)
        gradient = torch.autograd.grad(ctc_loss(params, bad), params)
        updated = tuple(p-.1*g for p, g in zip(params, gradient))
        self.assertLess(float(ctc_loss(updated, bad).detach()), float(ctc_loss(params, bad).detach()))
        self.assertGreater(float(ctc_loss(updated, good).detach()), float(ctc_loss(params, good).detach()))
        weights, _ = transfer_weights(params, [bad], [good], [bad], .1)
        self.assertEqual(weights.tolist(), [0.])

    def test_equal_transfer_recovers_uniform_mass_without_biasing_pauses(self):
        params = self.parameters()
        good = self.data(1)
        weights, _ = transfer_weights(params, [good, good, good], [good], [good], .1)
        for w in weights.tolist():
            self.assertAlmostEqual(w, 1/3)
        # The shipped-output objective covers every frame, including blank-won
        # frames, and is independent of the transcript (including no transcript).
        self.assertAlmostEqual(float(preservation_loss(params, good).detach()), 0., places=12)
        import torch
        good['target'] = torch.tensor([], dtype=torch.long)
        self.assertAlmostEqual(float(preservation_loss(params, good).detach()), 0., places=12)


class FrozenFit(unittest.TestCase):
    def fixture(self, out):
        import numpy as np

        np.savez(out/'initial-head.npz', weight=np.zeros((3, 2), dtype='float32'), bias=np.zeros(3, dtype='float32'))
        (out/'encoder-features.onnx').write_bytes(b'unit fixture provenance')
        views = []
        for i in range(7):
            for supervised in (True, False):
                path = out/f'{i}-{supervised}.npz'
                np.savez(path, features=np.asarray([[1., .5], [1., .4], [1., .3]], dtype='float32'),
                         baseline=np.full((3, 3), -np.log(3), dtype='float32'),
                         target=np.asarray([1] if supervised else [], dtype=np.int64))
                views.append(dict(recording=f'{i}-{supervised}', speaker=f's{i}', supervised=supervised,
                    partition='train' if i < 5 else 'validation', file=str(path), sha256=digest(path)))
        write_json(out/'plan.json', dict(recipe=RECIPE, views=views, excluded_speakers=['release-only'],
            initial_head_sha256=digest(out/'initial-head.npz'),
            feature_graph_sha256=digest(out/'encoder-features.onnx')))

    def test_matched_arms_have_same_data_updates_and_training_only_selection(self):
        import numpy as np

        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as directory:
            out = Path(directory)
            self.fixture(out)
            fit(out)
            report = json.loads((out/'selection.json').read_text())
            self.assertTrue(report['identical_updates'])
            self.assertTrue(report['identical_support_batches'])
            self.assertTrue(report['disjoint_query_speakers'])
            self.assertEqual(len(report['history']), RECIPE['epochs'])
            for row in report['history']:
                self.assertAlmostEqual(row['validation']['uniform'], row['validation']['weighted'], places=6)
                self.assertEqual(row['updates'], row['epoch']*2)
                for episode in json.loads((out/f"weights-{row['epoch']}.json").read_text()):
                    query = {key.split('-')[0] for key in episode['query']}
                    updated = {key.split('-')[0] for key in episode['support']+episode['preservation']}
                    self.assertFalse(query & updated)
            epoch = min(report['history'], key=lambda row: (row['validation']['weighted'], row['epoch']))['epoch']
            self.assertEqual(report['selected']['epoch'], epoch)
            for name in ('uniform', 'weighted'):
                self.assertTrue(report['arms'][name]['path'].endswith(f'{name}-{epoch}.npz'))
            a = np.load(report['arms']['uniform']['path'])
            b = np.load(report['arms']['weighted']['path'])
            np.testing.assert_allclose(a['weight'], b['weight'], atol=1e-7)
            # No expected text was supplied for any preservation control.
            for v in read_plan(out)['views']:
                if not v['supervised']:
                    self.assertEqual(len(np.load(v['file'])['target']), 0)

    def test_frozen_cache_and_excluded_speakers_are_enforced(self):
        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as directory:
            out = Path(directory)
            self.fixture(out)
            plan = read_plan(out)
            plan['excluded_speakers'].append('s0')
            write_json(out/'plan.json', plan)
            with self.assertRaisesRegex(ValueError, 'Training-speaker separation'):
                read_plan(out)
            plan['excluded_speakers'].remove('s0')
            write_json(out/'plan.json', plan)
            Path(plan['views'][0]['file']).write_bytes(b'changed features')
            with self.assertRaisesRegex(ValueError, 'Frozen training features'):
                read_plan(out)


if __name__ == '__main__':
    unittest.main()

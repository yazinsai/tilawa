"""Partition provenance and matched whole-sequence optimizer behavior."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from student_balanced_head import (RECIPE, EXPERIMENT, objective, read_plan,
    selected_plan, speaker_partitions, train_update)
from student_head_refit import ROOT, digest, has_clean_transcript, select_original_records, write_json


def fixture_plan(out):
    """Small complete recording/view provenance, without recitation data."""
    import numpy as np

    records = [dict(key=f'{source}/{speaker}/{phone}', speaker=f'{source}/{speaker}',
                    source=source) for source in ('original', 'help')
               for speaker in range(3) for phone in (1, 2)]
    partitions = speaker_partitions(records)
    cache = out/'features'
    cache.mkdir()
    views = []
    for i, record in enumerate(records):
        record['partition'] = partitions[record['speaker']]
        phone = 1+i % 2
        for view in range(RECIPE['views']):
            path = cache/f'{i}-{view}.npz'
            # The rival wins every frame, but the complete transcript is legal.
            p = np.full((5, 4), .05)
            p[:, 3-phone] = .85
            np.savez(path, features=np.eye(3, dtype='float32')[[0, 1, 2, 0, 1]],
                     baseline=np.log(p).astype('float32'), target=np.asarray([phone], dtype='int64'))
            views.append(dict(file=str(path), sha256=digest(path), recording=record['key'],
                speaker=record['speaker'], partition=record['partition'], supervised=True, view=view))
    np.savez(out/'initial-head.npz', weight=np.zeros((4, 3), dtype='float32'),
             bias=np.zeros(4, dtype='float32'))
    (out/'encoder-features.onnx').write_bytes(b'synthetic feature provenance')
    plan = dict(schema=1, id=EXPERIMENT, recipe=RECIPE, records=records, views=views,
        excluded_speakers=[], primary_sha256='synthetic', preparation_seconds=0.,
        initial_head_sha256=digest(out/'initial-head.npz'),
        feature_graph_sha256=digest(out/'encoder-features.onnx'))
    write_json(out/'plan.json', plan)
    return plan


class Speakers(unittest.TestCase):
    def test_excludes_complete_original_speakers_before_sampling(self):
        rows = [dict(key=str(i), shard='s', row=i, speaker=speaker, cluster=speaker,
                     split=split, bucket='clean-match', n_samples=16000)
                for i, (speaker, split) in enumerate([('a', 'train'), ('a', 'dev'),
                    ('b', 'train'), ('b', 'train'), ('c', 'train'), ('d', 'train')])]
        admitted, banned = select_original_records(rows, {('s', '2')}, 100, 30)
        self.assertEqual({r['speaker'] for r in admitted}, {'c', 'd'})
        self.assertEqual(banned, {'a', 'b'})

    def test_validation_separates_whole_training_speakers_in_both_sources(self):
        rows = [dict(speaker=f'{source}/{speaker}', source=source)
                for source in ('original', 'help') for speaker in range(10) for _ in range(2)]
        partitions = speaker_partitions(rows)
        for source in ('original', 'help'):
            self.assertEqual(sum(partitions[f'{source}/{i}'] == 'validation' for i in range(10)), 2)
        self.assertEqual(partitions, speaker_partitions(list(reversed(rows))))

    def test_scripted_deviations_or_repeats_never_become_verified_transcripts(self):
        self.assertTrue(has_clean_transcript(dict(kind='none')))
        for label in [dict(kind='wrong_word'), dict(kind='none', events=[{}]),
                      dict(kind='none', non_errors=[{}]), dict(kind='none', unsupported=[{}]),
                      dict(kind='none', scored=False)]:
            self.assertFalse(has_clean_transcript(label))


class Provenance(unittest.TestCase):
    def test_plan_rejects_missing_duplicate_or_cross_speaker_views_and_tampered_cache(self):
        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            out = Path(tmp)
            original = fixture_plan(out)
            self.assertEqual(read_plan(out), original)
            for change in ('missing', 'duplicate', 'speaker', 'partition', 'excluded'):
                plan = copy.deepcopy(original)
                if change == 'missing':
                    plan['views'].pop()
                elif change == 'duplicate':
                    plan['views'][-1] = plan['views'][0]
                elif change == 'speaker':
                    plan['views'][0]['speaker'] = plan['views'][-1]['speaker']
                elif change == 'partition':
                    plan['views'][0]['partition'] = 'other'
                else:
                    plan['excluded_speakers'] = [plan['views'][0]['speaker']]
                write_json(out/'plan.json', plan)
                with self.assertRaises(ValueError):
                    read_plan(out)
            write_json(out/'plan.json', original)
            Path(original['views'][0]['file']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                read_plan(out)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required for synthetic optimizer fixtures')
class Optimizer(unittest.TestCase):
    def test_complete_refit_freezes_controls_at_selected_exposure_and_refuses_rerun(self):
        import contextlib
        import io
        from unittest.mock import patch
        from student_balanced_head import fit

        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            out = Path(tmp)
            plan = fixture_plan(out)
            with patch('student.TOKENS', 'بتا'), contextlib.redirect_stdout(io.StringIO()):
                fit(out)
            selection = json.loads((out/'selection.json').read_text())
            self.assertEqual(set(selection['arms']), {'balanced', 'shuffled', 'length'})
            self.assertEqual(len(selection['history']), RECIPE['epochs'])
            training = [v for v in plan['views'] if v['partition'] == 'train']
            self.assertEqual(selection['final_exposures_per_arm'], len(training)*RECIPE['epochs'])
            self.assertEqual(len(set(selection['updates'].values())), 1)
            self.assertEqual(selection['grouping_sha256'], digest(out/'grouping.json'))
            self.assertEqual(selected_plan(out), plan)
            schedules = [json.loads(row) for row in (out/'batches.jsonl').read_text().splitlines()]
            self.assertEqual(len(schedules), RECIPE['epochs']*RECIPE['views'])
            changed = copy.deepcopy(selection)
            changed['updates']['length'] += 1
            write_json(out/'selection.json', changed)
            with self.assertRaisesRegex(ValueError, 'control exposure differs'):
                selected_plan(out)
            write_json(out/'selection.json', selection)
            (out/'batches.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'audit changed'):
                selected_plan(out)
            with self.assertRaisesRegex(ValueError, 'already started'):
                fit(out)

    def test_simultaneous_accumulation_matches_one_batch_gradient_and_momentum_step(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(14)
        head = torch.nn.Linear(3, 4)
        control = copy.deepcopy(head)
        data = [dict(features=torch.randn(5+i, 3),
                     baseline=torch.randn(5+i, 4).log_softmax(-1),
                     target=torch.tensor([1, 2])) for i in range(3)]
        a, b = [torch.optim.SGD(h.parameters(), lr=RECIPE['lr'], momentum=RECIPE['momentum'])
                for h in (head, control)]
        # Include a partial batch: its fixed divisor must not amplify samples.
        for _ in range(2):
            train_update(head, a, data, 8)
            b.zero_grad(set_to_none=True)
            (sum(objective(control, d) for d in data)/8).backward()
            b.step()
        for x, y in zip(head.parameters(), control.parameters()):
            torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-6)
        self.assertEqual(a.param_groups[0]['lr'], b.param_groups[0]['lr'])

    def test_full_inventory_sequence_objective_rejects_empty_or_impossible_targets(self):
        import torch
        h = torch.nn.Linear(3, 4)
        data = dict(features=torch.randn(2, 3), baseline=torch.randn(2, 4).log_softmax(-1),
                    target=torch.tensor([], dtype=torch.long))
        with self.assertRaises(ValueError):
            objective(h, data)
        data['target'] = torch.tensor([1, 1])
        with self.assertRaises(ValueError):
            objective(h, data)


if __name__ == '__main__':
    unittest.main()

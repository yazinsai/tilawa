"""Synthetic optimization checks; no practice audio or labels."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import student_trajectory_head as trajectory
from student_head_refit import ROOT, digest, has_clean_transcript, select_original_records, write_json


class Trajectory(unittest.TestCase):
    def test_average_is_equal_per_update_and_independent_of_live_parameters(self):
        import torch

        head = torch.nn.Linear(1, 1)
        average = trajectory.ParameterAverage()
        for w, b in [(0., 3.), (9., 6.), (3., 12.)]:
            with torch.no_grad():
                head.weight.fill_(w)
                head.bias.fill_(b)
            average.add(head)
        actual = average.head(head)
        self.assertEqual(average.count, 3)
        self.assertEqual(actual.weight.item(), 4.)
        self.assertEqual(actual.bias.item(), 7.)
        with torch.no_grad():
            head.weight.fill_(100.)
        self.assertEqual(average.head(head).weight.item(), 4.)
        with self.assertRaisesRegex(ValueError, 'empty'):
            trajectory.ParameterAverage().head(head)

    def test_training_convergence_requires_a_stable_tail_not_one_lucky_loss(self):
        self.assertFalse(trajectory.converged([1., .8, .8, .8]))
        self.assertFalse(trajectory.converged([1., .8, .8, .8, .8]))
        self.assertTrue(trajectory.converged([1., .8, .8, .8, .8, .8]))
        self.assertFalse(trajectory.converged([1., .8, .7, .6, .5, .4]))
        self.assertFalse(trajectory.converged([1., .8, .81, .8, .81, .8]))
        with self.assertRaises(ValueError):
            trajectory.converged([1., 1., 1., 1., float('nan')])

    def test_speakers_and_scripted_recordings_cannot_cross_training_partitions(self):
        records = [dict(speaker=f'{source}/{i}', source=source)
                   for source in ('original', 'help') for i in range(10)]
        mapping = trajectory.speaker_partitions(records+records)
        self.assertEqual(len(mapping), 20)
        self.assertEqual(sum(p == 'validation' for p in mapping.values()), 4)
        self.assertEqual(mapping, trajectory.speaker_partitions(list(reversed(records))))
        self.assertTrue(has_clean_transcript(dict(kind='none', split='train', events=[], non_errors=[])))
        for field in ('events', 'non_errors', 'unsupported'):
            self.assertFalse(has_clean_transcript(dict(kind='none', **{field: [{}]})))
        self.assertFalse(has_clean_transcript(dict(kind='wrong_word')))
        meta = [dict(speaker='a', split='train', shard='s', row=0, cluster=1,
                     bucket='clean-match', n_samples=16000, key='a'),
                dict(speaker='a', split='eval', shard='s', row=1, cluster=1,
                     bucket='clean-match', n_samples=16000, key='b'),
                dict(speaker='c', split='train', shard='s', row=2, cluster=2,
                     bucket='clean-match', n_samples=16000, key='c')]
        selected, banned = select_original_records(meta, set(), 10, 30)
        self.assertEqual([r['key'] for r in selected], ['c'])
        self.assertEqual(banned, {'a'})

    def test_preservation_uses_shipped_outputs_without_scripted_transcripts(self):
        import torch

        h = torch.nn.Linear(2, 3)
        x = torch.tensor([[1., 2.], [3., 1.]])
        baseline = h(x).detach().log_softmax(-1)
        d = dict(features=x, baseline=baseline, target=torch.tensor([], dtype=torch.long))
        self.assertAlmostEqual(float(trajectory.objective(h, d).detach()), 0., places=6)
        with torch.no_grad():
            h.bias[1] += 1.
        self.assertGreater(float(trajectory.objective(h, d).detach()), 0.)

    def test_tiny_fit_freezes_selection_and_matches_control_compute(self):
        import numpy as np
        import torch

        recipe = dict(trajectory.RECIPE, batch=1, preservation_batch=1, warmup_max_epochs=8,
                      convergence_relative=1., convergence_patience=2, tail_epochs=4,
                      average_last_epochs=[4, 2])
        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as temp:
            out = Path(temp)
            torch.manual_seed(4)
            h = torch.nn.Linear(2, 3)
            np.savez(out/'initial-head.npz', weight=h.weight.detach().numpy(), bias=h.bias.detach().numpy())
            (out/'encoder-features.onnx').write_bytes(b'frozen synthetic encoder provenance')
            views = []
            for i, (partition, supervised) in enumerate([('train', True), ('train', False),
                                                       ('validation', True), ('validation', False)]):
                x = np.asarray([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype='float32')+i*.1
                path = out/f'view-{i}.npz'
                lp = h(torch.from_numpy(x)).detach().log_softmax(-1).numpy()
                np.savez(path, features=x, baseline=lp,
                         target=np.asarray([1, 2] if supervised else [], dtype='int64'))
                views.append(dict(file=str(path), sha256=digest(path), speaker=f'speaker-{i}',
                                  partition=partition, supervised=supervised))
            plan = dict(id=trajectory.EXPERIMENT, recipe=recipe, views=views, excluded_speakers=[],
                initial_head_sha256=digest(out/'initial-head.npz'),
                feature_graph_sha256=digest(out/'encoder-features.onnx'))
            write_json(out/'plan.json', plan)
            # Observe actual loss inputs and optimizer steps independently of
            # the implementation's comparison report.
            pending, batches, rates = {}, {}, {}
            original_objective = trajectory.objective
            original_step = torch.optim.SGD.step

            def observed_objective(head, data):
                if torch.is_grad_enabled():
                    key = id(head.weight)
                    pending.setdefault(key, []).append((data['features'].numpy().tobytes(),
                                                        data['target'].numpy().tobytes()))
                return original_objective(head, data)

            def observed_step(optimizer, *args, **kwargs):
                key = id(optimizer.param_groups[0]['params'][0])
                batches.setdefault(key, []).append(pending.pop(key))
                rates.setdefault(key, []).append(optimizer.param_groups[0]['lr'])
                return original_step(optimizer, *args, **kwargs)

            with patch.object(trajectory, 'RECIPE', recipe), \
                    patch.object(trajectory, 'objective', observed_objective), \
                    patch.object(torch.optim.SGD, 'step', observed_step):
                trajectory.fit(out)
                selection = json.loads((out/'selection.json').read_text())
                self.assertEqual(len(batches), 2)
                primary, control = sorted(batches, key=lambda k: -len(batches[k]))
                self.assertEqual(len(batches[primary]), selection['updates'])
                self.assertEqual(len(batches[control]), recipe['tail_epochs'])
                self.assertEqual(batches[primary][-recipe['tail_epochs']:], batches[control])
                self.assertEqual(rates[primary], [recipe['lr']]*selection['updates'])
                self.assertEqual(rates[control], [trajectory.continuation_lr(i)
                                                 for i in range(recipe['tail_epochs'])])
                self.assertTrue(selection['identical_minibatches'])
                self.assertTrue(selection['identical_update_counts'])
                self.assertEqual(selection['average_counts'], {'4': 4, '2': 2})
                self.assertEqual(set(selection['arms']), {'averaged', 'terminal', 'ordinary'})
                losses = selection['validation']
                best = min([4, 2], key=lambda n: (losses[f'average-last-{n}'], -n))
                self.assertEqual(selection['selected_last_epochs'], best)
                for ck in selection['arms'].values():
                    self.assertEqual(ck['sha256'], digest(ck['path']))
                # The ordinary annealing schedule affects only its control;
                # the single averaged trajectory retains the fixed tail rate.
                self.assertGreater(trajectory.continuation_lr(0), trajectory.continuation_lr(3))
                self.assertTrue(json.loads((out/'convergence.json').read_text())['converged'])
                path = Path(views[0]['file'])
                path.write_bytes(b'changed feature cache')
                with self.assertRaisesRegex(ValueError, 'cache changed'):
                    trajectory.read_plan(out)


if __name__ == '__main__':
    unittest.main()

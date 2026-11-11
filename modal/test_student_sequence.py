"""Exact-path checks; tiny fixtures run locally without any acoustic workload."""
import itertools
import math
import unittest

from student_sequence import (collapse, ctc_score, transcriptions, candidate_union,
                              speaker_groups, sequence_probabilities, sequence_disagreement,
                              admitted_none, head_objective, phone_head, matched_refit)


def logs(rows):
    return [[math.log(p) for p in row] for row in rows]


def enumerate_mass(rows):
    result = {}
    for path in itertools.product(range(len(rows[0])), repeat=len(rows)):
        seq = collapse(path)
        mass = math.prod(rows[t][p] for t, p in enumerate(path))
        result[seq] = result.get(seq, 0.) + mass
    return result


class ExactTranscriptions(unittest.TestCase):
    def test_every_transcription_matches_enumerated_frame_paths(self):
        rows = [[.4, .35, .25], [.2, .5, .3], [.3, .2, .5], [.6, .3, .1]]
        masses = enumerate_mass(rows)
        for seq, mass in masses.items():
            self.assertAlmostEqual(math.exp(ctc_score(logs(rows), seq)), mass, places=12)
        self.assertAlmostEqual(sum(math.exp(ctc_score(logs(rows), s)) for s in masses), 1.)

    def test_duplicate_phone_occurrences_require_blank(self):
        self.assertEqual(collapse([1, 1, 0, 1]), (1, 1))
        self.assertEqual(ctc_score(logs([[.1, .9], [.1, .9]]), (1, 1)), -math.inf)
        self.assertAlmostEqual(math.exp(ctc_score(logs([[.1, .9]] * 3), (1, 1))), .081)

    def test_complete_beam_is_exact_and_cap_is_deterministic(self):
        rows = [[.4, .35, .25], [.2, .5, .3], [.3, .2, .5]]
        expected = tuple(sorted(enumerate_mass(rows), key=lambda s: (-enumerate_mass(rows)[s], s)))
        self.assertEqual(transcriptions(logs(rows), 100, len(expected)), expected)
        self.assertEqual(transcriptions(logs(rows), 100, 3), expected[:3])

    def test_union_keeps_whole_ordered_alternatives_from_both_models(self):
        teacher = logs([[.05, .9, .05], [.9, .05, .05], [.05, .05, .9]])
        student = logs([[.9, .05, .05], [.9, .05, .05], [.05, .05, .9]])
        self.assertEqual(transcriptions(teacher, 1, 1), ((1, 2),))
        self.assertEqual(transcriptions(student, 1, 1), ((2,),))
        self.assertEqual(candidate_union(teacher, student, 1, 1), ((1, 2), (2,)))

    def test_invalid_candidate_and_limits_fail(self):
        with self.assertRaises(ValueError):
            ctc_score(logs([[.5, .5]]), (0,))
        with self.assertRaises(ValueError):
            transcriptions(logs([[.5, .5]]), 1, 2)
        for rows in ([[0., 0.]], [[math.nan, 0.]], [[0., -1.], [0.]]):
            with self.assertRaises(ValueError):
                transcriptions(rows)
        with self.assertRaises(ValueError):
            candidate_union(logs([[.5, .5]]), logs([[.2, .4, .4]]))

    def test_speaker_identity_is_stable_and_excluded_whole(self):
        speakers = [f"training-person-{i}" for i in range(20)]
        self.assertEqual(speaker_groups(speakers), speaker_groups(speakers[::-1] + speakers))
        self.assertEqual(set(speaker_groups(speakers).values()), {"fit", "selection"})
        with self.assertRaises(ValueError):
            speaker_groups(speakers, [speakers[0]])

    def test_none_enrollment_excludes_holdout_speakers_and_repeats(self):
        good = dict(id="synthetic", split="train", kind="none", speaker="fit", scored=True,
                    events=[], non_errors=[], unsupported=[], wav="x.wav", ayahs=["synthetic"])
        bad = [dict(good, id="repeat", non_errors=[{"kind": "repeat"}]),
               dict(good, id="script", kind="skipped_word"),
               dict(good, id="unscored", scored=False),
               dict(good, id="excluded", excluded=True)]
        self.assertEqual(admitted_none([good, *bad]), [good])
        self.assertEqual(admitted_none([good, dict(good, split="held-out")]), [])
        self.assertEqual(admitted_none([good], ["fit"]), [])
        with self.assertRaises(ValueError):
            admitted_none([good, good])


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU torch is required for differentiable-head checks")
class DifferentiableDistribution(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_other_is_exact_complement_not_a_truncated_renormalization(self):
        rows = [[.4, .35, .25], [.2, .5, .3], [.3, .2, .5]]
        lp = torch.tensor(logs(rows), dtype=torch.float64)
        p = sequence_probabilities(lp, [(), (1, 2)])
        masses = enumerate_mass(rows)
        torch.testing.assert_close(p, torch.tensor([masses[()], masses[(1, 2)],
            1 - masses[()] - masses[(1, 2)]], dtype=torch.float64))
        for candidates in ([(), ()], [(1.5,)], [(0,)]):
            with self.assertRaises(ValueError):
                sequence_probabilities(lp, candidates)
        with self.assertRaises(ValueError):
            sequence_probabilities(torch.zeros_like(lp), [()])

    def test_gradient_matches_full_enumeration(self):
        logits = torch.tensor(logs([[.4, .35, .25], [.2, .5, .3], [.3, .2, .5]]),
                              dtype=torch.float64, requires_grad=True)
        lp = logits.log_softmax(-1)
        teacher = torch.tensor(logs([[.3, .5, .2], [.5, .25, .25], [.3, .2, .5]]), dtype=torch.float64)
        candidates = [(), (1, 2)]
        loss = sequence_disagreement(lp, teacher, candidates)
        got = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        masses = [lp.new_zeros(()) for _ in range(3)]
        for path in itertools.product(range(3), repeat=3):
            seq = collapse(path)
            bucket = candidates.index(seq) if seq in candidates else 2
            masses[bucket] = masses[bucket] + sum(lp[t, p] for t, p in enumerate(path)).exp()
        q = sequence_probabilities(teacher, candidates)
        exact = (q * (q.log() - torch.stack(masses).log())).sum()
        want = torch.autograd.grad(exact, logits)[0]
        torch.testing.assert_close(loss, exact)
        torch.testing.assert_close(got, want)

    def test_timing_and_frame_count_need_not_match(self):
        a = torch.tensor(logs([[.1, .9], [.8, .2]]), dtype=torch.float64)
        b = a.flip(0)
        self.assertAlmostEqual(float(sequence_disagreement(a, b, [(), (1,)])), 0., places=12)
        c = torch.tensor(logs([[.5, .5]] * 3), dtype=torch.float64)
        self.assertTrue(torch.isfinite(sequence_disagreement(a, c, [(), (1,), (1, 1)])))

    def test_impossible_student_sequence_does_not_produce_nan_gradient(self):
        logits = torch.tensor([[0., 1.]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor(logs([[.1, .9]] * 3), dtype=torch.float64)
        loss = sequence_disagreement(logits.log_softmax(-1), t, [(1,), (1, 1)])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_feature_teacher_and_shipped_evidence_are_frozen(self):
        head = phone_head([[.1, .2], [.3, -.1], [-.1, .2]], [0., 0., 0.])
        features = torch.arange(16, dtype=torch.float32).reshape(8, 2).requires_grad_()
        original = head(features).log_softmax(-1).detach().requires_grad_()
        teacher = original.detach().clone().requires_grad_()
        item = dict(features=features, shipped=original, teacher=teacher, target=torch.tensor([1, 2]))
        loss = head_objective(head, item, "sequence", [(1, 2), ()])
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(original.grad)
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(head.weight.grad.abs().sum()), 0.)
        with self.assertRaises(ValueError):
            head_objective(head, dict(item, target=torch.tensor([1] * 5)), "sequence", [(1, 2)])

    def test_matched_arms_share_order_data_and_selected_update_count(self):
        from unittest.mock import patch
        import student_sequence as module
        head = phone_head([[.1, .2], [.3, -.1], [-.1, .2]], [0., 0., 0.])
        rows = []
        for i, speaker in enumerate(["fit", "fit", "selection"]):
            x = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 20 + i
            lp = head(x).log_softmax(-1).detach()
            rows.append(dict(speaker=speaker, views=[dict(features=x, shipped=lp, teacher=lp,
                                                        target=torch.tensor([1, 2]))]))
        calls, checks = [], []
        objective = module.head_objective
        def record(head, item, arm, candidates):
            calls.append((arm, id(item)))
            return objective(head, item, arm, candidates)
        def validate(head, selection, beam, retain):
            self.assertEqual([r["speaker"] for r in selection], ["selection"])
            epoch = len(checks) // 2
            checks.append(epoch)
            return dict(per=[4., 2., 1., 3.][epoch], objective=0.)
        with patch.object(module, "head_objective", side_effect=record):
            heads, report = matched_refit(head, rows, dict(fit="fit", selection="selection"),
                                         validate, 4, 4, epochs=3, learning_rate=.001)
        self.assertEqual(set(heads), {"sequence", "framewise"})
        self.assertEqual(report["selectedEpoch"], 2)
        self.assertEqual(report["selectedUpdates"], 4)
        self.assertEqual(report["trainingUpdates"], 6)
        self.assertEqual(len(calls), 12)
        for a, b in zip(calls[::2], calls[1::2]):
            self.assertEqual((a[0], b[0]), ("sequence", "framewise"))
            self.assertEqual(a[1], b[1])


if __name__ == "__main__":
    unittest.main()

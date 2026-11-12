"""Edit-risk contracts against exhaustive tiny CTC distributions; no lab data."""
import importlib.util
import itertools
import math
import unittest
from unittest.mock import patch

from student_edit_risk import (admitted_none, candidate_union, collapse, ctc_score,
    edit_risk, head_objective, levenshtein, matched_refit, phone_head,
    sequence_log_probabilities, speaker_groups, transcriptions)

TORCH = importlib.util.find_spec("torch") is not None


def distribution(rows):
    result = {}
    for path in itertools.product(range(len(rows[0])), repeat=len(rows)):
        hyp = collapse(path)
        probability = math.exp(sum(rows[t][p] for t, p in enumerate(path)))
        result[hyp] = result.get(hyp, 0.) + probability
    return result


class LatticeTests(unittest.TestCase):
    def setUp(self):
        self.rows = [[math.log(p) for p in row] for row in
                     ((.3, .5, .2), (.6, .1, .3), (.2, .3, .5))]

    def test_forward_sums_merge_collapsed_paths_and_repeated_labels(self):
        oracle = distribution(self.rows)
        self.assertEqual(collapse((1, 1, 0, 1)), (1, 1))
        for hyp, probability in oracle.items():
            self.assertAlmostEqual(math.exp(ctc_score(self.rows, hyp)), probability, places=13)
        self.assertEqual(ctc_score(self.rows[:2], (1, 1)), -math.inf)
        self.assertAlmostEqual(sum(math.exp(ctc_score(self.rows, h)) for h in oracle), 1.)

    def test_unpruned_beam_matches_exhaustive_transcription_ranking(self):
        oracle = distribution(self.rows)
        expected = tuple(sorted(oracle, key=lambda h: (-oracle[h], h)))
        self.assertEqual(transcriptions(self.rows, 100, len(expected)), expected)

    def test_reference_free_unique_bounded_candidates(self):
        union = candidate_union(self.rows, self.rows, 4, 3)
        self.assertEqual(len(union), len(set(union)))
        self.assertLessEqual(len(union), 6)
        # Proposal APIs have no transcript argument; scoring different references
        # leaves the acoustic proposal exactly the same.
        for ref in ((1,), (2, 1), (1, 1, 2)):
            [levenshtein(h, ref) for h in union]
            self.assertEqual(candidate_union(self.rows, self.rows, 4, 3), union)

    def test_invalid_probability_rows_and_bounds_rejected(self):
        for rows in ([[0., 0.]], [[math.nan, -1.]], [[-1., -1.], [-1.]]):
            with self.assertRaises(ValueError):
                transcriptions(rows)
        for beam, retain in ((0, 0), (2, 3), (2, 0)):
            with self.assertRaises(ValueError):
                transcriptions(self.rows, beam, retain)

    def test_edit_distance_rewards_recovery_and_penalizes_invention_and_order(self):
        reference = (1, 2, 3)
        self.assertEqual([levenshtein(h, reference) for h in
                          ((), (3,), (2, 3), reference, (1, 2, 3, 2), (3, 2, 1))],
                         [3, 2, 1, 0, 1, 2])

    def test_training_speaker_exclusion_and_strict_none_enrollment(self):
        row = dict(id="take", speaker="learner", split="train", kind="none", scored=True,
                   events=[], non_errors=[], unsupported=[], wav="take.wav", ayahs=["1:1"])
        self.assertEqual(admitted_none([row]), [row])
        self.assertEqual(admitted_none([row, dict(row, split="test", id="held")]), [])
        for change in (dict(kind="repeat"), dict(non_errors=[{}]), dict(events=[{}]),
                       dict(excluded=True), dict(scored=False)):
            self.assertEqual(admitted_none([dict(row, **change)]), [])
        with self.assertRaises(ValueError):
            speaker_groups(["learner"], ["learner"])
        groups = speaker_groups(["learner", "other", "learner"])
        self.assertEqual(groups, speaker_groups(["other", "learner"]))


@unittest.skipUnless(TORCH, "Torch required on supervisor")
class RiskTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        self.logits = torch.tensor([[.1, .8, -.3], [.4, -.1, .5], [-.2, .3, .9]],
                                   dtype=torch.float64, requires_grad=True)

    def exhaustive_risk(self, logits, target):
        torch = self.torch
        lp = logits.log_softmax(-1)
        terms = []
        for path in itertools.product(range(lp.shape[1]), repeat=len(lp)):
            terms.append(sum(lp[t, p] for t, p in enumerate(path)).exp() *
                         levenshtein(collapse(path), target) / max(1, len(target)))
        return torch.stack(terms).sum()

    def test_full_distribution_risk_and_gradient_match_exhaustive_paths(self):
        lp = self.logits.log_softmax(-1)
        candidates = tuple(distribution(lp.detach().tolist()))
        got = edit_risk(lp, candidates, (1, 2))
        expected = self.exhaustive_risk(self.logits, (1, 2))
        self.assertAlmostEqual(float(got["conditional"].detach()), float(expected.detach()), places=12)
        self.assertAlmostEqual(float(got["retainedMass"].detach()), 1., places=12)
        self.assertAlmostEqual(float(got["otherMass"].detach()), 0., places=12)
        a = self.torch.autograd.grad(got["conditional"], self.logits, retain_graph=True)[0]
        b = self.torch.autograd.grad(expected, self.logits)[0]
        self.torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_omitted_mass_and_bounds_do_not_claim_conditional_risk_is_full(self):
        lp = self.logits.log_softmax(-1)
        candidates = ((), (1, 2), (1, 1))
        oracle = distribution(lp.detach().tolist())
        got = edit_risk(lp, candidates, (1, 2))
        retained = sum(oracle[h] for h in candidates)
        self.assertAlmostEqual(float(got["retainedMass"].detach()), retained, places=12)
        self.assertAlmostEqual(float(got["otherMass"].detach()), 1 - retained, places=12)
        full = float(self.exhaustive_risk(self.logits, (1, 2)))
        self.assertLessEqual(float(got["lower"].detach()), full)
        self.assertGreaterEqual(float(got["upper"].detach()), full)
        self.assertAlmostEqual(float(got["coverage"].detach()), -math.log(retained) / len(lp))

    def test_partial_lattice_objective_gradient_matches_finite_difference(self):
        candidates = ((), (1,), (1, 2), (2, 1))
        objective = edit_risk(self.logits.log_softmax(-1), candidates, (1, 2))["objective"]
        gradient = self.torch.autograd.grad(objective, self.logits)[0]
        for frame, phone in itertools.product(range(3), range(3)):
            plus, minus = self.logits.detach().clone(), self.logits.detach().clone()
            plus[frame, phone] += 1e-5
            minus[frame, phone] -= 1e-5
            a = edit_risk(plus.log_softmax(-1), candidates, (1, 2))["objective"]
            b = edit_risk(minus.log_softmax(-1), candidates, (1, 2))["objective"]
            self.assertAlmostEqual(float(gradient[frame, phone]), float((a - b) / 2e-5), places=7)

    def test_rare_lattice_is_stable_without_probability_floor(self):
        logits = self.torch.zeros((1200, 3), dtype=self.torch.float64, requires_grad=True)
        got = edit_risk(logits.log_softmax(-1), ((), (1,)), (1,))
        self.assertTrue(self.torch.isfinite(got["objective"]))
        self.assertEqual(float(got["retainedMass"].detach()), 0.)
        self.assertEqual(float(got["otherMass"].detach()), 1.)
        got["objective"].backward()
        self.assertTrue(self.torch.isfinite(logits.grad).all())

    def test_duplicates_impossible_sequences_and_blank_are_handled(self):
        lp = self.logits.log_softmax(-1)
        with self.assertRaises(ValueError):
            sequence_log_probabilities(lp, ((1,), (1,)))
        with self.assertRaises(ValueError):
            edit_risk(lp, ((0,),), (1,))
        scores = sequence_log_probabilities(lp, ((), (1, 1, 1)))
        self.assertTrue(self.torch.isneginf(scores[1]))
        with self.assertRaises(ValueError):
            edit_risk(lp, ((1, 1, 1),), (1,))

    def item(self):
        features = self.torch.tensor([[1., .2], [.1, .4], [.2, .9]], requires_grad=True)
        head = phone_head([[0., 0.], [.3, .1], [.1, .4]], [0., 0., 0.])
        shipped = head(features).log_softmax(-1).detach().requires_grad_()
        return head, dict(features=features, shipped=shipped, target=self.torch.tensor([1, 2]))

    def test_control_is_ordinary_ctc_with_shared_preservation_and_frozen_evidence(self):
        head, item = self.item()
        lp = head(item["features"].detach()).log_softmax(-1)
        expected = self.torch.nn.functional.ctc_loss(lp[:, None], item["target"],
            self.torch.tensor([3]), self.torch.tensor([2]), reduction="sum") / 3
        ordinary = head_objective(head, item, "ctc")
        self.torch.testing.assert_close(ordinary, expected)
        candidates = candidate_union(item["shipped"].tolist(), lp.detach().tolist(), 16, 16)
        risk = head_objective(head, item, "risk", candidates)
        self.torch.testing.assert_close(risk - ordinary,
            edit_risk(lp, candidates, (1, 2))["objective"], check_dtype=False)
        risk.backward()
        self.assertIsNone(item["features"].grad)
        self.assertIsNone(item["shipped"].grad)
        self.assertTrue(self.torch.isfinite(head.weight.grad).all())

    def test_matched_updates_refresh_candidates_and_selection_sees_only_training_holdout(self):
        head, item = self.item()
        rows = [dict(speaker="fit", views=[item]), dict(speaker="selection", views=[item])]
        calls, proposals = [], []
        def validate(model, selection, beam, retain):
            self.assertEqual([r["speaker"] for r in selection], ["selection"])
            calls.append(model.weight.detach().clone())
            return dict(per=1., objective=float((model.weight.detach() ** 2).sum()))
        import student_edit_risk as module
        original = module.candidate_union
        def propose(baseline, current, beam, retain):
            proposals.append(current)
            return original(baseline, current, beam, retain)
        with patch.object(module, "candidate_union", side_effect=propose):
            heads, report = matched_refit(head, rows, {"fit": "fit", "selection": "selection"},
                validate, 16, 16, epochs=2, learning_rate=.02)
        self.assertEqual(set(heads), {"risk", "ctc"})
        self.assertEqual(report["trainingUpdates"], 2)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(proposals), 4)
        self.assertNotEqual(proposals[0], proposals[2])
        self.assertTrue(all(h["updates"] == h["epoch"] for h in report["history"]))


@unittest.skipUnless(TORCH, "Torch required to import the training runner")
class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        import student_edit_risk_train as runner
        self.runner = runner
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.help = self.root / "help.jsonl"
        self.transcribed = self.root / "transcribed.jsonl"
        (self.root / "none.wav").write_bytes(b"synthetic none audio identity")
        (self.root / "read.wav").write_bytes(b"synthetic transcribed audio identity")
        self.none = dict(id="none", speaker="help-speaker", split="train", kind="none", scored=True,
            events=[], non_errors=[], unsupported=[], wav="none.wav", ayahs=["1:1"])
        self.additional = dict(speaker="transcribed-speaker", split="train", authorized=True, verified=True,
            wav="read.wav", phones=runner.TOKENS[:2], sha256=runner.digest(self.root / "read.wav"))
        self.reference = dict(ayahs=[dict(s=1, a=1, p=runner.TOKENS[:2])])
        self.write = lambda p, rows: p.write_text("\n".join(json.dumps(r) for r in rows))
        self.write(self.help, [self.none])
        self.write(self.transcribed, [self.additional])

    def enroll(self):
        return self.runner.recordings(self.help, self.transcribed, [], self.reference)

    def test_verified_recordings_require_audio_identity_and_authorization(self):
        rows, excluded = self.enroll()
        self.assertEqual({r["source"] for r in rows}, {"help-none", "transcribed"})
        self.assertEqual(excluded, [])
        for change in (dict(authorized=False), dict(verified=False), dict(sha256="wrong")):
            self.write(self.transcribed, [dict(self.additional, **change)])
            with self.assertRaises(ValueError):
                self.enroll()

    def test_excluded_speaker_is_discovered_across_corpora_before_enrollment(self):
        # A nontraining record in the second corpus also excludes its help take.
        self.write(self.transcribed, [self.additional,
            dict(self.additional, speaker="help-speaker", split="release")])
        with self.assertRaisesRegex(ValueError, "Both authorized"):
            self.enroll()

    def test_duplicate_audio_and_path_escape_are_rejected(self):
        self.write(self.transcribed, [dict(self.additional, wav="none.wav",
            sha256=self.runner.digest(self.root / "none.wav"))])
        with self.assertRaisesRegex(ValueError, "Repeated audio"):
            self.enroll()
        with self.assertRaisesRegex(ValueError, "leaves"):
            self.runner.contained(self.root, "../escaped.wav")

    def test_lattice_bounds_do_not_depend_on_recorded_transcript(self):
        import torch
        lp = torch.tensor([[.3, .5, .2], [.6, .1, .3], [.2, .3, .5]]).log()
        rows = [dict(speaker="training-selection", views=[dict(shipped=lp, target=(1, 2))])]
        before = self.runner.choose_limits(rows)
        rows[0]["views"][0]["target"] = (2, 2, 2, 2)
        self.assertEqual(before, self.runner.choose_limits(rows))


if __name__ == "__main__":
    unittest.main()

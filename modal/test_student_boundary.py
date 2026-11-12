"""Small exhaustive oracles for the training-only boundary objective."""

import copy
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from student_boundary import (BoundaryCTC, authorized_none, objectives, recording_update,
                              speaker_partition, spelling)


def collapse(path):
    return [c for i, c in enumerate(path) if c and (i == 0 or c != path[i - 1])]


def paths(probabilities, words, cuts):
    out = []
    edges = (0, *cuts, len(probabilities))
    for path in itertools.product(range(probabilities.shape[1]), repeat=len(probabilities)):
        if all(collapse(path[a:b]) == word for word, a, b in zip(words, edges, edges[1:])):
            # No implicit CTC reset between equal articulations.
            if any(path[t - 1] != 0 and path[t - 1] == path[t] for t in cuts):
                continue
            out.append(math.prod(float(probabilities[t, c]) for t, c in enumerate(path)))
    return sum(out)


def exhaustive(phone, letter, phone_words, letter_words):
    cuts = list(itertools.combinations(range(1, len(phone)), len(phone_words) - 1))
    p = [paths(phone, phone_words, c) for c in cuts]
    l = [paths(letter, letter_words, c) for c in cuts]
    return dict(shared=math.log(sum(a * b for a, b in zip(p, l)) / len(cuts)),
                independent=math.log(sum(p) * sum(l) / len(cuts) ** 2))


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def compare(self, phone_words, letter_words, frames=4):
        p = torch.randn(frames, 3, dtype=torch.float64).log_softmax(-1)
        l = torch.randn(frames, 3, dtype=torch.float64).log_softmax(-1)
        got = BoundaryCTC(phone_words, letter_words, 3, 3).partitions(p, l)
        want = exhaustive(p.exp(), l.exp(), phone_words, letter_words)
        for name in got:
            self.assertAlmostEqual(got[name].item(), want[name], places=11)

    def test_exact_shared_and_independent_marginals(self):
        self.compare([[1], [2]], [[2], [1]])
        self.compare([[1, 2], [1]], [[2], [1, 2]])
        self.compare([[1], [2], [1]], [[2], [1], [2]])

    def test_repeated_occurrences_within_and_across_words(self):
        self.compare([[1, 1]], [[2, 2]])
        self.compare([[1], [1]], [[2], [1]])
        self.compare([[1], [2]], [[2], [2]])
        self.compare([[1], [1]], [[2], [2]])

    def test_single_word_is_product_of_ordinary_ctc(self):
        p = torch.randn(7, 3, dtype=torch.float64).log_softmax(-1)
        l = torch.randn(7, 3, dtype=torch.float64).log_softmax(-1)
        pw, lw = [[1, 1, 2]], [[2, 1]]
        values = BoundaryCTC(pw, lw, 3, 3).partitions(p, l)
        def ctc(lp, target):
            return -torch.nn.functional.ctc_loss(lp[:, None], torch.tensor(target),
                torch.tensor([len(lp)]), torch.tensor([len(target)]), reduction="sum")
        want = ctc(p, pw[0]) + ctc(l, lw[0])
        torch.testing.assert_close(values["shared"], want, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(values["shared"], values["independent"], atol=1e-11, rtol=1e-11)

    def test_heads_can_use_different_within_word_timings(self):
        # Both align the entire one-word interval, but emit on different frames.
        p = torch.tensor([[.05, .9, .05], [.9, .05, .05], [.9, .05, .05]], dtype=torch.float64).log()
        l = p.flip(0)
        got = BoundaryCTC([[1]], [[1]], 3, 3).partitions(p, l)
        self.assertGreater(got["shared"].item(), -1.)

    def test_incompatible_boundary_evidence_is_penalized(self):
        p = torch.tensor([[.01, .98, .01], [.01, .01, .98], [.01, .01, .98], [.01, .01, .98]], dtype=torch.float64).log()
        l = torch.tensor([[.01, .98, .01], [.01, .98, .01], [.01, .98, .01], [.01, .01, .98]], dtype=torch.float64).log()
        got = BoundaryCTC([[1], [2]], [[1], [2]], 3, 3).partitions(p, l)
        self.assertLess(got["shared"].item(), got["independent"].item() - 2)

    def test_both_gradients_and_unreachable_states(self):
        lattice = BoundaryCTC([[1, 1], [2]], [[2], [1, 2]], 3, 3)
        p = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        l = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        for arm in ("shared", "independent"):
            self.assertTrue(torch.autograd.gradcheck(
                lambda a, b: lattice.partitions(a.log_softmax(-1), b.log_softmax(-1))[arm],
                (p, l), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_impossible_transcript_is_not_zero_loss(self):
        lp = torch.randn(2, 3).log_softmax(-1)
        with self.assertRaisesRegex(ValueError, "no legal"):
            BoundaryCTC([[1, 1]], [[2]], 3, 3).partitions(lp, lp)

    def test_bad_shapes_and_transcripts(self):
        with self.assertRaises(ValueError):
            BoundaryCTC([[0]], [[1]], 3, 3)
        with self.assertRaises(ValueError):
            BoundaryCTC([[1]], [[1], [2]], 3, 3)
        with self.assertRaises(ValueError):
            BoundaryCTC([[]], [[1]], 3, 3)
        with self.assertRaises(ValueError):
            BoundaryCTC([[1]], [[1]], 3, 3).partitions(torch.zeros(3, 3), torch.zeros(2, 3))

    def test_matched_compute_uses_only_the_chosen_objective_gradient(self):
        features = torch.randn(4, 2, dtype=torch.float64)
        shipped = torch.randn(4, 3, dtype=torch.float64).log_softmax(-1)
        lattice = BoundaryCTC([[1], [2]], [[2], [1]], 3, 3)
        initial = torch.nn.ModuleDict({name: torch.nn.Linear(2, 3).double() for name in ("phone", "letter")})
        for arm in ("shared", "independent"):
            expected, actual = copy.deepcopy(initial), copy.deepcopy(initial)
            opt = torch.optim.SGD(expected.parameters(), lr=.01)
            opt.zero_grad()
            objectives(expected, features, shipped, lattice, 4)[arm].backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 5.)
            opt.step()
            recording_update(actual, torch.optim.SGD(actual.parameters(), lr=.01), arm,
                             features, shipped, lattice, 4)
            for a, b in zip(actual.parameters(), expected.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


class AdmissionTests(unittest.TestCase):
    def test_arabic_spelling(self):
        self.assertEqual(spelling("ٱلرَّحْمَـٰنِ"), "الرحمن")
        self.assertEqual(spelling("سُؤَالٌ"), "سؤال")
        self.assertEqual(spelling("إِيَّاكَ"), "إياك")
        self.assertEqual(spelling("لَهُۥ"), "له")
        self.assertEqual(spelling("فِيهِۦ"), "فيه")

    def test_none_only_not_mistake_scripts_or_repeats(self):
        row = dict(split="train", kind="none", speaker="train-speaker", ayahs=["reference"],
                   events=[], non_errors=[], unsupported=[], excluded=False)
        self.assertTrue(authorized_none(row))
        for key, value in [("split", "eval"), ("kind", "added_word"), ("events", [{}]),
                           ("non_errors", [{}]), ("unsupported", [{}]), ("excluded", True),
                           ("speaker", ""), ("ayahs", [])]:
            self.assertFalse(authorized_none(dict(row, **{key: value})))

    def test_speaker_partition_not_recording_partition(self):
        speakers = ["a", "b", "c", "d", "e", "f"]
        split = speaker_partition(speakers)
        self.assertEqual(split, speaker_partition(speakers[::-1] + speakers))
        self.assertEqual(set(split.values()), {"train", "validation"})
        with self.assertRaises(ValueError):
            speaker_partition(["a", "a"])

    def test_cross_corpus_heldout_is_excluded_before_any_admission(self):
        from student_boundary_fit import training_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(id="synthetic", speaker="held", split="train", kind="none", ayahs=["synthetic:ayah"],
                       events=[], non_errors=[], unsupported=[], wav="a.wav")
            (root / "labels.jsonl").write_text(json.dumps(row))
            (root / "corpus.jsonl").write_text(json.dumps(dict(speaker="held", split="release")))
            with self.assertRaisesRegex(ValueError, "No authorized"):
                training_rows(root, {"ayahs": []}, root / "corpus.jsonl")

    def test_connected_word_transcripts_keep_duration_separate_from_occurrences(self):
        from student_boundary_fit import training_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(id="synthetic", speaker="fit", split="train", kind="none", ayahs=["synthetic:ayah"],
                       events=[], non_errors=[], unsupported=[], wav="a.wav")
            (root / "labels.jsonl").write_text(json.dumps(row))
            (root / "a.wav").write_bytes(b"synthetic-audio-identity")
            reference = dict(ayahs=[dict(s="synthetic", a="ayah", w=["بَ", "بَ"], p="ببََببََ", o=[0, 4])])
            rows, excluded = training_rows(root, reference)
            self.assertEqual(rows[0]["words"], [dict(phones="بَ", letters="ب")] * 2)
            self.assertEqual(excluded, [])

    def test_audio_identity_and_root_containment(self):
        from student_boundary_fit import contained, training_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "escapes"):
                contained(root, "../outside.wav")
            row = dict(id="synthetic", speaker="fit", split="train", kind="none", ayahs=["synthetic:ayah"],
                       events=[], non_errors=[], unsupported=[], wav="a.wav")
            (root / "labels.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(dict(row, id="duplicate")))
            (root / "a.wav").write_bytes(b"synthetic-audio-identity")
            ref = dict(ayahs=[dict(s="synthetic", a="ayah", w=["بَ"], p="بَ", o=[0])])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                training_rows(root, ref)


if __name__ == "__main__":
    unittest.main()

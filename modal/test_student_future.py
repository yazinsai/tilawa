import copy
import itertools
import unittest

import numpy as np

from student_future import (admitted_none, continuation_pool, input_deadline,
                            marginal_targets, output_chunk, probabilities, disagreement_rows,
                            replace_future, speaker_partition, aligned_pronunciation_ends)


def manifest():
    return dict(sampleRate=16000, frameMs=40, window=45, shift=32,
                fbank=dict(frameShiftMs=10, frameLengthMs=25, snipEdges=True))


class FutureTargetsTest(unittest.TestCase):
    def test_deadline_protects_whole_chunk_and_analysis_window(self):
        m = manifest()
        self.assertEqual(output_chunk(m), 8)
        self.assertEqual([input_deadline(i, m) for i in (0, 7, 8, 15, 16)],
                         [7440, 7440, 12560, 12560, 17680])
        for key, value in (("shift", 0), ("window", 31), ("frameMs", 41)):
            with self.assertRaises(ValueError):
                output_chunk({**m, key: value})
        m["fbank"]["snipEdges"] = False
        with self.assertRaises(ValueError):
            input_deadline(0, m)

    def test_only_exclusive_future_changes_without_fade_or_loop(self):
        wave, donor = np.arange(10000, dtype=np.float32), np.full(10000, -7.)
        got = replace_future(wave, donor, 7440)
        np.testing.assert_array_equal(got[:7440], wave[:7440])
        np.testing.assert_array_equal(got[7440:], donor[:2560])
        self.assertEqual(got.shape, wave.shape)
        np.testing.assert_array_equal(wave, np.arange(10000))
        with self.assertRaises(ValueError):
            replace_future(wave, donor[:2559], 7440)
        np.testing.assert_array_equal(replace_future(wave, donor, 12000), wave)

    def test_average_probabilities_recomputes_frontend_on_original_clock(self):
        wave = np.zeros(10000, dtype=np.float32)
        donors = [np.ones(10000), np.full(10000, 2.)]
        calls = []

        def teacher(audio):
            calls.append(audio.copy())
            p = {0: [.01, .99], 1: [.9, .1], 2: [.2, .8]}[int(audio[-1])]
            return np.tile(np.log(p), (16, 1))

        def student(audio):
            # Each row sees all samples through its chunk's exact deadline.
            return np.array([[audio[:min(len(audio), input_deadline(i, manifest()))].sum()]
                             for i in range(16)])

        ordinary, marginal, audit = marginal_targets(teacher, wave, donors, 16, manifest(),
                                                     infer_student=student, student_rows=student(wave))
        self.assertEqual(len(calls), 3)
        np.testing.assert_allclose(ordinary, np.tile([.01, .99], (16, 1)))
        np.testing.assert_allclose(marginal[:8], np.tile([.55, .45], (8, 1)))
        np.testing.assert_array_equal(marginal[8:], ordinary[8:])  # No future at EOF.
        self.assertEqual(audit, [dict(start=0, stop=8, deadline=7440, draws=2)])
        logits_average = probabilities((np.log([[.9, .1]]) + np.log([[.2, .8]])) / 2)
        self.assertGreater(abs(float(logits_average[0, 0]) - .55), .04)

    def test_audit_detects_nonlocal_student_and_teacher_clock_loss(self):
        wave, donors = np.zeros(10000), [np.ones(10000)]
        teacher = lambda audio: np.zeros((16, 2))
        student = lambda audio: np.full((16, 2), audio.sum())
        with self.assertRaises(AssertionError):
            marginal_targets(teacher, wave, donors, 16, manifest(),
                             infer_student=student, student_rows=student(wave))
        with self.assertRaises(ValueError):
            marginal_targets(lambda audio: np.zeros((7, 2)), wave, donors, 16, manifest())

    def test_multiple_chunks_preserve_absolute_rows(self):
        wave = np.zeros(16000)
        def teacher(audio):
            rows = np.tile([-2., -1.], (24, 1))
            rows[:, 0] += np.arange(24) / 10 + audio[-1]
            return rows
        ordinary, marginal, audit = marginal_targets(teacher, wave, [np.ones(16000)], 24, manifest())
        expected = probabilities(teacher(np.ones(16000)))
        np.testing.assert_allclose(marginal[:16], expected[:16])
        np.testing.assert_array_equal(marginal[16:], ordinary[16:])
        self.assertEqual([r["start"] for r in audit], [0, 8])

    def test_training_admission_and_grouped_donors(self):
        def row(i, speaker):
            return dict(id=str(i), speaker=speaker, split="train", kind="none", scored=True,
                        events=[], non_errors=[], unsupported=[])
        clean, repeat = row(0, "a"), {**row(1, "b"), "non_errors": ["repeat"]}
        held = {**row(2, "a"), "split": "eval"}
        self.assertEqual(admitted_none([clean, repeat]), [clean])
        self.assertEqual(admitted_none([clean, held]), [])
        with self.assertRaises(ValueError):
            admitted_none([clean, copy.deepcopy(clean)])
        records = [row(i, s) for i, s in enumerate("abcdefghij")]
        groups = speaker_partition([r["speaker"] for r in records])
        self.assertEqual(groups, speaker_partition(list(reversed("abcdefghij"))))
        self.assertEqual(list(groups.values()).count("selection"), 2)
        for r in records:
            pool = continuation_pool(records, groups, r)
            self.assertTrue(pool)
            self.assertTrue(all(p["id"] != r["id"] and groups[p["speaker"]] == groups[r["speaker"]]
                                for p in pool))
        with self.assertRaises(ValueError):
            speaker_partition([None] + list("abcde"))

    def test_sparse_supervision_keeps_unselected_targets_and_partial_tail(self):
        wave, donors = np.zeros(20000), [np.ones(20000)]
        def teacher(audio):
            return np.tile([audio[-1], 0.], (25, 1))
        ordinary, marginal, audit = marginal_targets(teacher, wave, donors, 25, manifest(), chunks=[8])
        np.testing.assert_array_equal(marginal[:8], ordinary[:8])
        np.testing.assert_array_equal(marginal[16:], ordinary[16:])
        self.assertTrue((marginal[8:16, 0] > ordinary[8:16, 0]).all())
        self.assertEqual([a["start"] for a in audit], [8])
        for chunks in ([1], [8, 8], [32], []):
            with self.assertRaises(ValueError):
                marginal_targets(teacher, wave, donors, 25, manifest(), chunks=chunks)

    def test_premise_counts_same_frame_identity_not_time_shift(self):
        student = np.eye(3)[[0, 0, 2, 0]]
        teacher = np.eye(3)[[1, 1, 1, 2]]
        marginal = np.eye(3)[[0, 2, 2, 0]]
        self.assertEqual(disagreement_rows(student, teacher, marginal, {1}), [0, 2])
        self.assertEqual(disagreement_rows(student, teacher, marginal, {2}), [3])

    def test_held_pronunciation_is_protected_beyond_native_deadline(self):
        path = [1] * 21 + [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        lp = np.full((len(path), 3), -20.)
        lp[np.arange(len(path)), path] = 0
        ends = aligned_pronunciation_ends(lp, np.array([1, 2]), [0, 1], manifest())
        self.assertEqual(int(ends[:8].max()), 13840)
        wave, calls = np.zeros(25000), []
        def teacher(audio):
            calls.append(audio.copy())
            return lp
        _, _, audit = marginal_targets(teacher, wave, [np.ones(25000)], len(lp), manifest(),
                                       chunks=[0], pronunciation_ends={0: 13840})
        np.testing.assert_array_equal(calls[1][:13840], wave[:13840])
        self.assertTrue(calls[1][13840:].all())
        self.assertEqual(audit[0]["nativeDeadline"], 7440)
        self.assertEqual(audit[0]["pronunciationEnd"], 13840)

    def test_canonical_word_alignment_matches_exhaustive_ctc_paths(self):
        for target in ([1, 2], [1, 1]):
            lp = np.random.default_rng(18).normal(size=(6, 3))
            def collapse(path):
                return [v for i, v in enumerate(path) if v and (i == 0 or path[i - 1] != v)]
            paths = [p for p in itertools.product(range(3), repeat=6) if collapse(p) == target]
            best = max(paths, key=lambda p: lp[np.arange(6), p].sum())
            # Each target occurrence is a different canonical word, including
            # equal phones separated by the mandatory CTC blank.
            split = next(i for i in range(1, 6) if best[i] == target[1] and
                         (target[0] != target[1] or (best[i - 1] == 0 and target[0] in best[:i - 1])))
            last = [max(i for i in range(split) if best[i] == target[0]),
                    max(i for i in range(split, 6) if best[i] == target[1])]
            ends = aligned_pronunciation_ends(lp, np.array(target), [0, 1], manifest())
            for word, frame in enumerate(last):
                self.assertEqual(ends[frame], (frame + 1) * 640 + 400)
        with self.assertRaises(ValueError):
            aligned_pronunciation_ends(np.zeros((2, 3)), np.array([1, 1]), [0], manifest())


if __name__ == "__main__":
    unittest.main()

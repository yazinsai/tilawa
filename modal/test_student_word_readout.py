"""Run on the supplied supervisor: python3 -m unittest discover -s modal -p test_student_word_readout.py."""

import copy
import unittest

from student_word_readout import (
    TrainingRecording,
    cache_recording,
    correspondence,
    fit_readouts,
    speaker_partition,
    validate_caches,
)


class ToyEar:
    eos = 99
    artifact_sha256 = "a" * 64
    generation_sha256 = "b" * 64

    def __init__(self, generated=(1, 9, 2)):
        self.generated = generated
        self.calls = []

    def generate(self, audio):
        self.calls.append(("generate", audio))
        return self.generated

    def states(self, audio, history):
        self.calls.append(("states", audio, tuple(history)))
        return [tuple(history[:i]) for i in range(len(history) + 1)]


def recording(speaker="learner", identity="take", tokens=(1, 2), **kwargs):
    return TrainingRecording(identity, speaker, tokens,
                             kwargs.get("source", "authorized-transcribed"),
                             kwargs.get("transcript_basis", "verified-performed"))


def partition_speaker(partition):
    return next(f"training-speaker-{i}" for i in range(1000)
                if speaker_partition(f"training-speaker-{i}") == partition)


def paired_caches():
    return [cache_recording(ToyEar(), recording(partition_speaker(p), p), object(), set())
            for p in ("fit", "selection")]


class CorrespondenceTests(unittest.TestCase):
    def test_inserted_output_keeps_the_recorded_continuation(self):
        result = correspondence((1, 2, 3), (1, 9, 2, 3))
        self.assertEqual(result.operations, ("match", "insert", "match", "match"))
        self.assertEqual(result.next_recorded, (0, 1, 1, 2, 3))

    def test_deleted_token_is_recovery_target_before_next_output(self):
        result = correspondence((1, 2, 3), (1, 3))
        self.assertEqual(result.operations, ("match", "delete", "match"))
        self.assertEqual(result.next_recorded, (0, 1, 3))

    def test_consecutive_deletions_target_the_first_missing_token(self):
        result = correspondence((1, 2, 3, 4), (1, 4))
        self.assertEqual(result.next_recorded, (0, 1, 4))

    def test_repeated_tokens_retain_distinct_sequence_positions(self):
        result = correspondence((1, 2, 1, 3), (1, 1, 3))
        self.assertEqual(result.next_recorded, (0, 1, 3, 4))

    def test_ambiguous_duplicate_uses_documented_diagonal_tie_break(self):
        result = correspondence((1, 2), (1, 1, 2))
        self.assertEqual(result.operations, ("insert", "match", "match"))
        self.assertEqual(result.next_recorded, (0, 0, 1, 2))

    def test_early_end_does_not_supervise_eos_over_trailing_deletions(self):
        self.assertEqual(correspondence((1, 2, 3), (1,)).next_recorded, (0, 1))
        self.assertEqual(correspondence((1, 2, 3), ()).next_recorded, (0,))

    def test_trailing_insertions_are_supervised_as_eos(self):
        self.assertEqual(correspondence((1, 2), (1, 2, 9, 8)).next_recorded,
                         (0, 1, 2, 2, 2))

    def test_exact_and_empty_sequences_have_a_final_state(self):
        self.assertEqual(correspondence((1, 2), (1, 2)).next_recorded, (0, 1, 2))
        self.assertEqual(correspondence((), ()).next_recorded, (0,))


class CacheTests(unittest.TestCase):
    def test_both_histories_use_identical_audio_without_expected_passage(self):
        ear, audio = ToyEar(), object()
        cache = cache_recording(ear, recording(), audio, set())
        self.assertEqual(ear.calls, [("generate", audio), ("states", audio, (1, 2)),
                                     ("states", audio, (1, 9, 2))])
        self.assertEqual(cache["teacher"]["targets"], (1, 2, 99))
        self.assertEqual(cache["rollout"]["targets"], (1, 2, 2, 99))

    def test_empty_generation_targets_recorded_onset(self):
        cache = cache_recording(ToyEar(()), recording(), object(), set())
        self.assertEqual(cache["rollout"]["targets"], (1,))

    def test_release_speaker_is_rejected_before_decoder_execution(self):
        ear = ToyEar()
        with self.assertRaisesRegex(ValueError, "Release-held-out"):
            cache_recording(ear, recording(), object(), {"learner"})
        self.assertEqual(ear.calls, [])

    def test_help_train_accepts_only_none_labelled_transcripts(self):
        recording(source="help-train", transcript_basis="none-labelled").validate(set())
        for basis in ("planted-skip", "teacher-guessed", "verified-performed"):
            with self.subTest(basis=basis), self.assertRaisesRegex(ValueError, "Mistake scripts"):
                recording(source="help-train", transcript_basis=basis).validate(set())

    def test_unverified_and_nontraining_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            recording(transcript_basis="canonical-assumed").validate(set())
        with self.assertRaises(ValueError):
            recording(source="practice").validate(set())
        with self.assertRaises(ValueError):
            recording(tokens=()).validate(set())

    def test_prompt_or_eos_cannot_enter_content_history(self):
        with self.assertRaisesRegex(ValueError, "EOS"):
            cache_recording(ToyEar((1, 99)), recording(), object(), set())
        with self.assertRaisesRegex(ValueError, "EOS"):
            cache_recording(ToyEar(), recording(tokens=(1, 99)), object(), set())

    def test_hashes_are_hex_digests_and_prefix_state_count_is_checked(self):
        ear = ToyEar()
        ear.generation_sha256 = "z" * 64
        with self.assertRaisesRegex(ValueError, "hashes"):
            cache_recording(ear, recording(), object(), set())
        ear = ToyEar()
        ear.states = lambda audio, history: []
        with self.assertRaisesRegex(ValueError, "every prefix"):
            cache_recording(ear, recording(), object(), set())

    def test_fit_revalidates_exclusions_and_saved_targets(self):
        caches = paired_caches()
        with self.assertRaisesRegex(ValueError, "Release-held-out"):
            validate_caches(caches, {caches[0]["speaker"]})
        caches[0]["rollout"]["targets"] = (1, 9, 2, 99)
        with self.assertRaisesRegex(ValueError, "recorded continuation"):
            validate_caches(caches, set())

    def test_recording_duplication_and_mixed_ears_are_rejected(self):
        caches = paired_caches()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_caches(caches + [copy.deepcopy(caches[0])], set())
        caches[1]["ear_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "one pinned"):
            validate_caches(caches, set())

    def test_partition_cannot_be_reassigned_to_move_a_speaker(self):
        caches = paired_caches()
        fit, selection = validate_caches(caches, set())
        self.assertEqual(len(fit), 1)
        self.assertEqual(len(selection), 1)
        caches[0]["partition"] = "selection"
        with self.assertRaisesRegex(ValueError, "fixed speaker split"):
            validate_caches(caches, set())


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch fitting checks require the supervisor training image")
class FittingTests(unittest.TestCase):
    def test_matched_heads_fit_without_updating_base_projection_or_cache(self):
        torch.manual_seed(1)
        caches = paired_caches()
        for c in caches:
            for arm in ("teacher", "rollout"):
                c[arm]["states"] = torch.randn(len(c[arm]["targets"]), 3, requires_grad=True)
        projection = torch.nn.Linear(3, 100)
        before = {k: v.detach().clone() for k, v in projection.state_dict().items()}
        states_before = [c[arm]["states"].detach().clone() for c in caches
                         for arm in ("teacher", "rollout")]
        results = fit_readouts(caches, projection, excluded_speakers=set(), rank=2, epochs=2)
        self.assertEqual(set(results), {"teacher", "rollout"})
        for result in results.values():
            self.assertTrue(torch.isfinite(torch.tensor(result["selection_loss"])))
            self.assertIn(result["epoch"], (1, 2))
            self.assertGreater(result["head"][-1].weight.abs().sum().item(), 0)
        for name, value in projection.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(all(p.grad is None for p in projection.parameters()))
        for old, history in zip(states_before, [c[arm] for c in caches
                                                for arm in ("teacher", "rollout")]):
            self.assertTrue(torch.equal(old, history["states"]))
            self.assertIsNone(history["states"].grad)


if __name__ == "__main__":
    unittest.main()

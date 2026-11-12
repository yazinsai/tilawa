"""python -m unittest discover -s modal -p test_student_bridge.py

Policy tests need only stdlib. Tensor/model tests run in the authorized train image.
"""

import copy
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "tools")]
from student_bridge_data import (assistant_config, budget_remaining, confined_file,
                                 draw_seed, help_none, release_speakers, require_matched_compute, speaker_partition)
from student_bridge_train import distillation_loss, freeze

HAS_TORCH = importlib.util.find_spec("torch") is not None


class TrainingPolicies(unittest.TestCase):
    def records(self):
        return [dict(key=f"{s}:{i}", speaker=s, split="train", supervised=True)
                for s in ("a", "b", "c", "d", "e") for i in range(3)]

    def test_a_speaker_with_any_release_recording_is_denied_in_both_partitions(self):
        records = self.records() + [dict(key="release", speaker="a", split="test", supervised=False)]
        train, selection = speaker_partition(records, {"b"}, 42)
        self.assertEqual({r["speaker"] for r in train + selection}, {"c", "d", "e"})

    def test_gold_denial_cannot_be_hidden_by_bucket_or_clip_filtering(self):
        records = self.records() + [dict(key="gold", speaker="a", split="train", supervised=False, gold=True)]
        train, selection = speaker_partition(records, set(), 42)
        self.assertNotIn("a", {r["speaker"] for r in train + selection})

    def test_selection_speakers_are_disjoint_and_stable_after_row_reordering(self):
        records = self.records()
        expected = speaker_partition(records, set(), 42)
        random.Random(99).shuffle(records)
        self.assertEqual(expected, speaker_partition(records, set(), 42))
        train, selection = expected
        self.assertFalse({r["speaker"] for r in train} & {r["speaker"] for r in selection})
        self.assertEqual(len(train + selection), 15)

    def test_unknown_speakers_and_duplicate_recordings_fail_closed(self):
        for bad in ({"speaker": ""}, {"speaker": None}, {"key": "a:0"}):
            records = self.records()
            records[1] |= bad
            with self.assertRaises(ValueError):
                speaker_partition(records, set(), 42)

    def test_training_only_selection_cannot_fall_back_to_test_when_too_small(self):
        with self.assertRaises(ValueError):
            speaker_partition([dict(key="one", speaker="one", split="train", supervised=True)], set(), 42)

    def test_release_speaker_is_denied_even_when_their_release_clip_is_absent_from_s1(self):
        denied = release_speakers(dict(slices=dict(gold=[dict(speaker="tlog-cluster-a", source="tlog-bucket-review")],
                                                 silver=[dict(speaker="learner", source="help-eval-clean")])))
        self.assertEqual(denied, {"s1:a", "help:learner"})
        records = [{**r, "speaker": "s1:" + r["speaker"]} for r in self.records()]
        train, val = speaker_partition(records, denied, 42)
        self.assertNotIn("s1:a", {r["speaker"] for r in train + val})

    def test_missing_release_speaker_metadata_does_not_silently_authorize_training(self):
        with self.assertRaises(ValueError):
            release_speakers(dict(slices=dict(gold=[dict(speaker=None, source="help-eval-clean")])))

    def test_planted_scripts_and_repeats_are_not_transcript_supervision(self):
        clean = dict(kind="none", split="train", scored=True, events=[], non_errors=[], unsupported=[])
        self.assertTrue(help_none(clean))
        for field, value in (("kind", "repeat"), ("kind", "substitution"), ("split", "eval"),
                             ("events", [{"errorClass": "wrong_word"}]), ("non_errors", [{"kind": "repeat"}]),
                             ("unsupported", [{"rule": "tajweed"}]), ("scored", False), ("excluded", "review")):
            self.assertFalse(help_none({**clean, field: value}), field)

    def test_same_augmentation_draw_in_both_arms_independent_of_order(self):
        visits = [("a", 0), ("b", 0), ("a", 1)]
        expected = {(k, v): draw_seed(42, k, v) for k, v in visits}
        self.assertEqual(expected, {(k, v): draw_seed(42, k, v) for k, v in reversed(visits)})
        self.assertEqual(len(set(expected.values())), 3)

    def test_assistant_widens_capacity_without_changing_clock_or_shipped_config(self):
        original = dict(feedforward_dim="512,768", encoder_dim="192,256", num_encoder_layers="2,2")
        before = copy.deepcopy(original)
        assistant = assistant_config(original)
        self.assertEqual(original, before)
        self.assertEqual(assistant["feedforward_dim"], "1536,2304")
        self.assertEqual(assistant["encoder_dim"], original["encoder_dim"])
        self.assertEqual(assistant["num_encoder_layers"], original["num_encoder_layers"])

    def test_direct_budget_includes_assistant_and_target_generation(self):
        bridge = dict(assistant=12., assistant_targets=3., student=10.)
        self.assertEqual(require_matched_compute(bridge, dict(student=26., last_update=2.))["bridgeGpuSeconds"], 25.)
        for spent in (10., 24.9, 27.1):
            with self.assertRaises(ValueError):
                require_matched_compute(bridge, dict(student=spent, last_update=2.))

    def test_budget_does_not_accept_nonfinite_or_untrained_results(self):
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                budget_remaining(value, 10)
        self.assertTrue(budget_remaining(9.9, 10))
        self.assertFalse(budget_remaining(10, 10))
        with self.assertRaises(ValueError):
            require_matched_compute(dict(assistant=1., assistant_targets=1., student=1.), dict(student=3., last_update=0.))

    def test_asset_path_cannot_escape_authorized_training_root(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp) / "training"
            root.mkdir()
            (root / "audio.wav").write_bytes(b"audio")
            self.assertEqual(confined_file(root, "audio.wav"), (root / "audio.wav").resolve())
            for path in ("missing.wav", "../secret", "/outside/audio.wav"):
                with self.assertRaises(ValueError):
                    confined_file(root, path)


@unittest.skipUnless(HAS_TORCH, "tensor tests require the authorized train image")
class TensorBehavior(unittest.TestCase):
    def test_frozen_assistant_is_inference_only_and_never_receives_gradients(self):
        import torch
        class Supervisor(torch.nn.Linear):
            def streaming_eval(self, chunk, left):
                self.geometry = (chunk, left)
        teacher = freeze(Supervisor(3, 3))
        student = torch.nn.Linear(3, 3)
        x = torch.randn(2, 3)
        loss = (student(x) - teacher(x)).square().sum()
        loss.backward()
        self.assertFalse(teacher.training)
        self.assertEqual(teacher.geometry, (16, 128))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()))

    def fixture(self):
        import torch
        logits = torch.tensor([[[0., 1., -1.], [1., 0., -1.], [2., 0., -2.]]], requires_grad=True)
        b = dict(transcript=torch.tensor([1]), transcript_lens=torch.tensor([1]),
                 teacher_lens=torch.tensor([1]), ids=torch.tensor([[[1, 0, 2, 0, 0]]]),
                 probs=torch.tensor([[[.6, .3, .1, 0., 0.]]]))
        old = torch.tensor([[[.2, .7, .1], [.5, .4, .1], [.3, .6, .1]]]).log()
        return logits, torch.tensor([2]), old, b

    def test_padding_has_no_gradient_from_ctc_kd_or_preservation(self):
        import torch
        logits, lens, old, b = self.fixture()
        loss = distillation_loss(logits.log_softmax(-1), lens, old, b, dict(ctc=1., kd=1., preservation=.5))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logits.grad[:, 2].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[:, :2].abs().sum()), 0.)

    def test_supervisory_frame_mapping_clamps_shorter_teacher_without_losing_weak_alternatives(self):
        import torch
        logits, lens, old, b = self.fixture()
        lp = logits.log_softmax(-1)
        loss = distillation_loss(lp, lens, old, b, dict(ctc=0., kd=1., preservation=0.))
        expected = -(lp[0, :2] * torch.tensor([.3, .6, .1])).sum(-1).mean()
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        b["teacher_lens"] = torch.tensor([0])
        with self.assertRaises(ValueError):
            distillation_loss(lp, lens, old, b, dict(ctc=0., kd=1., preservation=0.))

    def test_configurable_model_keeps_inventory_lengths_and_shipped_presets(self):
        import torch
        import student_train as T
        original = copy.deepcopy(T.SIZES)
        config = dict(num_encoder_layers="1,1,1,1,1,1", feedforward_dim="32,32,32,32,32,32",
                      encoder_dim="16,16,16,16,16,16", encoder_unmasked_dim="16,16,16,16,16,16",
                      num_heads="2,2,2,2,2,2", query_head_dim="8", value_head_dim="4", pos_head_dim="2",
                      pos_dim=8, cnn_module_kernel="3,3,3,3,3,3")
        small = T.build_model("full", config).eval()
        wide = T.build_model("full", assistant_config(config)).eval()
        small.streaming_eval(16, 128)
        wide.streaming_eval(16, 128)
        with torch.no_grad():
            a, al = small(torch.randn(1, 93, 80), torch.tensor([93]))
            b, bl = wide(torch.randn(1, 93, 80), torch.tensor([93]))
        self.assertEqual(T.SIZES, original)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape[-1], 43)
        self.assertTrue(torch.equal(al, bl))
        self.assertGreater(sum(p.numel() for p in wide.parameters()), sum(p.numel() for p in small.parameters()))


if __name__ == "__main__":
    unittest.main()

"""Numerical and preservation checks for whole-recording acoustic Group DRO.

  python3 -m unittest discover -s tools -p test_student_group_dro.py
Torch checks run in the training image; numpy-only checks also run locally.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal"))
from student_group_dro import AcousticGroups, GroupObjective, clean_help, descriptors, group_plan, partition


class AcousticTests(unittest.TestCase):
    def test_cli_maps_parent_argument_to_training_function(self):
        from unittest.mock import patch
        import student_group_dro as module
        with patch.object(sys, "argv", ["student_group_dro.py", "train", "--data", "data",
                                       "--parent", "parent.pt", "--out", "fit"]), patch.object(module, "fit") as run:
            module.main()
        self.assertEqual(run.call_args.kwargs["parent_path"], Path("parent.pt"))
        self.assertNotIn("parent", run.call_args.kwargs)

    def test_whole_recording_descriptor_retains_pauses(self):
        speech = np.sin(np.arange(16000) * .08).astype(np.float32)
        with_pause = np.concatenate([speech, np.zeros(16000)])
        a, b = descriptors(speech), descriptors(with_pause)
        self.assertTrue(np.isfinite(b).all())
        self.assertGreater(b[2], a[2] + .4)
        self.assertGreater(b[1], a[1])

    def test_finite_silence_and_reject_malformed_audio(self):
        self.assertTrue(np.isfinite(descriptors(np.zeros(16000))).all())
        for x in (np.array([np.nan] * 16000), np.zeros((10000, 2)), np.zeros(2)):
            with self.assertRaises(ValueError):
                descriptors(x)

    def test_acoustic_clustering_and_prediction_do_not_use_text(self):
        x = np.array([[0, 0], [0, .1], [10, 10], [10, 10.1]], dtype=float)
        a = AcousticGroups().fit(x, 2)
        labels = a.predict(x)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(a.predict(x[[2, 0]]).tolist(), labels[[2, 0]].tolist())
        before = a.centers.copy()
        a.predict(np.array([[100, 100]]))
        np.testing.assert_array_equal(before, a.centers)

    def test_partition_keeps_all_speaker_takes_together(self):
        labels = {s: partition(s) for s in ("speaker-a", "speaker-b", "speaker-c")}
        self.assertEqual([partition("speaker-a")] * 10, [labels["speaker-a"]] * 10)
        with self.assertRaises(ValueError):
            partition("")

    def test_only_usable_none_training_takes_are_eligible(self):
        row = dict(split="train", kind="none", scored=True, events=[], non_errors=[], unsupported=[])
        self.assertTrue(clean_help(row))
        for update in (dict(split="eval"), dict(kind="substitution"), dict(kind="repeat"),
                       dict(scored=False), dict(events=[{}]), dict(non_errors=[{}]),
                       dict(unsupported=[{}]), dict(excluded="review")):
            self.assertFalse(clean_help({**row, **update}))
        self.assertFalse(clean_help({k: v for k, v in row.items() if k != "scored"}))

    def test_mean_control_recovers_recording_prevalence(self):
        obj = GroupObjective([90, 10], False)
        before = obj.weights.copy()
        obj.update([1, 10])
        np.testing.assert_allclose(obj.weights, [.9, .1])
        np.testing.assert_array_equal(obj.weights, before)
        self.assertAlmostEqual(float(obj.weights @ np.array([1, 10])), 1.9)

    def test_adversary_moves_mass_to_the_worse_rare_group(self):
        obj = GroupObjective([90, 10], True)
        q = obj.update([1, 10])
        self.assertGreater(q[1], q[0])
        self.assertAlmostEqual(q.sum(), 1.)
        self.assertTrue((q > 0).all())
        # Adding a common offset does not change the adversary.
        shifted = GroupObjective([90, 10], True).update([101, 110])
        np.testing.assert_allclose(q, shifted)
        for _ in range(100):
            q = obj.update([1, 10])
        self.assertGreater(q[1], .999)

    def test_equal_group_losses_and_invalid_groups(self):
        obj = GroupObjective([9, 1], True)
        np.testing.assert_allclose(obj.update([2, 2]), [.5, .5])
        with self.assertRaises(ValueError):
            GroupObjective([1, 0], True)
        with self.assertRaises(ValueError):
            obj.update([np.inf, 1])

    def test_matched_batch_plan_has_complete_groups_and_same_samples(self):
        labels = [0] * 90 + [1] * 10
        a = list(group_plan(labels, 10, 2, 7))
        self.assertEqual(a, list(group_plan(labels, 10, 2, 7)))
        for batch in a:
            self.assertEqual([labels[i] for i in batch], [0, 0, 1, 1])
        with self.assertRaises(ValueError):
            list(group_plan([0, 2], 1, 1, 0))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch checks require the training image")
class AdapterTests(unittest.TestCase):
    def test_zero_init_parent_is_preserved_and_merge_preserves_adapted_output(self):
        import torch
        from torch import nn
        from student_group_dro import base_only, install_adapters, merge_adapters
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 4))
                self.ctc_output = nn.Linear(4, 2)
            def forward(self, x):
                return self.ctc_output(self.encoder(x))
        torch.manual_seed(1)
        model = Tiny().eval()
        x = torch.randn(2, 7, 3)
        parent = model(x).detach()
        names = install_adapters(model, 2, 0)
        self.assertEqual(len(names), 2)
        torch.testing.assert_close(parent, model(x), rtol=0, atol=0)
        for name, p in model.named_parameters():
            self.assertEqual(p.requires_grad, name.endswith((".up", ".down")))
        model.encoder[0].up.data.fill_(.05)
        changed = model(x).detach()
        self.assertFalse(torch.equal(changed, parent))
        with base_only(model):
            torch.testing.assert_close(parent, model(x), rtol=0, atol=0)
        torch.testing.assert_close(changed, model(x), rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            with base_only(model):
                raise RuntimeError("interrupted parent pass")
        torch.testing.assert_close(changed, model(x), rtol=0, atol=0)
        merged = merge_adapters(model)
        torch.testing.assert_close(changed, merged(x), rtol=1e-5, atol=1e-6)
        restored = Tiny().eval()
        restored.load_state_dict(merged.state_dict(), strict=True)
        torch.testing.assert_close(changed, restored(x), rtol=1e-5, atol=1e-6)

    def test_recording_normalization_ignores_padding_and_preserves_pause_gradient(self):
        import torch
        from student_group_dro import recording_losses
        # Both records are the same three-frame sequence; second has extra padding.
        raw = torch.tensor([[[2., 0.], [0., 2.], [2., 0.], [5., -5.]],
                            [[2., 0.], [0., 2.], [2., 0.], [-5., 5.]]], requires_grad=True)
        lp = raw.log_softmax(-1)
        parent = lp.detach().exp()
        loss = recording_losses(lp, parent, torch.tensor([3, 3]), torch.tensor([1, 1]),
                                torch.tensor([1, 1]), kd_weight=2.)
        torch.testing.assert_close(loss[0], loss[1])
        loss.sum().backward()
        self.assertGreater(float(raw.grad[:, 0].abs().sum()), 0.)
        self.assertEqual(float(raw.grad[:, 3].abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()

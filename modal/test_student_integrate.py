import unittest
import json
import tempfile
from pathlib import Path

import torch

from student_integrate import attach_heads, decode_events, objective, pool
from student_integrate_assets import copy_auxiliaries, digest


class IntegrationTest(unittest.TestCase):
    def test_overlap_conserves_mass_and_gradient(self):
        mass = torch.tensor([[0.75, 0.75, 0.5]], requires_grad=True)
        features = torch.tensor([[[1.0], [3.0], [5.0]]])
        pooled, weights = pool(mass, features, torch.tensor([2]))
        torch.testing.assert_close(weights.sum(2), mass)
        torch.testing.assert_close(pooled, torch.tensor([[[1.5], [4.0]]]))
        pooled.square().sum().backward()
        self.assertTrue(torch.isfinite(mass.grad).all())
        self.assertGreater(float(mass.grad.abs().sum()), 0)

    def test_affine_classifier_commutes_with_complete_pooling(self):
        mass = torch.tensor([[0.25, 0.75]])
        features = torch.tensor([[[1.0, 2.0], [4.0, -1.0]]])
        head = torch.nn.Linear(2, 3)
        pooled, weights = pool(mass, features, torch.tensor([1]))
        torch.testing.assert_close(head(pooled), torch.bmm(weights.transpose(1, 2), head(features)))

    def test_raw_decoder_does_not_force_a_target_count(self):
        scores = torch.tensor([[0.0, 3.0]] * 4)
        self.assertEqual(len(decode_events(torch.tensor([0.25] * 4), scores)), 1)
        self.assertEqual(len(decode_events(torch.tensor([0.75] * 4), scores)), 3)
        self.assertEqual(decode_events(torch.zeros(4), scores), [])
        self.assertEqual(decode_events(torch.zeros(0), scores[:0]), [])

    def test_zero_length_silence_pool(self):
        pooled, weights = pool(torch.zeros(2, 4), torch.ones(2, 4, 3), torch.zeros(2, dtype=torch.long))
        self.assertEqual(tuple(pooled.shape), (2, 0, 3))
        self.assertEqual(tuple(weights.shape), (2, 4, 0))

    def test_both_arms_train_count_identity_and_independent_ctc(self):
        class Acoustic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ctc_output = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(2, 3))

            def encode(self, features, lengths):
                return features, lengths

        batch = dict(feats=torch.ones(1, 4, 2), feat_lens=torch.tensor([4]),
                     targets=torch.tensor([1, 2]), target_lens=torch.tensor([2]))
        for arm in ("cif", "ctc"):
            model = attach_heads(Acoustic())
            losses = objective(model, batch, arm)
            self.assertEqual(float(losses["count"].detach()), 0.5)
            losses["loss"].backward()
            for head in (model.phone_mass, model.phone_output, model.ctc_output[1]):
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                self.assertGreater(float(head.weight.grad.abs().sum()), 0)

    def test_silence_supervises_zero_count_without_phone_targets(self):
        class Acoustic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ctc_output = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(2, 3))

            def encode(self, features, lengths):
                return features, lengths

        model = attach_heads(Acoustic())
        batch = dict(feats=torch.ones(1, 4, 2), feat_lens=torch.tensor([4]),
                     targets=torch.zeros(0, dtype=torch.long), target_lens=torch.tensor([0]))
        losses = objective(model, batch, "cif")
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertGreater(float(model.phone_mass.bias.grad), 0)

    def test_primary_replacement_preserves_complete_auxiliary_bundle(self):
        temporary_root = Path(__file__).resolve().parents[1] / ".autolab"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            base, out = root / "base", root / "out"
            base.mkdir()
            out.mkdir()
            (base / "manifest.json").write_text(json.dumps(dict(file="model.onnx")))
            (base / "model.onnx").write_bytes(b"old-primary")
            (out / "model.onnx").write_bytes(b"new-primary")
            (base / "teacher.onnx").write_bytes(b"installed-teacher-graph")
            with self.assertRaises(ValueError):
                copy_auxiliaries(base, out)
            (base / "teacher.onnx.data").write_bytes(b"installed-teacher-weights")
            assets = copy_auxiliaries(base, out)
            self.assertEqual((out / "model.onnx").read_bytes(), b"new-primary")
            self.assertEqual(set(assets), {"teacher.onnx", "teacher.onnx.data"})
            self.assertEqual(assets["teacher.onnx.data"]["sha256"], digest(base / "teacher.onnx.data"))
            self.assertEqual((out / "teacher.onnx.data").read_bytes(), b"installed-teacher-weights")


if __name__ == "__main__":
    unittest.main()

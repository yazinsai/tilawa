"""Run on the training image: python -m unittest discover -s modal -p test_student_specialists.py."""

import copy
import itertools
import unittest

import torch

from student_specialists import (
    DenseFeedforward, RoutedFeedforward, ctc_regime_occupancy,
    regime_map, replace_upper_block, router_loss,
)


class SpecialistsTest(unittest.TestCase):
    def block(self):
        from zipformer import FeedforwardModule
        torch.manual_seed(7)
        return FeedforwardModule(8, 12, dropout=0.0).eval()

    def test_identity_and_dense_parameter_equality(self):
        block = self.block()
        routed, dense = RoutedFeedforward(block).eval(), DenseFeedforward(block).eval()
        x = torch.randn(13, 2, 8)
        with torch.no_grad():
            # Identity must hold for nonuniform as well as uniform routing.
            routed.router.weight.normal_()
            routed.router.bias.normal_()
            torch.testing.assert_close(routed(x), block(x), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(dense(x), block(x), rtol=1e-5, atol=1e-6)
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(routed), count(dense))

    def test_router_and_experts_receive_gradients(self):
        routed = RoutedFeedforward(self.block()).train()
        x = torch.randn(6, 2, 8)
        out = routed(x)
        occupancy = torch.zeros(2, 6, 3)
        occupancy[0, :, 0] = 1
        occupancy[1, :, 1] = 1
        (out.square().mean() + router_loss(routed.route_logits, occupancy,
                                          torch.tensor([6, 6]))).backward()
        self.assertGreater(routed.router.weight.grad.abs().sum().item(), 0)
        for expert in routed.experts:
            self.assertGreater(expert.in_proj.weight.grad.abs().sum().item(), 0)

    def test_router_padding_has_no_loss(self):
        logits = torch.randn(8, 2, 3)
        occupancy = torch.rand(2, 8, 3)
        occupancy /= occupancy.sum(-1, keepdim=True)
        lengths = torch.tensor([8, 4])
        before = router_loss(logits, occupancy, lengths)
        logits[4:, 1] += torch.tensor([1000., -1000., 0.])
        torch.testing.assert_close(before, router_loss(logits, occupancy, lengths))
        self.assertEqual(float(router_loss(logits, occupancy, lengths * 0)), 0.)
        with self.assertRaisesRegex(ValueError, "frame clocks"):
            router_loss(logits, occupancy[:, ::2], lengths)

    def test_identity_in_deployed_streaming_cache_loop(self):
        from unittest.mock import patch
        import student_train as T
        from student_export import wrap

        tiny = dict(num_encoder_layers="1,1,1,1,1,1",
                    feedforward_dim="32,32,32,32,32,32",
                    encoder_dim="16,16,16,16,16,16",
                    encoder_unmasked_dim="16,16,16,16,16,16")
        with patch.dict(T.SIZES, {"specialist-test": tiny}), patch.dict(T.COMMON,
                num_heads="2,2,2,2,2,2", query_head_dim="8", value_head_dim="4",
                pos_head_dim="2", pos_dim=8, cnn_module_kernel="3,3,3,3,3,3"):
            base = T.build_model("specialist-test").eval()
            changed = copy.deepcopy(base)
            block = replace_upper_block(changed)
            expected, got = wrap(base, 16, 64), wrap(changed, 16, 64)
            a, b = expected.init_states(), got.init_states()
            self.assertEqual([tuple(s.shape) for s in a], [tuple(s.shape) for s in b])
            with torch.no_grad():
                for _ in range(4):
                    x = torch.randn(1, 45, 80)
                    ap, a = expected(x, a)
                    bp, b = got(x, b)
                    self.assertEqual(ap.shape, (1, 8, 43))
                    torch.testing.assert_close(ap, bp, rtol=1e-5, atol=1e-5)
                    for old, new in zip(a, b):
                        torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-5)
                # The internal final layer is already downsampled to 40 ms;
                # router supervision must not repeat its labels at 20 ms.
                block.train()
                lp, lengths = changed(torch.randn(1, 90, 80), torch.tensor([90]))
                self.assertEqual(block.route_logits.shape[:2], (lp.shape[1], 1))
                occupancy = torch.zeros_like(lp[..., :3])
                occupancy[..., 0] = 1
                self.assertTrue(torch.isfinite(router_loss(block.route_logits, occupancy, lengths)))
            # Export a changed full encoder, including its real recurrent
            # caches, rather than proving parity only for an isolated FF.
            import tempfile
            from pathlib import Path
            from student_export import export_onnx
            from student_specialist_train import streaming_parity
            changed.eval()
            with torch.no_grad():
                block.router.weight.normal_()
                block.experts[0].out_proj.weight.add_(.02)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "stream.onnx"
                states = export_onnx(changed, str(path), 16, 64)
                streaming_parity(changed, path, torch.randn(120, 80), states, left=64)

    def test_occupancy_matches_exhaustive_ctc_paths(self):
        # Repeated phones are deliberately included: no cross-blank merging.
        classes = torch.tensor([2, 0, 1])
        lp = torch.tensor([[.1, .7, .2], [.6, .3, .1], [.2, .6, .2],
                           [.3, .2, .5]]).log()
        for transcript in ([1, 1], [1, 2], []):
            expected = torch.zeros(4, 3)
            z = 0.
            for path in itertools.product(range(3), repeat=4):
                collapsed, previous = [], -1
                for label in path:
                    if label and label != previous:
                        collapsed.append(label)
                    previous = label
                if collapsed != transcript:
                    continue
                prob = lp.exp()[torch.arange(4), list(path)].prod().item()
                z += prob
                for t, label in enumerate(path):
                    expected[t, classes[label]] += prob
            got = ctc_regime_occupancy(lp, torch.tensor(transcript, dtype=torch.long), classes)
            torch.testing.assert_close(got, expected / z)
        self.assertIsNone(ctc_regime_occupancy(lp[:2], torch.tensor([1, 1]), classes))

    def test_inventory_classes(self):
        self.assertEqual(regime_map("بتمنَں۾ؙۜڇ").tolist(), [2, 0, 0, 1, 1, 1, 1, 1, 2, 2, 0])

    def test_export_contains_audio_only_router(self):
        import tempfile
        import numpy as np
        import onnx
        import onnxruntime as ort

        routed = RoutedFeedforward(self.block()).eval()
        # Test a nontrivial trained function, not just the identity initialization.
        with torch.no_grad():
            routed.router.weight.normal_()
            routed.experts[0].out_proj.weight.add_(.1)
        x = torch.randn(7, 1, 8)
        with tempfile.TemporaryDirectory() as d:
            path = d + "/block.onnx"
            torch.onnx.export(routed, (x,), path, input_names=["audio"],
                              output_names=["residual"], opset_version=17, dynamo=False)
            graph = onnx.load(path)
            self.assertEqual([i.name for i in graph.graph.input], ["audio"])
            self.assertIn("Softmax", {n.op_type for n in graph.graph.node})
            got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
                None, {"audio": x.numpy()})[0]
            np.testing.assert_allclose(got, routed(x).detach().numpy(), rtol=1e-4, atol=1e-5)


class TrainingInputsTest(unittest.TestCase):
    def test_only_authorized_none_takes_and_safe_audio_paths(self):
        import json
        import tempfile
        from pathlib import Path
        import numpy as np
        import soundfile as sf
        from student_specialist_train import none_takes

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "wav").mkdir()
            sf.write(root / "wav/take.wav", np.zeros(1600, dtype=np.float32), 16000)
            record = dict(speaker="training-speaker", split="train", kind="none",
                          ayahs=["1:1"], wav="wav/take.wav", events=[], non_errors=[],
                          unsupported=[], scored=True)
            rows = [record, {**record, "kind": "repeat"}, {**record, "scored": False},
                    {**record, "excluded": "review"}, {**record, "events": [{"errorClass": "wrong_word"}]},
                    {**record, "speaker": "held-speaker", "split": "eval"}]
            labels = root / "labels.jsonl"
            labels.write_text("\n".join(map(json.dumps, rows)))
            ref = {"ayahs": [{"s": 1, "a": 1, "p": "مَ"}]}
            self.assertEqual([x[0] for x in none_takes(root, ref)], ["training-speaker"])
            labels.write_text(json.dumps(record) + "\n" + json.dumps({**record, "split": "eval"}))
            with self.assertRaisesRegex(ValueError, "both training and held-out"):
                none_takes(root, ref)
            labels.write_text(json.dumps({**record, "wav": "../outside.wav"}))
            with self.assertRaisesRegex(ValueError, "supplied wav"):
                none_takes(root, ref)

    def test_paired_augmentation_and_fixed_validation_draws(self):
        import numpy as np
        from student_specialist_train import PairedData

        wave = np.sin(np.arange(16000, dtype=np.float32) * .1) * .1
        clips = dict(ref=[], n_samples=np.array([], dtype=np.int64))
        cfg = dict(rir_p=0., noise_p=0., spec_aug=True)
        a = PairedData(clips, [("speaker", wave, "مَ")], cfg, {}, 17)
        b = PairedData(clips, [("speaker", wave, "مَ")], cfg, {}, 17)
        fixed = a.item(0, 0)["feats"]
        a.epoch = b.epoch = 2
        torch.testing.assert_close(a[0]["feats"], b[0]["feats"], rtol=0, atol=0)
        torch.testing.assert_close(a.item(0, 0)["feats"], fixed, rtol=0, atol=0)
        self.assertFalse(torch.equal(a[0]["feats"], fixed))

    def test_verification_assets_are_required_and_copied(self):
        import tempfile
        from pathlib import Path
        from student_specialist_train import copy_verification_assets

        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "source", Path(d) / "bundle"
            src.mkdir()
            dst.mkdir()
            with self.assertRaises(FileNotFoundError):
                copy_verification_assets(src, dst)
            for name in ("teacher.onnx", "teacher.onnx.data"):
                (src / name).write_bytes(name.encode())
            copy_verification_assets(src, dst)
            for name in ("teacher.onnx", "teacher.onnx.data"):
                self.assertEqual((src / name).read_bytes(), (dst / name).read_bytes())


if __name__ == "__main__":
    unittest.main()

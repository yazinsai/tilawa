"""CPU unit checks; run in the supplied training image, with pinned icefall."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from scaling import BiasNorm
from student_scale_bypass import (ScaleBiasNorm, copy_insertion_assets, install_scale_bypass,
                                  learner_records, partition_records, train_pair, verify_streaming_export)


class ScaleBypassTests(unittest.TestCase):
    def test_initial_output_and_gradients_match_original_at_each_channel_axis(self):
        for axis in (-1, -2, 0, 1):
            shape = [3, 5, 7]
            shape[axis] = 4
            torch.manual_seed(7)
            original = nn.Sequential(BiasNorm(4, channel_dim=axis)).double()
            model = copy.deepcopy(original)
            install_scale_bypass(model)
            a = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            expected, actual = original(a), model(b)
            self.assertTrue(torch.equal(actual, expected))
            expected.square().sum().backward()
            actual.square().sum().backward()
            torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-9)
            for name in ("bias", "log_scale"):
                torch.testing.assert_close(getattr(original[0], name).grad, getattr(model[0], name).grad,
                                           atol=1e-9, rtol=1e-9)
            self.assertGreater(float(model[0].scale_projection.weight.grad.abs().sum()), 0)

    def test_residual_is_the_actual_biased_channel_log_rms(self):
        for axis in (-1, 1):
            model = nn.Sequential(BiasNorm(3, channel_dim=axis)).double().eval()
            baseline = copy.deepcopy(model)
            install_scale_bypass(model)
            model[0].scale_projection.weight.data.copy_(torch.tensor([[0.2], [-0.3], [0.5]]))
            x = torch.arange(1, 25, dtype=torch.float64).reshape(2, 4, 3)
            if axis == 1:
                x = x.transpose(1, 2)
            bias = baseline[0].bias.reshape([1, 3, 1] if axis == 1 else [1, 1, 3])
            log_rms = ((x - bias).square().mean(axis, keepdim=True)).log() * 0.5
            weights = model[0].scale_projection.weight.reshape([1, 3, 1] if axis == 1 else [1, 1, 3])
            torch.testing.assert_close(model(x) - baseline(x), log_rms * weights)

    def test_scale_distinguishes_equal_normalized_patterns_control_does_not(self):
        models = {}
        for mode in ("scale", "constant"):
            m = nn.Sequential(BiasNorm(3)).double().eval()
            m[0].bias.data.zero_()
            install_scale_bypass(m, mode)
            m[0].scale_projection.weight.data.fill_(0.1)
            models[mode] = m
        x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        torch.testing.assert_close(models["constant"](x), models["constant"](x * 4))
        torch.testing.assert_close(models["scale"](x * 4) - models["scale"](x),
                                   torch.full_like(x, 0.1 * torch.log(torch.tensor(4.0)).item()))
        self.assertEqual(sum(p.numel() for p in models["constant"].parameters()),
                         sum(p.numel() for p in models["scale"].parameters()))

    def test_every_site_is_upgraded_and_original_state_keys_survive(self):
        model = nn.Sequential(BiasNorm(4), nn.Sequential(BiasNorm(4), nn.Linear(4, 4)))
        state = copy.deepcopy(model.state_dict())
        self.assertEqual(install_scale_bypass(model), ["0", "1.0"])
        for key, value in state.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        self.assertEqual(sum(isinstance(m, ScaleBiasNorm) for m in model.modules()), 2)
        with self.assertRaises(ValueError):
            install_scale_bypass(model)
        with self.assertRaises(ValueError):
            install_scale_bypass(nn.Linear(4, 4))
        with self.assertRaises(ValueError):
            install_scale_bypass(nn.Sequential(BiasNorm(4)), "energy")

    def test_rejected_installation_does_not_partially_change_other_sites(self):
        installed = nn.Sequential(BiasNorm(4))
        install_scale_bypass(installed)
        model = nn.Sequential(BiasNorm(4), installed)
        with self.assertRaises(ValueError):
            install_scale_bypass(model)
        self.assertIs(type(model[0]), BiasNorm)

    def test_insertion_assets_are_copied_exactly_and_missing_files_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "base", Path(tmp) / "export"
            source.mkdir()
            dest.mkdir()
            (source / "teacher.onnx").write_bytes(b"existing teacher graph")
            with self.assertRaises(FileNotFoundError):
                copy_insertion_assets(source, dest)
            self.assertFalse(list(dest.iterdir()))
            (source / "teacher.onnx.data").write_bytes(b"unchanged external weights")
            hashes = copy_insertion_assets(source, dest)
            self.assertEqual(set(hashes), {"teacher.onnx", "teacher.onnx.data"})
            for name in hashes:
                self.assertEqual((source / name).read_bytes(), (dest / name).read_bytes())

    def test_state_roundtrip_retains_mode_and_projection(self):
        first = nn.Sequential(BiasNorm(4))
        install_scale_bypass(first, "constant")
        first[0].scale_projection.weight.data.fill_(0.3)
        second = nn.Sequential(BiasNorm(4))
        install_scale_bypass(second, "constant")
        second.load_state_dict(first.state_dict())
        x = torch.randn(2, 4)
        self.assertTrue(torch.equal(first(x), second(x)))

    def test_validation_partition_cannot_share_speakers(self):
        records = [dict(speaker=f"training-{i}") for i in range(30)]
        train, validation = partition_records(records)
        self.assertTrue(train and validation)
        self.assertFalse({r["speaker"] for r in train} & {r["speaker"] for r in validation})
        with self.assertRaises(ValueError):
            train_pair("not-opened.pt", [records[0]], [records[0]], "unused", None)

    def test_learner_admission_rejects_held_out_scripts_repeats_and_exclusions(self):
        import json
        import numpy as np
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sf.write(root / "take.wav", np.zeros(1600, dtype=np.float32), 16000)
            base = dict(id="ordinary", split="train", kind="none", speaker="training",
                        events=[], non_errors=[], ayahs=["1:1"], wav="take.wav")
            rejected = [dict(split="val"), dict(split="test"), dict(kind="omitted_word"),
                        dict(events=[{"type": "added_word"}]), dict(non_errors=[{"type": "repeat"}]),
                        dict(excluded=True), dict(exclude=True), dict(unsupported=True)]
            rows = [base] + [{**base, **change, "id": f"rejected-{i}"} for i, change in enumerate(rejected)]
            (root / "labels.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            reference = root / "reference.json"
            reference.write_text(json.dumps(dict(ayahs=[dict(s=1, a=1, p="بت")])))
            admitted = learner_records(root, reference)
            self.assertEqual([row["key"] for row in admitted], ["ordinary"])
            self.assertEqual(admitted[0]["speaker"], "help:training")

    def test_nonzero_bypass_exports_to_ordinary_onnx_operations(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        from icefall.utils import get_onnx_export_kwargs

        for mode in ("scale", "constant"):
            model = nn.Sequential(BiasNorm(4)).eval()
            install_scale_bypass(model, mode)
            model[0].scale_projection.weight.data.fill_(0.2)
            x = torch.randn(2, 5, 4)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "model.onnx")
                torch.onnx.export(model, (x,), path, input_names=["x"], output_names=["y"],
                                  opset_version=17, **get_onnx_export_kwargs())
                graph = onnx.load(path)
                self.assertTrue(all(node.domain in ("", "ai.onnx") for node in graph.graph.node))
                actual = ort.InferenceSession(path).run(None, {"x": x.numpy()})[0]
                np.testing.assert_allclose(actual, model(x).detach().numpy(), rtol=1e-5, atol=1e-6)

    def test_full_acoustic_model_preserves_initial_outputs_and_exports_warmed_states(self):
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from student_export import export_onnx, wrap
        import student_train as training

        size = dict(num_encoder_layers="1,1,1,1,1,1", feedforward_dim="32,32,32,32,32,32",
                    encoder_dim="16,16,16,16,16,16", encoder_unmasked_dim="16,16,16,16,16,16")
        common = dict(num_heads="2,2,2,2,2,2", query_head_dim="8", value_head_dim="4", pos_head_dim="2",
                      pos_dim=8, cnn_module_kernel="3,3,3,3,3,3")
        with patch.dict(training.SIZES, {"scale-test": size}), patch.dict(training.COMMON, common):
            torch.manual_seed(0)
            original = training.build_model("scale-test").eval()
            model = training.build_model("scale-test", "scale").eval()
            missing, unexpected = model.load_state_dict(original.state_dict(), strict=False)
            self.assertTrue(missing)
            self.assertTrue(all(".scale_projection.weight" in key for key in missing))
            self.assertFalse(unexpected)
            plain, bypass = wrap(original, 16, 64), wrap(model, 16, 64)
            a, b = plain.init_states(), bypass.init_states()
            with torch.no_grad():
                for _ in range(3):
                    x = torch.randn(1, 45, 80)
                    expected, a = plain(x, a)
                    actual, b = bypass(x, b)
                    self.assertTrue(torch.equal(expected, actual))
                    for left, right in zip(a, b):
                        self.assertTrue(torch.equal(left, right))
            for module in model.modules():
                if isinstance(module, ScaleBiasNorm):
                    module.scale_projection.weight.data.fill_(0.01)
            with tempfile.TemporaryDirectory() as tmp:
                fp32, int8 = Path(tmp) / "fp32.onnx", Path(tmp) / "int8.onnx"
                states = export_onnx(model, str(fp32), 16, 64)
                quantize_dynamic(str(fp32), str(int8), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
                parity = verify_streaming_export(model, fp32, int8, states, 16, 64)
                self.assertEqual(parity["windows"], 3)
                self.assertLess(parity["max_abs_probability_error"]["fp32"], 1e-4)


if __name__ == "__main__":
    unittest.main()

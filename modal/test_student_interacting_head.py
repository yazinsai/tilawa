"""Behavior and exported-primary parity checks; no practice inputs required.

python -m unittest discover -s modal -p test_student_interacting_head.py -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from student_interacting_head import (
    DenseHead, InteractingCodeHead, fixed_code, heads, matched_widths,
    objective, partition, phone_log_probs,
)
from student_interacting_export import digest, expose, graft, insertion_assets, session, stream
from student_interacting_train import admitted_help


class PhoneHeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.code = fixed_code(43)

    def test_exact_parameter_control_and_identity_initialization(self):
        bottleneck, hidden = matched_widths(256, 43)
        self.assertEqual((bottleneck, hidden), (52, 79))
        models = (InteractingCodeHead(256, self.code, bottleneck), DenseHead(256, 43, hidden))
        x, shipped = torch.randn(11, 256), torch.randn(11, 43).log_softmax(-1)
        for model in models:
            self.assertEqual(sum(p.numel() for p in model.parameters()), 23700)
            torch.testing.assert_close(model(x), torch.zeros_like(shipped), rtol=0, atol=0)
            lp = phone_log_probs(model, x, shipped, torch.zeros(256), torch.ones(256))
            torch.testing.assert_close(lp.exp(), shipped.exp())
        self.assertEqual(torch.unique(self.code, dim=0).shape[0], 43)
        torch.testing.assert_close(self.code.sum(-1), torch.zeros(43))
        self.assertEqual(torch.unique(models[0].pairs, dim=0).shape[0], 43)

    def test_pairwise_coefficients_change_odds_nonlinearly(self):
        # Disable singles and isolate one acoustic-dependent pair coefficient.
        model = InteractingCodeHead(2, self.code, 1).double()
        pair = next(i for i in range(model.pairs.shape[1]) if model.pairs[0, i] != model.pairs[1, i])
        with torch.no_grad():
            model.bottleneck.weight.copy_(torch.tensor([[.7, -.4]]))
            model.bottleneck.bias.fill_(.3)
            model.interaction.weight[pair, 0] = 2.
        x = torch.tensor([.4, -.2], dtype=torch.float64, requires_grad=True)
        odds = lambda value: model(value)[0] - model(value)[1]
        hessian = torch.autograd.functional.hessian(odds, x)
        self.assertGreater(abs(float(hessian[0, 1])), .001)
        self.assertNotEqual(float(odds(x)), 0.)
        # Adding a shared logit offset cannot remove the pair's phone contrast.
        lp = (model(x) + 100).log_softmax(-1)
        self.assertAlmostEqual(float(lp[0] - lp[1]), float(odds(x)), places=5)

    def test_ctc_backpropagates_into_head_without_training_observations(self):
        model = InteractingCodeHead(4, self.code, 3)
        x = torch.randn(12, 4)
        shipped = torch.randn(12, 43).log_softmax(-1)
        target = torch.tensor([1, 1, 2])  # Repeated phone requires an intervening blank.
        lp = phone_log_probs(model, x, shipped, torch.zeros(4), torch.ones(4))
        objective(lp, target, shipped, 1.).backward()
        self.assertGreater(float(model.single.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.interaction.weight.grad.abs().sum()), 0.)
        self.assertIsNone(x.grad)
        self.assertIsNone(shipped.grad)
        with self.assertRaisesRegex(ValueError, "complete phone transcript"):
            objective(lp[:3], target, shipped[:3], 1.)

    def test_speaker_partitions_and_performed_transcript_admission(self):
        groups = partition(["a", "b", "c", "a", "d", "e"])
        self.assertEqual(groups, partition(["e", "d", "c", "b", "a"]))
        self.assertEqual(set(groups.values()), {0, 1, 2})
        def row(speaker, kind="none", split="train", **kwargs):
            return dict(speaker=speaker, kind=kind, split=split, **kwargs)
        data = [row("a"), row("a", split="eval"), row("b", kind="substitution"),
                row("c", non_errors=[{"kind": "repeat"}]), row("d"), row("e", excluded="review"),
                row("f", scored=False), row("g", events=[{"errorClass": "added_word"}])]
        self.assertEqual([r["speaker"] for r in admitted_help(data)], ["d"])


def toy_bundle(path):
    """Stateful, named CTC graph with a native feature width matching production.

    This makes graft/cache checks inexpensive and independent of a private model.
    """
    width, vocab = 256, 43
    rng = np.random.default_rng(11)
    graph = helper.make_graph([
        helper.make_node("Slice", ["x", "start", "end", "axis", "stride"], ["subsampled"]),
        helper.make_node("MatMul", ["subsampled", "embedding"], ["embedded"]),
        helper.make_node("Add", ["embedded", "cache"], ["features"], name="encoder"),
        helper.make_node("MatMul", ["features", "weight"], ["scores"], name="ctc_output.1/MatMul"),
        helper.make_node("LogSoftmax", ["scores"], ["log_probs"], axis=-1),
        helper.make_node("Identity", ["features"], ["new_cache"]),
    ], "toy", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 29, 80]),
                helper.make_tensor_value_info("cache", TensorProto.FLOAT, [1, 4, width])],
       [helper.make_tensor_value_info("log_probs", TensorProto.FLOAT, [1, 4, vocab]),
        helper.make_tensor_value_info("new_cache", TensorProto.FLOAT, [1, 4, width])],
       [numpy_helper.from_array(rng.normal(0, .01, (width, vocab)).astype("float32"), "weight"),
        numpy_helper.from_array(rng.normal(0, .01, (80, width)).astype("float32"), "embedding"),
        *[numpy_helper.from_array(np.array([value], dtype="int64"), name)
          for name, value in (("start", 0), ("end", 16), ("axis", 1), ("stride", 4))]])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, str(path / "model.onnx"))
    m = dict(schema=1, id="unit-test", file="model.onnx", blank=0, frameMs=40,
             tokens="".join(chr(0x621 + i) for i in range(42)), fbank=dict(mels=80, padValue=-23.0259),
             sampleRate=16000, window=29, shift=16,
             states=[dict(name="cache", dtype="float32", shape=[1, 4, width])],
             sha256=digest(path / "model.onnx"))
    (path / "manifest.json").write_text(json.dumps(m))
    return m


class PrimaryExportTest(unittest.TestCase):
    def test_interacting_and_control_primary_graphs_match_torch_and_keep_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            m = toy_bundle(source)
            source_manifest_bytes = (source / "manifest.json").read_bytes()
            primary_hash = digest(source / "model.onnx")
            (source / "teacher.onnx.data").write_bytes(b"unit-test auxiliary data")
            (source / "auxiliary").mkdir()
            (source / "auxiliary" / "weights.data").write_bytes(b"nested data")
            _, feature, width, extractor = expose(source, Path(directory) / "exposed")
            self.assertEqual(width, 256)
            baseline, frozen = session(source / "model.onnx"), session(extractor)
            for arm, head in heads(width, 43).items():
                torch.manual_seed(33)
                with torch.no_grad():
                    # Nonzero residuals ensure this verifies the trained-head path.
                    for p in head.parameters(): p.normal_(0, .04)
                center, scale = np.full(width, .2, dtype="float32"), np.full(width, 1.3, dtype="float32")
                out = Path(directory) / arm
                path = graft(source, head, center, scale, out, arm)
                deployed = session(path, optimize=True)
                state = np.zeros((1, 4, width), dtype="float32")
                rng = np.random.default_rng(2)
                for _ in range(4):
                    feeds = dict(x=rng.normal(size=(1, 29, 80)).astype("float32"), cache=state)
                    expected = baseline.run(["log_probs", "new_cache"], feeds)
                    raw, cache, x = frozen.run(["log_probs", "new_cache", feature], feeds)
                    np.testing.assert_array_equal(raw, expected[0])
                    np.testing.assert_array_equal(cache, expected[1])
                    lp, next_cache = deployed.run(["log_probs", "new_cache"], feeds)
                    with torch.no_grad():
                        want = phone_log_probs(head, torch.tensor(x), torch.tensor(raw),
                                               torch.tensor(center), torch.tensor(scale)).numpy()
                    np.testing.assert_allclose(lp, want, rtol=1e-5, atol=1e-5)
                    np.testing.assert_array_equal(next_cache, cache)
                    np.testing.assert_allclose(np.exp(lp).sum(-1), 1., atol=1e-6)
                    state = cache
                new_m = json.loads((out / "manifest.json").read_text())
                self.assertEqual(new_m["tokens"], m["tokens"])
                self.assertEqual(new_m["states"], m["states"])
                self.assertEqual(new_m["sha256"], digest(path))
                self.assertEqual((out / "teacher.onnx.data").read_bytes(), b"unit-test auxiliary data")
                self.assertEqual((out / "auxiliary" / "weights.data").read_bytes(), b"nested data")
            self.assertEqual(digest(source / "model.onnx"), primary_hash)
            self.assertEqual((source / "manifest.json").read_bytes(), source_manifest_bytes)

    def test_stream_drains_partial_windows_with_the_sdk_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            m = toy_bundle(source)
            _, feature, width, extractor = expose(source, Path(directory) / "exposed")
            frozen = session(extractor)
            for frames in (9, 28, 29, 30, 45, 91):
                fbank = np.random.default_rng(frames).normal(size=(frames, 80)).astype("float32")
                lp, features = stream(frozen, m, fbank, feature)
                total = ((frames - 7) // 2 + 1) // 2
                self.assertEqual(lp.shape, (total, 43))
                self.assertEqual(features.shape, (total, width))
                # Independently emulate the SDK: drain every full input window,
                # then pad and emit only the remaining valid rows at end().
                state = np.zeros((1, 4, width), dtype="float32")
                expected, consumed, emitted = [], 0, 0
                while frames - consumed >= m["window"] or emitted < total:
                    view = fbank[consumed:consumed + m["window"]]
                    x = np.full((1, m["window"], 80), m["fbank"]["padValue"], dtype="float32")
                    x[0, :len(view)] = view
                    rows, state = frozen.run(["log_probs", "new_cache"], dict(x=x, cache=state))
                    keep = len(rows[0]) if len(view) == m["window"] else min(len(rows[0]), total - emitted)
                    expected.append(rows[0, :keep])
                    consumed += m["shift"]
                    emitted += keep
                np.testing.assert_array_equal(lp, np.concatenate(expected))

    def test_export_rejects_invalid_normalization_and_in_place_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            toy_bundle(source)
            head = heads(256, 43)["interacting"]
            center, scale = np.zeros(256, dtype="float32"), np.ones(256, dtype="float32")
            with self.assertRaisesRegex(ValueError, "separate"):
                graft(source, head, center, scale, source, "interacting")
            center[0] = np.nan
            with self.assertRaisesRegex(ValueError, "normalization"):
                graft(source, head, center, scale, Path(directory) / "out", "interacting")

    def test_feature_extraction_fails_for_an_unidentifiable_head(self):
        from student_interacting_export import native_features
        graph = helper.make_model(helper.make_graph([], "absent", [], []))
        with self.assertRaisesRegex(ValueError, "uniquely identify"):
            native_features(graph)

    def test_required_insertion_assets_cannot_silently_disappear(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaisesRegex(ValueError, "insertion teacher"):
                insertion_assets(source)
            graph = helper.make_model(helper.make_graph([], "teacher", [], []))
            onnx.save(graph, str(source / "teacher.onnx"))
            with self.assertRaisesRegex(ValueError, "data file"):
                insertion_assets(source)
            (source / "teacher.onnx.data").write_bytes(b"unchanged")
            self.assertEqual(set(insertion_assets(source)), {"teacher.onnx", "teacher.onnx.data"})
            weight = numpy_helper.from_array(np.ones(2, dtype="float32"), "external")
            weight.ClearField("raw_data")
            weight.data_location = TensorProto.EXTERNAL
            for key, value in (("location", "missing.data"), ("length", "8")):
                entry = weight.external_data.add()
                entry.key, entry.value = key, value
            graph.graph.initializer.append(weight)
            onnx.save(graph, str(source / "teacher.onnx"))
            with self.assertRaisesRegex(ValueError, "external weights"):
                insertion_assets(source)


if __name__ == "__main__":
    unittest.main()

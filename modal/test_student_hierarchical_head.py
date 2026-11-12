"""Behavior and exported-primary parity checks; no practice inputs required.

python -m unittest discover -s modal -p test_student_hierarchical_head.py -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from student_hierarchical_head import HierarchicalHead, heads, objective, partition, phone_log_probs
from student_hierarchical_tree import phonetic_tree, matched_widths
from student_hierarchical_export import digest, expose, graft, insertion_assets, session, stream
from student_hierarchical_train import admitted_help
from student import TOKENS


class PhoneHeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_parameter_control_and_identity_initialization(self):
        self.assertEqual(matched_widths(256, TOKENS), (25, 326, 97800))
        x = torch.randn(11, 256)
        shipped = torch.randn(11, 43).log_softmax(-1)
        for model in heads(256, TOKENS).values():
            self.assertEqual(sum(p.numel() for p in model.parameters()), 97800)
            lp = phone_log_probs(model, x, shipped, torch.zeros(256), torch.ones(256))
            torch.testing.assert_close(lp.exp(), shipped.exp())
            torch.testing.assert_close(lp.exp().sum(-1), torch.ones(11))

    def test_within_family_discrimination_is_nonlinear_and_conditionally_independent(self):
        model = HierarchicalHead(2, TOKENS, 1).double()
        branch = next(i for i, n in enumerate(model.nodes) if n.name == "closures")
        classifier = model.branches[branch]
        with torch.no_grad():
            classifier.hidden.weight.copy_(torch.tensor([[.7, -.4]]))
            classifier.hidden.bias.fill_(.3)
            classifier.output.weight[0, 0] = 2.
        x = torch.tensor([.4, -.2], dtype=torch.float64, requires_grad=True)
        shipped = torch.full((43,), -10., dtype=torch.float64).log_softmax(-1)
        left, right = [c[0] for c in model.nodes[branch].children[:2]]
        def odds(value):
            lp = model(value, shipped)
            return lp[left] - lp[right]
        hessian = torch.autograd.functional.hessian(odds, x)
        self.assertGreater(abs(float(hessian[0, 1])), .001)
        before = model(x, shipped)
        # Changing another family's nonlinear classifier cannot affect these
        # within-family odds or the family's total mass; parent routing is fixed.
        unrelated = next(i for i, n in enumerate(model.nodes) if n.name == "front")
        with torch.no_grad():
            model.branches[unrelated].output.weight.fill_(.8)
            model.branches[unrelated].output.weight[0].fill_(1.6)
        after = model(x, shipped)
        torch.testing.assert_close(before[left] - before[right], after[left] - after[right])
        family = [c[0] for c in model.nodes[branch].children]
        torch.testing.assert_close(before[family], after[family])
        torch.testing.assert_close(after.exp().sum(), torch.tensor(1., dtype=torch.float64))

    def test_ctc_trains_all_nodes_without_training_frozen_observations(self):
        model = HierarchicalHead(4, TOKENS, 3)
        x = torch.randn(100, 4, requires_grad=True)
        shipped = torch.randn(100, 43).log_softmax(-1).detach().requires_grad_()
        target = torch.tensor([1, 1, *range(2, 43)])
        for step in range(2):
            model.zero_grad()
            lp = phone_log_probs(model, x, shipped, torch.zeros(4), torch.ones(4))
            objective(lp, target, shipped, 1.).backward()
            for classifier in model.branches:
                self.assertGreater(float(classifier.output.weight.grad.abs().sum()), 0.)
                self.assertTrue(torch.isfinite(classifier.output.weight.grad).all())
                if step:
                    self.assertGreater(float(classifier.hidden.weight.grad.abs().sum()), 0.)
            with torch.no_grad():
                for p in model.parameters(): p -= .01 * p.grad
        self.assertIsNone(x.grad)
        self.assertIsNone(shipped.grad)
        with self.assertRaisesRegex(ValueError, "complete phone transcript"):
            objective(lp[:3], target, shipped[:3], 1.)

    def test_extreme_branch_scores_remain_normalized(self):
        model = HierarchicalHead(2, TOKENS, 2)
        with torch.no_grad():
            for classifier in model.branches:
                classifier.output.weight.normal_(0, 100.)
        lp = model(torch.randn(8, 2), (torch.randn(8, 43) * 100).log_softmax(-1))
        self.assertTrue(torch.isfinite(lp).all())
        torch.testing.assert_close(lp.logsumexp(-1), torch.zeros(8), atol=1e-5, rtol=0)

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
             tokens=TOKENS, fbank=dict(mels=80, padValue=-23.0259),
             sampleRate=16000, window=29, shift=16,
             states=[dict(name="cache", dtype="float32", shape=[1, 4, width])],
             sha256=digest(path / "model.onnx"))
    (path / "manifest.json").write_text(json.dumps(m))
    teacher = helper.make_model(helper.make_graph([], "teacher", [], []))
    onnx.save(teacher, str(path / "teacher.onnx"))
    (path / "teacher.onnx.data").write_bytes(b"unit-test auxiliary data")
    return m


class PrimaryExportTest(unittest.TestCase):
    def test_quantized_source_exposes_float_features_and_retains_stream_state(self):
        from onnxruntime.quantization import QuantType, quantize_dynamic
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            m = toy_bundle(source)
            quantize_dynamic(str(source / "model.onnx"), str(source / "quantized.onnx"),
                             op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
            m.update(file="quantized.onnx", sha256=digest(source / "quantized.onnx"))
            (source / "manifest.json").write_text(json.dumps(m))
            _, feature, width, extractor = expose(source, Path(directory) / "exposed")
            self.assertEqual(width, 256)
            frozen = session(extractor)
            model = heads(width, TOKENS)["hierarchical"]
            with torch.no_grad():
                for classifier in model.branches:
                    classifier.output.weight.normal_(0, .1)
            center, scale = np.zeros(width, dtype="float32"), np.ones(width, dtype="float32")
            out = Path(directory) / "readout"
            deployed = session(graft(source, model, center, scale, out, "hierarchical"), optimize=True)
            state = np.zeros((1, 4, width), dtype="float32")
            for step in range(3):
                feeds = dict(x=np.random.default_rng(step).normal(size=(1, 29, 80)).astype("float32"), cache=state)
                raw, state, x = frozen.run(["log_probs", "new_cache", feature], feeds)
                got, cache = deployed.run(["log_probs", "new_cache"], feeds)
                with torch.no_grad():
                    want = phone_log_probs(model, torch.tensor(x), torch.tensor(raw),
                                           torch.tensor(center), torch.tensor(scale)).numpy()
                np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
                np.testing.assert_array_equal(cache, state)

    def test_hierarchical_and_control_primary_graphs_match_torch_and_keep_caches(self):
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
            for arm, head in heads(width, TOKENS).items():
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
            head = heads(256, TOKENS)["hierarchical"]
            center, scale = np.zeros(256, dtype="float32"), np.ones(256, dtype="float32")
            with self.assertRaisesRegex(ValueError, "separate"):
                graft(source, head, center, scale, source, "hierarchical")
            center[0] = np.nan
            with self.assertRaisesRegex(ValueError, "normalization"):
                graft(source, head, center, scale, Path(directory) / "out", "hierarchical")

    def test_feature_extraction_fails_for_an_unidentifiable_head(self):
        from student_hierarchical_export import native_features
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

"""Behavior tests for unchanged-weight streaming attention graph surgery."""

import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from student_attention_recheck import entropy_temperature, recheck_graph, spread_scale


def fixture():
    info = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    return helper.make_model(helper.make_graph([
        helper.make_node("MatMul", ["q", "k"], ["content"], name="/encoder/layer/self_attn_weights/MatMul"),
        helper.make_node("Add", ["content", "position"], ["logits"]),
        helper.make_node("Where", ["mask", "padding", "logits"], ["masked"]),
        helper.make_node("Softmax", ["masked"], ["weights"], axis=-1),
        helper.make_node("MatMul", ["weights", "values"], ["context"]),
        helper.make_node("Identity", ["k"], ["cached_key"]),
    ], "attention", [info("q", [2, 1, 2, 2]), info("k", [2, 1, 2, 3]),
                     info("position", [2, 1, 2, 3]), info("values", [2, 1, 3, 2]),
                     helper.make_tensor_value_info("mask", TensorProto.BOOL, [2, 1, 2, 3])],
       [info("context", [2, 1, 2, 2]), info("cached_key", [2, 1, 2, 3])],
       initializer=[numpy_helper.from_array(np.array(-1000, np.float32), "padding")]),
       opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


class AttentionRecheckTest(unittest.TestCase):
    def setUp(self):
        self.model = fixture()
        self.scales = {"/encoder/layer/self_attn_weights/MatMul": [2, 3]}
        rng = np.random.default_rng(42)
        self.feed = {v.name: rng.normal(size=[d.dim_value for d in v.type.tensor_type.shape.dim]).astype(np.float32)
                     for v in self.model.graph.input if v.name != "mask"}
        self.feed["mask"] = np.zeros((2, 1, 2, 3), dtype=bool)

    def run_graph(self, graph, feed=None):
        return ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feed or self.feed)

    def test_content_selection_ignores_norm_while_cache_and_values_retain_it(self):
        original_bytes = self.model.SerializeToString()
        graph = recheck_graph(self.model, self.scales)
        self.assertEqual(self.model.SerializeToString(), original_bytes)
        self.assertEqual(list(graph.graph.input), list(self.model.graph.input))
        self.assertEqual(list(graph.graph.output), list(self.model.graph.output))
        self.assertEqual(graph.graph.initializer[0], self.model.graph.initializer[0])
        a, cache = self.run_graph(graph)
        feed = dict(self.feed)
        feed["q"] = feed["q"] * np.array([1, 7], np.float32).reshape(1, 1, 2, 1)
        feed["k"] = feed["k"] * np.array([2, 9, .1], np.float32).reshape(1, 1, 1, 3)
        b, changed_cache = self.run_graph(graph, feed)
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_array_equal(cache, self.feed["k"])
        np.testing.assert_array_equal(changed_cache, feed["k"])
        feed["values"] = feed["values"] * 4
        c, _ = self.run_graph(graph, feed)
        np.testing.assert_allclose(c, 4 * a, atol=1e-6)
        original, _ = self.run_graph(self.model, feed)
        self.assertGreater(np.max(np.abs(original - c)), .01)

    def test_control_preserves_norm_preference_and_one_is_exact_identity(self):
        graph = recheck_graph(self.model, {next(iter(self.scales)): [1, 1]}, temperature=True)
        a, cache = self.run_graph(graph)
        original, original_cache = self.run_graph(self.model)
        np.testing.assert_array_equal(a, original)
        np.testing.assert_array_equal(cache, original_cache)
        feed = dict(self.feed)
        feed["k"] = feed["k"] * np.array([1, 8, 1], np.float32).reshape(1, 1, 1, 3)
        changed, _ = self.run_graph(graph, feed)
        self.assertGreater(np.max(np.abs(a - changed)), .01)

    def test_zero_norm_is_finite_and_preserves_original_cache(self):
        feed = dict(self.feed)
        feed["q"] = np.zeros_like(feed["q"])
        feed["k"] = np.zeros_like(feed["k"])
        context, cache = self.run_graph(recheck_graph(self.model, self.scales), feed)
        self.assertTrue(np.isfinite(context).all())
        np.testing.assert_array_equal(cache, feed["k"])

    def test_padding_mask_is_unchanged_in_both_arms(self):
        feed = dict(self.feed)
        feed["mask"] = np.ones_like(feed["mask"])
        feed["mask"][..., 1] = False
        for temperature in [False, True]:
            graph = recheck_graph(self.model, self.scales, temperature=temperature)
            context, _ = self.run_graph(graph, feed)
            expected = np.repeat(feed["values"][..., 1:2, :], 2, axis=-2)
            np.testing.assert_allclose(context, expected, atol=1e-6)

    def test_calibration_matches_spread_with_position_covariance(self):
        rng = np.random.default_rng(3)
        unit = rng.normal(size=(20, 8))
        position = .7 * unit + rng.normal(size=(20, 8))
        original = 4.2 * unit + position
        scale = spread_scale(unit, position, original)
        self.assertAlmostEqual(scale, 4.2, places=10)
        self.assertAlmostEqual(entropy_temperature(original, original * .3), .3, places=10)

    def test_incomplete_or_invalid_calibration_fails(self):
        with self.assertRaises(ValueError):
            recheck_graph(self.model, {})
        with self.assertRaises(ValueError):
            recheck_graph(self.model, {next(iter(self.scales)): [0, 1]})


if __name__ == "__main__":
    unittest.main()

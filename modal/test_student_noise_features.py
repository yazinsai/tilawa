"""Small graph fixtures for frozen feature capture and head-only export."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from student_noise_features import (ROOT, digest, feature_runtime, linear_head,
    export, replace_head_parameters, stream_features, write_json)

class Export(unittest.TestCase):
    def fixture(self, quantized):
        import numpy as np
        import onnx
        from onnx import helper as h, numpy_helper as nh, TensorProto as T

        rng = np.random.default_rng(831)
        weight = rng.normal(size=(80, 3)).astype('float32')
        bias = rng.normal(size=3).astype('float32')
        constants = [nh.from_array(bias, 'ctc_output.1.bias')]
        if quantized:
            scale = np.asarray(np.max(np.abs(weight))/127, dtype='float32')
            q = np.round(weight/scale).astype('int8')
            constants += [nh.from_array(q, 'weight_quantized'), nh.from_array(scale, 'weight_scale'),
                          nh.from_array(np.asarray(0, dtype='int8'), 'weight_zero_point')]
            nodes = [h.make_node('DynamicQuantizeLinear', ['x'], ['q', 'qs', 'qz']),
                     h.make_node('MatMulInteger', ['q', 'weight_quantized', 'qz', 'weight_zero_point'], ['integer'],
                                 name='/ctc_output/ctc_output.1/MatMul_quant'),
                     h.make_node('Cast', ['integer'], ['floating'], to=T.FLOAT),
                     h.make_node('Mul', ['qs', 'weight_scale'], ['joint_scale']),
                     h.make_node('Mul', ['floating', 'joint_scale'], ['product'])]
        else:
            constants.append(nh.from_array(weight, 'weight'))
            nodes = [h.make_node('MatMul', ['x', 'weight'], ['product'], name='/ctc_output/ctc_output.1/MatMul')]
        nodes += [h.make_node('Add', ['product', 'ctc_output.1.bias'], ['logits']),
                  h.make_node('LogSoftmax', ['logits'], ['log_probs'], axis=-1),
                  h.make_node('Identity', ['cache'], ['new_cache'])]
        graph = h.make_graph(nodes, 'head-fixture',
            [h.make_tensor_value_info('x', T.FLOAT, [1, 4, 80]), h.make_tensor_value_info('cache', T.FLOAT, [1])],
            [h.make_tensor_value_info('log_probs', T.FLOAT, [1, 4, 3]), h.make_tensor_value_info('new_cache', T.FLOAT, [1])], constants)
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        return model

    def test_feature_capture_reproduces_float_and_quantized_head(self):
        import numpy as np
        import onnx

        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            for quantized in (False, True):
                source = Path(tmp)/str(quantized)
                source.mkdir()
                onnx.save(self.fixture(quantized), source/'model.onnx')
                write_json(source/'manifest.json', dict(file='model.onnx', sha256=digest(source/'model.onnx'),
                    tokens='بت', window=4, shift=4, chunk=8,
                    states=[dict(name='cache', dtype='float32', shape=[1])], fbank=dict(padValue=-23.0259)))
                out = source/'features'
                out.mkdir()
                manifest, head, runtime = feature_runtime(source, out)
                x = np.random.default_rng(42).normal(size=(1, 4, 80)).astype('float32')
                lp, features = runtime.run(['log_probs', head['feature']], dict(x=x, cache=np.zeros(1, dtype='float32')))
                logits = features@head['weight']+head['bias']
                peak = logits.max(-1, keepdims=True)
                reconstructed = logits-peak-np.log(np.exp(logits-peak).sum(-1, keepdims=True))
                np.testing.assert_allclose(reconstructed, lp, atol=.0001, rtol=0)

    def test_refit_changes_only_head_constants_and_preserves_stream_state(self):
        import numpy as np
        import onnxruntime as ort

        for quantized in (False, True):
            graph = self.fixture(quantized)
            before = graph.SerializeToString()
            original_nodes = [n.SerializeToString() for n in graph.graph.node]
            head = linear_head(graph, 3)
            changed = replace_head_parameters(graph, head['weight']*.7, head['bias']+1.)
            self.assertEqual([n.SerializeToString() for n in graph.graph.node], original_nodes)
            self.assertEqual(len(changed), 4 if quantized else 2)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            a, b = [ort.InferenceSession(g, options, providers=['CPUExecutionProvider']) for g in (before, graph.SerializeToString())]
            feeds = dict(x=np.random.default_rng(42).normal(size=(1, 4, 80)).astype('float32'),
                         cache=np.asarray([17.], dtype='float32'))
            p, q = a.run(None, feeds), b.run(None, feeds)
            np.testing.assert_array_equal(p[1], q[1])
            self.assertFalse(np.allclose(p[0], q[0]))

    def test_refit_retains_per_channel_weight_quantization(self):
        import numpy as np
        from onnx import numpy_helper

        graph = self.fixture(True)
        tensors = {t.name: t for t in graph.graph.initializer}
        scale = numpy_helper.to_array(tensors['weight_scale'])
        tensors['weight_scale'].CopyFrom(numpy_helper.from_array(np.repeat(scale, 3), 'weight_scale'))
        tensors['weight_zero_point'].CopyFrom(numpy_helper.from_array(np.zeros(3, dtype='int8'), 'weight_zero_point'))
        head = linear_head(graph, 3)
        weight = head['weight']*np.asarray([.5, 1., 2.])
        replace_head_parameters(graph, weight, head['bias'])
        found = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
        self.assertEqual(found['weight_scale'].shape, (3,))
        self.assertEqual(found['weight_zero_point'].shape, (3,))
        np.testing.assert_allclose(found['weight_scale'], np.max(np.abs(weight), axis=0)/127, rtol=1e-6)

    def test_cached_stream_trims_final_padding_and_carries_encoder_state(self):
        import numpy as np

        class Runtime:
            calls = 0

            def run(self, outputs, feeds):
                self.calls += 1
                np.testing.assert_array_equal(feeds['cache'], [self.calls-1])
                offset = (self.calls-1)*8
                features = np.arange(offset, offset+8, dtype='float32')[None, :, None]
                return [np.repeat(features, 3, axis=-1),
                        np.asarray([self.calls], dtype='float32'), features]

        manifest = dict(window=45, shift=32, chunk=16,
                        states=[dict(name='cache', shape=[1], dtype='float32')],
                        fbank=dict(padValue=-23.0259))
        # 45 real fbank frames yield 10 output rows, despite 16 padded rows in
        # two model windows. This is the SDK's end-of-stream boundary.
        runtime = Runtime()
        features, baseline = stream_features(runtime, manifest, np.zeros((45, 80), dtype='float32'), 'features')
        self.assertEqual(runtime.calls, 2)
        self.assertEqual(features.shape, (10, 1))
        self.assertEqual(baseline.shape, (10, 3))
        np.testing.assert_array_equal(features[:, 0], np.arange(10))
        with self.assertRaises(ValueError):
            stream_features(Runtime(), manifest, np.zeros((4, 80), dtype='float32'), 'features')

    def test_export_requires_and_retains_complete_inherited_assets(self):
        import numpy as np
        import onnx

        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            root = Path(tmp)
            source, out = root/'source', root/'training'
            source.mkdir()
            out.mkdir()
            graph = self.fixture(True)
            onnx.save(graph, source/'model.onnx')
            manifest = dict(file='model.onnx', sha256=digest(source/'model.onnx'),
                tokens='بت', window=4, shift=4, chunk=8,
                states=[dict(name='cache', dtype='float32', shape=[1])], fbank=dict(padValue=-23.0259))
            write_json(source/'manifest.json', manifest)
            auxiliary = source/'auxiliary.data'
            auxiliary.write_bytes(b'synthetic auxiliary weights')
            registry = root/'packages/sdk/src/registry.ts'
            registry.parent.mkdir(parents=True)
            registry.write_text(f'file: "auxiliary.data", bytes: {auxiliary.stat().st_size}, sha256: "{digest(auxiliary)}"')
            head = linear_head(graph, 3)
            checkpoint = out/'head.npz'
            np.savez(checkpoint, weight=head['weight'].T*.9, bias=head['bias'])
            plan = dict(primary_sha256=manifest['sha256'])
            write_json(out/'plan.json', plan)
            selected = dict(path=str(checkpoint), sha256=digest(checkpoint))
            write_json(out/'selection.json', dict(plan_sha256=digest(out/'plan.json'),
                arms=dict(ctc=selected, crossed=selected)))
            with patch('student_noise_features.ROOT', root):
                # Corrupt declared assets block export before either arm exists.
                auxiliary.write_bytes(b'incomplete')
                with self.assertRaisesRegex(ValueError, 'asset unavailable'):
                    export(source, out, lambda _: plan)
                self.assertFalse((out/'crossed-bundle').exists())
                auxiliary.write_bytes(b'synthetic auxiliary weights')
                export(source, out, lambda _: plan)
            for arm in ('ctc', 'crossed'):
                bundle = out/(arm+'-bundle')
                candidate = json.loads((bundle/'manifest.json').read_text())
                self.assertEqual(candidate['sha256'], digest(bundle/'model.onnx'))
                self.assertEqual(candidate['tokens'], manifest['tokens'])
                self.assertEqual(candidate['states'], manifest['states'])
                self.assertEqual(digest(bundle/'auxiliary.data'), digest(auxiliary))

if __name__ == "__main__":
    unittest.main()

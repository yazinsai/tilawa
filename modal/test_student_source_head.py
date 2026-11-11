"""Real small-graph parity and matched-optimizer tests, with no recitation data."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from student_source_families import CopyPolicy
from student_source_head import (RECIPE, ROOT, clean_target, digest, export, feature_runtime, fit, gradient_effect,
                                linear_head, replace_head_parameters, stream_features, write_json)


class Export(unittest.TestCase):
    def test_clean_transcripts_keep_identity_but_collapse_held_phone_runs(self):
        self.assertEqual(clean_target('ببااابا', 'با'), [1, 2, 1, 2])
        with self.assertRaises(KeyError):
            clean_target('ج', 'با')

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

    def test_export_requires_complete_auxiliary_bundle_and_preserves_manifest_contract(self):
        import numpy as np
        import onnx

        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            root = Path(tmp)
            source, out = root/'source', root/'refit'
            source.mkdir()
            out.mkdir()
            graph = self.fixture(True)
            onnx.save(graph, source/'model.onnx')
            manifest = dict(schema=1, id='synthetic', file='model.onnx',
                sha256=digest(source/'model.onnx'), bytes=(source/'model.onnx').stat().st_size,
                tokens='بت', blank=0, sampleRate=16000, frameMs=40, window=4, shift=4, chunk=8,
                states=[dict(name='cache', dtype='float32', shape=[1])],
                fbank=dict(mels=80, padValue=-23.0259), supported=['wrong_word'])
            write_json(source/'manifest.json', manifest)
            head = linear_head(graph, 3)
            np.savez(out/'initial-head.npz', weight=head['weight'].T, bias=head['bias'])
            (out/'encoder-features.onnx').write_bytes(b'synthetic frozen feature provenance')
            write_json(out/'plan.json', dict(recipe=RECIPE, grouping_policy=asdict(CopyPolicy()), views=[],
                primary_sha256=manifest['sha256'], initial_head_sha256=digest(out/'initial-head.npz'),
                feature_graph_sha256=digest(out/'encoder-features.onnx')))
            arms = {}
            for name, factor in [('original', .8), ('balanced', .7)]:
                checkpoint = out/(name+'.npz')
                np.savez(checkpoint, weight=head['weight'].T*factor, bias=head['bias'])
                arms[name] = dict(path=str(checkpoint), sha256=digest(checkpoint))
            write_json(out/'selection.json', dict(plan_sha256=digest(out/'plan.json'), arms=arms))
            auxiliary = {'teacher.onnx': b'synthetic teacher graph', 'teacher.onnx.data': b'synthetic weights'}
            import hashlib
            registry = root/'packages/sdk/src/registry.ts'
            registry.parent.mkdir(parents=True)
            registry.write_text('\n'.join('file: "%s", bytes: %d, sha256: "%s"' %
                (file, len(value), hashlib.sha256(value).hexdigest()) for file, value in auxiliary.items()))
            with patch('student_source_head.ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Inherited SDK asset unavailable or corrupt'):
                    export(source, out)
                self.assertFalse((out/'balanced-bundle').exists())
                for file, value in auxiliary.items():
                    (source/file).write_bytes(value)
                (source/'teacher.onnx.data').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'Inherited SDK asset unavailable or corrupt'):
                    export(source, out)
                (source/'teacher.onnx.data').write_bytes(auxiliary['teacher.onnx.data'])
                export(source, out)
            for name in arms:
                bundle = out/(name+'-bundle')
                exported = json.loads((bundle/'manifest.json').read_text())
                for key, value in manifest.items():
                    if key not in ('id', 'bytes', 'sha256'):
                        self.assertEqual(exported[key], value)
                self.assertEqual(exported['sha256'], digest(bundle/'model.onnx'))
                for file, value in auxiliary.items():
                    self.assertEqual((bundle/file).read_bytes(), value)
                candidate = onnx.load(str(bundle/'model.onnx'))
                self.assertEqual([n.SerializeToString() for n in graph.graph.node],
                                 [n.SerializeToString() for n in candidate.graph.node])


class MatchedFit(unittest.TestCase):
    def test_corrective_gradient_and_update_count_are_measured_separately(self):
        import numpy as np
        import torch  # available in the supervisor training image

        (ROOT/'.autolab').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.autolab') as tmp:
            out = Path(tmp)
            np.savez(out/'initial-head.npz', weight=np.zeros((3, 2), dtype='float32'), bias=np.zeros(3, dtype='float32'))
            (out/'encoder-features.onnx').write_bytes(b'frozen synthetic cache provenance')
            views, families = [], {}
            for key, family, feature, target, partition, speaker in [
                ('a1', 'a', 1, 1, 'train', 's1'), ('a2', 'a', 1, 1, 'train', 's1'),
                ('a3', 'a', 1, 1, 'train', 's1'), ('b', 'b', 2, 2, 'train', 's2'),
                ('v1', 'v1', 1, 1, 'validation', 'v1'), ('v2', 'v2', 2, 2, 'validation', 'v2')]:
                file = out/(key+'.npz')
                np.savez(file, features=np.asarray([[feature, 0]], dtype='float32'),
                         baseline=np.full((1, 3), -np.log(3), dtype='float32'), target=np.asarray([target], dtype=np.int64))
                views.append(dict(recording=key, speaker=speaker, partition=partition, file=str(file),
                                  sha256=digest(file), original_weight=1))
                families[key] = family
            write_json(out/'plan.json', dict(recipe=RECIPE, grouping_policy=asdict(CopyPolicy()), views=views,
                family_by_recording=families, initial_head_sha256=digest(out/'initial-head.npz'),
                feature_graph_sha256=digest(out/'encoder-features.onnx')))
            fit(out)
            effect = gradient_effect(out, [[2., 0.]], [2], [1])
            self.assertLess(effect['original'], 0)
            self.assertGreater(effect['balanced'], 0)
            self.assertTrue(effect['balancing_only_correction'])
            selected = json.loads((out/'selection.json').read_text())
            self.assertTrue(selected['identical_updates'])
            self.assertTrue(selected['identical_minibatches'])
            self.assertEqual(len(selected['history']), RECIPE['epochs'])
            self.assertEqual(selected['selected']['updates'], selected['selected']['epoch'])
            for name in ('original', 'balanced'):
                self.assertTrue(selected['arms'][name]['path'].endswith(f"{name}-{selected['selected']['epoch']}.npz"))


if __name__ == '__main__':
    unittest.main()

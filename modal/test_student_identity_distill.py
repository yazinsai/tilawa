"""Synthetic checks for loss factorization, full-transcript occupancy and export."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import student_identity_distill as D


class IdentityDistillationTests(unittest.TestCase):
    def test_transcript_occupancy_matches_all_complete_ctc_paths(self):
        probs = np.array([[.7, .2, .1], [.1, .8, .1], [.6, .1, .3], [.2, .2, .6]])
        for target in ([1], [1, 2], [1, 1]):
            total = 0.
            occupied = np.zeros(len(probs))
            for path in itertools.product(range(3), repeat=len(probs)):
                collapsed, previous = [], -1
                for symbol in path:
                    if symbol and symbol != previous:
                        collapsed.append(symbol)
                    previous = symbol
                if collapsed == target:
                    mass = np.prod([probs[t, symbol] for t, symbol in enumerate(path)])
                    total += mass
                    occupied += mass*(np.array(path) != 0)
            got = D.ctc_occupancy(np.log(probs), target)
            self.assertEqual(got.shape, (len(probs), len(target)))
            np.testing.assert_allclose(got.sum(1), occupied/total, atol=1e-12)

    def test_occupancy_excludes_actual_silence_and_rejects_impossible_transcript(self):
        lp = np.array([[0., -np.inf, -np.inf], [-np.inf, 0., -np.inf], [0., -np.inf, -np.inf]])
        np.testing.assert_array_equal(D.ctc_occupancy(lp, [1])[:, 0], [0, 1, 0])
        with self.assertRaisesRegex(ValueError, 'complete'):
            D.ctc_occupancy(np.log(np.full((2, 3), 1/3)), [1, 1])

    def test_probability_reconstruction_preserves_blank_and_speech(self):
        rng = np.random.default_rng(4)
        base = np.log(D.softmax(rng.normal(size=(50, 7)))).astype('float32')
        identity = rng.normal(size=(50, 6))*4
        result = D.reconstruct(base, identity)
        np.testing.assert_array_equal(result[:, 0], base[:, 0])
        np.testing.assert_allclose(np.exp(result[:, 1:]).sum(1), -np.expm1(base[:, 0]), atol=1e-7)
        silence = D.reconstruct(np.array([[0., -np.inf, -np.inf]]), np.array([[1., 3.]]))
        np.testing.assert_array_equal(silence, [[0., -np.inf, -np.inf]])

    @staticmethod
    def item():
        return dict(features=np.array([[1., .5], [-.2, .3]]),
            baseline=np.log(np.array([[.9, .04, .06], [.5, .3, .2]])),
            teacher_identity=np.array([[.8, .2], [.1, .9]]),
            teacher_speech=np.array([.01, .01]), occupancy=np.array([.7, .2]))

    def test_ordinary_kd_attenuates_identity_gradient_by_teacher_speech(self):
        w, b = np.zeros((2, 2)), np.zeros(2)
        with patch.dict(D.RECIPE, anchor=0.):
            a = D.objective(w, b, self.item(), True, True)
            o = D.objective(w, b, self.item(), False, True)
        np.testing.assert_allclose(o[1], .01*a[1], atol=1e-12)
        np.testing.assert_allclose(o[2], .01*a[2], atol=1e-12)
        # The gradient and wanted identity are zero on unoccupied frames.
        item = self.item()
        item['occupancy'] = np.array([1., 0.])
        first = D.objective(w, b, item, True, True)
        item['teacher_identity'][1] = [1., 0.]
        second = D.objective(w, b, item, True, True)
        np.testing.assert_array_equal(first[1], second[1])

    def test_analytic_gradient_matches_numerical_gradient(self):
        w = np.array([[.2, .1], [-.3, .4]])
        b = np.array([.1, -.2])
        for conditional in [False, True]:
            _, dw, db = D.objective(w, b, self.item(), conditional, True)
            for array, derivative in [(w, dw), (b, db)]:
                for at in np.ndindex(array.shape):
                    original = array[at]
                    array[at] = original + 1e-6
                    high = D.objective(w, b, self.item(), conditional)
                    array[at] = original - 1e-6
                    low = D.objective(w, b, self.item(), conditional)
                    array[at] = original
                    self.assertAlmostEqual(derivative[at], (high-low)/2e-6, places=7)

    def test_matched_fit_uses_training_partition_and_same_selected_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root/'initial-head.npz', weight=np.zeros((2, 2)), bias=np.zeros(2))
            views = []
            for speaker, partition in [('training', 'train'), ('reserved', 'validation')]:
                path = root/(speaker+'.npz')
                np.savez(path, **self.item())
                views.append(dict(speaker=speaker, partition=partition, file=str(path)))
            plan = dict(views=views)
            (root/'plan.json').write_text('{}')
            with patch.object(D, 'read_plan', return_value=plan), patch.dict(D.RECIPE, epochs=3, batch=1):
                D.fit(root)
            selection = json.loads((root/'selection.json').read_text())
            self.assertTrue(selection['identical_updates'])
            self.assertTrue(selection['identical_minibatches'])
            epoch = selection['selected']['epoch']
            for name, head in selection['arms'].items():
                self.assertEqual(Path(head['path']).name, f'{name}-{epoch}.npz')
                self.assertEqual(D.digest(head['path']), head['sha256'])
            a = np.load(selection['arms']['conditional']['path'])
            b = np.load(selection['arms']['ordinary']['path'])
            self.assertFalse(np.array_equal(a['weight'], b['weight']))

    def test_partition_validation_rejects_speaker_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['initial-head.npz', 'encoder-features.onnx']:
                (root/name).write_bytes(b'synthetic')
            plan = dict(recipe=D.RECIPE, initial_head_sha256=D.digest(root/'initial-head.npz'),
                feature_graph_sha256=D.digest(root/'encoder-features.onnx'), excluded_speakers=[],
                views=[dict(speaker='same', partition='train', file=str(root/'initial-head.npz'),
                            sha256=D.digest(root/'initial-head.npz')),
                       dict(speaker='same', partition='validation', file=str(root/'initial-head.npz'),
                            sha256=D.digest(root/'initial-head.npz'))])
            (root/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'partition'):
                D.read_plan(root)


class GraphTests(unittest.TestCase):
    @staticmethod
    def graph(quantized):
        import onnx
        from onnx import helper as H, numpy_helper as N, TensorProto as T
        weights = np.array([[.1, .2, -.3], [.2, -.1, .4]], dtype='float32')
        bias = np.array([.3, -.2, .1], dtype='float32')
        init = [N.from_array(bias, 'ctc_output.bias')]
        if quantized:
            init += [N.from_array(np.round(weights/.01).astype('int8'), 'ctc_weight_quantized'),
                     N.from_array(np.array(.01, dtype='float32'), 'ctc_weight_scale'),
                     N.from_array(np.array(0, dtype='int8'), 'ctc_weight_zero_point')]
            nodes = [H.make_node('DynamicQuantizeLinear', ['x'], ['qx', 'xs', 'xz']),
                     H.make_node('MatMulInteger', ['qx', 'ctc_weight_quantized', 'xz', 'ctc_weight_zero_point'],
                                 ['integer'], name='ctc_output/MatMul_quant'),
                     H.make_node('Cast', ['integer'], ['float'], to=T.FLOAT),
                     H.make_node('Mul', ['xs', 'ctc_weight_scale'], ['scale']),
                     H.make_node('Mul', ['float', 'scale'], ['projection'])]
        else:
            init.append(N.from_array(weights, 'ctc_weight'))
            nodes = [H.make_node('MatMul', ['x', 'ctc_weight'], ['projection'], name='ctc_output/MatMul')]
        nodes += [H.make_node('Add', ['projection', 'ctc_output.bias'], ['logits']),
                  H.make_node('LogSoftmax', ['logits'], ['log_probs'], axis=-1),
                  H.make_node('Identity', ['cache'], ['new_cache'])]
        return H.make_model(H.make_graph(nodes, 'synthetic',
            [H.make_tensor_value_info('x', T.FLOAT, [1, 8, 2]), H.make_tensor_value_info('cache', T.INT64, [1])],
            [H.make_tensor_value_info('log_probs', T.FLOAT, [1, 8, 3]), H.make_tensor_value_info('new_cache', T.INT64, [1])], init),
            opset_imports=[H.make_opsetid('', 17)], ir_version=10)

    def test_added_float_and_integer_heads_preserve_every_original_output(self):
        import onnx
        import onnxruntime as ort
        for quantized in [False, True]:
            graph = self.graph(quantized)
            nodes = [n.SerializeToString() for n in graph.graph.node]
            tensors = [t.SerializeToString() for t in graph.graph.initializer]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            a = ort.InferenceSession(graph.SerializeToString(), opts, providers=['CPUExecutionProvider'])
            weight = np.array([[2., -2.], [-1., 1.]], dtype='float32')
            D.add_identity_head(graph, weight, np.array([.4, -.4], dtype='float32'))
            self.assertEqual(nodes, [n.SerializeToString() for n in list(graph.graph.node)[:len(nodes)]])
            self.assertEqual(tensors, [t.SerializeToString() for t in list(graph.graph.initializer)[:len(tensors)]])
            onnx.checker.check_model(graph)
            b = ort.InferenceSession(graph.SerializeToString(), opts, providers=['CPUExecutionProvider'])
            x = np.random.default_rng(7).normal(size=(1, 8, 2)).astype('float32')
            feed = dict(x=x, cache=np.array([52], dtype='int64'))
            base = a.run(None, feed)
            changed = b.run(None, feed)
            for original, preserved in zip(base, changed):
                np.testing.assert_array_equal(original, preserved)
            if quantized:
                q, scale, zero = ort.InferenceSession(
                    self.quantizer_graph().SerializeToString(), opts, providers=['CPUExecutionProvider']).run(None, dict(x=x))
                x = (q.astype('float32')-zero)*scale
            np.testing.assert_allclose(changed[-1], x@weight+np.array([.4, -.4]), atol=1e-6)
            with self.assertRaisesRegex(ValueError, 'already'):
                D.add_identity_head(graph, weight, np.zeros(2))

    @staticmethod
    def quantizer_graph():
        from onnx import helper as H, TensorProto as T
        return H.make_model(H.make_graph([H.make_node('DynamicQuantizeLinear', ['x'], ['qx', 'xs', 'xz'])], 'quantizer',
            [H.make_tensor_value_info('x', T.FLOAT, [1, 8, 2])],
            [H.make_tensor_value_info('qx', T.UINT8, [1, 8, 2]), H.make_tensor_value_info('xs', T.FLOAT, []),
             H.make_tensor_value_info('xz', T.UINT8, [])]), opset_imports=[H.make_opsetid('', 17)], ir_version=10)

    def test_export_rejects_missing_inherited_assets_before_writing_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                D.inherited_assets(Path(directory))

    def test_export_keeps_complete_bundle_and_frozen_manifest_contract(self):
        import onnx
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, out = root/'installed', root/'training'
            model.mkdir()
            out.mkdir()
            onnx.save(self.graph(True), str(model/'model.onnx'))
            (model/'teacher-data').write_bytes(b'inherited fixture')
            manifest = dict(tokens='ab', file='model.onnx', sha256=D.digest(model/'model.onnx'),
                inventory='fixture', frameMs=40, window=45, shift=32, blank=0,
                states=[dict(name='cache', shape=[1], dtype='int64')])
            (model/'manifest.json').write_text(json.dumps(manifest))
            (out/'plan.json').write_text('{}')
            assets = {'teacher-data': D.digest(model/'teacher-data')}
            plan = dict(primary_sha256=manifest['sha256'], inherited_assets=assets)
            checkpoints = {}
            for name in ['ordinary', 'conditional']:
                path = out/(name+'.npz')
                np.savez(path, weight=np.array([[1., -1.], [.2, -.2]], dtype='float32'),
                         bias=np.array([.1, -.1], dtype='float32'))
                checkpoints[name] = dict(path=str(path), sha256=D.digest(path))
            (out/'selection.json').write_text(json.dumps(dict(plan_sha256=D.digest(out/'plan.json'), arms=checkpoints)))
            with patch.object(D, 'read_plan', return_value=plan), patch.object(D, 'inherited_assets', return_value=assets):
                # Production exporter supplies 80-band windows; the synthetic
                # graph supplies two features. Wrap only the generated input.
                import onnxruntime as ort
                factory = ort.InferenceSession
                class FixtureSession:
                    def __init__(self, *args, **kwargs):
                        self.runtime = factory(*args, **kwargs)
                    def get_outputs(self):
                        return self.runtime.get_outputs()
                    def run(self, names, feeds):
                        feeds = dict(feeds, x=feeds['x'][:, :8, :2].copy())
                        return self.runtime.run(names, feeds)
                with patch.object(ort, 'InferenceSession', FixtureSession):
                    D.export(model, out)
            for name in checkpoints:
                bundle = out/(name+'-bundle')
                exported = json.loads((bundle/'manifest.json').read_text())
                for key in ['tokens', 'inventory', 'frameMs', 'window', 'shift', 'blank', 'states']:
                    self.assertEqual(manifest[key], exported[key])
                self.assertEqual(exported['sha256'], D.digest(bundle/'model.onnx'))
                self.assertEqual(assets['teacher-data'], D.digest(bundle/'teacher-data'))


if __name__ == '__main__':
    unittest.main()

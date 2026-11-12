import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from student_pareto import (admitted_none, common_descent, ctc_contributions,
                           ctc_occupancies, group_ids, log_softmax, matched_refit,
                           speaker_partition)


def temporary_directory():
    root = Path(__file__).resolve().parents[1] / '.autolab/pareto-test-tmp'
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class CTCContributionsTest(unittest.TestCase):
    def test_forward_backward_matches_independent_path_enumeration(self):
        lp = log_softmax(np.random.default_rng(2).normal(size=(5, 3)))
        for target in ([], [1], [1, 1], [1, 2]):
            mass, occupancy = 0., np.zeros_like(lp)
            for path in itertools.product(range(3), repeat=5):
                collapsed, prev = [], -1
                for label in path:
                    if label and label != prev:
                        collapsed.append(label)
                    prev = label
                if collapsed != target:
                    continue
                p = np.exp(sum(lp[t, label] for t, label in enumerate(path)))
                mass += p
                for t, label in enumerate(path):
                    occupancy[t, label] += p
            loss, got = ctc_occupancies(lp, np.asarray(target, dtype=np.int64))
            self.assertAlmostEqual(loss, -np.log(mass), places=12)
            np.testing.assert_allclose(got, occupancy / mass, atol=1e-12)

    def test_complete_head_gradient_and_group_softmax_competition(self):
        rng = np.random.default_rng(9)
        x, w, b = rng.normal(size=(7, 2)), rng.normal(size=(3, 2)), rng.normal(size=3)
        target, groups = np.array([1, 2, 1]), np.array([2, 0, 1])
        loss, pieces = ctc_contributions(x, w, b, target, groups)
        theta = np.concatenate((w, b[:, None]), axis=1)
        numerical = np.zeros_like(theta)
        for index in np.ndindex(theta.shape):
            losses = []
            for sign in (1., -1.):
                v = theta.copy()
                v[index] += sign * 1e-5
                losses.append(ctc_contributions(x, v[:, :-1], v[:, -1], target, groups)[0])
            numerical[index] = (losses[0] - losses[1]) / 2e-5
        np.testing.assert_allclose(pieces.sum(0), numerical, atol=1e-9, rtol=1e-7)
        self.assertGreater(np.linalg.norm(pieces[0, 2]), 0.)  # consonants compete with blank
        self.assertGreater(np.linalg.norm(pieces[2, 1]), 0.)  # blank competes with consonants

    def test_matches_pytorch_original_ctc_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch oracle runs in the supervisor runtime')
        rng = np.random.default_rng(11)
        x = rng.normal(size=(9, 4))
        w, b = rng.normal(size=(3, 4)), rng.normal(size=3)
        y = np.array([1, 1, 2], dtype=np.int64)
        loss, pieces = ctc_contributions(x, w, b, y, np.array([2, 0, 1]))
        tw, tb = torch.tensor(w, requires_grad=True), torch.tensor(b, requires_grad=True)
        lp = (torch.tensor(x) @ tw.T + tb).log_softmax(-1)
        reference = torch.nn.functional.ctc_loss(lp[:, None], torch.tensor(y),
            torch.tensor([len(x)]), torch.tensor([len(y)]), reduction='sum') / len(x)
        reference.backward()
        self.assertAlmostEqual(loss, reference.item(), places=12)
        np.testing.assert_allclose(pieces.sum(0), np.concatenate((tw.grad.numpy(), tb.grad.numpy()[:, None]), 1), atol=1e-12)

    def test_invalid_and_impossible_transcripts_fail(self):
        lp = log_softmax(np.ones((2, 3)))
        for y in (np.array([0]), np.array([3]), np.array([1, 1]), np.array([1.])):
            with self.assertRaises(ValueError):
                ctc_occupancies(lp, y)
        with self.assertRaises(ValueError):
            ctc_occupancies(lp + 1, np.array([1]))

    def test_inventory_has_three_disjoint_groups(self):
        groups = group_ids('بَں۾ۜ')
        np.testing.assert_array_equal(groups, [2, 0, 1, 0, 0, 1])


class ProjectionTest(unittest.TestCase):
    def test_no_conflict_preserves_original_gradient(self):
        gradients = np.array([[1., 0.], [0., 2.], [1., 1.]])
        np.testing.assert_array_equal(common_descent(gradients), gradients.sum(0))

    def test_conflict_projects_to_nearest_boundary(self):
        gradients = np.array([[1., 0.], [-2., 1.], [0., 0.]])
        # Sum=(-1,1), first halfspace x>=0, second y>=2x.
        np.testing.assert_allclose(common_descent(gradients), [0., 1.], atol=1e-12)

    def test_dependent_and_opposing_constraints(self):
        for gradients in (np.array([[1., 0.], [-2., 0.], [0., 3.]]),
                          np.array([[1., 0.], [-2., 0.], [-4., 0.]])):
            direction = common_descent(gradients)
            np.testing.assert_allclose(direction, [0., gradients.sum(0)[1]], atol=1e-12)
        np.testing.assert_array_equal(common_descent(np.zeros((3, 2))), [0., 0.])

    def test_random_projections_match_independent_quadratic_solver(self):
        try:
            from scipy.optimize import minimize
        except ImportError:
            self.skipTest('Independent QP oracle needs scipy')
        rng = np.random.default_rng(42)
        for _ in range(30):
            gradients = rng.normal(size=(3, 6))
            total = gradients.sum(0)
            result = minimize(lambda v: .5 * np.sum((v - total) ** 2), np.zeros(6),
                jac=lambda v: v - total, constraints=[dict(type='ineq', fun=lambda v: gradients @ v,
                jac=lambda v: gradients)], method='SLSQP', options=dict(ftol=1e-12, maxiter=100))
            self.assertTrue(result.success)
            direction = common_descent(gradients)
            np.testing.assert_allclose(direction, result.x, atol=1e-7, rtol=1e-7)
            self.assertGreaterEqual(float((gradients @ direction).min()), -1e-10)


class ProvenanceAndControlTest(unittest.TestCase):
    def test_nontraining_speaker_excluded_before_enrollment(self):
        good = dict(id='take', speaker='a', split='train', kind='none', scored=True,
                    wav='take.wav', ayahs=['1:1'], events=[], non_errors=[], unsupported=[])
        self.assertEqual(admitted_none([good]), [good])
        self.assertEqual(admitted_none([good, dict(good, split='eval')]), [])
        self.assertEqual(admitted_none([dict(good, events=[{}])]), [])
        self.assertEqual(admitted_none([dict(good, non_errors=[{}])]), [])
        with self.assertRaises(ValueError):
            admitted_none([good, good])

    def test_speaker_group_is_independent_of_recording_order_and_count(self):
        speakers = ['a', 'b', 'c', 'd', 'e', 'f']
        self.assertEqual(speaker_partition(speakers), speaker_partition(speakers[::-1] + ['a']))

    def test_additional_corpus_holdouts_exclude_none_speakers_globally(self):
        from student_pareto_train import recordings
        with temporary_directory() as tmp:
            root = Path(tmp)
            none = dict(id='none', split='train', speaker='held', kind='none', scored=True,
                        events=[], non_errors=[], unsupported=[], wav='missing.wav', ayahs=['toy:ayah'])
            (root / 'labels.jsonl').write_text(json.dumps(none))
            (root / 'transcribed.jsonl').write_text(json.dumps(dict(speaker='held', split='release')))
            with self.assertRaisesRegex(ValueError, 'Need training-speaker none'):
                recordings(root / 'labels.jsonl', root / 'transcribed.jsonl', [],
                           dict(ayahs=[dict(s='toy', a='ayah', p='بَ')]))

    def test_audio_enrollment_rejects_escape_and_unverified_transcripts(self):
        from student_pareto_train import contained, recordings
        with temporary_directory() as tmp:
            root = Path(tmp)
            corpus = root / 'corpus'
            corpus.mkdir()
            (root / 'outside.wav').write_bytes(b'outside')
            with self.assertRaises(ValueError):
                contained(corpus, '../outside.wav')
            (corpus / 'none.wav').write_bytes(b'none')
            none = dict(id='none', split='train', speaker='fit', kind='none', scored=True,
                        events=[], non_errors=[], unsupported=[], wav='none.wav', ayahs=['toy:ayah'])
            (corpus / 'labels.jsonl').write_text(json.dumps(none))
            (corpus / 'transcribed.jsonl').write_text(json.dumps(dict(speaker='other', split='train',
                authorized=True, verified=False, phones='بَ', wav='other.wav')))
            with self.assertRaisesRegex(ValueError, 'authorized performed transcript'):
                recordings(corpus / 'labels.jsonl', corpus / 'transcribed.jsonl', [],
                           dict(ayahs=[dict(s='toy', a='ayah', p='بَ')]))

    def test_control_has_identical_data_update_count_and_shared_selection_epoch(self):
        groups = {'a': 'fit', 'b': 'fit', 'c': 'selection'}
        rows = [dict(speaker=s, views=[dict(features=np.array([[i]]), target=np.array([1]))])
                for i, s in enumerate(groups)]
        seen = []
        def gradient(x, w, b, y, ids):
            seen.append(float(x[0, 0]))
            return 1., np.ones((3, 2, 2))
        def validate(head, selection):
            self.assertEqual([r['speaker'] for r in selection], ['c'])
            return dict(objective=float(head[0].sum()))
        with patch('student_pareto.ctc_contributions', side_effect=gradient):
            heads, report = matched_refit((np.zeros((2, 1)), np.zeros(2)), rows,
                                         groups, 'ب', validate, 2, .01, 8)
        self.assertEqual(seen[:2], seen[2:4])
        self.assertEqual(seen[4:6], seen[6:8])
        self.assertEqual(report['selectedEpoch'], 2)
        self.assertEqual(report['selectedUpdates'], 4)
        np.testing.assert_allclose(heads['ordinary'][0], heads['pareto'][0])


class StreamingExportTest(unittest.TestCase):
    def test_quantized_native_head_extraction_and_replacement(self):
        try:
            import onnx
            from onnx import helper as h, numpy_helper as nh
            from onnx.reference import ReferenceEvaluator
        except ImportError:
            self.skipTest('ONNX graph tests run in the supervisor runtime')
        from student_pareto_export import head_parameters, replace_head
        weight = np.array([[.2, -.1, .3], [.4, .2, -.2]], dtype=np.float32)
        bias = np.array([.1, .2, -.1], dtype=np.float32)
        nodes = [h.make_node('Identity', ['x'], ['feature'], name='encoder'),
                 h.make_node('Add', ['state', 'one'], ['new_state'], name='state_update'),
                 h.make_node('DynamicQuantizeLinear', ['feature'], ['q_x', 'x_scale', 'x_zero']),
                 h.make_node('MatMulInteger', ['q_x', 'weight_quantized', 'x_zero', 'weight_zero_point'],
                             ['raw'], name='ctc_output/MatMulInteger'),
                 h.make_node('Cast', ['raw'], ['cast'], to=onnx.TensorProto.FLOAT),
                 h.make_node('Mul', ['x_scale', 'weight_scale'], ['scale']),
                 h.make_node('Mul', ['cast', 'scale'], ['linear']),
                 h.make_node('Add', ['linear', 'bias'], ['logits']),
                 h.make_node('LogSoftmax', ['logits'], ['log_probs'], axis=-1)]
        tensors = [nh.from_array(np.ones(1, np.float32), 'one'), nh.from_array(bias, 'bias'),
                   nh.from_array(np.rint(weight / .1).astype(np.int8), 'weight_quantized'),
                   nh.from_array(np.array(.1, np.float32), 'weight_scale'),
                   nh.from_array(np.array(0, np.int8), 'weight_zero_point')]
        graph = h.make_graph(nodes, 'quantized', [h.make_tensor_value_info('x', 1, [1, 2, 2]),
            h.make_tensor_value_info('state', 1, [1])], [h.make_tensor_value_info('log_probs', 1, [1, 2, 3]),
            h.make_tensor_value_info('new_state', 1, [1])], tensors,
            value_info=[h.make_tensor_value_info('feature', 1, [1, 2, 2])])
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)])
        w, b = head_parameters(model)
        np.testing.assert_allclose(w, weight.T, atol=1e-7)
        np.testing.assert_array_equal(b, bias)
        feeds = dict(x=np.array([[[1., 2.], [.2, -.1]]], np.float32), state=np.array([5.], np.float32))
        before = ReferenceEvaluator(model).run(None, feeds)
        after_model = replace_head(model, w + .1, b + .2)
        after = ReferenceEvaluator(after_model).run(None, feeds)
        want = log_softmax((feeds['x'] @ (w + .1).T + b + .2).reshape(2, 3))
        np.testing.assert_allclose(after[0][0], want, atol=1e-6)
        np.testing.assert_array_equal(after[1], before[1])
        self.assertTrue(any(n.name == 'encoder' for n in after_model.graph.node))
        self.assertFalse(any('ctc_output' in n.name for n in after_model.graph.node))

    def test_caching_drains_valid_frames_and_preserves_carried_state(self):
        from student_pareto_export import stream
        class Runtime:
            def __init__(self):
                self.calls = []
            def run(self, names, feeds):
                self.calls.append({k: v.copy() for k, v in feeds.items()})
                offset = int(feeds['cache'][0])
                values = np.arange(offset, offset + 8, dtype=np.float32)
                return [np.tile(values[None, :, None], (1, 1, 3)),
                        np.array([offset + 8], dtype=np.int64), values[None, :, None]]
        manifest = dict(fbank=dict(mels=80, padValue=-23.0259), window=45, shift=32,
                        states=[dict(name='cache', dtype='int64', shape=[1])])
        runtime = Runtime()
        lp, features = stream(runtime, manifest, np.ones((80, 80), dtype=np.float32), 'features')
        self.assertEqual(len(lp), 18)  # physical valid rows, not three padded 8-row chunks
        np.testing.assert_array_equal(features[:, 0], np.arange(18))
        np.testing.assert_array_equal(lp[:, 0], np.arange(18))
        self.assertEqual(len(runtime.calls), 3)
        np.testing.assert_array_equal(runtime.calls[2]['cache'], [16])
        np.testing.assert_array_equal(runtime.calls[2]['x'][0, :16], np.ones((16, 80)))
        np.testing.assert_allclose(runtime.calls[2]['x'][0, 16:], manifest['fbank']['padValue'])

    def test_native_graft_keeps_states_auxiliary_assets_and_head_dimensions(self):
        try:
            import onnx
            from onnx import helper, numpy_helper, TensorProto
        except ImportError:
            self.skipTest('ONNX graph tests run in the supervisor runtime')
        from student_pareto_export import digest, expose, graft, head_parameters, insertion_assets, load_source
        with temporary_directory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            weight = np.arange(240, dtype=np.float32).reshape(80, 3) / 1000
            bias = np.array([.1, .2, .3], dtype=np.float32)
            nodes = [helper.make_node('Identity', ['x'], ['features']),
                     helper.make_node('MatMul', ['features', 'w'], ['linear'], name='ctc_output/MatMul'),
                     helper.make_node('Add', ['linear', 'b'], ['logits']),
                     helper.make_node('LogSoftmax', ['logits'], ['log_probs'], axis=-1),
                     helper.make_node('Identity', ['cache'], ['new_cache'])]
            graph = helper.make_graph(nodes, 'native',
                [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 80]),
                 helper.make_tensor_value_info('cache', TensorProto.INT64, [1])],
                [helper.make_tensor_value_info('log_probs', TensorProto.FLOAT, [1, 2, 3]),
                 helper.make_tensor_value_info('new_cache', TensorProto.INT64, [1])],
                initializer=[numpy_helper.from_array(weight, 'w'), numpy_helper.from_array(bias, 'b')],
                value_info=[helper.make_tensor_value_info('features', TensorProto.FLOAT, [1, 2, 80])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
            onnx.save(model, source / 'model.onnx')
            teacher = helper.make_model(helper.make_graph(
                [helper.make_node('Identity', ['weights'], ['out'])], 'aux', [],
                [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1])],
                initializer=[numpy_helper.from_array(np.ones(1, dtype=np.float32), 'weights')]))
            onnx.save_model(teacher, source / 'teacher.onnx', save_as_external_data=True,
                            all_tensors_to_one_file=True, location='teacher.onnx.data', size_threshold=0)
            manifest = dict(id='toy', file='model.onnx', sha256=digest(source / 'model.onnx'),
                blank=0, frameMs=40, sampleRate=16000, tokens='بَ', fbank=dict(mels=80))
            (source / 'manifest.json').write_text(json.dumps(manifest))
            _, native, _, _ = load_source(source)
            got_w, got_b = head_parameters(native)
            np.testing.assert_array_equal(got_w, weight.T)
            np.testing.assert_array_equal(got_b, bias)
            # ONNX's reference evaluator is independent of the graft implementation.
            from onnx.reference import ReferenceEvaluator
            x = np.random.default_rng(2).normal(size=(1, 2, 80)).astype(np.float32)
            feeds = dict(x=x, cache=np.array([17], dtype=np.int64))
            _, _, _, exposed_path = expose(source, root / 'features')
            before = ReferenceEvaluator(native).run(None, feeds)
            exposed = ReferenceEvaluator(str(exposed_path)).run(None, feeds)
            for a, b in zip(before, exposed[:2]):
                np.testing.assert_array_equal(a, b)
            path = graft(source, (weight.T * .9, bias), root / 'grafted', 'pareto')
            after = ReferenceEvaluator(str(path)).run(None, feeds)
            np.testing.assert_array_equal(after[1], before[1])
            want = log_softmax((x @ (weight * .9) + bias).reshape(2, 3))
            np.testing.assert_allclose(after[0][0], want, atol=1e-6)
            self.assertEqual(insertion_assets(root / 'grafted'), insertion_assets(source))
            exported = json.loads((root / 'grafted/manifest.json').read_text())
            for key in ('tokens', 'blank', 'frameMs', 'sampleRate', 'fbank'):
                self.assertEqual(manifest[key], exported[key])
            self.assertEqual(exported['sha256'], digest(path))
            (source / 'teacher.onnx.data').write_bytes(b'')
            with self.assertRaises(ValueError):
                insertion_assets(source)


if __name__ == '__main__':
    unittest.main()

"""Small CPU tests of actual exported rounding, acceptance and streaming parity."""
import copy
import json
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import torch

from student_quantized_head import QuantizedHead, digest, predict, quantize_control, runtime, stream_chunks
from student_export_effective import (actual_loss, discrete_step, eligible, losses,
                                      fit, proposals, speaker_partition, verify_product)


def model_fixture(directory):
    width, vocab = 3, 3
    tensor = lambda name, dims: helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
    feature = 'encoder/features'
    weights = np.array([[1., -.4, .25], [.17, -1., .38], [-.63, .14, .71]], dtype=np.float32)
    nodes = [helper.make_node('Add', ['x', 'cache'], [feature], name='encoder/features'),
             helper.make_node('Add', ['cache', 'advance'], ['new_cache'], name='encoder/state'),
             helper.make_node('MatMul', [feature, 'weight'], ['linear'], name='ctc_output/MatMul'),
             helper.make_node('Add', ['linear', 'bias'], ['logits'], name='ctc_output/Add'),
             helper.make_node('LogSoftmax', ['logits'], ['log_probs'], axis=-1, name='ctc_output/LogSoftmax')]
    graph = helper.make_graph(nodes, 'fixture', [tensor('x', [1, None, width]), tensor('cache', [1, 1, width])],
        [tensor('log_probs', [1, None, vocab]), tensor('new_cache', [1, 1, width])],
        [numpy_helper.from_array(weights, 'weight'),
         numpy_helper.from_array(np.array([.1, -.2, .3], np.float32), 'bias'),
         numpy_helper.from_array(np.full((1, 1, width), .03, np.float32), 'advance')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 10
    return quantize_control(model, directory)


def recording(head, seed=0):
    rng = np.random.default_rng(seed)
    chunks = [(rng.normal(0, .1, (1, 4, 3)).astype('float32'), 4),
              (rng.normal(0, 2., (1, 4, 3)).astype('float32'), 2)]
    anchor = predict(runtime(head.graph), head.feature, chunks)
    return dict(speaker='eligible-speaker', target=torch.tensor([1, 2]),
                chunks=[(torch.from_numpy(x), keep) for x, keep in chunks], anchor=torch.from_numpy(anchor))


class ExportEffectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.model = model_fixture(cls.directory / 'fixture')
        cls.head = QuantizedHead(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_extracted_and_grafted_head_preserve_primary_and_carried_states(self):
        head = self.head
        before = runtime(head.source)
        after = runtime(head.graft(head.graph))
        isolated = runtime(head.graph)
        exposed = runtime(head.exposed())
        rng = np.random.default_rng(7)
        state = np.zeros((1, 1, 3), dtype=np.float32)
        for scale in (.05, 20., .7):
            feeds = {'x': rng.normal(0, scale, (1, 4, 3)).astype('float32'), 'cache': state}
            a, b = before.run(None, feeds), after.run(None, feeds)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
            features = exposed.run([head.feature], feeds)[0]
            np.testing.assert_array_equal(a[0], isolated.run(['log_probs'], {head.feature: features})[0])
            state = a[1]

    def test_dynamic_activation_ranges_are_kept_per_full_chunk(self):
        row = recording(self.head)
        chunks = [(x.numpy(), keep) for x, keep in row['chunks']]
        exact = predict(runtime(self.head.graph), self.head.feature, chunks)
        joined = np.concatenate([x for x, _ in chunks], axis=1)
        wrong = runtime(self.head.graph).run(['log_probs'], {self.head.feature: joined})[0][0]
        wrong = np.concatenate([wrong[:4], wrong[4:6]])
        self.assertGreater(float(abs(exact - wrong).max()), 1e-4)
        # Removing invalid drain rows also changes the final chunk scale.
        cropped = chunks[-1][0][:, :2]
        full = runtime(self.head.graph).run(['log_probs'], {self.head.feature: chunks[-1][0]})[0][:, :2]
        shortened = runtime(self.head.graph).run(['log_probs'], {self.head.feature: cropped})[0]
        self.assertGreater(float(abs(full - shortened).max()), 1e-4)

    def test_small_float_update_disappears_in_real_export(self):
        head = self.head
        baseline = head.floating(head.weights())
        changed = head.weights().copy()
        changed[0, 1] += float(head.scale.item()) * .05
        changed_graph = head.floating(changed)
        a = quantize_control(baseline, self.directory / 'baseline')
        b = quantize_control(changed_graph, self.directory / 'changed')
        tensors = lambda g: {i.name: numpy_helper.to_array(i) for i in g.graph.initializer}
        for name, value in tensors(a).items():
            np.testing.assert_array_equal(value, tensors(b)[name])
        row = recording(head)
        chunks = [(x.numpy(), keep) for x, keep in row['chunks']]
        floating_change = abs(predict(runtime(baseline), head.feature, chunks) -
                              predict(runtime(changed_graph), head.feature, chunks)).max()
        self.assertGreater(float(floating_change), 0.)
        np.testing.assert_array_equal(predict(runtime(a), head.feature, chunks),
                                      predict(runtime(b), head.feature, chunks))

    def test_coordinate_bounds_and_signed_saturation(self):
        codes = np.array([[127, -128, 1, -2]], np.int8)
        original = np.array([[127, -128, 0, 0]], np.int8)
        gradient = np.array([[-10., 10., -1., 1.]])
        self.assertEqual(proposals(codes, original, gradient, .1, 8, 2), [((0, 2), 2)])
        np.testing.assert_array_equal(codes, [[127, -128, 1, -2]])

    def test_accepted_moves_improve_actual_recording_objective(self):
        head = self.head
        rows = [recording(head, i) for i in range(2)]
        codes = head.initial_codes.copy()
        accepted_count = 0
        for _ in range(4):
            before = actual_loss(head, head.with_codes(codes), rows)
            updated, score, accepted = discrete_step(head, codes, rows, count=codes.size)
            actual = actual_loss(head, head.with_codes(updated), rows)
            self.assertAlmostEqual(score, actual, places=7)
            self.assertLessEqual(actual, before)
            self.assertLessEqual(int(abs(updated.astype(int) - head.initial_codes.astype(int)).max()), 2)
            self.assertEqual(int(np.count_nonzero(updated != codes)), int(accepted))
            if accepted:
                self.assertLess(actual, before)
                accepted_count += 1
            codes = updated
        self.assertGreater(accepted_count, 0)

    def test_surrogate_direction_cannot_accept_an_actual_loss(self):
        head = self.head
        codes = head.initial_codes.copy()
        # Proposals still come from transcript gradients; an actual evaluator
        # that finds no improvement must reject every suggested direction.
        with patch('student_export_effective.actual_loss', return_value=3.):
            got, score, accepted = discrete_step(head, codes, [recording(head)], count=codes.size)
        self.assertFalse(accepted)
        self.assertEqual(score, 3.)
        np.testing.assert_array_equal(got, codes)

    def test_code_edit_changes_no_encoder_or_bias_parameter(self):
        codes = self.head.initial_codes.copy()
        codes[0, 0] -= 1
        grafted = self.head.graft(self.head.with_codes(codes))
        before = {i.name: i.SerializeToString() for i in self.model.graph.initializer}
        after = {i.name: i.SerializeToString() for i in grafted.graph.initializer}
        self.assertEqual([name for name in before if before[name] != after[name]], [self.head.weight_name])
        self.assertEqual(after.keys(), before.keys())
        original_nodes = [n.SerializeToString() for n in self.model.graph.node if not n.name.startswith('ctc_output')
                          and n.op_type != 'DynamicQuantizeLinear']
        for node in original_nodes:
            self.assertIn(node, [n.SerializeToString() for n in grafted.graph.node])

    def test_speaker_exclusion_precedes_enrollment(self):
        def entry(speaker):
            return dict(speaker=speaker, source='help-train-none', held_out=False, split='train',
                        transcript_kind='performed', phones='ab', audio='take.wav', sha256='known',
                        label='none', events=[], non_errors=[], unsupported=[])
        train = entry('ordinary')
        held = entry('release')
        duplicate = dict(held, split='test')
        self.assertEqual(eligible([train, held, duplicate], []), [train])
        self.assertEqual(eligible([train, held], ['release']), [train])
        for field, value in [('label', 'wrong_word'), ('non_errors', ['repeat']), ('events', ['addition'])]:
            with self.assertRaises(ValueError):
                eligible([dict(train, **{field: value})], [])
        groups = speaker_partition([entry(str(i)) for i in range(10)])
        self.assertEqual(set(groups.values()), {'fit', 'selection'})
        self.assertEqual(groups, speaker_partition([entry(str(i)) for i in reversed(range(10))]))

    def test_transcript_loss_repeated_labels_and_pause_preservation(self):
        row = recording(self.head)
        row['target'] = torch.tensor([1, 1])
        lp = row['anchor'].clone().requires_grad_()
        ctc, kl = losses(lp, row)
        expected = torch.nn.functional.ctc_loss(lp[:, None], row['target'], [len(lp)], [2],
                                                reduction='sum') / 2
        torch.testing.assert_close(ctc, expected)
        self.assertAlmostEqual(float(kl.detach()), 0., places=6)
        (ctc + kl).backward()
        self.assertTrue(torch.isfinite(lp.grad).all())

    def test_nonquantized_head_fails_closed(self):
        with self.assertRaises(ValueError):
            QuantizedHead(self.head.floating(self.head.weights()))

    def test_control_and_discrete_export_selected_heads_and_complete_bundles(self):
        source, cache = self.directory / 'source', self.directory / 'cache'
        source.mkdir()
        cache.mkdir()
        onnx.save(self.model, source / 'model.onnx')
        onnx.save(self.model, cache / 'source.onnx')
        # Synthetic teacher assets test bundle identity only; no listening claim.
        onnx.save(self.head.graph, source / 'teacher.onnx')
        (source / 'teacher.onnx.data').write_bytes(b'fixture-external-data')
        manifest = dict(id='fixture', file='model.onnx', sha256=digest(source / 'model.onnx'), quantizedFrom='old-source',
                        states=[dict(name='cache', shape=[1, 1, 3], dtype='float32')])
        (source / 'manifest.json').write_text(json.dumps(manifest))
        rows = [recording(self.head, i) for i in range(10)]
        for i, row in enumerate(rows):
            row['speaker'] = str(i)
            row['windows'] = [row['chunks'][0][0].clone(), row['chunks'][1][0].clone() - .03]
        groups = speaker_partition(rows)
        for row in rows:
            row['partition'] = groups[row['speaker']]
        torch.save(rows, cache / 'recordings.pt')
        metadata = dict(source=str(source), manifest=manifest, partitions=groups, recordings=len(rows),
                        cache_sha256=digest(cache / 'recordings.pt'), elapsed_s=0., started_unix=time.time(),
                        source_onnx_sha256=digest(cache / 'source.onnx'))
        (cache / 'cache.json').write_text(json.dumps(metadata))
        for discrete, arm in [(False, 'control-int8'), (True, 'discrete')]:
            out = self.directory / arm
            fit(Namespace(cache=str(cache), out=str(out), steps=2, lr=.001, proposals=8, bound=2), discrete)
            result = json.loads((out / 'refit.json').read_text())
            self.assertEqual(result['partitions'], groups)
            self.assertFalse(result['selection_uses_practice'])
            product = out / arm
            exported = json.loads((product / 'manifest.json').read_text())
            self.assertEqual(exported['sha256'], digest(product / 'model.onnx'))
            self.assertEqual(exported['derivedFrom'], manifest['sha256'])
            self.assertNotIn('quantizedFrom', exported)
            for filename in ('teacher.onnx', 'teacher.onnx.data'):
                self.assertEqual(digest(source / filename), digest(product / filename))
            self.assertEqual(result['bias_frozen'], True)
            if discrete:
                codes = np.load(out / 'selected-codes.npy')
                self.assertLessEqual(int(abs(codes.astype(int) - self.head.initial_codes.astype(int)).max()), 2)
            else:
                weights = np.load(out / 'selected-weights.npy')
                expected = self.head.graft(self.head.floating(weights))
                actual = onnx.load(out / 'control-fp32/model.onnx')
                self.assertEqual(actual.SerializeToString(), expected.SerializeToString())

    def test_stream_drain_keeps_full_padded_chunk_and_native_clock(self):
        source = copy.deepcopy(self.model)
        # A tiny frontend takes four regularly spaced rows from each SDK window.
        # Its arithmetic is not a Zipformer oracle; it tests cache/drain plumbing.
        source.graph.input[0].CopyFrom(helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 29, 80]))
        first = next(n for n in source.graph.node if n.name == 'encoder/features')
        self.assertEqual(first.op_type, 'Add')
        first.input[0] = 'encoder/subsampled'
        source.graph.node.insert(0, helper.make_node('Slice', ['x', 'starts', 'ends', 'axes', 'strides'],
                                                   ['encoder/subsampled'], name='encoder/subsample'))
        for name, value in [('starts', [0, 0, 0]), ('ends', [1, 8, 3]),
                            ('axes', [0, 1, 2]), ('strides', [1, 2, 1])]:
            source.graph.initializer.append(numpy_helper.from_array(np.array(value, np.int64), name))
        del source.graph.value_info[:]
        head = QuantizedHead(source)
        manifest = dict(chunk=8, window=29, shift=16,
                        fbank=dict(padValue=-23.0259),
                        states=[dict(name='cache', shape=[1, 1, 3], dtype='float32')])
        fbank = np.random.default_rng(11).normal(size=(26, 80)).astype('float32')
        chunks, anchor = stream_chunks(head.exposed(), manifest, fbank, head.feature)
        self.assertEqual([keep for _, keep in chunks], [4, 1])
        self.assertEqual([x.shape for x, _ in chunks], [(1, 4, 3), (1, 4, 3)])
        self.assertEqual(len(anchor), 5)
        windows = []
        for j in range(2):
            window = np.full((1, 29, 80), -23.0259, dtype=np.float32)
            view = fbank[j * 16:j * 16 + 29]
            window[0, :len(view)] = view
            windows.append(torch.from_numpy(window))
        row = dict(chunks=[(torch.from_numpy(x), keep) for x, keep in chunks], windows=windows)
        verify_product(head, head.graph, manifest, [row])
        codes = head.initial_codes.copy()
        codes[1, 0] -= 1
        verify_product(head, head.with_codes(codes), manifest, [row])
        # An accidental timestamp/frame shift fails product parity.
        broken = copy.deepcopy(row)
        broken['chunks'][0] = (broken['chunks'][0][0] + .3, 4)
        with self.assertRaises(AssertionError):
            verify_product(head, head.graph, manifest, [broken])


if __name__ == '__main__':
    unittest.main()

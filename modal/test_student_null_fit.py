"""Training partition, preservation, CPU budget and actual Zipformer replay tests."""

import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from student_null_fit import (Budget, acoustic_loss, cache_features, file_hash, fit, frozen_hash,
                              group_partition, help_records, replay)
from student_null_attention import install_query_attention


class TrainingContractTests(unittest.TestCase):
    def test_partition_is_grouped_deterministic_and_order_independent(self):
        records = [{'speaker': f'speaker-{s}'} for s in range(10) for _ in range(3)]
        a = group_partition(records)
        b = group_partition(list(reversed(records)))
        self.assertEqual(a, list(reversed(b)))
        groups = {}
        for r, split in zip(records, a):
            groups.setdefault(r['speaker'], set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(set(a), {'fit', 'validation'})
        with self.assertRaisesRegex(ValueError, 'three'):
            group_partition([{'speaker': 'only'}])

    def test_none_admission_excludes_scripts_repeats_and_entire_held_speaker(self):
        reference = {'ayahs': [{'s': 1, 'a': 1, 'p': 'بت'}]}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'audio.wav').write_bytes(b'synthetic test audio placeholder')
            ordinary = dict(id='ordinary', speaker='train-speaker', split='train', kind='none', scored=True,
                            wav='audio.wav', ayahs=['1:1'], events=[], non_errors=[], unsupported=[])
            rows = [ordinary,
                    {**ordinary, 'id':'mistake', 'kind':'skipped_word'},
                    {**ordinary, 'id':'event', 'events':[{'class':'omitted_word'}]},
                    {**ordinary, 'id':'repeat', 'non_errors':[{'class':'repeat'}]},
                    {**ordinary, 'id':'unscored', 'scored':False},
                    {**ordinary, 'id':'excluded', 'excluded':True},
                    {**ordinary, 'id':'held-train', 'speaker':'held'},
                    {**ordinary, 'id':'held-eval', 'speaker':'held', 'split':'eval'}]
            (root / 'labels.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            admitted = help_records(root, reference)
            self.assertEqual([r['key'] for r in admitted], ['help:ordinary'])
            self.assertEqual(admitted[0]['target'], 'بت')
            (root / 'labels.jsonl').write_text(json.dumps({**ordinary, 'wav':'../escape.wav'}))
            with self.assertRaisesRegex(ValueError, 'inside'):
                help_records(root, reference)

    def test_complete_recording_ctc_and_preservation_are_finite(self):
        logits = torch.randn(1, 8, 5, requires_grad=True)
        lp = logits.log_softmax(-1)
        record = {'target':[1, 2], 'baseline':lp.detach().clone()}
        ctc, kl = acoustic_loss(lp, record)
        self.assertTrue(torch.isfinite(ctc))
        self.assertLess(abs(float(kl.detach())), 1e-6)
        (ctc + kl).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        altered = record['baseline'] + torch.randn_like(lp) * .1
        record['baseline'] = altered.log_softmax(-1)
        _, kl = acoustic_loss(lp.detach(), record)
        self.assertGreater(float(kl), 0)

    def test_cpu_budget_persists_and_cannot_reset_by_resuming(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch('student_null_fit.time.process_time', side_effect=[10.0, 12.5]):
                Budget(root).check()
            self.assertEqual(json.loads((root / 'cpu-budget.json').read_text())['cpu_seconds'], 2.5)
            with patch('student_null_fit.time.process_time', side_effect=[100.0, 104.0]):
                Budget(root).check()
            self.assertEqual(json.loads((root / 'cpu-budget.json').read_text())['cpu_seconds'], 6.5)
            with patch('student_null_fit.time.process_time', side_effect=[0.0, 86400.0]):
                with self.assertRaisesRegex(RuntimeError, 'CPU day'):
                    Budget(root).check()


@unittest.skipUnless(importlib.util.find_spec('zipformer'), 'requires pinned Zipformer in supervisor train image')
class ZipformerReplayTests(unittest.TestCase):
    def setUp(self):
        from zipformer import Zipformer2
        from subsampling import Conv2dSubsampling
        torch.manual_seed(17)
        torch.set_num_threads(1)
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_embed = Conv2dSubsampling(80, 16, dropout=0.0)
                self.encoder = Zipformer2(output_downsampling_factor=2,
                    downsampling_factor=(1, 2, 2), num_encoder_layers=(1, 1, 2), encoder_dim=(16, 24, 16),
                    encoder_unmasked_dim=(16, 24, 16), query_head_dim=4, value_head_dim=2,
                    pos_head_dim=2, pos_dim=8, num_heads=2, feedforward_dim=32, cnn_module_kernel=3,
                    dropout=0.0, causal=True, chunk_size=(8,), left_context_frames=(16,))
                self.ctc_output = nn.Linear(24, 5)
            def streaming_eval(self, chunk, left):
                self.encoder.chunk_size = (chunk,)
                self.encoder.left_context_frames = (left,)
        self.base = Tiny().eval().requires_grad_(False)
        self.feats = torch.randn(93, 80)
        self.manifest = {'chunk':8, 'leftContext':16, 'window':29, 'shift':16, 'fbank':{'padValue':-23.0259}}

    def full_stream(self, model):
        # Load the production wrapper without constructing Modal resources.
        source = ast.parse((Path(__file__).parent / 'student_export.py').read_text())
        definition = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'wrap')
        namespace = {'PAD': 13}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), 'student_export.py', 'exec'), namespace)
        wrap = namespace['wrap']
        from torch.nn import functional as F
        s = wrap(model, 8, 16)
        states = s.init_states()
        need = ((len(self.feats) - 7) // 2 + 1) // 2
        windows = -(-need // 4)
        total = 29 + (windows - 1) * 16
        padded = F.pad(self.feats, (0, 0, 0, total - len(self.feats)), value=-23.0259)
        outs = []
        for start in range(0, total - 29 + 1, 16):
            lp, states = s(padded[None, start:start+29], states)
            outs.append(lp)
        return torch.cat(outs, 1)[:, :need]

    def test_cached_prefix_matches_complete_stream_with_adapted_upper_state(self):
        with torch.no_grad():
            cached = cache_features(self.base, self.feats, self.manifest)
            torch.testing.assert_close(cached['baseline'], self.full_stream(self.base), atol=1e-6, rtol=1e-5)
        for mode in ('null', 'temperature'):
            model = copy.deepcopy(self.base)
            adapter = install_query_attention(model, mode)
            adapter.score_weight.data.normal_()
            adapter.score_bias.data.zero_()
            got = replay(model, cached, self.manifest)
            expected = self.full_stream(model)
            torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
            got.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in (adapter.score_weight, adapter.score_bias)))
            self.assertEqual(frozen_hash(model), frozen_hash(self.base))

    def test_temperature_identity_matches_complete_unchanged_stream(self):
        original = self.full_stream(self.base)
        temperature = copy.deepcopy(self.base)
        install_query_attention(temperature, 'temperature')
        torch.testing.assert_close(original, self.full_stream(temperature), atol=2e-6, rtol=1e-5)

    def test_whole_primary_streaming_onnx_export_keeps_clock_and_state_shapes(self):
        import numpy as np
        import onnxruntime as ort
        source = ast.parse((Path(__file__).parent / 'student_export.py').read_text())
        functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ('wrap', 'export_onnx', 'stream_onnx')]
        namespace = {'PAD':13, 'LOG_EPS':-23.0259}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'student_export.py', 'exec'), namespace)
        base_shapes = [list(s.shape) for s in namespace['wrap'](self.base, 8, 16).init_states()]
        for mode in ('null', 'temperature'):
            model = copy.deepcopy(self.base)
            adapter = install_query_attention(model, mode)
            adapter.score_weight.data.normal_()
            adapter.score_bias.data.zero_()
            with tempfile.TemporaryDirectory() as d:
                path = str(Path(d) / 'model.onnx')
                states = namespace['export_onnx'](model, path, 8, 16)
                self.assertEqual([s['shape'] for s in states], base_shapes)
                got = namespace['stream_onnx'](path, self.feats, states, 8)
                expected = self.full_stream(model).detach().numpy()[0]
                self.assertEqual(got.shape[1], expected.shape[1])
                np.testing.assert_allclose(got[:len(expected)], expected, atol=2e-5, rtol=2e-5)

    def test_matched_fit_selection_resumption_and_frozen_checkpoint(self):
        from types import SimpleNamespace
        from student_null_fit import atomic_json, save_torch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = root / 'base.pt'
            checkpoint.write_bytes(b'synthetic initial checkpoint provenance')
            with torch.no_grad():
                cached = cache_features(self.base, self.feats, self.manifest)
            cached['target'] = [1, 2]
            (root / 'cache').mkdir()
            index = []
            for key, speaker, partition in [('fit', 'speaker-fit', 'fit'), ('val', 'speaker-val', 'validation')]:
                for view in (0, 1):
                    file = f'{key}-{view}.pt'
                    save_torch(root / 'cache' / file, cached)
                    index.append({'key':key, 'speaker':speaker, 'view':view, 'file':file, 'partition':partition})
            atomic_json(root / 'design.json', {'checkpoint_sha256':file_hash(checkpoint), 'geometry':self.manifest,
                'seed':20261112, 'epochs':2, 'preservation':[1.0],
                'data':[{'key':'fit', 'speaker':'speaker-fit', 'partition':'fit'},
                        {'key':'val', 'speaker':'speaker-val', 'partition':'validation'}]})
            atomic_json(root / 'cache-index.json', index)
            args = SimpleNamespace(output=root, checkpoint=checkpoint)
            ck = {'cfg':{'size':'tiny'}, 'model':self.base.state_dict()}
            with patch('student_null_fit.PRESERVATION', (1.0,)), patch('student_null_fit.EPOCHS', 2):
                fit(args, self.base, ck, self.manifest, Budget(root))
                for mode in ('null', 'temperature'):
                    selected = torch.load(root / f'{mode}-selected.pt', weights_only=False)
                    self.assertEqual(selected['selection']['updates'], 2 * selected['selection']['epoch'])
                    self.assertEqual(selected['cfg']['query_attention'], mode)
                    loaded = copy.deepcopy(self.base)
                    install_query_attention(loaded, mode)
                    loaded.load_state_dict(selected['model'], strict=True)
                    self.assertEqual(frozen_hash(loaded), frozen_hash(self.base))
                before = {m:torch.load(root / f'{m}-selected.pt', weights_only=False) for m in ('null', 'temperature')}
                fit(args, self.base, ck, self.manifest, Budget(root))
                for mode, original in before.items():
                    after = torch.load(root / f'{mode}-selected.pt', weights_only=False)
                    self.assertEqual(after['selection'], original['selection'])
                    for name in original['model']:
                        self.assertTrue(torch.equal(after['model'][name], original['model'][name]))
                    # Exercise the production checkpoint loader, isolating its
                    # Modal declaration and fixed remote path from this test.
                    source = ast.parse((Path(__file__).parent / 'student_train.py').read_text())
                    loader = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'load_student')
                    namespace = {'build_model': lambda size:copy.deepcopy(self.base)}
                    exec(compile(ast.Module(body=[loader], type_ignores=[]), 'student_train.py', 'exec'), namespace)
                    with patch('torch.load', return_value=after):
                        loaded, _ = namespace['load_student']('synthetic')
                    self.assertEqual(frozen_hash(loaded), frozen_hash(self.base))
                    torch.testing.assert_close(self.full_stream(loaded), replay(loaded, cached, self.manifest))
                changed = [{**r, 'speaker':'speaker-fit'} if r['partition'] == 'validation' else r for r in index]
                atomic_json(root / 'cache-index.json', changed)
                with self.assertRaisesRegex(ValueError, 'paired data'):
                    fit(args, self.base, ck, self.manifest, Budget(root))


if __name__ == '__main__':
    unittest.main()

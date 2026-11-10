"""Small CPU invariants and ONNX parity; no audio or scoring fixtures."""

import ast
import contextlib
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from student_memory import AcousticMemory, DurationNeutralAttention, install_memory, visibility
from student_none import append_none, none_records


class PlainAttention(nn.Module):
    """Minimal standard attention oracle with icefall's projected-key cache."""
    def __init__(self):
        super().__init__()
        self.dropout, self.num_heads, self.query_head_dim = 0, 1, 2
        self.in_proj = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.in_proj.weight.copy_(torch.eye(2).repeat(2, 1))

    def forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
        q, k = self.in_proj(x).chunk(2, dim=-1)
        scores = torch.einsum("qbd,kbd->bqk", q, k)
        allowed = visibility(x, key_padding_mask, attn_mask)
        return scores.masked_fill(~allowed, -1000).softmax(-1).unsqueeze(0)

    def streaming_forward(self, x, pos_emb, cached_key, left_context_len, key_padding_mask):
        q, k = self.in_proj(x).chunk(2, dim=-1)
        k = torch.cat([cached_key, k])
        scores = torch.einsum("qbd,kbd->bqk", q, k)
        allowed = visibility(x, key_padding_mask, key_len=k.shape[0])
        weights = scores.masked_fill(~allowed, -1000).softmax(-1).unsqueeze(0)
        return weights, k[-left_context_len:]


class MemoryTests(unittest.TestCase):
    def memory(self, keys, scores, allowed=None, query=None):
        keys = torch.tensor(keys, dtype=torch.float32).reshape(1, 1, -1, 1)
        scores = torch.tensor(scores, dtype=torch.float32).reshape(1, 1, 1, -1)
        if allowed is None:
            allowed = torch.ones(1, 1, keys.shape[2], dtype=torch.bool)
        if query is None:
            query = torch.zeros(1, 1, 1, 1)
        m = AcousticMemory(1, 1, 40).eval()
        return m, keys, scores, allowed, query

    def test_repetition_has_one_unit_of_mass(self):
        # Repeating a sustained identical key from 1 to 40 frames cannot buy
        # more mass than a singleton consonant at the same content score.
        for n in [1, 4, 40]:
            m, k, s, a, q = self.memory([0] * n + [3], [0] * (n + 1))
            got = m(s.softmax(-1), q, k, a).flatten()
            torch.testing.assert_close(got[:n].sum(), torch.tensor(.5))
            torch.testing.assert_close(got[-1], torch.tensor(.5))

    def test_expansion_matches_explicit_hierarchy_and_keeps_details(self):
        m, k, s, a, q = self.memory([0, .1, 3, 3.1, 3.2], [0, 2, -1, 1, 3])
        groups = [s.flatten()[:2], s.flatten()[2:]]
        group_scores = torch.stack([g.logsumexp(0) - np.log(len(g)) for g in groups])
        expected = torch.cat([p * g.softmax(0) for p, g in zip(group_scores.softmax(0), groups)])
        got = m(s.softmax(-1), q, k, a).flatten()
        torch.testing.assert_close(got, expected)
        # Frame values are retained: this is not pooling values before query.
        values = torch.tensor([9., -2., 4., 1., 7.])
        torch.testing.assert_close(got @ values, expected @ values)

    def test_singletons_equal_ordinary_attention(self):
        m, k, s, a, q = self.memory([0, 3, 6], [-2, 0, 3])
        torch.testing.assert_close(m(s.softmax(-1), q, k, a), s.softmax(-1))

    def test_similarity_cannot_join_nonadjacent_keys(self):
        m, k, s, a, q = self.memory([0, 3, 0], [0, 0, 0])
        torch.testing.assert_close(m.connectivity(k).squeeze(), torch.eye(3))

    def test_visible_duration_excludes_padding_and_future(self):
        m, k, s, a, q = self.memory([0, 0, 0, 3], [0, 0, 0, 0])
        a = torch.tensor([[[False, False, True, True]]])
        w = s.masked_fill(~a.unsqueeze(0), -1000).softmax(-1)
        torch.testing.assert_close(m(w, q, k, a).flatten(), torch.tensor([0., 0., .5, .5]))
        # A prefix query sees a singleton group even if that group extends into
        # the next unseen chunk; no lookahead is added by the memory operation.
        a = torch.tensor([[[True, False, False, False]]])
        w = s.masked_fill(~a.unsqueeze(0), -1000).softmax(-1)
        torch.testing.assert_close(m(w, q, k, a).flatten(), torch.tensor([1., 0., 0., 0.]))

    def test_interval_counts_match_dense_membership_with_joint_gradients(self):
        torch.manual_seed(9)
        m = AcousticMemory(2, 3, 40).train()
        keys = torch.randn(2, 2, 7, 3)
        query = torch.randn(2, 2, 3, 3)
        allowed = torch.tensor([[[False, True, True, True, False, False, False],
                                 [True, True, True, True, True, False, False],
                                 [False, False, False, False, True, True, True]]]*2)
        scores = torch.randn(2, 2, 3, 7)
        weights = scores.masked_fill(~allowed.unsqueeze(0), -1000).softmax(-1)
        with torch.no_grad():
            m.duration_query.fill_(.1)
        got = m(weights, query, keys, allowed)
        count = torch.matmul(allowed.unsqueeze(0).float(), m.connectivity(keys)).clamp_min(1)
        annotation = ((query * m.duration_query[:, None, None]).sum(-1).unsqueeze(-1)
                      * torch.log1p(count * .04))
        mass = weights / count * annotation.exp()
        expected = mass / mass.sum(-1, keepdim=True)
        torch.testing.assert_close(got, expected)
        values = torch.randn_like(got)
        for actual, dense in zip(torch.autograd.grad((got*values).sum(), tuple(m.parameters()), retain_graph=True),
                                 torch.autograd.grad((expected*values).sum(), tuple(m.parameters()))):
            torch.testing.assert_close(actual, dense)

    def test_all_hidden_and_underflow_remain_finite(self):
        m, k, s, a, q = self.memory([0, 0, 3], [-1000, -1000, 0])
        a.zero_()
        got = m(s.softmax(-1), q, k, a)
        self.assertTrue(torch.isfinite(got).all())
        self.assertEqual(got.count_nonzero(), 0)

    def test_duration_is_retrievable_independent_of_key_repetition(self):
        got = []
        for n in [1, 8]:
            m, k, s, a, q = self.memory([0] * n + [3], [0] * (n + 1), query=torch.ones(1, 1, 1, 1))
            with torch.no_grad():
                m.duration_query.fill_(1)
            got.append(m(s.softmax(-1), q, k, a).flatten()[:n].sum())
        self.assertGreater(got[1], got[0])

    def test_grouping_and_duration_receive_joint_gradients(self):
        m, k, s, a, q = self.memory([0, .3, 3], [0, 1, 2], query=torch.ones(1, 1, 1, 1))
        m.train()
        k.requires_grad_()
        s.requires_grad_()
        result = m(s.softmax(-1), q, k, a)
        (result.flatten() * torch.tensor([0., 1., -1.])).sum().backward()
        for parameter in [m.log_metric, m.join_bias, m.duration_query, k, s]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum(), 0)
        # The relaxation never changes forward scores, even during training.
        m.eval()
        torch.testing.assert_close(m(s.softmax(-1), q, k, a), result)

    def test_bfloat16_group_ids_do_not_coalesce_after_256(self):
        m = AcousticMemory(1, 1, 40).eval()
        k = (torch.arange(600) % 2 * 4).to(torch.bfloat16).reshape(1, 1, -1, 1)
        torch.testing.assert_close(m.connectivity(k).squeeze(), torch.eye(600))

    def test_masks_refuse_hidden_holes(self):
        x = torch.zeros(3, 1, 2)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            visibility(x, torch.tensor([[False, True, False]]))

    def test_streaming_cache_and_full_window_agree(self):
        base = PlainAttention()
        adapted = DurationNeutralAttention(copy.deepcopy(base), 40).eval()
        x = torch.tensor([[0., 0.], [0., .1], [3., 3.], [3., 3.1]]).unsqueeze(1)
        pad = torch.zeros(1, 4, dtype=torch.bool)
        full = adapted(x, None)
        streamed, new_cache = adapted.streaming_forward(x[2:], None, x[:2], 2, pad)
        torch.testing.assert_close(streamed, full[:, :, 2:])
        _, plain_cache = base.streaming_forward(x[2:], None, x[:2], 2, pad)
        torch.testing.assert_close(new_cache, plain_cache)

    def test_install_only_upper_stages_and_preserve_state(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn_weights = PlainAttention()
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.downsampling_factor = (1, 4, 2)
        model.encoder.encoders = nn.ModuleList()
        for _ in range(3):
            stage = nn.Module()
            stage.layers = nn.ModuleList([Layer()])
            model.encoder.encoders.append(stage)
        before = copy.deepcopy(model)
        install_memory(model)
        self.assertIsInstance(model.encoder.encoders[0].layers[0].self_attn_weights, PlainAttention)
        for i, ms in [(1, 80), (2, 40)]:
            weights = model.encoder.encoders[i].layers[0].self_attn_weights
            self.assertEqual(weights.memory.frame_ms, ms)
            torch.testing.assert_close(weights.base.in_proj.weight,
                                       before.encoder.encoders[i].layers[0].self_attn_weights.in_proj.weight)
        checkpoint = copy.deepcopy(model.state_dict())
        install_memory(before)
        before.load_state_dict(checkpoint, strict=True)
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_memory(before)

    def test_onnx_export_matches_normal_and_truncated_memory(self):
        import onnxruntime as ort

        m, k, s, a, q = self.memory([0, .1, 3, 3.1], [0, 1, 2, 3], query=torch.ones(1, 1, 1, 1))
        with torch.no_grad():
            m.duration_query.fill_(.2)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "memory.onnx")
            torch.onnx.export(m, (s.softmax(-1), q, k, a), path, opset_version=17, dynamo=False,
                              input_names=["weights", "query", "keys", "allowed"], output_names=["result"])
            runtime = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            for mask in [a, torch.tensor([[[False, True, True, True]]]), torch.zeros_like(a)]:
                w = s.masked_fill(~mask.unsqueeze(0), -1000).softmax(-1)
                got = runtime.run(None, {"weights": w.numpy(), "query": q.numpy(), "keys": k.numpy(), "allowed": mask.numpy()})[0]
                torch.testing.assert_close(torch.from_numpy(got), m(w, q, k, mask))


class TrainingTests(unittest.TestCase):
    def test_continuations_require_preservation_and_identical_shipped_augmentation(self):
        tree = ast.parse(Path(__file__).with_name("student_train.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_continuation")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "student_train.py", "exec"), namespace)
        validate = namespace["validate_continuation"]
        cfg = {"duration_memory": True, "preservation_weight": 1., "preservation_run": "ordinary",
               "target": "reference", "streaming_chunk": 16, "streaming_left": 128, "augmentation_seed": 7,
               "help_none_dir": "/supplied/help-train", "help_reference": "/supplied/ref.json", "init": "",
               "noise_p": .5, "snr_db": (5., 25.), "rir_p": .2, "spec_aug": True}
        baseline = {**cfg, "snr_db": [5., 25.]}
        for memory in [True, False]:
            validate({**cfg, "duration_memory": memory}, baseline)
        for field, bad in [("preservation_weight", 0), ("preservation_weight", -1), ("target", "teacher"),
                           ("preservation_run", ""), ("streaming_chunk", None), ("augmentation_seed", None),
                           ("help_none_dir", ""), ("help_reference", ""), ("init", "librispeech-m"),
                           ("noise_p", 0), ("rir_p", 0), ("snr_db", (15., 25.)), ("spec_aug", False)]:
            with self.subTest(field=field, value=bad), self.assertRaises(ValueError):
                validate({**cfg, field: bad}, baseline)
        # Ordinary legacy training has no new requirements.
        validate({"duration_memory": False, "preservation_weight": 0})

    def test_none_audio_append_and_real_collation_preserve_native_buffers(self):
        import io
        import soundfile as sf

        reference = {"ayahs": [{"s": 1, "a": 1, "p": "ab"}]}
        template = {"speaker": "learner", "kind": "none", "split": "train",
                    "scored": True, "events": [], "non_errors": [], "unsupported": [], "ayahs": ["1:1"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "reference.json").write_text(json.dumps(reference))
            waves = [np.zeros(1600, dtype=np.float32), np.full(3200, .25, dtype=np.float32)]
            labels = []
            for i, wave in enumerate(waves):
                sf.write(root / f"take{i}.wav", wave, 16000, subtype="FLOAT")
                labels.append({**template, "id": str(i), "wav": f"take{i}.wav"})
            (root / "labels.jsonl").write_text("\n".join(map(json.dumps, labels)))
            native = b"native-audio"
            clips = {"ref": ["c"], "hyp": ["c"], "target": ["c"], "key": ["native"],
                     "n_samples": np.array([4000]), "n_frames": np.array([2]),
                     "flac": np.frombuffer(native, np.uint8), "flac_off": np.array([0, len(native)]),
                     "ids": np.ones(10, np.uint8), "probs": np.full(10, .2, np.float16), "top_off": np.array([0, 10])}
            self.assertEqual(append_none(clips, root, root / "reference.json"), 2)
            self.assertEqual(clips["flac"][:len(native)].tobytes(), native)
            np.testing.assert_array_equal(clips["n_frames"], [2, 0, 0])
            np.testing.assert_array_equal(clips["top_off"], [0, 10, 15, 20])
            self.assertEqual(clips["target"], ["c", "ab", "ab"])
            for i, expected in enumerate(waves, 1):
                audio = clips["flac"][clips["flac_off"][i]:clips["flac_off"][i+1]].tobytes()
                got, rate = sf.read(io.BytesIO(audio), dtype="float32")
                np.testing.assert_array_equal(got, expected)
                self.assertEqual(rate, 16000)
                self.assertEqual(clips["n_samples"][i], len(expected))
            # Exercise production collation with the exact appended teacher rows.
            tree = ast.parse(Path(__file__).with_name("student_train.py").read_text())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "collate")
            namespace = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "student_train.py", "exec"), namespace)
            items = []
            for i in range(3):
                lo, hi = clips["top_off"][i:i+2]
                items.append({"i": i, "feats": torch.zeros(i+2, 80), "target": torch.tensor([1]),
                              "top_ids": torch.from_numpy(clips["ids"][lo:hi].reshape(-1, 5).astype(np.int64)),
                              "top_p": torch.from_numpy(clips["probs"][lo:hi].reshape(-1, 5).astype(np.float32)),
                              "n_teacher": int(clips["n_frames"][i])})
            batch = namespace["collate"](items)
            torch.testing.assert_close(batch["n_teacher"], torch.tensor([2, 0, 0]))
            self.assertEqual(tuple(batch["top_ids"].shape), (3, 2, 5))
            self.assertEqual(batch["top_p"][1:].count_nonzero(), 0)

    def test_none_admission_rejects_scripts_and_split_speaker_leaks(self):
        reference = {"ayahs": [{"s": 1, "a": 1, "p": "ab"}]}
        template = {"id": "take", "speaker": "learner", "kind": "none", "split": "train",
                    "scored": True, "events": [], "non_errors": [], "unsupported": [],
                    "ayahs": ["1:1"], "wav": "take.wav"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps(template))
            self.assertEqual(none_records(directory, reference)[0]["target"], "ab")
            for modification in [{"kind": "repeat"}, {"events": [{"errorClass": "wrong_word"}]},
                                 {"non_errors": [{"kind": "repeat"}]}, {"excluded": True},
                                 {"split": "eval"}, {"scored": False}]:
                path.write_text(json.dumps({**template, **modification}))
                with self.assertRaisesRegex(ValueError, "no eligible"):
                    none_records(directory, reference)
            path.write_text(json.dumps(template) + "\n" + json.dumps({**template, "split": "eval"}))
            with self.assertRaisesRegex(ValueError, "no eligible"):
                none_records(directory, reference)

    def test_actual_training_loss_masks_missing_teacher_and_padded_rows(self):
        # Load the production loss without executing Modal image declarations.
        tree = ast.parse(Path(__file__).with_name("student_train.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "losses")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "student_train.py", "exec"), namespace)
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.tensor([[[0., 2., -1.], [2., 0., -1.], [99., -99., 0.]],
                                                        [[0., -1., 2.], [2., 0., -1.], [0., 2., -1.]]]))
            def forward(self, features, lengths):
                return self.logits.log_softmax(-1), torch.tensor([2, 3])
        model, anchor = Model(), Model().requires_grad_(False)
        batch = {"feats": torch.zeros(2, 3, 80), "feat_lens": torch.tensor([3, 3]),
                 "targets": torch.tensor([1, 2]), "target_lens": torch.tensor([1, 1]),
                 "n_teacher": torch.tensor([2, 0]),
                 "top_ids": torch.tensor([[[1]*5, [0]*5], [[0]*5, [0]*5]]),
                 "top_p": torch.tensor([[[.2]*5, [.2]*5], [[0.]*5, [0.]*5]])}
        cfg = {"ctc_weight": 1., "kd_weight": .5, "kd_delay": 0, "preservation_weight": 2.}
        # Only disable CUDA autocast on these tiny CPU tensors.
        original = torch.autocast
        torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()
        try:
            loss = namespace["losses"](model, batch, cfg, torch.device("cpu"), anchor)
            expected = -model.logits.log_softmax(-1)[0, :2, [1, 0]].diagonal().mean()
            torch.testing.assert_close(loss["kd"], expected)
            self.assertLess(abs(loss["preservation"].item()), 1e-6)
            loss["loss"].backward()
            self.assertEqual(model.logits.grad[0, 2].count_nonzero(), 0)
            self.assertIsNone(anchor.logits.grad)
            with self.assertRaisesRegex(ValueError, "frozen listener"):
                namespace["losses"](model, batch, cfg, torch.device("cpu"))
        finally:
            torch.autocast = original


if __name__ == "__main__":
    unittest.main()

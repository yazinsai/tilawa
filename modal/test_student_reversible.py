"""Small CPU tests; native integration additionally requires pinned icefall.

PYTHONPATH=modal:tools:$PYTHONPATH python -m unittest discover -s modal -p 'test_student_reversible*.py' -v
"""

import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from student_reversible import AcousticBlock, AcousticStack
from student_reversible_train import acoustic_losses, eligible_none, original_records, paired_models, partition_records


def nontrivial(block):
    for function in (block.f, block.g):
        nn.init.normal_(function.output.weight, std=0.2)
        nn.init.normal_(function.output.bias, std=0.1)
    return block


class CouplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)

    def test_nontrivial_inverse_preserves_full_state_with_masks_and_history(self):
        block = nontrivial(AcousticBlock(8, 12, 3)).double()
        x = torch.randn(11, 2, 8, dtype=torch.float64)
        states = [torch.randn_like(state) for state in block.init_states(2, x.device, x.dtype)]
        mask = torch.ones_like(x)
        mask[3:6, :, :4] = 0
        y, _ = block(x, states, mask)
        self.assertEqual(y.shape, x.shape)
        self.assertGreater((y - x).abs().max().item(), 0.01)
        torch.testing.assert_close(block.inverse(y, states, mask), x, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(y[3:6, :, :4], x[3:6, :, :4])

    def test_streaming_matches_full_and_inverse_uses_incoming_caches(self):
        block = nontrivial(AcousticBlock(8, 12, 4)).double()
        x = torch.randn(23, 2, 8, dtype=torch.float64)
        expected, expected_caches = block(x)
        caches = block.init_states(2, x.device, x.dtype)
        outputs = []
        for start, end in ((0, 1), (1, 6), (6, 9), (9, 23)):
            old = [cache.clone() for cache in caches]
            y, caches = block(x[start:end], caches)
            torch.testing.assert_close(block.inverse(y, old), x[start:end], rtol=1e-10, atol=1e-10)
            # Inference never mutates the caller's cached state.
            if start == 0:
                self.assertTrue(all(not state.any() for state in old))
            outputs.append(y)
        torch.testing.assert_close(torch.cat(outputs), expected, rtol=1e-10, atol=1e-10)
        for actual, expected in zip(caches, expected_caches):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_residual_control_can_destroy_information(self):
        class NegativeIdentity(nn.Module):
            def forward(self, x, states):
                return -x, states

        block = AcousticBlock(2, 2, 2, "residual")
        block.f = NegativeIdentity()
        block.g = NegativeIdentity()
        states = [torch.zeros(1) for _ in range(6)]
        a = torch.tensor([[[1., 1.]]])
        b = torch.tensor([[[5., 5.]]])
        torch.testing.assert_close(block(a, states)[0], block(b, states)[0])
        block.mode = "reversible"
        y, _ = block(a, states)
        torch.testing.assert_close(block.inverse(y, states), a)
        self.assertFalse(torch.equal(y, block(b, states)[0]))

    def test_equal_parameters_and_gradients_in_both_functions(self):
        reversible = nontrivial(AcousticBlock(8, 12, 3))
        residual = copy.deepcopy(reversible)
        residual.mode = "residual"
        self.assertEqual(sum(p.numel() for p in reversible.parameters()), sum(p.numel() for p in residual.parameters()))
        x = torch.randn(8, 2, 8, requires_grad=True)
        reversible(x)[0].square().sum().backward()
        for function in (reversible.f, reversible.g):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in function.parameters()))
            self.assertGreater(function.input.weight.grad.abs().sum().item(), 0)
        self.assertGreater(x.grad[..., :4].abs().sum().item(), 0)
        self.assertGreater(x.grad[..., 4:].abs().sum().item(), 0)

    def test_stack_six_cache_routing_and_padding_preserve_state(self):
        stack = AcousticStack(8, 12, 3, 2, "reversible").double()
        for layer in stack.layers:
            nontrivial(layer)
        x = torch.randn(17, 2, 8, dtype=torch.float64)
        pad = torch.zeros(2, 17, dtype=torch.bool)
        pad[1, 13:] = True
        expected = stack(x, src_key_padding_mask=pad)
        caches = stack.init_states(2, x.device)
        outputs = []
        for start, end in ((0, 5), (5, 10), (10, 17)):
            actual, caches = stack.streaming_forward(x[start:end], caches, 64, pad[:, :end])
            self.assertEqual(len(caches), 12)
            outputs.append(actual)
        torch.testing.assert_close(torch.cat(outputs), expected, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(expected[13:, 1], x[13:, 1])

    def test_invalid_modes_and_odd_channels(self):
        for channels, mode in ((7, "reversible"), (8, "unknown")):
            with self.assertRaises(ValueError):
                AcousticBlock(channels, 12, 3, mode)
        with self.assertRaises(ValueError):
            AcousticBlock(8, 12, 3, "residual").inverse(torch.zeros(4, 1, 8))
        with self.assertRaises(ValueError):
            AcousticBlock(8, 12, 3)(torch.zeros(4, 1, 8), [torch.zeros(1)])


class TrainingSafetyTests(unittest.TestCase):
    def test_excluded_original_clip_blocks_its_entire_speaker(self):
        rows = [dict(shard="fixture", row=i, cluster=speaker, bucket="clean-match", split="train",
                     n_samples=16000, ref="a", hyp="a", flac=b"audio", top5_ids=b"ids", top5_p=b"probs")
                for i, speaker in ((0, "excluded-speaker"), (1, "excluded-speaker"), (2, "training-speaker"))]
        table = types.SimpleNamespace(to_pylist=lambda: rows)
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.read_table = lambda *args, **kwargs: table
        arrow = types.ModuleType("pyarrow")
        arrow.parquet = parquet
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.parquet").touch()
            exclude = root / "exclude.txt"
            exclude.write_text("fixture:0\n")
            with patch.dict(sys.modules, {"pyarrow": arrow, "pyarrow.parquet": parquet}), patch("student.spliced_target", lambda ref, hyp: ref):
                records = original_records(root, exclude, 0, 41)
        self.assertEqual([r["speaker"] for r in records], ["tlog:training-speaker"])

    def test_none_labels_and_speaker_exclusions(self):
        clean = dict(split="train", kind="none", speaker="training-speaker", events=[], non_errors=[])
        self.assertTrue(eligible_none(clean, set()))
        self.assertFalse(eligible_none(clean, {"training-speaker"}))
        for field, value in (("split", "test"), ("kind", "repeat"), ("kind", "wrong_word"),
                             ("exclude", True), ("excluded", True), ("unsupported", ["vowel"]),
                             ("events", [{"class": "added_word"}]), ("non_errors", ["repeat"]), ("speaker", None)):
            self.assertFalse(eligible_none({**clean, field: value}, set()), (field, value))

    def test_partition_is_deterministic_and_speaker_disjoint(self):
        records = [dict(key=f"clip-{i}-{j}", speaker=f"speaker-{i}") for i in range(30) for j in range(3)]
        train, val = partition_records(records)
        self.assertTrue(train and val)
        self.assertFalse({r["speaker"] for r in train} & {r["speaker"] for r in val})
        reverse_train, reverse_val = partition_records(list(reversed(records)))
        self.assertEqual({r["key"] for r in train}, {r["key"] for r in reverse_train})
        self.assertEqual({r["key"] for r in val}, {r["key"] for r in reverse_val})

    def test_kd_excludes_none_takes_without_teacher_and_is_finite(self):
        lp = torch.randn(2, 5, 4).log_softmax(-1).requires_grad_()
        batch = dict(targets=torch.tensor([1, 2]), target_lens=torch.tensor([1, 1]), n_teacher=torch.tensor([5, 1]),
                     top_ids=torch.zeros(2, 5, 5, dtype=torch.long), top_p=torch.zeros(2, 5, 5),
                     has_teacher=torch.tensor([True, False]))
        batch["top_p"][0, :, 0] = 1
        loss, metrics = acoustic_losses(lp, torch.tensor([5, 5]), batch, {"kd_weight": 1}, lp.detach())
        self.assertAlmostEqual(metrics["kd"], float(-lp[0, :, 0].mean().detach()), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(lp.grad).all())


@unittest.skipUnless(importlib.util.find_spec("zipformer"), "Pinned icefall required for native integration")
class NativeIntegrationTests(unittest.TestCase):
    def test_shipped_prefix_config_rebuild_clock_and_warmed_states(self):
        import student_train as T
        from student_export import wrap

        old_sizes, old_common = copy.deepcopy(T.SIZES), copy.deepcopy(T.COMMON)
        try:
            T.SIZES["test"] = dict(num_encoder_layers="1,1,1,1,1,1", feedforward_dim="32,32,32,32,32,32",
                                   encoder_dim="16,16,16,16,16,16", encoder_unmasked_dim="16,16,16,16,16,16")
            T.COMMON.update(num_heads="2,2,2,2,2,2", query_head_dim="8", value_head_dim="4", pos_head_dim="2", pos_dim=8,
                            cnn_module_kernel="3,3,3,3,3,3")
            base = T.build_model("test")
            ck = dict(cfg=dict(size="test"), model=base.state_dict())
            _, arms = paired_models(ck, 41)
            for model in arms.values():
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        torch.testing.assert_close(parameter, base.state_dict()[name], rtol=0, atol=0)
                rebuilt = T.build_model("test", model.encoder.encoders[-1].encoder.layers[0].mode)
                rebuilt.load_state_dict(model.state_dict())
                wrapper, ordinary = wrap(model, 16, 64), wrap(base, 16, 64)
                states, ordinary_states = wrapper.init_states(), ordinary.init_states()
                for _ in range(3):
                    x = torch.randn(1, 45, 80)
                    with torch.no_grad():
                        result, states = wrapper(x, states)
                        reference, ordinary_states = ordinary(x, ordinary_states)
                    self.assertEqual(result.shape, reference.shape)
                    self.assertEqual(result.shape[1], 8)
                    self.assertEqual(len(states), len(ordinary_states))
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertTrue(all(torch.isfinite(s).all() for s in states))
        finally:
            T.SIZES.clear()
            T.SIZES.update(old_sizes)
            T.COMMON.clear()
            T.COMMON.update(old_common)


if __name__ == "__main__":
    unittest.main()

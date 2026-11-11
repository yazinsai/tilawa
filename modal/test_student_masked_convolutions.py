"""Run in the supervisor's pinned train image: unittest discovery in modal/."""

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from subsampling import Conv2dSubsampling
from zipformer import SimpleDownsample, Zipformer2

from student_masked_convolutions import (
    MaskedConv2dSubsampling, downsample_valid, install_validity_masks, padding_mask,
)
from student_masked_refit import (
    ReadingDataset, check_partitions, copy_insertion_assets, export_bundle,
    learner_records, original_records, partition_records, reading_batch, upper_parameters,
)


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_embed = Conv2dSubsampling(80, 32, layer1_channels=4,
                                             layer2_channels=8, layer3_channels=16, dropout=0)
        self.encoder = Zipformer2(
            output_downsampling_factor=2, downsampling_factor=(1, 2, 4),
            num_encoder_layers=(1, 1, 1), encoder_dim=(32, 32, 32),
            encoder_unmasked_dim=(32, 32, 32), query_head_dim=8, pos_head_dim=4,
            value_head_dim=4, pos_dim=12, num_heads=(2, 2, 2),
            feedforward_dim=(64, 64, 64), cnn_module_kernel=(7, 7, 7),
            dropout=0, causal=True, chunk_size=(16,), left_context_frames=(64,),
        )
        self.ctc_output = torch.nn.Linear(32, 43)

    def forward(self, features, lengths):
        x, lengths = self.encoder_embed(features, lengths)
        x, lengths = self.encoder(x.transpose(0, 1), lengths, padding_mask(lengths, x.shape[1]))
        return self.ctc_output(x.transpose(0, 1)).log_softmax(-1), lengths

    def streaming_eval(self, chunk=16, left=64):
        self.encoder.chunk_size, self.encoder.left_context_frames = (chunk,), (left,)


class ValidityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def embedding(self):
        base = Conv2dSubsampling(80, 32, layer1_channels=4, layer2_channels=8,
                                layer3_channels=16, dropout=0).eval()
        with torch.no_grad():
            # Deliberately non-silent hidden padding, independent of a checkpoint.
            for module in base.modules():
                if isinstance(module, torch.nn.Conv2d) and module.bias is not None:
                    module.bias.fill_(0.4)
            base.convnext.pointwise_conv2.weight.mul_(20)
        masked = copy.deepcopy(base)
        masked.__class__ = MaskedConv2dSubsampling
        return base, masked

    def test_embedding_matches_isolated_item_with_nonsilent_padding(self):
        base, masked = self.embedding()
        features = torch.randn(2, 153, 80)
        lengths = torch.tensor([81, 153])
        features[0, 81:] = -23.0259
        with torch.no_grad():
            alone, alone_lengths = base(features[:1, :81], lengths[:1])
            batch, batch_lengths = masked(features, lengths)
            unmasked, _ = base(features, lengths)
        self.assertEqual(batch_lengths.tolist(), [(81 - 7) // 2, (153 - 7) // 2])
        torch.testing.assert_close(batch[0, :alone_lengths[0]], alone[0], rtol=1e-4, atol=1e-5)
        self.assertGreater(float((unmasked[0, :alone_lengths[0]] - alone[0]).abs().max()), 1e-4)
        self.assertEqual(int(torch.count_nonzero(batch[0, alone_lengths[0]:])), 0)

    def test_arbitrary_suffix_cannot_affect_valid_outputs_or_receive_gradient(self):
        _, masked = self.embedding()
        features = torch.randn(2, 131, 80, requires_grad=True)
        lengths = torch.tensor([80, 131])
        changed = features.detach().clone()
        changed[0, 80:] = torch.randn_like(changed[0, 80:]) * 500
        actual, lens = masked(features, lengths)
        expected, _ = masked(changed, lengths)
        torch.testing.assert_close(actual[0, :lens[0]], expected[0, :lens[0]], rtol=0, atol=0)
        actual[0, :lens[0]].square().sum().backward()
        self.assertEqual(int(torch.count_nonzero(features.grad[0, 80:])), 0)
        self.assertGreater(int(torch.count_nonzero(features.grad[0, :80])), 0)

    def test_stride_lengths_at_both_parities(self):
        _, masked = self.embedding()
        lengths = torch.arange(15, 41)
        output, got = masked(torch.randn(len(lengths), 40, 80), lengths)
        torch.testing.assert_close(got, (lengths - 7) // 2)
        self.assertEqual(output.shape[1], (40 - 7) // 2)

    def test_partial_downsampling_repeats_each_items_last_valid_frame(self):
        src = torch.randn(19, 3, 8, requires_grad=True)
        lengths = torch.tensor([5, 13, 19])
        for ds in (2, 4, 8):
            with self.subTest(ds=ds):
                module = SimpleDownsample(8, ds, dropout=0, causal=True)
                with torch.no_grad():
                    module.bias.copy_(torch.arange(ds) / ds)
                actual, got_lengths = downsample_valid(module, src, lengths)
                for i, length in enumerate(lengths):
                    expected = module(src[:length, i:i + 1])
                    torch.testing.assert_close(actual[:got_lengths[i], i], expected[:, 0])
                    self.assertEqual(int(torch.count_nonzero(actual[got_lengths[i]:, i])), 0)
                changed = src.detach().clone()
                changed[5:, 0] = 1000
                torch.testing.assert_close(downsample_valid(module, changed, lengths)[0][:got_lengths[0], 0],
                                           actual[:got_lengths[0], 0])
        actual[:got_lengths[0], 0].sum().backward()
        self.assertEqual(int(torch.count_nonzero(src.grad[5:, 0])), 0)

    def test_downsampling_rejects_empty_item(self):
        module = SimpleDownsample(8, 2, dropout=0, causal=True)
        with self.assertRaises(ValueError):
            downsample_valid(module, torch.randn(7, 1, 8), torch.tensor([0]))

    def test_full_model_boundary_and_state_dict_roundtrip(self):
        base = TinyStudent().eval()
        masked = copy.deepcopy(base)
        sites = install_validity_masks(masked)
        self.assertEqual(len(sites), 4)
        self.assertEqual(list(base.state_dict()), list(masked.state_dict()))
        restored = TinyStudent().eval()
        install_validity_masks(restored)
        restored.load_state_dict(masked.state_dict(), strict=True)
        features = torch.randn(2, 157, 80)
        lengths = torch.tensor([81, 157])
        with torch.no_grad():
            alone, lone_lens = base(features[:1, :81], lengths[:1])
            batch, batch_lens = masked(features, lengths)
            again, _ = restored(features, lengths)
        torch.testing.assert_close(batch[0, :lone_lens[0]], alone[0], rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(again, batch, rtol=0, atol=0)
        torch.testing.assert_close(batch_lens, ((lengths - 7) // 2 + 1) // 2)
        with self.assertRaises(ValueError):
            install_validity_masks(masked)

    def test_streaming_outputs_and_warmed_caches_are_unchanged(self):
        base = TinyStudent().eval()
        masked = copy.deepcopy(base)
        install_validity_masks(masked)
        models = (base, masked)
        states = [m.encoder.get_init_states(1, torch.device("cpu")) +
                  [m.encoder_embed.get_init_states(1, torch.device("cpu")), torch.zeros(1, dtype=torch.long)]
                  for m in models]
        with torch.no_grad():
            for _ in range(3):
                features = torch.randn(1, 45, 80)
                outputs = []
                for i, model in enumerate(models):
                    x, lens, pad = model.encoder_embed.streaming_forward(features, torch.tensor([45]), states[i][-2])
                    processed = states[i][-1]
                    mask = (processed[:, None] <= torch.arange(64)[None]).flip(1)
                    mask = torch.cat((mask, torch.zeros(1, 16, dtype=torch.bool)), dim=1)
                    x, _, cache = model.encoder.streaming_forward(x.transpose(0, 1), lens, states[i][:-2], mask)
                    outputs.append(model.ctc_output(x.transpose(0, 1)).log_softmax(-1))
                    states[i] = cache + [pad, processed + lens]
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                for original, treated in zip(*states):
                    torch.testing.assert_close(original, treated, rtol=0, atol=0)

    def test_streaming_export_uses_inherited_computation(self):
        model = TinyStudent().eval()
        install_validity_masks(model)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, output = root / "base", root / "export"
            base.mkdir()
            # Byte-copy fixture assets; the actual SDK integration test checks
            # the real insertion teacher's registry hashes and loads the bundle.
            for name in ("teacher.onnx", "teacher.onnx.data"):
                (base / name).write_bytes(name.encode())
            result = export_bundle(model, {"state": {"step": 1}}, output, "test-masked", base, left=64)
            self.assertEqual(result["frameMs"], 40)
            self.assertEqual(result["window"], 45)
            self.assertEqual(result["shift"], 32)
            report = json.loads((output / "export-parity.json").read_text())
            self.assertEqual(report["windows"], 3)
            self.assertLess(report["max_abs_probability_error"]["fp32"], 1e-4)
            for name in ("teacher.onnx", "teacher.onnx.data"):
                self.assertEqual((base / name).read_bytes(), (output / name).read_bytes())


class RefitTests(unittest.TestCase):
    def test_upper_blocks_and_head_only_are_trainable(self):
        model = TinyStudent()
        parameters = upper_parameters(model)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(names)
        self.assertEqual(len(names), len(parameters))
        self.assertTrue(all(n.startswith(("encoder.encoders.1.", "encoder.encoders.2.", "ctc_output."))
                            for n in names))
        self.assertTrue(any(n.startswith("encoder.encoders.1.") for n in names))
        self.assertTrue(any(n.startswith("encoder.encoders.2.") for n in names))
        self.assertTrue(any(n.startswith("ctc_output.") for n in names))

    def test_speaker_partition_is_deterministic_and_disjoint(self):
        records = [{"speaker": f"speaker-{i // 3}", "key": str(i)} for i in range(90)]
        train, validation = partition_records(records)
        self.assertTrue(train and validation)
        check_partitions(train, validation)
        reversed_train, reversed_val = partition_records(list(reversed(records)))
        self.assertEqual({r["key"] for r in train}, {r["key"] for r in reversed_train})
        self.assertEqual({r["key"] for r in validation}, {r["key"] for r in reversed_val})
        with self.assertRaises(ValueError):
            check_partitions(train, validation + train[:1])

    def test_learner_admission_rejects_scripts_repeats_and_held_out_speakers(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sf.write(root / "ordinary.wav", np.zeros(8000, dtype=np.float32), 16000)
            (root / "ref.json").write_text(json.dumps({"ayahs": [{"s": 1, "a": 1, "p": "بَسَمَ"}]}))
            template = dict(id="ordinary", speaker="training-speaker", split="train", kind="none",
                            events=[], non_errors=[], ayahs=["1:1"], wav="ordinary.wav")
            variants = [template, {**template, "id": "script", "kind": "wrong_word"},
                        {**template, "id": "event", "events": [{}]},
                        {**template, "id": "repeat", "non_errors": [{}]},
                        {**template, "id": "held-out", "split": "test", "speaker": "release-speaker"},
                        {**template, "id": "exclude", "excluded": True},
                        {**template, "id": "unsupported", "unsupported": True}]
            (root / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in variants))
            admitted = learner_records(root, root / "ref.json")
            self.assertEqual([r["key"] for r in admitted], ["ordinary"])
            self.assertEqual(admitted[0]["speaker"], "help:training-speaker")
            self.assertEqual(admitted[0]["target"], "بَسَمَ")

    def test_original_training_data_excludes_gold_and_non_training_splits(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = dict(shard="training", bucket="clean-match", split="train", cluster=4,
                            n_samples=8000, ref="بَسَمَ", hyp="بَسَمَ", flac=b"fixture")
            rows = [{**template, "row": 0}, {**template, "row": 1},
                    {**template, "row": 2, "split": "test"},
                    {**template, "row": 3, "bucket": "other"},
                    {**template, "row": 4, "n_samples": 400}]
            pq.write_table(pa.Table.from_pylist(rows), root / "train.parquet")
            (root / "exclusions.txt").write_text("training:1\n")
            admitted = original_records(root, root / "exclusions.txt")
            self.assertEqual([r["key"] for r in admitted], ["training:0"])
            self.assertEqual(admitted[0]["speaker"], "tlog:4")

    def test_dataset_reuses_exact_augmented_tensors_and_lengths(self):
        import numpy as np
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, np.sin(np.arange(8000) * 0.05).astype(np.float32), 16000, format="WAV")
        record = dict(audio=buffer.getvalue(), target="بَسَمَ")
        aug = {key: dict(data=np.ones(800, dtype=np.float32), offsets=np.array([0, 800]))
               for key in ("noise", "rir")}
        dataset = ReadingDataset([record], aug)
        first, second = dataset[0], dataset[0]
        torch.testing.assert_close(first["feats"], second["feats"], rtol=0, atol=0)
        b = reading_batch([first, dict(feats=first["feats"][:15], target=first["target"])])
        self.assertEqual(b["lens"].tolist(), [len(first["feats"]), 15])
        self.assertTrue(torch.all(b["feats"][1, 15:] == -23.0259))

    def test_missing_auxiliary_asset_fails_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, output = root / "base", root / "out"
            base.mkdir()
            output.mkdir()
            (base / "teacher.onnx").write_bytes(b"graph")
            with self.assertRaises(FileNotFoundError):
                copy_insertion_assets(base, output)
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

"""Training integration on tiny fixtures; no practice recordings or checkpoints."""
import copy
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

from student_routing_train import acoustic_objective, corpus_hash, load_partition, packed_clips, state_hash, validate_recipe


def recipe():
    return dict(schema=1, split="train", training_keys=["train-shard:0", "help:clean-train"],
                validation_keys=["valid-shard:0", "help:clean-valid"],
                speakers_by_key={"train-shard:0": "reader-a", "help:clean-train": "learner-a",
                                 "valid-shard:0": "reader-b", "help:clean-valid": "learner-b"},
                excluded_speakers=["release-reader"], checkpoint_sha256="a" * 64,
                checkpoint="/student/prior-runs/initial/ckpt.pt", base_bundle="/lab/data/models/initial",
                corpus_sha256="b" * 64, teacher_sha256="c" * 64,
                pairs=[dict(student="encoder.attention", teacher="encoder.layers.0.self_attn", downsample=1)],
                calibration_updates=2, updates=3, validate_every=1, seed=0, base_lr=.01,
                aux_weight=1., preservation_weight=1., ctc_weight=1., kd_weight=1., chunk=16, left=128,
                real_pause_examples=1, selection="minimum-validation-ctc-plus-preservation")


class RecipeTests(unittest.TestCase):
    def test_complete_disjoint_training_recipe_is_accepted(self):
        self.assertIsNotNone(validate_recipe(recipe()))

    def test_release_speaker_and_cross_partition_alias_are_rejected(self):
        for replacement in ("release-reader", "reader-a"):
            r = recipe()
            r["speakers_by_key"]["valid-shard:0"] = replacement
            with self.assertRaises(ValueError):
                validate_recipe(r)

    def test_incomplete_or_changed_contract_is_rejected(self):
        for key, value in (("teacher_sha256", ""), ("checkpoint", ""), ("base_bundle", ""), ("split", "eval"), ("chunk", 32),
                           ("real_pause_examples", 0), ("aux_weight", float('nan')),
                           ("selection", "best-practice-score"), ("calibration_updates", 0)):
            r = recipe()
            r[key] = value
            with self.assertRaises(ValueError, msg=key):
                validate_recipe(r)

    def test_missing_speaker_identity_and_keys_are_rejected(self):
        r = recipe()
        r["speakers_by_key"]["valid-shard:0"] = None
        with self.assertRaises(ValueError):
            validate_recipe(r)
        r = recipe()
        r["training_keys"].append(r["training_keys"][0])
        with self.assertRaises(ValueError):
            validate_recipe(r)

    def test_state_fingerprint_detects_weights_and_names(self):
        model = nn.Linear(2, 3)
        initial = state_hash(model)
        self.assertEqual(state_hash(copy.deepcopy(model)), initial)
        with torch.no_grad():
            model.bias[0] += .01
        self.assertNotEqual(initial, state_hash(model))


class ObjectiveTests(unittest.TestCase):
    def test_dispatch_trains_routes_or_features_with_common_ctc_kd_preservation(self):
        for mode in ("ordinary", "feature", "routing"):
            x = torch.tensor([[.4, .6], [.9, .1]]).log().requires_grad_()
            route, feature = torch.tensor(2., requires_grad=True), torch.tensor(3., requires_grad=True)
            ctc = torch.tensor(1., requires_grad=True)
            teacher = torch.tensor([[.2, .8], [.7, .3]]).log().requires_grad_()
            anchor = torch.tensor([[.4, .6], [.9, .1]]).log().requires_grad_()

            class Arm:
                device = torch.device("cpu")
                model = nn.Linear(1, 1)

                def stream(self, feats, targets=None):
                    return x, route, feature

            class Anchor:
                def stream(self, feats):
                    return anchor, None, None

            training = types.SimpleNamespace(DEFAULTS={}, losses=lambda *args: {"ctc": ctc}, collate=lambda items: items)
            with patch.dict(sys.modules, {"student_train": training}):
                loss, metrics = acoustic_objective(Arm(), Anchor(), {"routing_feats": torch.ones(2, 1)},
                                                  (teacher, []), recipe(), mode)
            expected = 1 + float(-(teacher.detach().exp() * x.detach()).sum(-1).mean())
            expected += {"ordinary": 0, "feature": 3, "routing": 2}[mode]
            self.assertAlmostEqual(loss.item(), expected, places=6)
            loss.backward()
            self.assertEqual(route.grad.item(), 1 if mode == "routing" else 0)
            self.assertEqual(feature.grad.item(), 1 if mode == "feature" else 0)
            self.assertIsNone(teacher.grad)
            self.assertIsNone(anchor.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(ctc.grad.item(), 1)
            self.assertAlmostEqual(metrics["preservation"], 0, places=6)


class DataTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import soundfile as sf

        root = Path(".scratch/routing-unit-tmp")
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temp.name)
        self.help = self.root / "help-train"
        self.help.mkdir()
        self.s1 = self.root / "s1"
        self.s1.mkdir()
        self.wave = np.concatenate([np.zeros(600, dtype=np.float32),
                                    np.linspace(-.1, .1, 1000, dtype=np.float32), np.zeros(600, dtype=np.float32)])
        sf.write(self.help / "take.wav", self.wave, 16000, subtype="FLOAT")
        row = dict(shard="train-shard", row=0, split="train", bucket="clean-match", ref="بَ",
                   hyp="بَ", n_samples=2200, n_frames=4, flac=(self.help / "take.wav").read_bytes(),
                   top5_ids=bytes(20), top5_p=np.zeros(20, dtype=np.float16).tobytes())
        pq.write_table(pa.Table.from_pylist([row]), self.s1 / "train-shard.parquet")
        self.label = dict(id="clean-train", speaker="learner-a", split="train", kind="none",
                          events=[], non_errors=[], unsupported=[], scored=True, wav="take.wav", ayahs=["1:1"])
        self.write_label(self.label)
        sys.path.insert(0, str(Path("tools").resolve()))

    def tearDown(self):
        self.temp.cleanup()

    def write_label(self, row):
        (self.help / "labels.jsonl").write_text(json.dumps(row) + "\n")

    def load(self):
        return load_partition(recipe(), "training", self.root, self.s1)

    def test_none_ingestion_retains_all_pcm_including_pauses(self):
        import io
        import soundfile as sf

        rows = self.load()
        wave, sr = sf.read(io.BytesIO(rows[1]["flac"]), dtype="float32")
        self.assertEqual(sr, 16000)
        np.testing.assert_array_equal(wave, self.wave)
        self.assertTrue(rows[1]["ref"])
        packed = packed_clips(rows)
        self.assertEqual(packed["target"], [r["ref"] for r in rows])
        self.assertEqual(packed["flac_off"][-1], len(packed["flac"]))
        self.assertEqual(packed["top_off"][-1], len(packed["ids"]))
        original = corpus_hash(rows)
        changed = copy.deepcopy(rows)
        changed[1]["ref"] += "ب"
        self.assertNotEqual(corpus_hash(changed), original)

    def test_repeats_and_non_none_labels_never_become_clean_transcripts(self):
        for key, value in (("kind", "repeat"), ("kind", "added_word"), ("non_errors", [{"type": "repeat"}]),
                           ("unsupported", ["tajweed"]), ("events", [{"class": "vowel"}])):
            row = dict(self.label, **{key: value})
            self.write_label(row)
            with self.assertRaises(ValueError, msg=key):
                self.load()

    def test_eval_and_misattributed_speakers_are_rejected(self):
        for key, value in (("split", "eval"), ("speaker", "release-reader"), ("scored", False)):
            self.write_label(dict(self.label, **{key: value}))
            with self.assertRaises(ValueError):
                self.load()

    def test_paths_outside_authorized_corpus_and_missing_keys_fail(self):
        self.write_label(dict(self.label, wav="../outside.wav"))
        with self.assertRaises(ValueError):
            self.load()
        (self.help / "labels.jsonl").write_text("")
        with self.assertRaises(ValueError):
            self.load()

    def test_paired_augmentation_is_identical_for_all_arms_and_changes_by_epoch(self):
        # Exercise the actual production dataset without constructing the
        # Modal image/volume declarations or loading an acoustic model.
        tree = ast.parse(Path("modal/student_train.py").read_text())
        names = {"ClipDataset", "pick", "reverb", "spec_augment"}
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        namespace = {"io": io, "SR": 16000}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "student_train.py", "exec"), namespace)
        fake_audio = types.SimpleNamespace(compliance=types.SimpleNamespace(kaldi=types.SimpleNamespace(
            fbank=lambda wave, **kwargs: wave[0, :800].reshape(10, 80).clone())))
        clips = packed_clips(self.load())
        cfg = dict(seed=0, routing_paired=True, spec_aug=True, noise_p=0., rir_p=0.)
        # Gain still comes from the shipped augmentation chain. Small source
        # PCM includes real zero-valued pause regions before/after articulation.
        aug = {"noise": {}, "rir": {}}
        with patch.dict(sys.modules, {"torchaudio": fake_audio}):
            a = namespace["ClipDataset"](clips, cfg, True, aug)
            b = namespace["ClipDataset"](clips, cfg, True, aug)
            first, same = a[1], b[1]
            torch.testing.assert_close(first["wave"], same["wave"], rtol=0, atol=0)
            torch.testing.assert_close(first["feats"], same["feats"], rtol=0, atol=0)
            self.assertEqual(len(first["wave"]), len(self.wave))
            self.assertFalse(torch.equal(first["feats"], first["routing_feats"]))
            b.augmentation_epoch = 1
            self.assertFalse(torch.equal(first["wave"], b[1]["wave"]))
            plain = namespace["ClipDataset"](clips, dict(cfg, routing_paired=False), False, None)[1]
            self.assertNotIn("wave", plain)
            self.assertNotIn("routing_feats", plain)


if __name__ == "__main__":
    unittest.main()

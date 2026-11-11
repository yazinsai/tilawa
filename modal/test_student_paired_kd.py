"""Lightweight waveform/cache checks; torch integration runs in the train image.

python3 -m unittest discover -s modal -p test_student_paired_kd.py
"""
import ast
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student_paired_kd import AcousticInput, TargetCache, TopTargets


class ObservationTests(unittest.TestCase):
    def test_two_random_transforms_of_one_recording_cannot_share_targets(self):
        calls = []

        def infer(x):
            calls.append(x.waveform().copy())
            return TopTargets.materialize([[0, 1]], [[abs(float(x.waveform()[0])), 0.1]])

        cache = TargetCache(infer, "frozen-teacher-and-processor")
        source = np.array([0.25, -0.5], dtype=np.float32)
        clean = AcousticInput.materialize(source, 16000)
        gain = AcousticInput.materialize(source * 2, 16000)
        noise = AcousticInput.materialize(source + [0.01, -0.03], 16000)
        self.assertNotEqual(cache.get(clean), cache.get(gain))
        self.assertNotEqual(cache.get(clean), cache.get(noise))
        self.assertIs(cache.get(clean), cache.get(AcousticInput.materialize(source, 16000)))
        self.assertEqual(len(calls), 3)

    def test_extent_rate_and_teacher_identity_are_part_of_target_identity(self):
        a = AcousticInput.materialize([0.1, 0.2, 0.3], 16000, 2)
        b = AcousticInput.materialize([0.1, 0.2, 0.3], 16000, 3)
        c = AcousticInput.materialize([0.1, 0.2, 0.3], 8000, 2)
        self.assertEqual(a.valid_mask().tolist(), [True, True, False])
        self.assertEqual(len(a.waveform()), 2)
        self.assertEqual(len({a.cache_key("one"), b.cache_key("one"), c.cache_key("one"), a.cache_key("two")}), 4)

    def test_source_and_returned_buffer_mutation_cannot_change_cached_input_or_targets(self):
        wave = np.array([0.2, 0.4], dtype=np.float32)
        x = AcousticInput.materialize(wave, 16000)
        key = x.cache_key("teacher")
        wave[:] = 0
        self.assertEqual(key, x.cache_key("teacher"))
        with self.assertRaises(ValueError):
            x.waveform()[0] = 1
        ids, p = np.array([[1, 2]]), np.array([[0.7, 0.2]])
        target = TopTargets.materialize(ids, p)
        old = target.probabilities
        p[:] = 0
        ids[:] = 9
        self.assertEqual(target.probabilities, old)
        self.assertEqual(np.frombuffer(target.ids, dtype="<i8").tolist(), [1, 2])

    def test_eviction_recomputes_without_reusing_another_observation(self):
        calls = []
        def infer(x):
            calls.append(x)
            return TopTargets.materialize([[1]], [[1.0]])
        cache = TargetCache(infer, "teacher", capacity=1)
        a, b = (AcousticInput.materialize([v], 16000) for v in (0.1, 0.2))
        cache.get(a)
        cache.get(b)
        cache.get(a)
        self.assertEqual(calls, [a, b, a])

    def test_scripted_help_train_takes_cannot_enter_transcript_refitting(self):
        from student_paired_kd import HeadExample
        targets = TopTargets.materialize([[1]], [[1.0]])
        with self.assertRaisesRegex(ValueError, "not performed transcripts"):
            HeadExample("training-speaker", "help-train", "wrong_word", AcousticInput.materialize([0.1], 16000),
                        None, None, targets, targets)


def training_functions():
    # Import the real data/loss functions without constructing a Modal image or
    # accessing credentials/volumes. No substitutes for their tensor arithmetic.
    path = Path(__file__).with_name("student_train.py")
    tree = ast.parse(path.read_text())
    names = {"ClipDataset", "collate", "losses", "pick", "reverb", "spec_augment"}
    tree.body = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in names]
    namespace = {"io": io, "SR": 16000}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import soundfile
        except ImportError:
            raise unittest.SkipTest("torch and soundfile required in the Modal train image")
        cls.torch, cls.sf = torch, soundfile

    def test_dataset_teacher_and_student_share_one_augmented_waveform(self):
        torch = self.torch
        ns = training_functions()
        buf = io.BytesIO()
        self.sf.write(buf, np.linspace(-0.3, 0.3, 16000), 16000, format="FLAC")
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        clips = dict(ref=["ب"], hyp=["ب"], flac=data, flac_off=[0, len(data)], top_off=[0, 5],
                     ids=np.zeros(5, dtype=np.uint8), probs=np.ones(5, dtype=np.float16))
        seen = []
        def fbank(wave, **kwargs):
            seen.append(wave.numpy().copy()[0])
            return torch.ones(20, 80)
        ta = types.SimpleNamespace(compliance=types.SimpleNamespace(kaldi=types.SimpleNamespace(fbank=fbank)))
        cfg = dict(rir_p=0, noise_p=0, snr_db=(5, 25), spec_aug=True)
        ds = ns["ClipDataset"](clips, cfg, True, {}, seed=17)
        # A nonempty augmentation bank enables the shipped random gain step.
        ds.aug = {"unused": True}
        with patch.dict(sys.modules, {"torchaudio": ta}):
            item = ds[0]
            again = ds[0]
        observation = item["observation"]
        np.testing.assert_array_equal(seen[0], observation.waveform())
        self.assertEqual(observation, again["observation"])
        teacher_seen = []
        cache = TargetCache(lambda x: (teacher_seen.append(x.waveform().copy()) or
                            TopTargets.materialize(np.zeros((5, 5)), np.ones((5, 5)))), "teacher")
        cache.get(observation)
        np.testing.assert_array_equal(seen[0], teacher_seen[0])
        # SpecAugment modifies features, not the shared waveform.
        self.assertTrue((item["feats"] == 0).any())
        source, _ = self.sf.read(io.BytesIO(data.tobytes()), dtype="float32")
        self.assertFalse(np.array_equal(source, observation.waveform()))
        self.assertTrue(item["acoustic_mismatch"])
        ds.train = False
        with patch.dict(sys.modules, {"torchaudio": ta}):
            self.assertFalse(ds[0]["acoustic_mismatch"])

    def test_paired_loss_uses_new_targets_and_keeps_control_batch_and_padding_unchanged(self):
        torch = self.torch
        ns = training_functions()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.randn(2, 4, 43))
            def forward(self, feats, lens):
                return self.logits.log_softmax(-1), torch.tensor([2, 4])
        model = Model()
        obs = [AcousticInput.materialize([0.1], 16000), AcousticInput.materialize([0.2], 16000)]
        items = [dict(i=i, observation=obs[i], acoustic_mismatch=True, feats=torch.zeros(4, 80), target=torch.tensor([2]),
                      top_ids=torch.zeros(3, 5, dtype=torch.long), top_p=torch.ones(3, 5)) for i in range(2)]
        batch = ns["collate"](items)
        original = batch["top_ids"].clone()
        cache = TargetCache(lambda x: TopTargets.materialize(np.full((3, 5), 2 if x == obs[0] else 3),
                                                            np.ones((3, 5))), "teacher")
        cfg = dict(kd_weight=2, kd_delay=0, ctc_weight=1, paired_kd=True)
        paired = ns["losses"](model, batch, cfg, "cpu", cache)
        lp = model.logits.log_softmax(-1)
        expected = -(lp[0, :2, 2].sum() + lp[1, :4, 3].sum()) / 6
        torch.testing.assert_close(paired["kd"], expected)
        gradient = torch.autograd.grad(paired["kd"], model.logits)[0]
        self.assertEqual(float(gradient[0, 2:].abs().sum()), 0)
        torch.testing.assert_close(batch["top_ids"], original)
        control = ns["losses"](model, batch, {**cfg, "paired_kd": False}, "cpu")
        self.assertNotEqual(float(control["kd"]), float(paired["kd"]))
        with self.assertRaisesRegex(ValueError, "requires the teacher"):
            ns["losses"](model, batch, cfg, "cpu")
        batch["acoustic_mismatch"] = [False, False]
        unchanged = ns["losses"](model, batch, cfg, "cpu")
        torch.testing.assert_close(unchanged["kd"], control["kd"])
        batch["acoustic_mismatch"] = [True, False]
        mixed = ns["losses"](model, batch, cfg, "cpu", cache)
        expected = -(lp[0, :2, 2].sum() + lp[1, :4, 0].sum()) / 6
        torch.testing.assert_close(mixed["kd"], expected)

    def test_identical_targets_refit_identically_and_reject_release_speakers(self):
        from student_paired_kd import HeadExample, refit_heads
        torch = self.torch
        torch.manual_seed(1)
        head = torch.nn.Linear(2, 3)
        target = TopTargets.materialize([[0, 1, 2]] * 5, [[0.5, 0.3, 0.2]] * 5)
        obs = AcousticInput.materialize([0.1], 16000)
        items = [HeadExample(s, "help-train", "none", obs, torch.randn(5, 2), torch.tensor([1, 2]),
                             target, target) for s in ("fit", "select")]
        before = {k: v.clone() for k, v in head.state_dict().items()}
        result = refit_heads(head, items[:1], items[1:], training_speakers={"fit", "select"},
                             held_out_speakers={"release"}, steps=3)
        self.assertEqual(result["current"]["history"], result["paired"]["history"])
        for k, v in result["current"]["head"].state_dict().items():
            torch.testing.assert_close(v, result["paired"]["head"].state_dict()[k], rtol=0, atol=0)
            torch.testing.assert_close(head.state_dict()[k], before[k], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "release-held-out"):
            refit_heads(head, items[:1], items[1:], training_speakers={"fit", "select"},
                        held_out_speakers={"select"}, steps=3)
        with self.assertRaisesRegex(ValueError, "speaker-disjoint"):
            refit_heads(head, items[:1], items[:1], training_speakers={"fit"}, held_out_speakers=set(), steps=3)


if __name__ == "__main__":
    unittest.main()

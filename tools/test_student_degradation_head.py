"""Synthetic checks; python3 -m unittest discover -s tools -p test_student_degradation_head.py.

Torch/ONNX checks run on the authorized worker. No acoustic training or inference
on real recordings runs locally. Standard-library and tiny waveform checks also
run without those optional dependencies.
"""

from collections import Counter
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal"))
from student_degradation_head import (LEVELS, acoustic_loss, authorized_none, course_orders,
                                     phone_degrade, selected_fit, speaker_partition, train, training_rows, waveform_views)
from student_frozen_head import Encoder, digest, export, output_frames, readout

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ONNX = HAS_TORCH and all(importlib.util.find_spec(m) for m in ("onnx", "onnxruntime"))


class ExposureTests(unittest.TestCase):
    def test_first_course_advances_but_control_shuffles_identical_multiset(self):
        orders = course_orders(7, passes=2)
        a, b = orders.values()
        self.assertEqual(len(a), 7 * len(LEVELS) * 2)
        self.assertEqual(Counter(a), Counter(b))
        self.assertEqual(set(Counter(a).values()), {2})
        self.assertEqual([level for _, level in a], sorted(level for _, level in a))
        self.assertNotEqual(a, b)
        self.assertEqual(orders, course_orders(7, passes=2))

    def test_later_courses_share_order_and_keep_original_pause_examples(self):
        a, b = course_orders(7, course=1).values()
        self.assertEqual(a, b)
        self.assertEqual(Counter(a), Counter(course_orders(7)["continuation"]))
        self.assertEqual(sum(level == 0 for _, level in a), 7)
        self.assertEqual(sum(level == len(LEVELS) - 1 for _, level in a), 7)

    def test_invalid_schedule(self):
        for args in (dict(recordings=0), dict(recordings=2, passes=0),
                     dict(recordings=2, levels=(0., 1., .5)), dict(recordings=2, levels=(0., 0., 1.))):
            with self.assertRaises(ValueError):
                course_orders(**args)

    def test_speaker_split_is_grouped_and_order_independent(self):
        speakers = ["a", "b", "c", "d", "e", "f"]
        split = speaker_partition(speakers)
        self.assertEqual(split, speaker_partition(speakers[::-1] + speakers))
        self.assertEqual(list(split.values()).count("validation"), 2)
        with self.assertRaises(ValueError):
            speaker_partition(["a", "a"])

    def test_error_and_repeat_prompts_are_never_performed_transcripts(self):
        row = dict(split="train", kind="none", speaker="learner", ayahs=["1:1"])
        self.assertTrue(authorized_none(row))
        for change in (dict(split="eval"), dict(kind="repeat"), dict(kind="vowel"), dict(excluded=True),
                       dict(events=[{}]), dict(non_errors=[{}]), dict(unsupported=[{}])):
            self.assertFalse(authorized_none({**row, **change}))

    def test_held_out_speaker_is_excluded_across_both_catalogs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            base = dict(split="train", kind="none", ayahs=["1:1", "1:2"], wav="clean.wav")
            labels = [{**base, "id": "clean", "speaker": "safe"},
                      {**base, "id": "leak", "speaker": "mixed", "wav": "leak.wav"}]
            (root / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
            corpus = [dict(id="eval", speaker="mixed", split="eval"),
                      dict(id="prompt", speaker="safe", split="train", audio="prompt.wav", transcript="ب"),
                      dict(id="performed", speaker="ordinary", split="train", audio="ordinary.wav",
                           transcript="بََم", authorization="transcribed-training")]
            catalog = root / "training.jsonl"
            catalog.write_text("\n".join(json.dumps(r) for r in corpus))
            refs = dict(ayahs=[dict(s=1, a=1, p="ببَ"), dict(s=1, a=2, p="َم")])
            rows = training_rows(root, refs, catalog)
            self.assertEqual([r["id"] for r in rows], ["help-none:clean", "transcribed:performed"])
            # Equal identities across a real pause retain separate CTC occurrences.
            self.assertEqual(rows[0]["target"], rows[1]["target"])
            labels[0]["wav"] = "../escape.wav"
            (root / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in labels))
            with self.assertRaises(ValueError):
                training_rows(root, refs, catalog)


class WaveformTests(unittest.TestCase):
    def test_original_and_degraded_endpoints_have_identical_clock(self):
        import numpy as np
        a = np.arange(800, dtype=np.float32) / 800
        b = a[::-1].copy()
        views = waveform_views(a, b)
        np.testing.assert_array_equal(views[0], a)
        np.testing.assert_array_equal(views[-1], b)
        np.testing.assert_array_equal(views[2], .5 * a + .5 * b)
        self.assertTrue(all(v.shape == a.shape and v.dtype == np.float32 for v in views))
        views[0][0] = 4
        self.assertEqual(a[0], 0)
        with self.assertRaises(ValueError):
            waveform_views(a, b[:-1])
        with self.assertRaises(ValueError):
            waveform_views(a, b, (0., .5))

    def test_degradation_matches_existing_chain_without_mutating_bank(self):
        import numpy as np
        from lab_sessions import degrade
        rng = np.random.default_rng(0)
        wave = (rng.normal(size=1600) * .04).astype(np.float32)
        rir = np.array([0., 1., .2, -.1], np.float32)
        noise = rng.normal(size=1600).astype(np.float32)
        before = [a.copy() for a in (wave, rir, noise)]
        actual = phone_degrade(wave, rir, [noise], 14.)
        np.testing.assert_array_equal(actual, degrade(wave, rir, [noise.copy()], 14.))
        self.assertEqual(len(actual), len(wave))
        self.assertLessEqual(float(abs(actual).max()), .890001)
        for a, b in zip((wave, rir, noise), before):
            np.testing.assert_array_equal(a, b)


@unittest.skipUnless(HAS_TORCH, "torch checks require the authorized worker")
class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_ctc_matches_enumeration_including_repeated_phone_occurrences(self):
        import torch
        from student import ctc_collapse
        head = torch.nn.Linear(3, 3, bias=False).double()
        with torch.no_grad():
            head.weight.copy_(torch.eye(3, dtype=torch.double))
        features = torch.tensor([[.2, .5, -.3], [.4, -.1, .2], [.2, .7, -.1]], dtype=torch.double)
        lp = head(features).log_softmax(-1)
        view = dict(features=features, log_probs=lp.detach())
        for target in ([1, 2], [1, 1]):
            mass = sum(torch.stack([lp[t, p] for t, p in enumerate(path)]).sum().exp()
                       for path in itertools.product(range(3), repeat=3) if ctc_collapse(path) == list(target))
            torch.testing.assert_close(acoustic_loss(head, view, target), -mass.log() / len(target))

    def test_preservation_includes_blank_pause_frames(self):
        import torch
        head = torch.nn.Linear(2, 3)
        f = torch.zeros(5, 2)
        ordinary = dict(features=f, log_probs=torch.tensor([[0., 4., 0.]]).expand(5, 3).log_softmax(-1))
        pauses = {**ordinary, "log_probs": torch.tensor([[8., 0., 0.]]).expand(5, 3).log_softmax(-1)}
        self.assertNotEqual(acoustic_loss(head, ordinary, [1, 2]).item(), acoustic_loss(head, pauses, [1, 2]).item())
        acoustic_loss(head, pauses, [1, 2]).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertFalse(torch.isfinite(acoustic_loss(head, {**pauses, "features": f[:1], "log_probs": pauses["log_probs"][:1]}, [1, 1])))

    def test_same_example_has_same_loss_and_gradient_in_either_arm(self):
        import copy
        import torch
        torch.manual_seed(11)
        continuation = torch.nn.Linear(2, 3)
        shuffled = copy.deepcopy(continuation)
        view = dict(features=torch.randn(5, 2), log_probs=torch.randn(5, 3).log_softmax(-1))
        original = {k: v.clone() for k, v in view.items()}
        a, b = [acoustic_loss(h, view, [1, 2]) for h in (continuation, shuffled)]
        torch.testing.assert_close(a, b)
        a.backward()
        b.backward()
        for x, y in zip(continuation.parameters(), shuffled.parameters()):
            torch.testing.assert_close(x.grad, y.grad)
        for k in view:
            torch.testing.assert_close(view[k], original[k])

    def test_training_selects_common_complete_course(self):
        import argparse
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            torch.save(dict(weight=np.zeros((3, 2), np.float32), bias=np.zeros(3, np.float32)), root / "initial.pt")
            rows, speakers = [], {"a": "train", "b": "validation"}
            for i, (speaker, part) in enumerate(speakers.items()):
                views = [dict(features=np.full((5, 2), level, np.float32),
                              log_probs=torch.randn(5, 3).log_softmax(-1).numpy()) for level in LEVELS]
                file = f"record-{i}.pt"
                torch.save(dict(target=[1, 2], views=views), root / file)
                rows.append(dict(file=file, speaker=speaker, partition=part, cache_sha256=digest(root / file)))
            index = dict(complete=True, recordings=rows, speakers=speakers, source_sha256="fixture", levels=LEVELS,
                         started_at=time.time(), hours=1, initial_sha256=digest(root / "initial.pt"))
            (root / "index.json").write_text(json.dumps(index))
            args = argparse.Namespace(cache=str(root), out=str(root / "fit"), threads=1, seed=0,
                                      lr=1e-4, courses=2, stage_passes=2)
            train(args)
            fits = [torch.load(selected_fit(root / "fit", a), weights_only=False) for a in ("continuation", "shuffled")]
            self.assertEqual(fits[0]["selected_updates"], fits[1]["selected_updates"])
            self.assertEqual(fits[0]["selected_course"], fits[1]["selected_course"])
            history = json.loads((root / "fit" / "training.json").read_text())["history"]
            self.assertEqual([h["updates"] for h in history], [10, 20])
            index["speakers"]["b"] = "train"
            (root / "index.json").write_text(json.dumps(index))
            args.out = str(root / "invalid")
            with self.assertRaises(ValueError):
                train(args)
            selection_path = root / "fit" / "selection.json"
            selection = json.loads(selection_path.read_text())
            selection["selected_updates"] += 1
            selection_path.write_text(json.dumps(selection))
            with self.assertRaises(ValueError):
                selected_fit(root / "fit", "continuation")


@unittest.skipUnless(HAS_ONNX, "ONNX/torch checks require the authorized worker")
class ExportTests(unittest.TestCase):
    def test_quantized_head_export_keeps_encoder_state_and_public_clock(self):
        import numpy as np
        import onnx
        import torch
        from onnx import helper as h, numpy_helper as nh, TensorProto as t
        arrays = dict(starts=np.array([0, 0], np.int64), ends=np.array([32, 2], np.int64),
                      axes=np.array([1, 2], np.int64), steps=np.array([4, 1], np.int64),
                      head_quantized=np.array([[2, -1, 3], [1, 2, -2]], np.int8),
                      head_zero_point=np.array(0, np.int8), head_scale=np.array(.25, np.float32),
                      bias=np.array([.1, -.2, .3], np.float32), one=np.array([1], np.int64),
                      float_one=np.array([1.], np.float32))
        nodes = [h.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["hidden"]),
                 h.make_node("DynamicQuantizeLinear", ["hidden"], ["q", "scale", "zero"]),
                 h.make_node("MatMulInteger", ["q", "head_quantized", "zero", "head_zero_point"], ["integer"]),
                 h.make_node("Cast", ["integer"], ["float"], to=t.FLOAT),
                 h.make_node("Mul", ["scale", "head_scale"], ["scales"]),
                 h.make_node("Mul", ["float", "scales"], ["linear"]),
                 h.make_node("Add", ["linear", "bias"], ["logits"]),
                 h.make_node("LogSoftmax", ["logits"], ["log_probs"], axis=-1),
                 h.make_node("Add", ["processed", "one"], ["new_processed"]),
                 h.make_node("Add", ["cache", "float_one"], ["new_cache"])]
        graph = h.make_graph(nodes, "fixture", [h.make_tensor_value_info("x", t.FLOAT, [1, 45, 80]),
                            h.make_tensor_value_info("processed", t.INT64, [1]),
                            h.make_tensor_value_info("cache", t.FLOAT, [1])],
                            [h.make_tensor_value_info("log_probs", t.FLOAT, [1, 8, 3]),
                             h.make_tensor_value_info("new_processed", t.INT64, [1]),
                             h.make_tensor_value_info("new_cache", t.FLOAT, [1])],
                            [nh.from_array(v, k) for k, v in arrays.items()])
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
        model.ir_version = 9
        feature, weight, bias, _ = readout(model.graph, 3)
        self.assertEqual(feature, "hidden")
        np.testing.assert_array_equal(weight, arrays["head_quantized"].T * .25)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            onnx.save(model, root / "model.onnx")
            m = dict(file="model.onnx", sha256=digest(root / "model.onnx"), tokens="ab", frameMs=40,
                     window=45, shift=32, chunk=16, fbank=dict(mels=80, padValue=-23.0259),
                     states=[dict(name="processed", shape=[1], dtype="int64"), dict(name="cache", shape=[1], dtype="float32")])
            (root / "manifest.json").write_text(json.dumps(m))
            f = np.arange(100 * 80, dtype=np.float32).reshape(100, 80) / 8000
            got = Encoder(root, 1).stream(f)
            np.testing.assert_array_equal(got["features"], f[:output_frames(100) * 4:4, :2])
            np.testing.assert_array_equal(got["times_ms"], np.arange(output_frames(100)) * 40)
            checkpoint = dict(head=dict(weight=torch.from_numpy(weight + .02), bias=torch.from_numpy(bias)),
                              source_sha256=m["sha256"], selected_updates=10, selected_course=1,
                              arm="continuation", cache_sha256="test")
            torch.save(checkpoint, root / "head.pt")
            # Missing declared assets must fail before a usable bundle is created.
            with self.assertRaises(ValueError):
                export(root, root / "head.pt", root / "missing", 1)
            self.assertFalse((root / "missing").exists())
            (root / "teacher.onnx").write_bytes(b"unchanged auxiliary")
            (root / "teacher.onnx.data").write_bytes(b"unchanged external weights")
            export(root, root / "head.pt", root / "bundle", 1)
            bundle = root / "bundle"
            new = json.loads((bundle / "manifest.json").read_text())
            for key in ("tokens", "frameMs", "window", "shift", "chunk", "fbank", "states"):
                self.assertEqual(new[key], m[key])
            self.assertEqual(new["sha256"], digest(bundle / "model.onnx"))
            for name in ("teacher.onnx", "teacher.onnx.data"):
                self.assertEqual(digest(bundle / name), digest(root / name))
            verification = json.loads((bundle / "verification.json").read_text())
            self.assertEqual(verification["recurrent_states"], "exact")
            unchanged = {i.name: nh.to_array(i) for i in onnx.load(bundle / "model.onnx").graph.initializer}
            for name, value in arrays.items():
                np.testing.assert_array_equal(unchanged[name], value)


if __name__ == "__main__":
    unittest.main()

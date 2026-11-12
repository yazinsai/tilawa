"""Small synthetic checks; no recordings or practice-specific identities."""
import copy
import itertools
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from student_locality import (LowerCache, UpperRefit, directional_response,
    grounding_loss, occurrence_masks, occurrence_posteriors, occurrence_score,
    paired_refit, probe_directions, speaker_partition, transcript_score)
from student_locality_train import bundle_assets, contained, enroll


torch.set_num_threads(1)


def enumerate_states(lp, target):
    labels = [0]
    for phone in target:
        labels.extend((phone, 0))
    paths = []
    for path in itertools.product(range(len(labels)), repeat=len(lp)):
        if path[0] not in (0, 1) or path[-1] not in (len(labels) - 2, len(labels) - 1):
            continue
        valid = True
        for a, b in zip(path, path[1:]):
            if b == a or b == a + 1:
                continue
            if b == a + 2 and labels[b] != 0 and labels[b] != labels[a]:
                continue
            valid = False
            break
        if valid:
            weight = math.exp(sum(float(lp[t, labels[s]]) for t, s in enumerate(path)))
            paths.append((path, weight))
    z = sum(w for _, w in paths)
    gamma = np.zeros((len(lp), len(labels)))
    for path, weight in paths:
        for t, state in enumerate(path):
            gamma[t, state] += weight / z
    return math.log(z), gamma


class OccurrenceTests(unittest.TestCase):
    def test_forward_backward_matches_all_paths(self):
        torch.manual_seed(2)
        lp = torch.randn(4, 3, dtype=torch.double).log_softmax(-1)
        for target in ([1], [1, 2], [1, 1], [1, 2, 1]):
            with self.subTest(target=target):
                y = torch.tensor(target)
                z, gamma = occurrence_posteriors(lp, y)
                want, occupancy = enumerate_states(lp, target)
                self.assertAlmostEqual(z, want, places=10)
                np.testing.assert_allclose(gamma, occupancy, atol=1e-10)
                self.assertAlmostEqual(float(transcript_score(lp, y)), want, places=10)

    def test_repeated_phones_are_distinct_occurrences(self):
        lp = torch.zeros(3, 2, dtype=torch.double).log_softmax(-1)
        _, gamma = occurrence_posteriors(lp, torch.tensor([1, 1]))
        torch.testing.assert_close(gamma, torch.tensor([[0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.]], dtype=torch.double))

    def test_soft_neighbourhood_includes_interior_blanks(self):
        gamma = torch.full((4, 11), 1. / 11)
        own, local = occurrence_masks(gamma, 2, 7)
        torch.testing.assert_close(own, torch.full((7,), 1. / 11))
        torch.testing.assert_close(local, torch.full((7,), 5. / 11))
        self.assertTrue((local < 1).all())

    def test_edge_neighbourhood_excludes_outer_silence(self):
        gamma = torch.full((2, 7), 1. / 7)
        _, local = occurrence_masks(gamma, 0, 4)
        torch.testing.assert_close(local, torch.full((4,), 3. / 7))

    def test_invalid_transcripts_and_clock_fail(self):
        lp = torch.zeros(2, 3).log_softmax(-1)
        for y in ([1, 1], [], [0], [3]):
            with self.assertRaises(ValueError):
                occurrence_posteriors(lp, torch.tensor(y, dtype=torch.long))
        with self.assertRaises(ValueError):
            occurrence_masks(torch.zeros(2, 5), 0, 7)

    def test_deleted_occurrence_score_and_gradient(self):
        logits = torch.tensor([[0., 2., -1.], [1., 0., 2.], [0., 1., 2.]], requires_grad=True)
        lp = logits.log_softmax(-1)
        y = torch.tensor([1, 2])
        score = occurrence_score(lp, y, 0)
        full, _ = enumerate_states(lp.detach(), [1, 2])
        deleted, _ = enumerate_states(lp.detach(), [2])
        self.assertAlmostEqual(float(score.detach()), full - deleted, places=10)
        score.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_single_occurrence_can_be_deleted_to_empty(self):
        lp = torch.tensor([[.2, .8], [.6, .4]], dtype=torch.double).log()
        score = occurrence_score(lp, torch.tensor([1]), 0)
        self.assertAlmostEqual(float(score), math.log(.88) - math.log(.12), places=10)


class ContextPhone(nn.Module):
    def __init__(self):
        super().__init__()
        self.own = nn.Parameter(torch.tensor(.8, dtype=torch.double))
        self.remote = nn.Parameter(torch.tensor(.6, dtype=torch.double))

    def forward(self, cache, frames=None):
        x = (cache.frames if frames is None else frames)[:, 0, 0]
        phone = self.own * x + self.remote * x.roll(2)
        return torch.stack((torch.zeros_like(phone), phone), -1).log_softmax(-1)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.model = ContextPhone()
        self.cache = LowerCache(torch.tensor([.1, .8, .4, -.3], dtype=torch.double)[:, None, None], None, 16)
        self.target = torch.tensor([1])

    def test_finite_difference_matches_directional_autograd(self):
        x = self.cache.frames.clone().requires_grad_()
        direction = torch.tensor([1., 0., -1., 0.], dtype=torch.double)[:, None, None]
        score = occurrence_score(self.model(self.cache, x), self.target, 0)
        want = (torch.autograd.grad(score, x)[0] * direction).sum()
        got = directional_response(self.model, self.cache, self.target, 0, direction, epsilon=1e-4)
        torch.testing.assert_close(got, want, atol=1e-7, rtol=1e-7)
        got.backward()
        self.assertTrue(torch.isfinite(self.model.remote.grad))

    def test_soft_probe_masks_keep_own_and_remote_separate(self):
        own = torch.tensor([0., 1., .25, 0.], dtype=torch.double)
        local = torch.tensor([0., 1., 1., .5], dtype=torch.double)
        a, b = probe_directions(self.cache, own, local, 9)
        c, d = probe_directions(self.cache, own, local, 9)
        torch.testing.assert_close(a, c)
        torch.testing.assert_close(b, d)
        self.assertEqual(float(a[0]), 0.)
        self.assertEqual(float(b[1]), 0.)
        self.assertEqual(float(b[2]), 0.)
        self.assertGreater(float(b[3].abs()), 0.)
        self.assertFalse(a.requires_grad)

    def test_floor_penalizes_flattening_own_response(self):
        anchor = torch.tensor(2.)
        self.assertEqual(float(grounding_loss(torch.tensor(2.), torch.tensor(0.), anchor)), 0.)
        self.assertEqual(float(grounding_loss(torch.tensor(0.), torch.tensor(0.), anchor)), 1.)
        own = torch.tensor(1., requires_grad=True)
        remote = torch.tensor(1., requires_grad=True)
        grounding_loss(own, remote, anchor).backward()
        self.assertLess(float(own.grad), 0.)
        self.assertGreater(float(remote.grad), 0.)

    def test_optimization_reduces_remote_dependence_retaining_own_response(self):
        class SensitivePhone(nn.Module):
            def __init__(self):
                super().__init__()
                self.local = nn.Parameter(torch.tensor(1., dtype=torch.double))
                self.remote = nn.Parameter(torch.tensor(.5, dtype=torch.double))

            def forward(self, cache, frames=None):
                x = (cache.frames if frames is None else frames)[:, 0, 0]
                logit = self.local * x[0] + self.remote * x[3]
                # A single scored occurrence, with a following blank frame.
                logits = torch.stack((torch.stack((logit * 0, logit)),
                                      torch.stack((logit * 0 + 10, logit * 0))))
                return logits.log_softmax(-1)

        model = SensitivePhone()
        cache = LowerCache(torch.tensor([.2, 0., 0., .1], dtype=torch.double)[:, None, None], None, 16)
        own = torch.tensor([1., 0., 0., 0.], dtype=torch.double)[:, None, None]
        remote = torch.tensor([0., 0., 0., 1.], dtype=torch.double)[:, None, None]
        target = torch.tensor([1])
        anchor = directional_response(model, cache, target, 0, own).detach()
        before = directional_response(model, cache, target, 0, remote).detach().abs()
        optimizer = torch.optim.SGD(model.parameters(), lr=.02)
        for _ in range(20):
            optimizer.zero_grad()
            a = directional_response(model, cache, target, 0, own)
            b = directional_response(model, cache, target, 0, remote)
            grounding_loss(a, b, anchor).backward()
            optimizer.step()
        after = directional_response(model, cache, target, 0, remote).detach().abs()
        retained = directional_response(model, cache, target, 0, own).detach().abs()
        self.assertLess(float(after), float(before) * .7)
        self.assertGreaterEqual(float(retained), float(anchor.abs()) * .95)


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x, **kwargs):
        return x + self.linear(x).tanh()


class Pool(nn.Module):
    def forward(self, x):
        if len(x) % 2:
            x = torch.cat((x, x[-1:]))
        return (x[::2] + x[1::2]) / 2


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([Block(4) for _ in range(4)])
        self.encoder_dim = (4,) * 4
        self.output_downsampling_factor = 2
        self.downsample_output = Pool()

    def forward(self, x, lengths):
        for block in self.encoders:
            x = block(x, chunk_size=16, feature_mask=1., attn_mask=None)
        return self.downsample_output(x), (lengths + 1) // 2


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_embed = nn.Linear(3, 4)
        self.encoder = Encoder()
        self.ctc_output = nn.Linear(4, 3)

    def streaming_eval(self, chunk, left):
        pass

    def forward(self, features, lengths):
        x = self.encoder_embed(features).permute(1, 0, 2)
        x, lengths = self.encoder(x, lengths)
        return self.ctc_output(x.permute(1, 0, 2)).log_softmax(-1), lengths


class RefitTests(unittest.TestCase):
    def test_cached_upper_matches_original_and_prefix_stays_frozen(self):
        torch.manual_seed(4)
        model = Student()
        refit = UpperRefit(model)
        features = torch.randn(7, 3)
        original, lengths = model(features[None], torch.tensor([len(features)]))
        cache = refit.cache(features)
        got = refit(cache)
        torch.testing.assert_close(got, original[0, :lengths[0]])
        self.assertFalse(cache.frames.requires_grad)
        (-transcript_score(got, torch.tensor([1, 2]))).backward()
        for name, p in model.named_parameters():
            upper = name.startswith(("encoder.encoders.2.", "encoder.encoders.3.", "ctc_output."))
            self.assertEqual(p.requires_grad, upper, name)
            self.assertEqual(p.grad is not None, upper, name)

    def test_speaker_groups_and_exclusions(self):
        speakers = ["speaker-" + str(i) for i in range(10)]
        groups = speaker_partition(speakers)
        self.assertEqual(groups, speaker_partition(list(reversed(speakers)) + speakers))
        self.assertEqual(list(groups.values()).count("selection"), 2)
        with self.assertRaises(ValueError):
            speaker_partition(speakers, [speakers[0]])
        with self.assertRaises(ValueError):
            speaker_partition(speakers[:4])

    def test_matched_updates_do_not_consult_selection_probes(self):
        initial = ContextPhone()
        speakers = ["speaker-" + str(i) for i in range(5)]
        groups = speaker_partition(speakers)
        rows = []
        for speaker in speakers:
            cache = LowerCache(torch.tensor([.1, .8, .4, -.3], dtype=torch.double)[:, None, None], None, 16)
            y = torch.tensor([1])
            own = torch.ones_like(cache.frames)
            remote = torch.ones_like(cache.frames) * .2
            with torch.no_grad():
                shipped = initial(cache)
                anchor = directional_response(initial, cache, y, 0, own)
            probes = [dict(occurrence=0, own=own, remote=remote, anchor=anchor)]
            if groups[speaker] == "selection":
                probes = None  # Selection must use only ordinary transcript/preservation.
            rows.append(dict(speaker=speaker, views=[dict(cache=cache, target=y, shipped=shipped, probes=probes)]))
        before = copy.deepcopy(initial.state_dict())
        arms, report = paired_refit(initial, rows, groups, epochs=2, lr=.001)
        self.assertEqual(report["trainingUpdates"], 8)
        self.assertEqual(report["selectedUpdates"], report["selectedEpoch"] * 4)
        for name, tensor in initial.state_dict().items():
            torch.testing.assert_close(tensor, before[name])
        self.assertEqual(set(arms), {"ordinary", "grounded"})
        self.assertTrue(all(math.isfinite(v) for h in report["history"] for v in h["selection"].values()))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.none = dict(speaker="learner", split="train", kind="none", scored=True,
                         events=[], non_errors=[], unsupported=[], ayahs=["text"], wav="take.wav")
        self.transcript = dict(speaker="reader", split="train", authorized=True,
                               verified=True, phones="b", wav="speech.wav", sha256="checksum")

    def test_mistake_scripts_and_repeats_never_become_transcripts(self):
        for change in (dict(kind="skipped_word"), dict(events=[{"kind": "wrong_word"}]),
                       dict(non_errors=[{"kind": "repeat"}]), dict(unsupported=["tajweed"]),
                       dict(scored=False), dict(excluded=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                enroll([dict(self.none, **change)], [self.transcript], [], {"text": "b"})

    def test_holdout_in_other_corpus_excludes_entire_speaker(self):
        train = dict(self.transcript, speaker="learner")
        heldout = dict(self.transcript, speaker="learner", split="test")
        with self.assertRaises(ValueError):
            enroll([self.none], [train, heldout, self.transcript], [], {"text": "b"})
        with self.assertRaises(ValueError):
            enroll([self.none], [self.transcript], ["learner"], {"text": "b"})

    def test_authorization_and_verification_required(self):
        for change in (dict(authorized=False), dict(verified=False), dict(sha256="")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                enroll([self.none], [dict(self.transcript, **change)], [], {"text": "b"})
        rows, excluded = enroll([self.none], [self.transcript], [], {"text": "b"})
        self.assertEqual({r["source"] for r in rows}, {"help-none", "transcribed"})
        self.assertEqual(excluded, [])

    def test_enrolled_audio_cannot_escape_corpus(self):
        with self.assertRaises(ValueError):
            contained(Path.cwd(), "../unrelated.wav")


class AssetTests(unittest.TestCase):
    def test_missing_and_truncated_external_teacher_assets_fail(self):
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        scratch = Path.cwd() / ".scratch/student"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            root = Path(name)
            with self.assertRaises(ValueError):
                bundle_assets(root)
            model = helper.make_model(helper.make_graph(
                [helper.make_node("Add", ["x", "w"], ["y"])], "teacher",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
                [numpy_helper.from_array(np.ones((1, 2), dtype=np.float32), "w")]))
            onnx.save_model(model, str(root / "teacher.onnx"), save_as_external_data=True,
                            all_tensors_to_one_file=True, location="teacher.onnx.data", size_threshold=0)
            assets = bundle_assets(root)
            self.assertEqual(set(assets), {"teacher.onnx", "teacher.onnx.data"})
            (root / "teacher.onnx.data").write_bytes(b"x")
            with self.assertRaises(ValueError):
                bundle_assets(root)


if __name__ == "__main__":
    unittest.main()

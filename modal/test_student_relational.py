"""Tiny numerical oracles for occurrence relationships; no audio or lab data."""
import ast
import copy
import itertools
from pathlib import Path
import types
import unittest

import torch

from student_relational import (batch_objective, comparison_mask, deterministic_features,
                                distillation_loss, occurrence_weights, pool_occurrences)
from student_relational_train import validate_help_row, validate_recipe

torch.set_num_threads(1)


def enumerate_weights(lp, transcript):
    """Independent exhaustive CTC *state* paths, including blank placements."""
    states = [0]
    for phone in transcript:
        states.extend([phone, 0])
    paths = []
    def visit(path):
        if len(path) == len(lp):
            if path[-1] >= len(states) - 2:
                paths.append(path)
            return
        for dest in (path[-1], path[-1] + 1, path[-1] + 2):
            if dest >= len(states):
                continue
            if dest == path[-1] + 2 and (states[dest] == 0 or states[dest] == states[path[-1]]):
                continue
            visit(path + [dest])
    for start in (0, 1):
        visit([start])
    result = torch.zeros(len(transcript), len(lp), dtype=lp.dtype)
    normalizer = 0.0
    for path in paths:
        mass = torch.exp(sum(lp[t, states[s]] for t, s in enumerate(path)))
        normalizer += mass
        for t, s in enumerate(path):
            if s % 2:
                result[s // 2, t] += mass
    result /= normalizer
    return result / result.sum(1, keepdim=True)


class OccurrenceTest(unittest.TestCase):
    def test_forward_backward_matches_exhaustive_paths(self):
        torch.manual_seed(5)
        for transcript in ([1], [1, 2], [1, 1], [1, 2, 1]):
            lp = torch.randn(5, 3, dtype=torch.double).log_softmax(1)
            torch.testing.assert_close(occurrence_weights(lp, transcript), enumerate_weights(lp, transcript))

    def test_repeated_phones_remain_distinct(self):
        lp = torch.full((3, 2), -torch.inf, dtype=torch.double)
        lp[0, 1] = lp[1, 0] = lp[2, 1] = 0
        torch.testing.assert_close(occurrence_weights(lp, [1, 1]), torch.tensor([[1., 0., 0.], [0., 0., 1.]], dtype=torch.double))
        with self.assertRaisesRegex(ValueError, "no finite complete"):
            occurrence_weights(lp[:2], [1, 1])

    def test_pooling_has_unit_mass_and_gradients_only_to_features(self):
        lp = torch.randn(5, 3, dtype=torch.double, requires_grad=True).log_softmax(1)
        feats = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
        weights = occurrence_weights(lp, [1, 2])
        torch.testing.assert_close(weights.sum(1), torch.ones(2, dtype=torch.double))
        value = pool_occurrences(feats, lp, [1, 2]).sum()
        gf, glp = torch.autograd.grad(value, (feats, lp), allow_unused=True)
        torch.testing.assert_close(gf, weights.sum(0)[:, None].expand(-1, 4))
        self.assertIsNone(glp)

    def test_long_held_phone_is_one_occurrence(self):
        lp = torch.tensor([[-torch.inf, 0.]] * 8)
        torch.testing.assert_close(occurrence_weights(lp, [1]), torch.full((1, 8), 1 / 8))

    def test_padding_outside_valid_extent_cannot_affect_pool(self):
        lp = torch.randn(4, 3).log_softmax(1)
        feats = torch.randn(4, 2)
        joined = torch.cat((feats, torch.full((2, 2), 1e20)))
        torch.testing.assert_close(pool_occurrences(joined[:4], lp, [1, 2]), pool_occurrences(feats, lp, [1, 2]))


class RelationTest(unittest.TestCase):
    def setUp(self):
        self.phones = [1, 2, 1, 2]
        self.records = ["a", "a", "b", "b"]
        self.teacher = torch.tensor([[1., 0.], [0., 1.], [.9, .1], [.1, .9]], dtype=torch.double)

    def loss(self, student, teacher=None, mode="relational", projection=None):
        return distillation_loss(student, self.teacher if teacher is None else teacher,
                                 self.phones, self.records, mode, .2, projection)

    def test_excludes_every_same_recording_pair(self):
        mask, anchors = comparison_mask(self.phones, self.records, "cpu")
        self.assertTrue(anchors.all())
        self.assertEqual(mask.tolist(), [[False, False, True, True]] * 2 + [[True, True, False, False]] * 2)
        _, counts = self.loss(self.teacher)
        self.assertEqual(counts, dict(occurrences=4, anchors=4, comparisons=8, recordings=2))

    def test_teacher_geometry_preserved_across_rotation_and_scale(self):
        rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.double)
        student = (self.teacher @ rotation) * torch.tensor([[3.], [.5], [2.], [1.]])
        loss, _ = self.loss(student)
        self.assertAlmostEqual(float(loss), 0., places=12)
        # Relationship transfer also permits different feature dimensions.
        wide = torch.cat((student, torch.zeros(4, 2, dtype=torch.double)), 1)
        self.assertAlmostEqual(float(self.loss(wide)[0]), 0., places=12)

    def test_confusable_negatives_produce_discriminating_gradient(self):
        student = torch.tensor([[1., .2], [.2, 1.], [.2, 1.], [1., .2]], dtype=torch.double, requires_grad=True)
        teacher = self.teacher.clone().requires_grad_(True)
        loss, _ = self.loss(student, teacher)
        gs, gt = torch.autograd.grad(loss, (student, teacher), allow_unused=True)
        self.assertIsNone(gt)
        self.assertGreater(float(loss.detach()), 1.)
        self.assertTrue(torch.isfinite(gs).all())
        self.assertGreater(float(gs.norm()), 0.)
        self.assertLess(float(self.loss(student.detach() - .01 * gs)[0]), float(loss.detach()))
        torch.autograd.gradcheck(lambda s: self.loss(s)[0], (student,))

    def test_hint_control_uses_same_eligible_occurrences(self):
        projection = torch.nn.Linear(2, 2, bias=False, dtype=torch.double)
        with torch.no_grad():
            projection.weight.copy_(torch.eye(2, dtype=torch.double))
        hint, counts = self.loss(self.teacher, mode="hint", projection=projection)
        _, relational_counts = self.loss(self.teacher)
        self.assertEqual(counts, relational_counts)
        self.assertEqual(float(hint.detach()), 0.)
        with self.assertRaisesRegex(ValueError, "projection"):
            self.loss(self.teacher, mode="hint")

    def test_sparse_anchors_require_both_positive_and_negative(self):
        with self.assertRaisesRegex(ValueError, "no cross-recording anchors"):
            distillation_loss(self.teacher, self.teacher, [1] * 4, self.records, "relational", .2)
        with self.assertRaisesRegex(ValueError, "no cross-recording anchors"):
            distillation_loss(self.teacher, self.teacher, self.phones, ["a"] * 4, "relational", .2)
        mask, anchors = comparison_mask([1, 2, 1], ["a", "a", "b"], "cpu")
        self.assertEqual(anchors.tolist(), [False, False, True])
        self.assertEqual(mask[2].tolist(), [True, True, False])

    def test_reordering_batch_preserves_loss(self):
        student = self.teacher.flip(0)
        expected, _ = self.loss(student)
        order = [3, 0, 2, 1]
        actual, _ = distillation_loss(student[order], self.teacher[order], [self.phones[i] for i in order],
                                     [self.records[i] for i in order], "relational", .2)
        torch.testing.assert_close(expected, actual)

    def test_invalid_features_and_temperature_refused(self):
        for temperature in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                distillation_loss(self.teacher, self.teacher, self.phones, self.records, "relational", temperature)
        with self.assertRaises(ValueError):
            self.loss(torch.zeros_like(self.teacher))


class FakeStudent(torch.nn.Module):
    def __init__(self, vocab=3):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4, dtype=torch.double)
        self.ctc_output = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(4, vocab, dtype=torch.double))
    def forward_features(self, feats, lens):
        return self.encoder(feats), lens
    def forward(self, feats, lens):
        f, l = self.forward_features(feats, lens)
        return self.ctc_output(f).log_softmax(-1), l


class IntegrationTest(unittest.TestCase):
    def test_acoustic_kd_matches_original_top_five_loss(self):
        torch.manual_seed(9)
        model = FakeStudent(vocab=7).eval()
        anchor = copy.deepcopy(model).requires_grad_(False)
        batch = {"keys": ["a", "b"], "recordings": ["parent-a", "parent-b"],
                 "feats": torch.randn(2, 4, 3, dtype=torch.double),
                 "feat_lens": torch.tensor([3, 4]), "targets": torch.tensor([1, 2, 1, 2]),
                 "target_lens": torch.tensor([2, 2])}
        rows = [{"key": key, "log_probs": torch.randn(3, 7).log_softmax(1),
                 "features": torch.randn(3, 5, dtype=torch.double)} for key in batch["keys"]]
        _, terms, _ = batch_objective(model, anchor, rows, batch, "relational", .2,
                                      dict(ctc=1., kd=1., preservation=1., distillation=1.))
        # Execute the existing production loss without importing audio helpers.
        source = Path(__file__).parents[1] / "tools" / "student.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "soft_target_loss")
        ns = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)
        top_p, top_ids = torch.stack([r["log_probs"].exp() for r in rows]).topk(5, dim=-1)
        indices = torch.arange(4).clamp(max=2)
        lp, lens = model(batch["feats"], batch["feat_lens"])
        valid = torch.arange(4)[None] < lens[:, None]
        expected = ns["soft_target_loss"](lp.float(), top_ids[:, indices], top_p[:, indices], valid)
        torch.testing.assert_close(terms["kd"], expected)

    def test_joint_objective_preserves_masking_and_updates_encoder(self):
        torch.manual_seed(1)
        model = FakeStudent()
        anchor = copy.deepcopy(model).requires_grad_(False)
        batch = {"keys": ["a", "b"], "recordings": ["parent-a", "parent-b"],
                 "feats": torch.randn(2, 6, 3, dtype=torch.double),
                 "feat_lens": torch.tensor([4, 6]), "targets": torch.tensor([1, 2, 1, 2]),
                 "target_lens": torch.tensor([2, 2])}
        rows = [{"key": key, "log_probs": torch.randn(n, 3, dtype=torch.double).log_softmax(1),
                 "features": torch.randn(n, 5, dtype=torch.double)} for key, n in zip(batch["keys"], [5, 6])]
        weights = dict(ctc=1., kd=.2, preservation=2., distillation=.5)
        loss, terms, counts = batch_objective(model, anchor, rows, batch, "relational", .2, weights)
        self.assertAlmostEqual(float(terms["preservation"].detach()), 0., places=6)
        loss.backward()
        self.assertGreater(float(model.encoder.weight.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in anchor.parameters()))
        batch["feats"][0, 4:] = 1e10
        second, second_terms, second_counts = batch_objective(model, anchor, rows, batch, "relational", .2, weights)
        torch.testing.assert_close(loss, second)
        self.assertEqual(counts, second_counts)
        projection = torch.nn.Linear(4, 5, dtype=torch.double)
        hint, hint_terms, hint_counts = batch_objective(model, anchor, rows, batch, "hint", .2, weights, projection)
        for name in ("ctc", "kd", "preservation"):
            torch.testing.assert_close(hint_terms[name], second_terms[name])
        self.assertEqual(hint_counts, counts)
        hint.backward()
        self.assertGreater(float(projection.weight.grad.norm()), 0.)
        batch["recordings"] = ["shared-parent", "shared-parent"]
        with self.assertRaisesRegex(ValueError, "no cross-recording anchors"):
            batch_objective(model, anchor, rows, batch, "relational", .2, weights)
        batch["recordings"] = ["parent-a", "parent-b"]
        rows.reverse()
        with self.assertRaisesRegex(ValueError, "another recording"):
            batch_objective(model, anchor, rows, batch, "relational", .2, weights)

    def test_mode_flags_restored_after_exception(self):
        model = FakeStudent().train()
        model.encoder.eval()
        expected = [m.training for m in model.modules()]
        with self.assertRaises(RuntimeError):
            with deterministic_features(model):
                self.assertFalse(any(m.training for m in model.modules()))
                raise RuntimeError()
        self.assertEqual([m.training for m in model.modules()], expected)

    def test_production_forward_remains_original_phone_head(self):
        # Execute both actual methods with tiny embedding/encoder doubles.
        # Only the icefall padding helper is replaced, not the operator sequence.
        tree = ast.parse(Path(__file__).with_name("student_train.py").read_text())
        build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_model")
        student = next(n for n in build.body if isinstance(n, ast.ClassDef))
        forward = next(n for n in student.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        features = next(n for n in student.body if isinstance(n, ast.FunctionDef) and n.name == "forward_features")
        features.body = [n for n in features.body if not isinstance(n, ast.ImportFrom)]
        def make_pad_mask(lens):
            return torch.arange(int(lens.max()))[None] >= lens[:, None]
        ns = {"make_pad_mask": make_pad_mask}
        exec(compile(ast.Module(body=[features, forward], type_ignores=[]), "student_train.py", "exec"), ns)
        class Embed:
            def __call__(self, x, lens):
                return x * .7, lens
        class Encoder:
            def __call__(self, x, lens, mask):
                return x.masked_fill(mask.T[:, :, None], 0.) * 2., lens
        model = types.SimpleNamespace(encoder_embed=Embed(), encoder=Encoder(),
                                      ctc_output=torch.nn.Linear(3, 3, dtype=torch.double))
        model.forward_features = types.MethodType(ns["forward_features"], model)
        x, lens = torch.randn(2, 4, 3, dtype=torch.double), torch.tensor([3, 4])
        actual = ns["forward"](model, x, lens)
        embedded, embedded_lens = model.encoder_embed(x, lens)
        encoded, output_lens = model.encoder(embedded.permute(1, 0, 2), embedded_lens, make_pad_mask(embedded_lens))
        expected = model.ctc_output(encoded.permute(1, 0, 2)).log_softmax(-1), output_lens
        torch.testing.assert_close(actual, expected)


class EligibilityTest(unittest.TestCase):
    def recipe(self):
        return {"schema": 1, "selection": "minimum-validation-ctc", "steps": 2,
                "validate_every": 1, "temperature": .2, "learning_rate": .01, "real_pause_examples": 1,
                "training_speakers": ["s"], "validation_speakers": ["v"], "excluded_speakers": ["e"],
                "items": [{"key": str(i), "recording": str(i), "speaker": speaker, "partition": partition, "source": source,
                           "original_split": "train", "audio_sha256": "a" * 64}
                          for i, ((speaker, partition), source) in enumerate(itertools.product(
                              (("s", "training"), ("v", "validation")), ("authorized-quran", "help-none")))],
                "student_sha256": "b" * 64, "teacher_sha256": "c" * 64,
                "weights": dict(ctc=1., kd=1., preservation=1., distillation=1.),
                "training_batches": [[0, 1]]}

    def test_frozen_batches_cover_all_training_sources(self):
        recipe = self.recipe()
        self.assertEqual(len(validate_recipe(recipe)), 4)
        for plans in ([], [[0]], [[0, 0, 1]], [[0, 1, 2]]):
            with self.assertRaisesRegex(ValueError, "Batch plan"):
                validate_recipe({**recipe, "training_batches": plans})

    def test_none_label_and_original_split_are_required(self):
        item = {"help_id": "x", "speaker": "s", "transcript": "ab"}
        row = {"split": "train", "kind": "none", "scored": True, "events": [], "non_errors": [],
               "unsupported": [], "excluded": False, "speaker": "s", "ayahs": ["1:1"], "wav": "wav/x.wav"}
        self.assertEqual(validate_help_row(item, {"x": row}, {"1:1": "ab"}), "wav/x.wav")
        for name, value in (("kind", "repeat"), ("split", "eval"), ("non_errors", [{}]),
                            ("speaker", "held-out"), ("events", [{}]), ("scored", False)):
            with self.assertRaises(ValueError):
                validate_help_row(item, {"x": {**row, name: value}}, {"1:1": "ab"})

    def test_recipe_rejects_release_speaker_overlap(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_recipe({"schema": 1, "selection": "minimum-validation-ctc", "steps": 1,
                             "validate_every": 1, "temperature": .2, "learning_rate": .01, "real_pause_examples": 1,
                             "training_speakers": ["s"], "validation_speakers": ["v"], "excluded_speakers": ["s"]})


if __name__ == "__main__":
    unittest.main()

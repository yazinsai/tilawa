"""Small synthetic contracts, including execution of the pinned ScaledAdam."""
import hashlib
import importlib.util
import os
import sys
import time
import types
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from student_class_scales import ClassLocalHead, admit_recordings, matched_refit, recording_loss, speaker_partition

SHA256 = "7282cd6db1d234e086110feb1f39f076a76095a6b026b939cebc81f2ed0f5d45"


def optimizer_module():
    path = Path(os.environ["TILAWA_OPTIMIZER_SOURCE"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Tests need the trainer's pinned optimizer source")
    # Only the standalone optimizer file's unused test-seeding import needs
    # lhotse. Supply that import for small tests without altering its definitions.
    if importlib.util.find_spec("lhotse") is None:
        lhotse, utils = types.ModuleType("lhotse"), types.ModuleType("lhotse.utils")
        utils.fix_random_seed = torch.manual_seed
        sys.modules["lhotse"], sys.modules["lhotse.utils"] = lhotse, utils
    spec = importlib.util.spec_from_file_location("tested_pinned_optim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gradients(head, weight, bias):
    if isinstance(head, ClassLocalHead):
        for p, g in zip(head.rows, weight):
            p.grad = g.clone()
        for p, g in zip(head.biases, bias):
            p.grad = g.clone()
    else:
        head.weight.grad, head.bias.grad = weight.clone(), bias.clone()


def full_gradient(params):
    params = list(params)
    if len(params) == 2:
        return params[0].grad.clone(), params[1].grad.clone()
    n = len(params) // 2
    return torch.stack([p.grad for p in params[:n]]), torch.stack([p.grad for p in params[n:]])


class ClassScalesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.optim = optimizer_module()

    def setUp(self):
        torch.manual_seed(3)
        self.linear = torch.nn.Linear(4, 6)
        self.x = torch.randn(9, 4, requires_grad=True)
        self.view = dict(features=self.x, target=torch.tensor([1, 2, 2, 3]),
                         shipped=self.linear(self.x).detach().log_softmax(-1),
                         teacher=torch.randn(7, 6, requires_grad=True))

    def optimizer(self, head, clipping=2., **kwargs):
        return self.optim.ScaledAdam(head.named_parameters(), lr=.03, clipping_scale=clipping, **kwargs)

    def test_classifier_and_ctc_kd_gradient_parity(self):
        local = ClassLocalHead(self.linear)
        torch.testing.assert_close(local(self.x), self.linear(self.x), atol=0, rtol=0)
        values = []
        for head in (self.linear, local):
            torch.manual_seed(11)
            loss = recording_loss(head, [self.view], dropout=.1, ctc_weight=1., kd_weight=2., kd_delay=1)
            loss.backward()
            values.append(loss.detach())
        torch.testing.assert_close(*values, atol=0, rtol=0)
        torch.testing.assert_close(self.linear.weight.grad, torch.stack([p.grad for p in local.rows]), atol=0, rtol=0)
        torch.testing.assert_close(self.linear.bias.grad, torch.stack([p.grad for p in local.biases]), atol=0, rtol=0)
        self.assertIsNone(self.x.grad)
        self.assertIsNone(self.view["teacher"].grad)

    def test_objective_matches_shipped_reduction_and_teacher_mapping(self):
        got = recording_loss(self.linear, [self.view], dropout=0., ctc_weight=1., kd_weight=2., kd_delay=1)
        lp = self.linear(self.x.detach()).log_softmax(-1)
        ctc = F.ctc_loss(lp[:, None], self.view["target"], torch.tensor([9]), torch.tensor([4]),
                         reduction="sum", zero_infinity=True) / 9
        idx = (torch.arange(9) - 1).clamp(min=0, max=6)
        p, ids = self.view["teacher"].detach().softmax(-1)[idx].topk(5, -1)
        p /= p.sum(-1, keepdim=True).clamp_min(1e-6)
        expected = ctc - 2 * (p * lp.gather(-1, ids)).sum(-1).mean()
        torch.testing.assert_close(got, expected)

    def test_scale_units_and_coordinate_second_moments(self):
        local = ClassLocalHead(self.linear)
        a, b = self.optimizer(self.linear), self.optimizer(local)
        w, bias = torch.randn_like(self.linear.weight), torch.randn_like(self.linear.bias)
        for head, opt in ((self.linear, a), (local, b)):
            gradients(head, w, bias)
            opt.step()
        self.assertEqual(len(b.param_groups), 1)
        self.assertEqual(a.state[self.linear.weight]["param_rms"].shape, (1, 1, 1))
        self.assertEqual(a.state[self.linear.bias]["param_rms"].shape, (1, 1))
        self.assertEqual(b.state[local.rows[0]]["param_rms"].shape, (6, 1))
        self.assertNotIn("param_rms", b.state[local.biases[0]])
        torch.testing.assert_close(a.state[self.linear.weight]["exp_avg_sq"][0],
                                   b.state[local.rows[0]]["exp_avg_sq"], atol=0, rtol=0)
        torch.testing.assert_close(a.state[self.linear.bias]["exp_avg_sq"][0],
                                   b.state[local.biases[0]]["exp_avg_sq"], atol=0, rtol=0)

    def test_unrelated_row_and_bias_magnitudes_do_not_scale_local_updates(self):
        original = torch.nn.Linear(4, 6)
        changed = torch.nn.Linear(4, 6)
        changed.load_state_dict(original.state_dict())
        with torch.no_grad():
            changed.weight[1:] *= 10
            changed.bias[1:] *= 10
        heads = [original, changed, ClassLocalHead(original), ClassLocalHead(changed)]
        opts = [self.optimizer(h, None) for h in heads]
        w, bias = torch.randn_like(original.weight), torch.randn_like(original.bias)
        for _ in range(8):  # Includes two ordinary periodic size updates.
            for h, opt in zip(heads, opts):
                gradients(h, w, bias)
                opt.step()
        left, right = heads[2].folded(), heads[3].folded()
        torch.testing.assert_close(left.weight[0], right.weight[0], atol=0, rtol=0)
        torch.testing.assert_close(left.bias[0], right.bias[0], atol=0, rtol=0)
        self.assertGreater(float((original.weight[0] - changed.weight[0]).detach().abs().max()), .001)
        self.assertGreater(float((original.bias[0] - changed.bias[0]).detach().abs()), .001)

    def test_global_clipping_factor_applies_to_all_coordinates(self):
        base = self.optim.ScaledAdam
        class Spy(base):
            factors = []
            def _get_clipping_scale(self, group, batches):
                factor = super()._get_clipping_scale(group, batches)
                self.factors.append(factor)
                return factor
        local = ClassLocalHead(self.linear)
        opt = Spy(local.named_parameters(), lr=.03, clipping_scale=2.)
        w, bias = torch.ones_like(self.linear.weight), torch.ones_like(self.linear.bias)
        gradients(local, w, bias)
        opt.step()
        old_w = opt.state[local.rows[0]]["exp_avg_sq"].clone()
        old_b = opt.state[local.biases[0]]["exp_avg_sq"].clone()
        # The scalar batch sorts first and owns the whole-head norm history.
        opt.state[local.biases[0]]["model_norm_threshold"] = .001
        opt.state[local.biases[0]]["num_clipped"] = 0
        gradients(local, w * 100, bias * 100)
        opt.step()
        self.assertEqual(len(opt.factors), 1)
        factor = opt.factors[0]
        self.assertGreater(factor, 0.)
        self.assertLess(factor, 1.)
        beta = opt.param_groups[0]["betas"][1]
        torch.testing.assert_close(opt.state[local.rows[0]]["exp_avg_sq"],
                                   old_w * beta + (100 * factor) ** 2 * (1 - beta))
        torch.testing.assert_close(opt.state[local.biases[0]]["exp_avg_sq"],
                                   old_b * beta + (100 * factor) ** 2 * (1 - beta))

    def test_optimizer_writeback_and_folded_export_after_training(self):
        head = ClassLocalHead(self.linear)
        before = head(self.x).detach()
        opt = self.optimizer(head)
        for _ in range(5):
            opt.zero_grad()
            recording_loss(head, [self.view], dropout=0., ctc_weight=1., kd_weight=2.).backward()
            opt.step()
        self.assertGreater(float((before - head(self.x)).abs().max().detach()), 0.)
        torch.testing.assert_close(head(self.x), head.folded()(self.x), atol=0, rtol=0)

    def records(self):
        return [dict(speaker=f"speaker-{i}", source=source, views=[self.view])
                for i in range(10) for source in ("help-none", "transcribed")]

    def test_matched_updates_dropout_and_control_selected_checkpoint(self):
        base, optimizers, schedulers = self.optim.ScaledAdam, [], []
        class Spy(base):
            def __init__(inner, params):
                super().__init__(params, lr=0., clipping_scale=2.)
                inner.gradients = []
                optimizers.append(inner)
            def step(inner):
                inner.gradients.append(full_gradient(inner.param_groups[0]["params"]))
                return super().step()
        class Schedule:
            def __init__(inner, opt):
                inner.events = []
                schedulers.append(inner)
            def step_batch(inner, i):
                inner.events.append(("batch", i))
            def step_epoch(inner, i):
                inner.events.append(("epoch", i))
        _, report = matched_refit(self.linear, self.records(), Spy, Schedule, epochs=2, seed=7,
                                  kd_weight=2., deadline=time.time() + 60)
        a, b = optimizers
        self.assertEqual(len(a.gradients), len(b.gradients))
        self.assertEqual(schedulers[0].events, schedulers[1].events)
        for left, right in zip(a.gradients, b.gradients):
            for x, y in zip(left, right):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        selected = min(report["history"], key=lambda row: row["objectives"]["tensor"])["epoch"]
        self.assertEqual(report["selectedEpoch"], selected)
        self.assertEqual(report["selectedEpoch"], 0)
        self.assertEqual(len(a.gradients), 2 * sum(2 for g in report["groups"].values() if g == "fit"))

    def test_parity_and_deadline_fail_before_updates(self):
        records = self.records()
        records[0] = dict(records[0], views=[dict(self.view, shipped=self.view["shipped"] + torch.arange(6))])
        with self.assertRaises(AssertionError):
            matched_refit(self.linear, records, lambda _: self.fail("optimizer constructed"), None, epochs=1, seed=1)
        with self.assertRaises(TimeoutError):
            matched_refit(self.linear, self.records(), None, None, epochs=1, seed=1, deadline=0.)

    def test_speaker_partition_and_admission_do_not_leak(self):
        groups = speaker_partition([f"speaker-{i}" for i in range(10)])
        self.assertEqual(groups, speaker_partition(list(reversed(list(groups))) * 2))
        def row(speaker, source, **kw):
            return dict(speaker=speaker, source=source, split="train", kind="none", scored=True,
                        events=[], non_errors=[], unsupported=[], authorized=True, verified=True,
                        audioSha256=speaker + source, phones="abc", **kw)
        rows = [row("fit", "transcribed"), row("none", "help-none"), row("blocked", "help-none"),
                dict(row("blocked", "transcribed"), split="release"),
                dict(row("repeat", "help-none"), non_errors=[{}])]
        self.assertEqual([r["speaker"] for r in admit_recordings(rows, [])], ["fit", "none"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            admit_recordings(rows + [rows[0]], [])
        with self.assertRaisesRegex(ValueError, "authorized"):
            admit_recordings([dict(rows[0], verified=False), rows[1]], [])


if __name__ == "__main__":
    unittest.main()

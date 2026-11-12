"""Reciprocal gradients and matched controls on synthetic acoustic observations."""
import unittest

import torch

from student_peer_head import (
    PRESERVATION, agreement_loss, fit_heads, frozen_peers, heads, joint_update,
    phone_log_probs, preservation_loss, transcript_loss,
)


def recordings():
    generator = torch.Generator().manual_seed(8)
    return [dict(views=[dict(features=torch.randn(9, 4, generator=generator),
                            shipped=torch.randn(9, 3, generator=generator).log_softmax(-1),
                            target=torch.tensor([1, 2, 1])) for _ in range(3)]) for _ in range(3)]


class ReciprocalHeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_initializations_are_reproducible_without_rng_side_effects(self):
        state = torch.get_rng_state().clone()
        a, b = heads(4, 3), heads(4, 3)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        x = torch.randn(8, 4)
        for i in range(3):
            torch.testing.assert_close(a[i](x), b[i](x), rtol=0, atol=0)
        self.assertFalse(torch.equal(a[0](x), a[1](x)))
        self.assertFalse(torch.equal(a[1](x), a[2](x)))

    def test_agreement_targets_are_probability_means_and_stop_gradient(self):
        recipient = torch.randn(7, 3, requires_grad=True)
        peers = [torch.randn(7, 3, requires_grad=True) for _ in range(2)]
        lp = recipient.log_softmax(-1)
        target = torch.stack([p.detach().softmax(-1) for p in peers]).mean(0)
        loss = agreement_loss(lp, [p.log_softmax(-1) for p in peers])
        torch.testing.assert_close(loss, (target * (target.log() - lp)).sum(-1).mean())
        loss.backward()
        torch.testing.assert_close(recipient.grad, (lp.exp().detach() - target) / 7)
        self.assertTrue(all(p.grad is None for p in peers))
        with self.assertRaises(ValueError):
            agreement_loss(lp, [lp])

    def test_current_predictions_evolve_while_frozen_targets_stay_fixed(self):
        models = heads(4, 3)
        frozen = frozen_peers(models)
        x, center, scale = torch.randn(8, 4), torch.zeros(4), torch.ones(4)
        original = [phone_log_probs(m, x, center, scale).detach().clone() for m in models]
        with torch.no_grad():
            models[1].output.bias[1] += 2.
        current = [phone_log_probs(m, x, center, scale) for m in models]
        saved = [phone_log_probs(m, x, center, scale) for m in frozen]
        self.assertFalse(torch.equal(current[1], original[1]))
        for i in range(3):
            torch.testing.assert_close(saved[i], original[i], rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in frozen.parameters()))
        self.assertNotEqual(float(agreement_loss(current[0], current[1:])),
                            float(agreement_loss(current[0], saved[1:])))

    def test_joint_update_predicts_recipient_views_before_any_optimizer_step(self):
        models = heads(4, 3)
        rows = recordings()
        calls = [[] for _ in range(3)]
        hooks = [m.register_forward_pre_hook(lambda module, args, i=i: calls[i].append(args[0].clone()))
                 for i, m in enumerate(models)]
        class CheckedSGD(torch.optim.SGD):
            def step(self, closure=None):
                # All three models have predicted all nine recording views.
                self_calls = [len(values) for values in calls]
                if self_calls != [9, 9, 9]:
                    raise AssertionError(self_calls)
                if any(p.grad is None for m in models for p in m.parameters()):
                    raise AssertionError("Optimizer ran before all recipient gradients")
                return super().step(closure)
        optimizers = [CheckedSGD(m.parameters(), lr=.01) for m in models]
        joint_update(models, optimizers, rows, torch.zeros(4), torch.ones(4), 1.)
        expected = [v["features"] for row in rows for v in row["views"]]
        for values in calls:
            for got, want in zip(values, expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        for hook in hooks:
            hook.remove()

    def test_ctc_and_preservation_train_readout_only_and_reject_impossible_alignment(self):
        model = heads(4, 3)[0]
        x, shipped = torch.randn(8, 4, requires_grad=True), torch.randn(8, 3, requires_grad=True)
        center, scale = torch.zeros(4, requires_grad=True), torch.ones(4, requires_grad=True)
        lp = phone_log_probs(model, x, center, scale)
        target = torch.tensor([1, 1, 2])
        loss = transcript_loss(lp, target) + PRESERVATION * preservation_loss(lp, shipped)
        loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        self.assertTrue(all(v.grad is None for v in (x, shipped, center, scale)))
        with self.assertRaisesRegex(ValueError, "complete phone transcript"):
            transcript_loss(lp[:3], target)

    def test_controls_share_warmup_and_compute_but_diverge_after_reciprocal_updates(self):
        rows = recordings()
        results = {}
        for arm in ("live", "independent", "frozen"):
            results[arm] = fit_heads(arm, rows, 4, 3, torch.zeros(4), torch.ones(4),
                                     1., epochs=4, warmup=2)
        baseline_work = results["live"][1]
        for models, work in results.values():
            self.assertEqual(work, baseline_work)
            self.assertEqual(work["optimizerStepsPerHead"], 12)
            self.assertEqual(work["differentiableForwards"], 108)
            self.assertEqual(work["detachedPeerForwards"], 216)
            self.assertEqual(work["exposures"], [[4] * 3] * 3)
        for arm in ("independent", "frozen"):
            self.assertFalse(torch.equal(results["live"][0][0].output.weight,
                                         results[arm][0][0].output.weight))
        # With zero agreement, all three arms must be exactly the CTC control.
        zero = [fit_heads(arm, rows, 4, 3, torch.zeros(4), torch.ones(4), 0., epochs=3, warmup=1)[0]
                for arm in ("live", "independent", "frozen")]
        for peer in range(3):
            for comparison in zero[1:]:
                for a, b in zip(zero[0][peer].parameters(), comparison[peer].parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

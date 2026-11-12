"""Small CPU behavior tests: python3 -m unittest discover -s modal -p test_student_prefix.py."""

import itertools
import math
import unittest
from unittest.mock import patch

import torch
from student_prefix import (Prefix, alignment_prefixes, forward_states, prefix_ctc_loss,
                            refit_pair, speaker_partitions, streaming_features)


def enumerate_states(probs, target):
    """Independent CTC-path oracle, including active occurrence identity."""
    paths = []
    for path in itertools.product(range(probs.shape[1]), repeat=len(probs)):
        collapsed, states, prev = [], [], 0
        for label in path:
            if label and label != prev:
                collapsed.append(label)
            states.append(2 * len(collapsed) - (1 if label else 0))
            prev = label
        if collapsed == list(target[:len(collapsed)]):
            weight = torch.stack([probs[t, label] for t, label in enumerate(path)]).prod()
            paths.append((collapsed, states, weight))
    return paths


class PrefixTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_complete_sequence_partition_and_alignment_match_all_paths(self):
        for target in ([1], [1, 2], [1, 1], [2, 1, 2]):
            lp = torch.randn(5, 3, dtype=torch.double).log_softmax(-1)
            alpha = forward_states(lp, target)
            paths = [p for p in enumerate_states(lp.exp(), target) if p[0] == list(target)]
            total = sum(p[2] for p in paths)
            self.assertAlmostEqual(torch.logsumexp(alpha[-1, -2:], 0).exp().item(), total.item(), places=12)
            for prefix in alignment_prefixes(lp, target, [2, 3, 4], confidence=0.2):
                counts = torch.zeros(len(target) + 1, dtype=torch.double)
                for _, states, weight in paths:
                    counts[states[prefix.frames - 1] // 2] += weight / total
                expected = counts[list(prefix.counts)]
                expected /= expected.sum()
                torch.testing.assert_close(torch.tensor(prefix.weights, dtype=torch.double), expected)

    def test_partial_final_occurrence_probability_and_gradient_match_enumeration(self):
        for target in ([1, 2, 1], [1, 1, 2]):
            logits = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
            lp = logits.log_softmax(-1)
            prefix = Prefix(4, (1, 2), (0.4, 0.6))
            mass = logits.new_zeros(())
            for _, states, weight in enumerate_states(lp[:4].exp(), target):
                for k, w in zip(prefix.counts, prefix.weights):
                    if states[-1] in (2*k-1, 2*k, 2*k+1):
                        mass = mass + w * weight
            expected = -mass.log() / 4
            actual = prefix_ctc_loss(lp, target, [prefix])
            torch.testing.assert_close(actual, expected)
            wanted = torch.autograd.grad(expected, logits, retain_graph=True)[0]
            got = torch.autograd.grad(actual, logits)[0]
            torch.testing.assert_close(got, wanted)
            self.assertTrue(torch.isfinite(got).all())
            self.assertEqual(got[4].abs().sum().item(), 0)

    def test_marginalization_does_not_require_final_phone_or_allow_later_progress(self):
        def loss(path):
            lp = torch.full((len(path), 4), -torch.inf)
            lp[range(len(path)), path] = 0
            return prefix_ctc_loss(lp, [1, 2, 3], [Prefix(len(path)-1, (1,), (1.0,))])
        # Blank after the required phone, or an active optional phone, both pass.
        self.assertEqual(loss([1, 0, 0, 0]).item(), 0)
        self.assertEqual(loss([1, 2, 2, 0]).item(), 0)
        # A following phone or completion of the optional one cannot pass.
        for path in ([1, 2, 3, 0], [1, 2, 0, 0], [0, 0, 0, 0]):
            with self.assertRaisesRegex(ValueError, 'no feasible CTC path'):
                loss(path)

    def test_repeated_labels_cannot_skip_the_separating_blank(self):
        lp = torch.full((4, 3), -torch.inf)
        lp[:, 1] = 0
        with self.assertRaisesRegex(ValueError, 'no feasible CTC path'):
            prefix_ctc_loss(lp, [1, 1, 2], [Prefix(3, (2,), (1.0,))])
        lp[1] = torch.tensor([0, -torch.inf, -torch.inf])
        self.assertEqual(prefix_ctc_loss(lp, [1, 1, 2], [Prefix(3, (2,), (1.0,))]).item(), 0)

    def test_no_prefix_is_differentiable_zero_even_with_impossible_classes(self):
        lp = torch.tensor([[0., -torch.inf, -torch.inf]], requires_grad=True)
        loss = prefix_ctc_loss(lp, [1, 2], [])
        loss.backward()
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(lp.grad, torch.zeros_like(lp))

    def test_diffuse_alignment_omits_prefix_instead_of_admitting_the_whole_transcript(self):
        lp = torch.full((12, 3), -math.log(3))
        self.assertEqual(alignment_prefixes(lp, [1, 2, 1, 2, 1], [6], confidence=0.9999), ())

    def test_malformed_targets_and_noninternal_boundaries_fail(self):
        lp = torch.randn(5, 3).log_softmax(-1)
        for target in ([], [0], [3]):
            with self.assertRaises(ValueError):
                forward_states(lp, target)
        for frames in (0, 5, 6):
            with self.assertRaises(ValueError):
                alignment_prefixes(lp, [1, 2], [frames])
        for prefix in (Prefix(3, (1, 3), (0.5, 0.5)), Prefix(3, (1,), (0.4,)), Prefix(3, (0,), (1,))):
            with self.assertRaises(ValueError):
                prefix_ctc_loss(lp, [1, 2, 1, 2], [prefix])

    def test_speaker_partition_is_stable_and_disjoint(self):
        train, val = speaker_partitions(['a', 'b', 'a', 'c', 'd'])
        self.assertEqual((train, val), speaker_partitions(['d', 'c', 'b', 'a']))
        self.assertFalse(train & val)
        self.assertEqual(train | val, set('abcd'))

    def test_paired_refit_trains_only_head_on_training_speakers(self):
        head = torch.nn.Linear(3, 3, bias=False).double()
        with torch.no_grad():
            head.weight.copy_(torch.eye(3, dtype=torch.double))
        before = head.weight.detach().clone()
        path = [0, 1, 1, 0, 2, 2, 0, 1, 0, 0]
        features = torch.nn.functional.one_hot(torch.tensor(path), 3).double() * 5
        records = [dict(speaker=s, features=features, target=[1, 2, 1], source='help-train-none',
                        label='none', held_out=False, transcript_kind='performed') for s in ('a', 'b', 'c')]
        with patch('student_prefix.alignment_prefixes', wraps=alignment_prefixes) as align:
            best, report = refit_pair(head, records, chunk=4, epochs=2)
            # Validation transcripts must not select prefixes or advance the
            # random stream used to pick training prefixes.
            self.assertEqual(align.call_count, report['training_recordings'])
        self.assertEqual(set(best), {'control', 'prefix'})
        self.assertEqual(report['updates_per_arm'], 2 * report['training_recordings'])
        self.assertGreater(report['training_prefixes'], 0)
        self.assertFalse(set(report['train_speakers']) & set(report['validation_speakers']))
        torch.testing.assert_close(head.weight, before)
        self.assertIsNone(head.weight.grad)
        self.assertFalse(torch.equal(best['control']['state_dict']['weight'], best['prefix']['state_dict']['weight']))

    def test_refit_rejects_scripted_takes_and_unresolved_eligibility(self):
        head = torch.nn.Linear(2, 3)
        for field, value in (('label', 'skipped_word'), ('held_out', True), ('held_out', None),
                             ('source', 'practice'), ('transcript_kind', 'canonical')):
            records = [dict(speaker=s, features=torch.ones(8, 2), target=[1, 2], source='help-train-none',
                            label='none', held_out=False, transcript_kind='performed') for s in ('a', 'b')]
            records[0][field] = value
            with self.assertRaisesRegex(ValueError, 'eligible transcribed training'):
                refit_pair(head, records)

    def test_streaming_cache_carries_state_and_keeps_sdk_frame_count(self):
        class Embed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = []

            def get_init_states(self, batch, device):
                return torch.zeros(1, device=device)

            def streaming_forward(self, x, x_lens, cached_left_pad):
                self.seen.append((x.detach().clone(), cached_left_pad.item()))
                return x[:, :8, :1], torch.tensor([8]), cached_left_pad + 1

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.masks = []

            def get_init_states(self, batch, device):
                return [torch.zeros(1, device=device)]

            def streaming_forward(self, x, x_lens, states, src_key_padding_mask):
                self.masks.append(src_key_padding_mask.clone())
                return x[::2] + states[0], x_lens // 2, [states[0] + 1]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.ones(1))
                self.encoder_embed, self.encoder = Embed(), Encoder()

            def streaming_eval(self, chunk, left):
                self.geometry = chunk, left

        model = Model()
        feats = torch.ones(50, 80, requires_grad=True)
        cached = streaming_features(model, feats, chunk=8, left=16)
        self.assertEqual(cached.shape, (11, 1))
        self.assertFalse(cached.requires_grad)
        self.assertEqual(model.geometry, (8, 16))
        self.assertEqual([pad for _, pad in model.encoder_embed.seen], [0, 1, 2])
        self.assertAlmostEqual(model.encoder_embed.seen[-1][0][0, -1, 0].item(), -23.0259, places=4)
        self.assertEqual(model.encoder.masks[0][0, :16].sum().item(), 16)
        self.assertEqual(model.encoder.masks[1][0, :16].sum().item(), 8)
        torch.testing.assert_close(cached[:, 0], torch.tensor([1]*4 + [2]*4 + [3]*3, dtype=cached.dtype))


if __name__ == '__main__':
    unittest.main()

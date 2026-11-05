import unittest

import numpy as np
import torch
from student_accusation_loss import collapse_owned, masked_loss, viterbi_states, word_slots
from student_accusation_finetune import corrected_words, accusation_words, eos_batch


class SlotLossTest(unittest.TestCase):
    def test_eos_batch_trims_losses_to_real_frames(self):
        f, lengths, real = eos_batch([(torch.ones(55, 80), []), (torch.ones(137, 80), [])])
        self.assertEqual(real, [12, 33])
        self.assertEqual(tuple(f.shape), (2, 173, 80))
        self.assertEqual(lengths.tolist(), [173, 173])
        self.assertTrue(torch.all(f[0, 55:] == -23.0259))
        logits = torch.randn(2, 42, 4, requires_grad=True)
        lp = logits.log_softmax(-1)
        loss = sum(masked_loss(lp[i, :n], torch.randn(n, 4).log_softmax(-1), [])["loss"] for i, n in enumerate(real))
        loss.backward()
        self.assertEqual(float(logits.grad[0, 12:].abs().sum() + logits.grad[1, 33:].abs().sum()), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_multiword_graph_scope_cannot_be_truncated_to_first_word(self):
        ref = {"ayahs": [{"s": 1, "a": 2, "p": "abc", "o": [0, 1]}, {"s": 1, "a": 3, "p": "de", "o": [0]}]}
        event = {"location": {"surah": 1, "ayah": 2, "word": 2}, "expected": "bcde"}
        self.assertEqual(accusation_words(event, ref), {"1:2:2", "1:3:1"})
        with self.assertRaises(ValueError):
            accusation_words({**event, "expected": "bd"}, ref)

    def test_transcript_preserves_slot_ownership_and_rejects_unknown_edits(self):
        ref = {"ayahs": [{"s": 1, "a": 2, "w": ["first", "second", "third", "last"]}]}
        w = lambda n: {"surah": 1, "ayah": 2, "word": n}
        label = {"split": "train", "device": "phone", "scored": True, "ayahs": ["1:2"],
            "events": [{"errorClass": "wrong_word", "location": w(2), "observed": "new word"},
                       {"errorClass": "omitted_word", "location": w(3)}],
            "non_errors": [{"kind": "repeat", "location": w(4)}]}
        out = corrected_words(label, ref)
        self.assertEqual([x["text"] for x in out], ["first", "new", "word", "last", "last"])
        self.assertEqual([x["key"] for x in out], ["1:2:1", "1:2:2", "1:2:2", "1:2:4", "1:2:4"])
        with self.assertRaises(ValueError):
            corrected_words({**label, "split": "eval"}, ref)
        label["events"][0].pop("observed")
        with self.assertRaises(ValueError):
            corrected_words(label, ref)

    def test_ctc_gradient_stays_inside_slot_and_kl_is_complementary(self):
        logits = torch.randn(18, 4, requires_grad=True)
        lp = logits.log_softmax(-1)
        shipped = torch.randn(18, 4).log_softmax(-1)
        out = masked_loss(lp, shipped, [{"from": 5, "to": 12, "target": [1, 2]}])
        ctc_grad = torch.autograd.grad(out["ctc"], logits, retain_graph=True)[0]
        kl_grad = torch.autograd.grad(out["kl"], logits)[0]
        self.assertEqual(float(ctc_grad[:5].abs().sum() + ctc_grad[12:].abs().sum()), 0)
        self.assertGreater(float(ctc_grad[5:12].abs().sum()), 0)
        self.assertEqual(float(kl_grad[5:12].abs().sum()), 0)
        self.assertGreater(float(kl_grad[:5].abs().sum()), 0)

    def test_clean_clip_has_only_full_kl(self):
        lp = torch.randn(9, 3).log_softmax(-1)
        out = masked_loss(lp, lp, [])
        self.assertEqual(float(out["ctc"]), 0)
        self.assertTrue(out["outside"].all())
        self.assertAlmostEqual(float(out["loss"]), 0, places=6)

    def test_shared_phone_ownership_and_clipped_margin(self):
        target, owners = collapse_owned([1, 1, 2, 2, 3], [0, 1, 1, 1, 2])
        self.assertEqual(target, [1, 2, 3])
        self.assertEqual(owners, [{0, 1}, {1}, {2}])
        lp = np.full((9, 4), -20.0)
        lp[np.arange(9), [0, 1, 1, 0, 2, 0, 3, 3, 0]] = 0
        slots = word_slots(viterbi_states(lp, target), target, owners, {1}, margin=2)
        self.assertEqual(slots[0], {"word": 1, "from": 0, "to": 7, "target": [1, 2]})

    def test_repeated_labels_need_blank_and_impossible_alignment_rejects(self):
        lp = np.full((5, 3), -20.0)
        lp[np.arange(5), [0, 1, 0, 1, 0]] = 0
        path = viterbi_states(lp, [1, 1])
        self.assertIn(2, path)
        with self.assertRaises(ValueError):
            viterbi_states(lp[:1], [1, 1])
        with self.assertRaises(ValueError):
            word_slots(np.array([1, 3, 5]), [1, 2, 1], [{0}, {1}, {0}], {0})


if __name__ == "__main__":
    unittest.main()

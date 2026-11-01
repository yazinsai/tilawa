"""Numerical checks for the real-word acoustic embedding training path."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal"))
import numpy as np
import torch
from student_word_embedding import BINS, Lexicon, WordEncoder, collapse_with_offsets, contrastive_loss, temporal_pool, word_spans
from student_word_embedding_replay import compare, trim_gap


class WordEmbeddingTests(unittest.TestCase):
    def test_pooling_ignores_batch_padding_and_handles_short_words(self):
        torch.manual_seed(0)
        x = torch.randn(4, 8, 57)
        lengths = torch.tensor([1, 13, 24, 57])
        got = temporal_pool(x, lengths)
        expected = torch.stack([torch.nn.functional.adaptive_avg_pool1d(x[i, :, :n], BINS)
                                for i, n in enumerate(lengths)])
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)

    def test_word_boundaries_and_repeated_phone_offsets(self):
        lex = Lexicon({"ayahs": [{"o": [0, 4], "p": "بَااتُب", "iso": ["بَاا", "تُب"]}]})
        self.assertEqual(lex.locate("بَااتُب"), [(0, 4, "بَاا"), (4, 7, "تُب")])
        self.assertEqual(lex.locate("ااتُب"), [(2, 5, "تُب")])
        self.assertEqual(collapse_with_offsets("بَااتُب"), ("بَاتُب", [0, 1, 2, 4, 5, 6]))

    def test_forced_alignment_assigns_words_without_neighbours(self):
        from student import phoneme_ids
        lex = Lexicon({"ayahs": [{"o": [0, 4], "p": "بَااتُب", "iso": ["بَاا", "تُب"]}]})
        target, _ = collapse_with_offsets(lex.text)
        ids = np.tile(np.arange(5, dtype=np.uint8), (42, 1))
        p = np.full((42, 5), .001, np.float16)
        p[:, 0] = .99
        for i, token in enumerate(phoneme_ids(target)):
            ids[i * 7 + 2:i * 7 + 5, 0] = token
            ids[i * 7 + 2:i * 7 + 5, 1:] = [p for p in range(43) if p != token][:4]
        rows = word_spans(dict(ref=lex.text, top5_ids=ids.tobytes(), top5_p=p.tobytes()), lex)
        self.assertEqual([r["key"] for r in rows], ["بَاا", "تُب"])
        self.assertLessEqual(rows[0]["end"], rows[1]["start"])
        self.assertGreater(rows[0]["end"], rows[0]["start"])

    def test_head_returns_unit_vectors_and_finite_gradients(self):
        torch.manual_seed(0)
        model = WordEncoder(8)
        z = model(torch.randn(4, 8, BINS), torch.tensor([.3, .4, .8, 1.2]))
        torch.testing.assert_close(z.norm(dim=-1), torch.ones(4))
        loss = contrastive_loss(z, torch.tensor([0, 0, 1, 1]), torch.arange(4))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_augmentations_of_one_recording_are_not_positive_pairs(self):
        z = torch.eye(3, requires_grad=True)
        self.assertEqual(float(contrastive_loss(z, torch.zeros(3), torch.zeros(3))), 0.)

    def test_neighbour_prefix_and_missing_coverage_block_confirmation(self):
        artifact = dict(keys=["word", "neighbour"], bank=torch.tensor([[1., 0.], [0., 1.]]),
                        prefix_bank=torch.tensor([[1., 0.]]), prefix_keys=[1], report={"radius": .2})
        z = torch.tensor([1., 0.])
        self.assertFalse(compare(z, artifact, ["neighbour"])["accept"])
        artifact["prefix_bank"] = torch.tensor([[0., 1.]])
        self.assertTrue(compare(z, artifact, ["neighbour"])["accept"])
        self.assertFalse(compare(z, artifact, ["neighbour", "unseen"])["accept"])
        self.assertFalse(compare(torch.tensor([0., 1.]), artifact, ["neighbour"])["accept"])
        self.assertFalse(compare(torch.tensor([.5, -.866]), artifact, ["neighbour"])["accept"])

    def test_silence_has_no_word_embedding_crop(self):
        self.assertEqual(len(trim_gap(np.zeros(16000, np.float32))), 0)


if __name__ == "__main__":
    unittest.main()

"""Crop labels and actual bidirectional receptive field (training image)."""
import unittest
import numpy as np
from student_offline import canonical_alignment, crop_target


class Labels(unittest.TestCase):
    def test_silence_does_not_become_a_phone(self):
        path = [0, 0, 1, 1, 0, 0, 0, 2, 0]
        ids = np.tile(np.arange(3), (len(path), 1))
        probs = np.full(ids.shape, .001)
        probs[np.arange(len(path)), path] = .998
        spans = canonical_alignment(ids, probs, [1, 2], 3)
        self.assertEqual(spans, [(2, 4), (7, 8)])
        self.assertEqual(crop_target([1, 2], spans, 4, 7), [])
        self.assertEqual(crop_target([1, 2], spans, 3, 9), [2])

    def test_adjacent_same_labels_require_blank(self):
        ids = np.tile(np.arange(2), (5, 1))
        probs = np.array([[.01,.99],[.01,.99],[.99,.01],[.01,.99],[.99,.01]])
        spans = canonical_alignment(ids, probs, [1, 1], 2)
        self.assertEqual(spans, [(0, 2), (3, 4)])

    def test_fractional_crop_drops_partial_edges(self):
        self.assertEqual(crop_target([1,2,3], [(1,3),(5,6),(8,10)], 2.2, 9.5), [2])

    def test_future_audio_changes_early_frames(self):
        import torch
        import student_train as T
        from student_offline import build
        T.SIZES["test-offline"] = dict(num_encoder_layers="1,1,1,1,1,1", feedforward_dim="32,32,32,32,32,32",
            encoder_dim="16,16,16,16,16,16", encoder_unmasked_dim="16,16,16,16,16,16")
        old = T.COMMON.copy()
        try:
            T.COMMON.update(num_heads="2,2,2,2,2,2", query_head_dim="8", value_head_dim="4",
                pos_head_dim="2", pos_dim=8, cnn_module_kernel="3,3,3,3,3,3")
            torch.manual_seed(0)
            model = build("test-offline").eval()
            x = torch.randn(1, 401, 80)
            changed = x.clone(); changed[:, 300:] += 3
            with torch.no_grad():
                a, _ = model(x, torch.tensor([401]))
                b, _ = model(changed, torch.tensor([401]))
            self.assertGreater(float((a[:,:20]-b[:,:20]).abs().max()), 1e-5)
            self.assertFalse(model.encoder.causal)
        finally:
            T.COMMON.clear(); T.COMMON.update(old)
            del T.SIZES["test-offline"]


if __name__ == "__main__": unittest.main()

"""Synthetic training-label and gradient tests; no benchmark examples."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import VOCAB, phoneme_ids
from student_relabel import free_tokens, masked_ctc_log_probs, relabel, speech_insertions


def posterior(phones, confidence=.98):
    best = [0]
    for token in phoneme_ids(phones):
        best.extend([token, 0])
    ids = np.zeros((len(best), 5), dtype=np.uint8)
    probs = np.zeros((len(best), 5), dtype=np.float32)
    for i, token in enumerate(best):
        others = [j for j in range(VOCAB) if j != token][:4]
        ids[i] = [token, *others]
        probs[i] = [confidence, *[(1 - confidence) / 4] * 4]
    return ids, probs


def test_only_interior_word_shapes_are_candidates():
    ref = "بَستَم"
    assert speech_insertions(ref, "بَسكَدتَم")[2][0]["phones"] == "كَد"
    assert not speech_insertions(ref, "كَدبَستَم")[2]  # teacher edge already retained
    assert not speech_insertions(ref, "بَساِۥتَم")[2]  # held vowel, not two consonants


def test_free_tokens_collapse_duration_but_keep_separate_consonants():
    ids, probs = posterior("ببَست")
    tokens = free_tokens(ids, probs)
    assert [t["id"] for t in tokens] == phoneme_ids("بَست")
    assert len(tokens[0]["frames"]) == 2


def test_confident_speech_is_inserted_but_vad_negative_pause_is_untouched():
    ref, heard = "بَستَم", "بَسكَدتَم"
    ids, probs = posterior(heard)
    yes = relabel(ref, heard, ids, probs, np.ones(len(ids), dtype=bool))
    no = relabel(ref, heard, ids, probs, np.zeros(len(ids), dtype=bool))
    assert yes["target"] == heard and yes["mask"] == []
    assert no["target"] == ref and no["mask"] == []


def test_uncertain_speech_masks_only_canonical_blank_frames():
    from student_relabel import canonical_path

    ref, heard = "بَستَم", "بَسكَدتَم"
    ids, probs = posterior(heard, .7)
    result = relabel(ref, heard, ids, probs, np.ones(len(ids), dtype=bool))
    assert result["target"] == ref
    assert result["mask"]
    path = canonical_path(ids, probs, ref)
    assert all(path[f] == 0 for f in result["mask"])


def test_cache_disagreement_does_not_relabel():
    ids, probs = posterior("بَسجَدتَم")
    result = relabel("بَستَم", "بَسكَدتَم", ids, probs, np.ones(len(ids), dtype=bool))
    assert result["mismatch"] and not result["mask"] and not result["spans"]


def test_masked_frames_get_no_ctc_gradient_and_still_separate_repeats():
    logits = torch.zeros((1, 3, 3), requires_grad=True)
    masked = torch.tensor([[False, True, False]])
    lp = masked_ctc_log_probs(logits.log_softmax(-1), masked)
    loss = torch.nn.functional.ctc_loss(lp.transpose(0, 1), torch.tensor([1, 1]),
                                        torch.tensor([3]), torch.tensor([2]), reduction="sum")
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(logits.grad[0, 1], torch.zeros(3))
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 2].abs().sum() > 0


def test_empty_mask_is_exactly_the_original_objective():
    lp = torch.randn(2, 4, 5).log_softmax(-1)
    assert torch.equal(masked_ctc_log_probs(lp, torch.zeros((2, 4), dtype=torch.bool)), lp)


def test_training_masks_both_ctc_and_kd_gradients():
    from student_train import losses

    logits = torch.zeros((1, 3, 6), requires_grad=True)

    def model(feats, lens):
        return logits.log_softmax(-1), torch.tensor([3])

    batch = dict(feats=torch.zeros((1, 3, 80)), feat_lens=torch.tensor([3]),
                 targets=torch.tensor([1, 1]), target_lens=torch.tensor([2]),
                 n_teacher=torch.tensor([3]), top_ids=torch.tensor([[[1, 2, 3, 4, 5]] * 3]),
                 top_p=torch.tensor([[[.9, .025, .025, .025, .025]] * 3]),
                 loss_mask=torch.tensor([[False, True, False]]))
    out = losses(model, batch, dict(kd_delay=0, ctc_weight=1., kd_weight=2.), 'cpu')
    out['loss'].backward()
    assert torch.equal(logits.grad[0, 1], torch.zeros(6))
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 2].abs().sum() > 0


def test_overlay_rejects_stale_labels_and_keeps_unselected_items(tmp_path):
    import json
    import pytest
    from student_relabel import apply_labels

    clips = dict(key=['a', 'b'], target=['بَستَم', 'سَم'], n_frames=[20, 10])
    path = tmp_path / 'labels.jsonl'
    row = dict(key='a', base_target='بَستَم', target='بَسكَدتَم', n_frames=20, mask=[7, 8])
    path.write_text(json.dumps(row))
    assert apply_labels(clips, path) == dict(changed=1, masked=1, overlay_rows=1)
    assert clips['target'] == ['بَسكَدتَم', 'سَم']
    assert clips['loss_mask'] == [[7, 8], []]
    with pytest.raises(ValueError, match='Stale relabel'):
        apply_labels(clips, path)


def test_ctc_only_training_still_works_without_teacher_buffers():
    from student_train import losses

    logits = torch.zeros((1, 3, 3), requires_grad=True)

    def model(feats, lens):
        return logits.log_softmax(-1), torch.tensor([3])

    batch = dict(feats=torch.zeros((1, 3, 80)), feat_lens=torch.tensor([3]),
                 targets=torch.tensor([1, 2]), target_lens=torch.tensor([2]))
    out = losses(model, batch, dict(ctc_weight=1., kd_weight=0.), 'cpu')
    assert 'kd' not in out and torch.isfinite(out['loss'])
    out['loss'].backward()
    assert logits.grad.abs().sum() > 0

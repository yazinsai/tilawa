# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with numpy --with torch --with pytest pytest tools/test_student.py"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import TOKENS, corpus_per, ctc_collapse, ids_phonemes, mix_noise, phoneme_ids, reference_phonemes, soft_target_loss, teacher_frames  # noqa: E402

REF = json.loads((Path(__file__).resolve().parent.parent / "packages/core/assets/ref-v2.json").read_text())


def span(s1, a1, w1, s2, a2, w2):
    return {"kind": "span", "span": {"start": {"surah": s1, "ayah": a1, "word": w1}, "end": {"surah": s2, "ayah": a2, "word": w2}}}


def test_whole_ayah_span_is_the_ayah_phonemes():
    assert reference_phonemes(REF, span(1, 1, 1, 1, 1, 4)) == "بِسمِللَااهِررَحمَاانِررَحِۦۦم"


def test_partial_span_is_cut_at_word_offsets():
    assert reference_phonemes(REF, span(1, 1, 2, 1, 1, 3)) == "للَااهِررَحمَاانِ"


def test_phonemes_map_to_teacher_vocab_ids_with_blank_zero():
    # muaalem-model-v3_2 vocab.json phoneme level: ب=2, ِ=34, س=12, م=24.
    assert phoneme_ids("بِسم") == [2, 34, 12, 24]
    assert ids_phonemes([2, 34, 12, 24]) == "بِسم"
    assert TOKENS == REF["inventory"]["tokens"]


def test_student_frame_j_reads_teacher_frame_j_clamped_to_the_teacher_length():
    # 25 Hz both: student frame j is centred near fbank frame 4j+4, teacher frame i (w2v-BERT stack + adapter) near 4i+2.5.
    assert teacher_frames(4, 6).tolist() == [0, 1, 2, 3]
    assert teacher_frames(5, 4).tolist() == [0, 1, 2, 3, 3]
    assert teacher_frames(3, 6, delay=2).tolist() == [0, 0, 0]


def test_greedy_ctc_collapses_repeats_between_blanks_only():
    assert ctc_collapse([0, 2, 2, 0, 2, 5, 5, 0]) == [2, 2, 5]


def test_corpus_per_is_total_edits_over_total_reference_length():
    # "abd" vs "abc": 1 substitution; "x" vs "xy": 1 deletion; 2 edits over 5 reference phonemes.
    assert corpus_per([("abd", "abc"), ("x", "xy")]) == 0.4


def test_noise_is_mixed_at_the_requested_snr_and_looped_to_the_clip_length():
    import numpy as np

    rng = np.random.default_rng(0)
    clean = np.sin(np.arange(16000) / 5).astype(np.float32)
    noise = rng.standard_normal(4000).astype(np.float32)
    mixed = mix_noise(clean, noise, snr_db=10.0, rng=rng)
    added = mixed - clean
    assert len(mixed) == len(clean)
    # 10 dB: signal power is 10x the added noise power.
    assert abs(np.mean(clean**2) / np.mean(added**2) - 10.0) < 0.01


def test_soft_target_loss_is_cross_entropy_to_renormalised_top_k_over_valid_frames():
    import torch

    student = torch.tensor([[[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]]).log()
    top_ids = torch.tensor([[[1, 0], [2, 1]]])
    top_p = torch.tensor([[[0.6, 0.2], [0.9, 0.1]]])
    valid = torch.tensor([[True, False]])
    # Frame 0: teacher renormalises to {1: 0.75, 0: 0.25}; -(0.75 ln 0.25 + 0.25 ln 0.5) = 1.21301. Frame 1 is padding.
    assert abs(soft_target_loss(student, top_ids, top_p, valid).item() - 1.21301) < 1e-4

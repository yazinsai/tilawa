# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "torch", "pytest"]
# ///
"""uv run --python cpython-3.12-macos-aarch64-none --with numpy --with torch --with pytest pytest tools/test_student.py"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import TOKENS, corpus_per, ctc_collapse, ctc_token_spans, exclusions, held_out, ids_phonemes, mix_noise, phoneme_ids, reference_phonemes, soft_target_loss, splice_teacher_frames, spliced_target, teacher_frames, unfaithful_splice  # noqa: E402

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


def test_spliced_target_keeps_teacher_edges_reference_interior_and_collapses_runs():
    # Exact matches of >= 3: "abbc" and "def". Outside them the teacher ("pp", "s"); between them the reference
    # (drops the teacher's inserted x); then runs collapse (pp -> p, bb -> b).
    assert spliced_target("qqabbcdefrr", "ppabbcxdefs") == "pabcdefs"


def test_spliced_target_without_an_anchor_is_the_collapsed_teacher():
    assert spliced_target("abcdef", "xxyz") == "xyz"


def test_soft_target_loss_is_cross_entropy_to_renormalised_top_k_over_valid_frames():
    import torch

    student = torch.tensor([[[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]]).log()
    top_ids = torch.tensor([[[1, 0], [2, 1]]])
    top_p = torch.tensor([[[0.6, 0.2], [0.9, 0.1]]])
    valid = torch.tensor([[True, False]])
    # Frame 0: teacher renormalises to {1: 0.75, 0: 0.25}; -(0.75 ln 0.25 + 0.25 ln 0.5) = 1.21301. Frame 1 is padding.
    assert abs(soft_target_loss(student, top_ids, top_p, valid).item() - 1.21301) < 1e-4


def test_ctc_token_spans_keep_repeated_tokens_separated_by_blank():
    assert ctc_token_spans([0, 2, 2, 0, 2, 5, 5, 0]) == ([2, 2, 5], [(1, 3), (4, 5), (5, 7)])


def test_unfaithful_splices_have_as_heard_targets_and_edit_maps():
    import numpy as np

    host_frames = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0]
    donor_frames = [0, 11, 0, 12, 0, 13, 0, 14, 0]
    host = np.sin(np.arange(16000, dtype=np.float32) / 11)
    donor = np.sin(np.arange(8000, dtype=np.float32) / 13)

    inserted = unfaithful_splice(host, host_frames, donor, donor_frames, "insert", np.random.default_rng(0))
    assert inserted and inserted["target_ids"][:4] == [11, 12, 13, 14]
    assert inserted["target_ids"][4:] == list(range(1, 11))
    assert inserted["kd_edit"]["delta"] > 0

    deleted = unfaithful_splice(host, host_frames, donor, donor_frames, "delete", np.random.default_rng(1))
    assert deleted and len(deleted["target_ids"]) < 10
    assert deleted["kd_edit"]["delta"] < 0

    swapped = unfaithful_splice(host, host_frames, donor, donor_frames, "swap", np.random.default_rng(2))
    assert swapped and any(x >= 11 for x in swapped["target_ids"])
    assert len(swapped["wave"]) != len(host)


def test_splice_teacher_mapping_masks_edit_and_remaps_later_frames():
    import numpy as np

    edit = {"start": 4000, "end": 8000, "delta": 4000, "source_samples": 12000, "output_samples": 16000}
    frames, valid = splice_teacher_frames(16, 12, edit, pad_s=0)
    assert frames.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert valid.tolist() == [True] * 4 + [False] * 4 + [True] * 8
    assert np.all(np.diff(frames[8:]) >= 0)


def test_gold_clips_are_held_out_of_training_only():
    gold = exclusions("clean-00033-of-00411:734\n\nunclean-00146-of-00332:490\n")
    assert held_out(gold, "clean-00033-of-00411", 734, "train")
    assert not held_out(gold, "clean-00033-of-00411", 735, "train")
    assert not held_out(gold, "unclean-00146-of-00332", 490, "test")

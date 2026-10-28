# 12 Calibrate the independent audio opinion

Status: resolved
Type: task
Blocked by: 07

Spec: `../spec.md` decision 17; `docs/agents/audio-review.md`.

Run `modal/review_asr.py` on the 40 cases from ticket 07 before showing it the labels. Report its agreement with your verdicts. At 95% or better agreement it pre-screens later test set versions and training-data samples, and you hear only disagreements plus a 10% spot check. Below that, record why and keep listening fully manual.

## Comments

2026-10-28. **Fails; listening stays fully manual.** Full write-up: `docs/benchmarks/lab-review-asr-v1.md`.

- Models (independent of the teacher and student, on Modal GPU, label-blind with no prompt): `Qwen/Qwen3-ASR-1.7B` @ `7278e1e7` and `Systran/faster-whisper-large-v3` @ `edaa852e`. Each heard only the 40 v1-check windows. The comparison with the labelled text (word level, normalised, diacritics stripped) used a rule committed before any transcript existed (`033609f`).
- Agreement with your 40 `ok` verdicts: Qwen 8/40 (20%, 95% range 10–35%), Whisper 7/40 (18%, 9–32%), both combined 6/40. The bar is 38/40.
- Negative controls, using the same audio with wrong text (the planted edit undone in the text, or a one-word change planted in the text of clean windows):
  - flagged: Qwen 40/40, Whisper 36/40, combined 40/40;
  - one-word changes flagged: 31/31 for all three;
  - another item's text flagged: 40/40 for all three.

  These rates are high mostly because the models reject almost everything. Whisper accepted the text with the repeat removed on 4 of 6 repeat/restart items, because it writes the canonical verse.
- Causes on correct audio:
  - edge words of the tight windows dropped (10 items for Qwen, 12 for Whisper);
  - misheard words, worst on the phone copies;
  - Whisper hallucinating "اشتركوا في القناة" or "ترجمة نانسي قنقر" on 7 of 40.
- Exploratory, in-sample run with 1 s of extra audio per side: Qwen 17/40, Whisper 20/40, either-model-match 28/40. Still far off.
- Proposed bar for any future model: at least 95% agreement, at least 90% of one-edit and one-word controls flagged, and no repeat misses, reported together. It only ever covers words. Seams (clicks, level jumps, clipped onsets, unnatural joins), tajweed and pronunciation still need listening.
- Pre-screen mode is built (`modal run modal/review_asr.py::prescreen --queue ...`, queue in the ticket 06 format, splices kept for the seam listen by default, seeded 10% spot check). It is not to be used until a model passes. On v1-check it would keep 38 of 40 items.
- Records: `/lab/reviews/lab-v1.1/asr-qwen3-asr-1.7b.json`, `asr-whisper-large-v3.json`, `asr-calibration.json`, and `*.pad1.json` for the exploratory run. Tables: `docs/benchmarks/lab-review-asr-v1.json` and `-pad1.json`.
- Possible next step (speculative): instead of transcribing freely, force-align both candidate texts (labelled vs original) with an independent Arabic CTC model and compare their likelihoods.


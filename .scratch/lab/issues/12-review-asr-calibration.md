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


2026-10-29. **Nemotron 3.5 Transcribe: fails as an independent opinion; its likelihood arm passes the numbers but only covers edited windows.** Full write-up: `docs/benchmarks/lab-review-asr-v1.md`, section "Nemotron 3.5 Transcribe".

- Model and setup:
  - `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` @ `9d8cd6b7`, run in-process with Transformers (`144ecf0`) in bf16 on a Modal H100.
  - Label-blind, using the model card's verbatim prompt with "Language: Arabic", greedy decoding, reasoning off.
  - The same 40 windows (same audio sha256), the same rule and the same controls as the earlier runs. The code and rules were committed in `37ffc4d` before any calibration output existed.
- Arm 1, free transcription:
  - Agreement 23/40 (58%, 95% range 42–71%), against 8/40 for Qwen and 7/40 for Whisper.
  - One-edit flagged 38/40 (95%), one-word 29/31 (94%), shifted 40/40.
  - It is the first model to write every repeat and restart (6/6).
  - Both misses are added-word splices where the planted word was dropped: `p15-x8` came out as the canonical verse, and `e19-x12` was blended.
  - Other causes:
    - 9 mishearings, including on the phone copies;
    - 2 edge drops;
    - 3 hallucinations, all Quran- or dua-flavoured: a dua loop on `e15-x8`, a pastiche of other verses on `e15-phone-229`, and an invented continuation on `p03-studio-5`.
- Exploratory runs (in-sample):
  - The suggested recitation prompt: 26/40, all controls flagged, but its hallucinations moved to other windows.
  - 1 s padding: 22/40. It fixed the edge drops but made the model canonicalise more, collapsing the `p15-x5` repeat.
  - The CTC head read greedily: 0/40.
- Arm 2, likelihood comparison:
  - The model scored the labelled text against its one-edit and shifted controls (undiacritised, after arm 1 was saved), so it was shown the candidates.
  - LM pairwise, margin 2 nats: agreement 39/40 (98%, 87–100%), one-edit flagged 39/40, one-word 30/31, no repeat misses. That passes the bar's numbers.
  - Caveats:
    - agreement and one-edit flagged are the same 40 comparisons read both ways;
    - only the 26 edited windows have a real alternative text (25/26 on those);
    - the one miss (`p12-x7`, a planted `مِهَـٰدًا` losing to the canonical `كِفَاتًا` by 3.3 nats) is the dangerous kind;
    - at a 5-nat margin agreement would drop to 35/40.
  - The rest fail: LM near-best 30/40; CTC pairwise 36/40 with 2 repeat misses; CTC near-best 0/40.
- Wired as the default pre-screen (`modal run modal/review_asr.py::prescreen --queue ...`, `--models nemotron-likelihood`):
  - edited windows are cleared only if the labelled text beats the original by 2 nats;
  - clean windows go to the listener (no alternative text);
  - every splice keeps its seam listen, plus a seeded 10% spot check.
  - On `v1-check` you would still hear 35 of 40 windows, about 2 minutes saved per version. The test output was deleted.
  - Trust it only after the spot checks hold over the next one or two versions. It does nothing for training-data samples.
- Cost: about 20 min of H100 wall time in total, roughly $1.5 (estimate).
- Records: `/lab/reviews/lab-v1.1/asr-nemotron-3.5-transcribe.json` (arm 1 per item, plus the arm 2 scores under `likelihood`), `asr-nemotron-recitation-prompt.json`, `asr-nemotron-3.5-transcribe.pad1.json`, and the `asr-calibration.nemotron-*.json` summaries. Tables: `docs/benchmarks/lab-review-asr-v1-nemotron-*.json`.

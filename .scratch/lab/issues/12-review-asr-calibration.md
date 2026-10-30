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

2026-10-29. **NVIDIA Arabic FastConformer: fails both arms, not wired; the default pre-screen stays `nemotron-likelihood`.** Its CTC head is useful as an independent data check for some edit classes, but not for added words and repeats. Full write-up: `docs/benchmarks/lab-review-asr-v1.md`, section "NVIDIA Arabic FastConformer".

- Model and setup:
  - `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` @ `7f32349d`, NeMo 2.7.3, float32 on a Modal T4.
  - The same 40 windows (all sha256 identical to Nemotron's), the same rule and the same controls.
  - The code and every rule were committed in `5d019b0` before any calibration or data-check output existed.
  - Its training data includes everyayah, so its prior is the canonical text.
  - NeMo 2.7.3's `transcribe()` leaves the encoder and CTC head in train mode (dropout on). The smoke test caught this, and the module now re-evals after every decode.
- Arm 1, free transcription (RNNT, the decision decoder):
  - Agreement 23/40 (58%, 42–71%), tying Nemotron; one-edit flagged 39/40, one-word 31/31, shifted 40/40.
  - One repeat miss: `p05-x6`, where the repeat was written once.
  - No hallucinations. Of the 17 rejections, 10 are partial or dropped edge words, 3 mishearings (a seam and the phone copies), 3 dropped planted words or cut-short output, and 1 canonicalisation.
  - CTC greedy: 19/40, with the same miss, plus letter doubling.
- Arm 2, CTC pairwise likelihood (diacritised spellings, 2-nat margin):
  - Agreement 36/40 (90%, 77–96%), one-edit 37/40, one-word 29/31, shifted 40/40.
  - Repeat miss `p05-x6`: the text without the repeat wins by 98.9 nats.
  - The other misses are the added words `p15-x8` and `e19-x12`, where the canonical text wins, and `p09-x8`, unsure.
  - It gets Nemotron's dangerous miss right: on `p12-x7` the planted `مِهَادًا` beats `كِفَاتًا` by 30 nats.
  - Plain spellings are worse (35/40, plus an `e05-x7` restart miss).
  - Exploratory, in-sample: requiring both Nemotron and FastConformer to clear flags every control with no repeat misses, but agreement drops to 35/40, below the bar. As a pre-screen it would clear 21 of 26 edited windows, against Nemotron's 25, and it closes the `p12-x7` hole. Worth considering if `p12-x7`-type misses worry you more than 4 extra listens per version.
- Independent data check of every planted edit (studio; phone in brackets), said vs original by CTC, 3-nat margin, teacher-check window:
  - mistakes 270/320 (252): wrong word 81/94, mutashabihat 10/10, omitted 42/42, added 68/105, skipped 69/69; repeats 14/23, restarts 17/17;
  - the teacher passes 320/320 studio;
  - in studio none passes only FastConformer's check and none fails both. 3 of the failures are windows you heard as `ok` (`e19-x12`, `e08-x1`, `p15-x5`);
  - 46 of the 59 studio failures are added words or repeats, where the prior deletes inserted speech. Nothing points to a label error;
  - with 2 s of context (exploratory) mistakes reach 281/320, but restarts drop to 11/17.
- Decision against the bar: arm 1 and arm 2 both fail on agreement and on the repeat miss. Not wired.
- Cost: about 12 min of T4 container time in total, roughly $0.15 (estimate).
- Records:
  - on the volume: `/lab/reviews/lab-v1.1/asr-fastconformer-{rnnt,ctc,likelihood}.json`, `asr-calibration.fastconformer[-likelihood].json`, and `/lab/data/lab-v1.1/checks-fastconformer-v1[.ctx2].json`;
  - tables: `docs/benchmarks/lab-review-asr-v1-fastconformer[-likelihood].json` and `lab-checks-fastconformer-v1[-ctx2].json`;
  - code: `modal/review_fastconformer.py`.


2026-10-29, round 2. **Nemotron error analysis, prompt ablation, rasm matching and ROVER: nothing passes the bar. Best is ROVER at 27/40 with all controls held. The default stays `nemotron-likelihood`.** Full write-up: `docs/benchmarks/lab-review-asr-v1.md`, section "Round 2".

- Part A, the error analysis (from saved outputs, no GPU):
  - The comparison is not the bottleneck. Crediting every orthography and edge case, Nemotron would reach at most 26/40 (`verbatim`) or 28/40 (`recitation`).
  - Only `e20-x6` (`كلا علام` for `كالأعلام`) is a pure spelling miss.
  - The rest are 8 mishearings per run (consonant pairs ث/ذ, ت/ط, tanwin endings, lost /h/), canonical-prior drops at the planted word, and dua or verse-pastiche hallucinations.
  - 5 of 12 real errors sit at the seam.
  - Qwen's and Whisper's edge drops are whole words inside the window, so they are not comparison-fixable. Table: `docs/benchmarks/lab-review-asr-v1-errors.json`.
- Part B, the prompt ablation:
  - Results: `verbatim` 23, `recitation` 26, `mistakes` (no "Quran") 24, `quran` 26, `plain` 25. Controls held for all; only `recitation` catches `p15-x8`, and only through a hallucination.
  - Dropping "Quran" does not help, and the mistakes clause is noise. The +1 to +3 comes from leaving the model-card template.
  - An oracle best prompt per window gives 28/40.
- Part C, preregistered (but tuned on the same 40 windows):
  - C1, the rasm comparison with session-time edge context: 25/40 on `recitation`, 23/40 on `verbatim`.
    - It loses to short skeletons that make one-letter errors heavier.
    - It adds a new Qwen repeat miss (`p05-x6`), because a context word absorbs a stray heard word.
  - C2, ROVER over 0.9/1.0/1.1× × raw/denoised (`recitation` prompt): 27/40 (52–80%), one-edit 40/40, one-word 31/31, shifted 40/40, no repeat misses.
  - C1 + C2: 26/40. Adding Qwen3 and Whisper voters: 26/40.
  - DeepFilterNet3 costs 5 to 7 matches per run. Nemotron's errors are systematic, so the perturbations barely decorrelate them.
- Nothing was wired. As a pre-screen, ROVER would keep 28 of 40, against 35 of 40 for `nemotron-likelihood`, but it rests on an opinion that is wrong on a third of correct windows.
- Cost: about 13 min of H100 wall time plus CPU denoising, roughly $1 (estimate).
- Records on the volume: `/lab/reviews/lab-v1.1/asr-nemotron-{mistakes,quran,plain}-prompt.json`, `asr-nemotron-rover.json`, `asr-calibration.nemotron-prompt-ablation.json`, `asr-calibration.round2.json`, and `rover/denoised/*.wav`.
- Possible next step (speculative): free transcription has topped out. What remains is constrained decoding around the labelled text (arm 2's direction) or a model fine-tuned on non-canonical recitation.

Commercial APIs (`docs/benchmarks/lab-review-asr-v1.md`, "Commercial APIs", preregistered in `88db81a`). Soniox `stt-async-v5` (Arabic hint, strict) and Speechmatics `enhanced` heard the same WAVs with only the language, on 152 of your verdicts: 142 ok and 10 bad splices.
- Soniox has the best free-transcription opinion so far:
  - 114/142 on the round-1 rule and 130/142 on the new label-blind judge;
  - 37/40 on the v1-check windows, against 35/40 for `nemotron-likelihood`;
  - one-edit 40/40, one-word 31/31, shifted 40/40;
  - one canonicalisation (e02-x3) and no hallucinations.
- Speechmatics: 86/142 and 105/142, with 17 canonicalisations.
- Neither can see a bad splice. The spliced word is in the audio and both write it: judge flags 1/10 (Soniox) and 3/10 (Speechmatics). Only Nemotron's likelihood catches them, 9/10.
- Cost: $0.10 and $0.75 per hour. Total API spend $0.66.
- Records: `/lab/reviews/<testSet>/asr-{soniox,speechmatics}.json`.

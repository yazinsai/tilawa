# Experiment Results

Full benchmark results for all experiments, run on 2026-04-02. Two test corpora: **v1** (53 samples: user recordings, EveryAyah reference, RetaSy crowdsourced) and **v2** (43 samples: RetaSy expanded + EveryAyah multi-verse).

## Streaming results (TypeScript ONNX pipeline)

The shipped model (`fastconformer-phoneme v4-tlog`, quantized ONNX) tested with the TypeScript `RecitationTracker` feeding 300ms chunks. This is the pipeline that runs in the browser and React Native.

| Mode | Corpus | Recall | Precision | SeqAcc | Correct |
|---|---|---|---|---|---|
| **Streaming** (deferred emission, 2026-04-11) | v1 (53) | **78.6%** | **66.8%** | **47.2%** | 35-40/53 |
| **Streaming** (deferred emission, 2026-04-11) | v2 (43) | **82.7%** | **63.7%** | **46.5%** | 34-36/43 |
| Streaming (pre-2026-04-11 baseline) | v1 (53) | 78.9% | 53.8% | 26.4% | 35-44/53 |
| Streaming (pre-2026-04-11 baseline) | v2 (43) | 80.5% | 56.9% | 32.6% | 33/43 |
| Non-streaming | v1 (53) | 84.1% | 84.9% | 81.1% | 43/53 |
| Non-streaming | v2 (43) | 78.1% | 79.1% | 74.4% | 32/43 |

**Notes:**
- Streaming metrics are medians across 5 runs on v1 and 3 runs on v2. ONNX non-determinism is ±3-6 per run.
- **Deferred emission** (commit `63774dc`, 2026-04-11): auto-advanced `verse_match` messages are held until fresh audio produces primary word alignment on the next verse. If tracking stales, the pending emission is silently dropped with full state rollback. This prevents cascades where verse N completing triggers emission of N+1, N+2, ... without audio evidence.
- Net effect on v1: precision +13.0pp, SeqAcc +20.8pp, recall -0.3pp. Same pattern on v2 (blind check).
- 0 stable-pass → stable-fail regressions across 5 runs.
- Measurement tool: `npx tsx web/frontend/test/stability-report.ts --repeats=5 [--corpus=test_corpus_v2]` produces per-sample stability classification + JSON report.
- Streaming recall remains higher than non-streaming because auto-advance discovers continuation verses; precision was the weak point fixed by deferred emission.
- Non-streaming mode runs the full audio through the ONNX model once and does a single `matchVerse()` call.

## Streaming results (Python, 3s chunks)

All experiments tested with the Python `StreamingPipeline` -- audio chunked into 3s segments, each transcribed independently, accumulated text fed to `VerseTracker` for progressive verse matching. This mirrors the browser streaming pattern.

### Experiments with >0% streaming recall

| Experiment | Base Model | Fine-tuned | Type | Size | v1 Recall | v1 Prec | v1 SeqAcc | v1 Latency | v2 Recall | v2 Prec | v2 SeqAcc | v2 Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **tadabur-whisper-small** | FaisaI/tadabur-Whisper-Small | yes | arabic | 461 MB | **87%** | 58% | 42% | 3.3s | **84%** | 58% | 47% | 3.8s |
| **fastconformer-lm-fusion** | nvidia FastConformer | no | arabic | 115 MB | 82% | **66%** | **55%** | **0.8s** | 74% | **59%** | **53%** | **1.0s** |
| fastconformer-ctc-rescore | nvidia FastConformer | yes | arabic | 260 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| fastconformer-phoneme | nvidia FastConformer | yes | phoneme | 436 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| nvidia-fastconformer | nvidia FastConformer | no | arabic | 115 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| fastconformer-nbest-bruteforce | nvidia FastConformer | no | arabic | 550 MB | 80% | 61% | 49% | 0.8s | 77% | 60% | 51% | 1.0s |
| rabah-pruned-ctc/8L-ft-fn | rabah wav2vec2-xlsr-quran | yes | arabic | 145 MB | 71% | 55% | 42% | 2.7s | 65% | 49% | 40% | 3.4s |
| whisper-lora | openai/whisper-small + LoRA | yes | arabic | 485 MB | 64% | 40% | 19% | 5.6s | 72% | 49% | 37% | 6.3s |
| whisper-small | openai/whisper-small | no | arabic | 461 MB | 63% | 42% | 26% | 3.8s | 53% | 33% | 21% | 6.0s |
| rabah-pruned-ctc/12L-ft-es | rabah wav2vec2-xlsr-quran | yes | arabic | 193 MB | 61% | 41% | 25% | 3.4s | 56% | 40% | 33% | 4.4s |
| two-stage | moonshine-tiny + wav2vec2 | yes | arabic | 463 MB | 47% | 23% | 13% | 3.7s | 38% | 24% | 19% | 5.8s |
| distilled-ctc | wav2vec2-base (distilled) | yes | arabic | 360 MB | 7% | 7% | 6% | 0.5s | 5% | 3% | 2% | 0.5s |

**Notes:**
- w2v-phonemes cannot stream -- it only has `predict()` (whole-file classification), not a `transcribe()` function that works on chunks.
- Streaming recall is generally lower than batch because 3s audio segments often contain only partial verses, making matching harder.
- FastConformer variants dominate: best balance of streaming accuracy, speed, and size.
- `tadabur-whisper-small` has highest streaming recall (87%) but at 3x the latency of FastConformer.

## Batch results (Python benchmark runner)

Full-file transcription using each experiment's native Python pipeline. Metrics averaged across all samples in each corpus.

### Experiments with >0% recall

| Experiment | Base Model | Fine-tuned | Type | Size | v1 Recall | v1 Prec | v1 SeqAcc | v1 Latency | v2 Recall | v2 Prec | v2 SeqAcc | v2 Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **w2v-phonemes/large** | hetchyy/r7 | no | phoneme | 970 MB | **100%** | **100%** | **100%** | 15.2s | **95%** | **95%** | **95%** | 30.4s |
| **fastconformer-quran-lm-fusion** | nvidia FastConformer | no | arabic | 115 MB | 95% | 96% | **94%** | 7.2s | **95%** | **95%** | **95%** | 6.6s |
| **nvidia-fastconformer** | nvidia FastConformer | no | arabic | 115 MB | 95% | 95% | 92% | **0.7s** | 93% | 90% | 86% | **0.9s** |
| fastconformer-phoneme | nvidia FastConformer | yes | phoneme | 436 MB | 95% | 95% | 92% | 7.9s | 93% | 90% | 86% | 7.1s |
| fastconformer-ctc-rescore | nvidia FastConformer | yes | arabic | 260 MB | 95% | 95% | 92% | 7.3s | 93% | 90% | 86% | 6.7s |
| fastconformer-nbest-bruteforce | nvidia FastConformer | no | arabic | 550 MB | 95% | 95% | 92% | 0.6s | 93% | 90% | 86% | 0.9s |
| tadabur-whisper-small | FaisaI/tadabur-Whisper-Small | yes | arabic | 461 MB | 86% | 88% | 79% | 1.3s | 87% | 87% | 81% | 1.4s |
| whisper-lora | openai/whisper-small + LoRA | yes | arabic | 485 MB | 82% | 86% | 77% | 2.3s | 81% | 84% | 79% | 2.1s |
| rabah-pruned-ctc/8L-ft-fn | rabah wav2vec2-xlsr-quran | yes | arabic | 145 MB | 75% | 75% | 74% | 3.7s | 77% | 77% | 77% | 3.9s |
| whisper-small | openai/whisper-small | no | arabic | 461 MB | 73% | 76% | 68% | 1.0s | 50% | 50% | 47% | 1.1s |
| two-stage | moonshine-tiny + wav2vec2 | yes | arabic | 463 MB | 69% | 69% | 66% | 2.3s | 56% | 56% | 51% | 2.2s |
| rabah-pruned-ctc/12L-ft-es | rabah wav2vec2-xlsr-quran | yes | arabic | 193 MB | 63% | 63% | 60% | 5.3s | 67% | 67% | 67% | 5.2s |
| rabah-pruned-ctc/8L-ft-es | rabah wav2vec2-xlsr-quran | yes | arabic | 145 MB | 55% | 55% | 55% | 4.0s | 47% | 47% | 47% | 4.0s |
| rabah-pruned-ctc/6L-ft-es | rabah wav2vec2-xlsr-quran | yes | arabic | 121 MB | 54% | 54% | 51% | 3.3s | 56% | 56% | 56% | 3.1s |
| distilled-ctc | wav2vec2-base (distilled) | yes | arabic | 360 MB | 30% | 29% | 26% | 0.6s | 26% | 26% | 26% | 6.2s |

### Experiments with 0% recall (broken or inapplicable)

| Experiment | Base Model | Type | Size | Reason |
|---|---|---|---|---|
| contrastive | HuBERT + AraBERT | embedding | 900 MB | English encoder produces useless Arabic features |
| contrastive-v2 | HuBERT + AraBERT | embedding | 367 MB | Same fundamental issue as v1 |
| embedding-search | HuBERT + FAISS | embedding | 397 MB | HuBERT encodes speaker identity, not content |
| ctc-alignment | wav2vec2-xlsr-53-arabic | arabic | 1.2 GB | Transcribe path broken; predict() works but runner uses transcribe() |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran | arabic | 290 MB | All samples errored (model loading issue) |
| streaming-asr | mlx-whisper base | arabic | 145 MB | Requires mlx-whisper (not installed) |
| two-stage-faster-whisper-pruned | faster-whisper + pruned CTC | arabic | — | Requires faster-whisper (not installed) |
| w2v-phonemes/base | hetchyy/r15_95m | phoneme | 116 MB | Model too small; phoneme output is garbage |

## Rabah pruned CTC variants

All variants of `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final` with layer pruning and optional fine-tuning.

| Variant | Layers | Pruning | Fine-tuned | v1 Recall | v1 SeqAcc | v2 Recall | v2 SeqAcc | Latency | Size |
|---|---|---|---|---|---|---|---|---|---|
| 8L-ft-fn-int8 | 8 | first_n | yes | **75%** | **74%** | **77%** | **77%** | 3.7s | 145 MB |
| 12L-ft-es-int8 | 12 | evenly_spaced | yes | 63% | 60% | 67% | 67% | 5.3s | 193 MB |
| 12L-int8 | 12 | evenly_spaced | no | 62% | 62% | 51% | 51% | 5.5s | 193 MB |
| 8L-ft-es-int8 | 8 | evenly_spaced | yes | 55% | 55% | 47% | 47% | 4.0s | 145 MB |
| 6L-ft-es-int8 | 6 | evenly_spaced | yes | 54% | 51% | 56% | 56% | 3.3s | 121 MB |
| 8L-int8 | 8 | evenly_spaced | no | 2% | 2% | 0% | 0% | 4.0s | 145 MB |
| 6L-int8 | 6 | evenly_spaced | no | 0% | 0% | 0% | 0% | 3.2s | 121 MB |

**Key insight:** `first_n` pruning (keep layers 0-7) vastly outperforms `evenly_spaced` at the same layer count (75% vs 55% at 8 layers). Fine-tuning the CTC head is essential — unpruned+unfinetuned models score near 0%.

## TLOG data mix experiments

Fine-tuning the phoneme CTC head with varying amounts of TLOG (phone-recorded Quran recitation) data.

| Model | TLOG samples | Quality threshold | Streaming v1 | Streaming v2 | Notes |
|---|---|---|---|---|---|
| **v4-tlog** (best) | ~18K (5/verse) | 0.3 | **45/53 (84.9%)** | **32/43 (74.4%)** | |
| v4-tlog-heavy | ~53K (15/verse) | 0.3 | 36-38/53 (70%) | 25/43 (58%) | |
| v4-tlog-hq | ~74K (30/verse) | 0.5 | 29-31/53 (56%) | 23-24/43 (54%) | |
| v6-augmented | ~29K (5/verse) | unfiltered | 26/53 (49%) NS | — | +MUSAN noise +teacher relabel |

## v6-augmented (REGRESSION, 2026-04-04)

Fine-tuned with three data-side changes vs v4-tlog baseline: MUSAN noise augmentation, teacher pseudo-labels (wav2vec2 relabeling), and unfiltered TLOG (no quality filter on fresh Modal volume).

| Mode | Corpus | Correct | Baseline (v4-tlog) | Delta |
|---|---|---|---|---|
| Non-streaming | v1 (53) | **26/53 (49%)** | 43/53 (81%) | **-17** |
| Streaming | v1 (53) | crashed (ONNX mutex) | 40-44/53 (~77%) | — |

**Training details:**
- Data: 157K entries (71K Iqra + 55K TTS + 1.8K RetaSy + 29K TLOG unfiltered)
- Teacher relabeling: 75% relabeled by wav2vec2 ONNX teacher, 25% original labels (3 of 12 workers preempted)
- Augmentation: MUSAN noise (prob=0.5, 0-20dB SNR) + speed/gain/white_noise/shift/silence. RIR impulse disabled (multi-channel compat issue)
- Training: NeMo <2.7, torch 2.5.1, A100-80GB, best val_loss=58.39 at step 6500, early stopped (patience 6)

**Root cause analysis:**
1. Unfiltered TLOG (29K entries) — no quality filter since fresh volume had no checkpoint model. Previous best used filtered ~18K
2. Teacher pseudo-labels changed 75% of training labels — teacher model may disagree with ground truth, introducing label noise
3. Three changes combined = impossible to attribute regression to any single factor

**Lesson:** Never combine multiple data-side changes in one run. The plan called for staged runs but codec resynth was skipped and the remaining changes were combined. TLOG quality filtering is critical — unfiltered TLOG has ~38% bad samples.

## Methodology

- **Batch mode:** Each experiment's `transcribe()` function processes the full audio file. The `StreamingPipeline` then matches the transcript against all 6,236 Quran verses using Levenshtein distance. Recall/Precision/SeqAcc computed per-sample, then averaged.
- **Python streaming mode:** Audio chunked into 3s segments, each transcribed independently, accumulated text fed to `VerseTracker` for progressive verse matching.
- **TypeScript streaming mode:** The `RecitationTracker` processes 300ms audio chunks through the ONNX model, with 4.0s silence tail for discovery flush. Only available for the shipped ONNX phoneme model.
- **Latency:** Wall-clock time per sample including model loading warmup (first sample excluded).
- **Model size:** As reported by each experiment's `model_size()` function.
- All results from a single run on Apple Silicon (CPU). ONNX non-determinism can cause ±2-3 sample variance per run.

## Per-experiment details

### ctc-alignment

CTC forced alignment using `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (1.2 GB). Scores candidate verses directly against frame-level character logits using the CTC forward algorithm, bypassing the information loss of greedy decoding. Too large (6x) and too slow (5x) for on-device use.

### nvidia-fastconformer

`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` via NeMo. Best balance of accuracy, speed, and size. Fine-tune sweep (v1/v2a/v2b/v3c) did not improve baseline -- all variants regressed.

### fastconformer-ctc-rescore

Two-stage: FastConformer ASR (115 MB) + CTC re-score top-50 candidates using fine-tuned 8L Rabah CTC (145 MB). CTC re-scoring does **not** recover any failures -- both models fail on the same hard cases (short isolated letters, multi-verse passages).

### fastconformer-nbest-bruteforce

N-best beam search + CTC brute-force. **Worse than baseline** -- introduced 2 new failures. CTC beam search without a language model produces near-identical hypotheses. A Quran-specific LM or constrained decoding would be needed.

### fastconformer-quran-lm-fusion

FastConformer + pyctcdecode Quran language model. Best batch SeqAcc (94% v1, 95% v2) but LM adds latency and can't easily run in browser.

### fastconformer-phoneme

Fine-tuned FastConformer CTC head on 69-phoneme Buckwalter vocabulary. This is the shipped ONNX model (`fastconformer_phoneme_q8.onnx`, 131 MB). Trained on 71K Iqra + 55K TTS + 1.8K RetaSy + ~18K TLOG.

### w2v-phonemes

Phoneme-based matching using wav2vec2 CTC models. Large model (970 MB) achieves 100% batch accuracy on v1 -- proves the approach works perfectly -- but too large/slow for real-time. Cannot stream (no `transcribe()` function).

### tadabur-whisper-small

`FaisaI/tadabur-Whisper-Small` -- best Whisper fine-tune. Highest streaming recall (87%) but 3x slower than FastConformer.

### rabah-pruned-ctc

Layer-pruned `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final`. Key finding: `first_n` pruning (keep layers 0-7) gets 75% vs 55% for `evenly_spaced`. Fine-tuning essential -- unpruned models score near 0%.

### two-stage

Moonshine Tiny Arabic (103 MB) for fast ASR + CTC re-score top 50 candidates. Falls back to large CTC model. Blocked on small CTC model.

### whisper-lora / whisper-small

Whisper-small variants. LoRA adapter improves over base but both trail FastConformer significantly, especially in streaming.

### distilled-ctc (failed)

wav2vec2-base knowledge-distilled from large CTC. English-only pretraining means no Arabic speech representations.

### contrastive / contrastive-v2 / embedding-search (failed)

All failed due to English-pretrained audio encoders (HuBERT/wav2vec2-base) producing useless features for Arabic speech.

## Key findings

1. **FastConformer dominates for streaming.** Best speed/accuracy/size tradeoff across all viable experiments.
2. **CTC forced alignment is the most accurate batch approach** but too large (1.2 GB) for on-device.
3. **ASR quality is the bottleneck.** All ASR-based approaches fail on the same samples.
4. **English-pretrained audio encoders fail on Arabic.** wav2vec2-base, HuBERT cannot produce useful features.
5. **Layer pruning + fine-tuning works.** 24→8 layers recovers most accuracy (75% at 145 MB).
6. **Short verses are hard across all approaches.** Verses under 3-4 words don't provide enough signal.
7. **Matching quality matters more than decode strategy.** Multi-pass phoneme matching (fragment scoring, spans) improved Python batch from 79%→90% v1. Beam search (pyctcdecode) is worse than greedy for this model.
8. **Beam candidate injection into tracker causes regressions.** The verse/span trie (1.7M nodes, 2.2ms decode) works correctly but beam-matched verses override correct greedy results when injected as candidates.
9. **TLOG removal regresses.** v5-robust fine-tunes without TLOG both performed worse than v4-tlog. The 18K TLOG sweet spot provides essential phone-mic diversity.

---

## Phase A results (2026-04-03)

Pre-Phase-A baseline: **45/53 (84.9%)** v1, **32/43 (74.4%)** v2 (single runs, before variance was understood).
After A1+A2+A3: **38-44/53 (avg ~41, 77%)** v1 across 3 runs. ONNX non-determinism is ±3-6, much wider than initially assumed. v2 pending re-measure.

**Note:** The earlier 45/53 and 50/53 figures were single-run measurements that fell within the high end of ONNX variance. The realistic streaming v1 baseline is **40-44/53**.

### Implemented

- **A1 (Short-utterance rescue):** When greedy text < 5 chars, CTC-rescores all short-verse candidates (≤15 phoneme tokens). Fires when ≥2 tokenIds, cyclesSinceCommit > 1, and acoustic margin ≥ ACOUSTIC_CLEAR_MARGIN.
- **A2 (Span-aware commit):** When span match is committed (ayah_end set), emits all verses in span and enters tracking on last verse.
- **A3 (Acoustic override ranking):** (a) acousticBest now found by lowest acoustic score among feasible (was first feasible in text-sort order). (b) New `acousticDominant` condition: overrides when acoustic margin ≥ 0.5, candidate text ≥ VERSE_MATCH_THRESHOLD, and lengthFit ≥ 0.5.

### Consistent failures across all 3 runs
- retasy_016 (3:2): garbage CTC output
- retasy_021 (1:7): empty match, low-quality transcription
- retasy_022 (1:7): wrong verse cascade (82:11)
- retasy_023 (110:2): wrong verse match
- retasy_026 (103:3): wrong verse match
- multi_103_001_003: cascade after correct first verse

### Variance-sensitive (fail in some runs, pass in others)
- retasy_000, retasy_005, retasy_012, retasy_020, retasy_027
- user_ikhlas_2_3, multi_036_001_005, multi_067_001_004, multi_002_285_286

### Not implemented
- **A4 (Gated trie beam expansion):** Deferred — remaining failures are model quality, not candidate retrieval.

## Planned experiments (2026-04-03)

Previous baseline: **45/53 (84.9%)** v1, **32/43 (74.4%)** v2.

The 8 v1 failures broke down as: 3 empty results (model output too short/garbage for matching), 3 wrong verse matches (phonemes match a different verse), 2 partial multi-verse (missed first/last verse in sequence).

### Phase A: Inference/tracker fixes (no training)

#### Experiment A1: Short-utterance rescue path

**What:** When greedy text is <5 chars, fall back to `rawPhonemes`/`tokenIds` + CTC rescoring over a curated short-verse/opener candidate set instead of returning empty.

**Why:** The tracker skips discovery when text is too short (`tracker.ts:463`). The 3 "empty results" failures (ref_036001, retasy_021, and one more) hit this path. `quran-db.ts` already has short-query boosts that never get reached.

**Expected impact:** 2-3 of 8 failures. **Complexity:** 1-2 days. **Training:** No.

#### Experiment A2: Span-aware streaming commit

**What:** When discovery selects a span match (ayah_end set), emit all verses in the span or enter a synthetic span-tracking state. Currently only the first ayah is emitted while `lastEmittedRef` jumps to the span end.

**Why:** The commit path emits one verse (`tracker.ts:624`), updates `lastEmittedRef` to span end (`tracker.ts:636`), then enters tracking on only the first verse (`tracker.ts:662`). This mismatch directly explains the 2 partial multi-verse failures (multi_036_001_005, multi_055_001_004).

**Expected impact:** 2-3 of 8 failures. **Complexity:** ~1 day. **Training:** No.

#### Experiment A3: Fix acoustic override ranking

**What:** Keep text-sorted and acoustic-sorted candidate lists separate. Calibrate a combined score over `stage_a_score`, `acousticScore`, `acousticMargin`, `lengthFit`, and continuation bonus. Currently `_rankCandidates()` re-sorts by text score after pulling `acousticBest`, so the "acoustic override" isn't truly acoustic-first.

**Why:** The 3 wrong-verse failures (retasy_012, retasy_016, retasy_022) have correct verses with decent acoustic scores but lower text scores. Proper acoustic-first ranking could recover them.

**Expected impact:** 1-2 of 8 failures. **Complexity:** 0.5-1.5 days. **Training:** No.

#### Experiment A4: Gated trie beam candidate expansion

**What:** Use trie beam prefixes to expand the candidate surah set (not inject as direct candidates). When text confidence is low, beam-suggested surahs get added to `retrieveCandidates` search scope, then CTC/text consensus reranks. The beam infrastructure is already wired in `inference.ts`.

**Why:** Direct beam injection regressed because it bypassed tracker heuristics. Surah-level expansion is safer — it just broadens the search without overriding the ranking.

**Expected impact:** 1-3 of 8 failures. **Complexity:** 1-2 days. **Training:** No.

### Phase B: Model training

#### Experiment B1: Hard-example fine-tune (stage B from v4-tlog)

**What:** Start from the shipped v4-tlog checkpoint (not from scratch). Short low-LR second stage focused on: short/noisy RetaSy clips, clipped-start TLOG samples, huruf-muqatta'at/openers (36:1 "Ya Sin"), and the current failure buckets. Curriculum weighting by difficulty, not more TLOG volume.

**Why:** v4-tlog is the best checkpoint. v4-tlog-heavy and v4-tlog-hq both regressed from adding more TLOG. The next model experiment should be curriculum and hard-example weighting, not raw data scaling.

**Expected impact:** 2-4 of 8 failures. **Complexity:** 3-5 days + export/validation. **Training:** Yes (Modal A100).

#### Experiment B2: Streaming-like augmentation

**What:** Extend the training augmentor with: explicit start/end truncation (not just shift), mild reverb/room coloration, random short-window crops (1-6s from longer recordings with trimmed labels), and adjacent-ayah concatenations. Current augmentor only has speed/gain/white noise/shift/silence.

**Why:** The model never sees what streaming actually produces — partial windows with missing starts/ends. This directly matches the failure modes: empty outputs from clipped starts, wrong matches from partial audio.

**Expected impact:** 2-3 of 8 failures. **Complexity:** 2-4 days. **Training:** Yes (Modal A100).

### Phase C: Matching + distillation (lower priority)

#### Experiment C1: Phoneme n-gram anchor in browser matcher

**What:** Port the rare phoneme n-gram voting idea from `experiments/w2v-phonemes/` into the browser's `quran-db.ts`. When full-string `ratio()` is weak, add surah candidates from rare phoneme 5-grams before span scoring. Requires pre-computing a phoneme n-gram index as a browser asset.

**Why:** w2v-phonemes is perfect on long/multi and only fails on short/noisy. Rare n-gram anchoring is a good retrieval-side complement to the current matcher.

**Expected impact:** 1-2 of 8 failures. **Complexity:** 1-2 days + small JSON asset. **Training:** No.

#### Experiment C2: Teacher distillation (w2v-phonemes/large → FastConformer)

**What:** Use `w2v-phonemes/large` (100% batch accuracy, 970MB) as a phoneme teacher. Generate pseudo-labels or soft logits on TLOG/RetaSy clips, then distill into the current FastConformer student. Unlike the failed `train_distill_modal.py` (which used wav2vec2-base as student), this keeps the proven FastConformer architecture.

**Why:** The teacher captures the task perfectly. The failed distillation attempt mostly falsified the English wav2vec2-base student choice, not the distillation approach itself.

**Expected impact:** 1-3 of 8 failures. **Complexity:** 1-2 weeks. **Training:** Yes (Modal A100).

### Execution order

**Do first (inference-only, ~4 days total):** A1 → A2 → A3 → A4

**Then (training, ~1 week):** B1 → B2

**If needed:** C1 → C2

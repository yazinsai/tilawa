# Benchmark results

Two test corpora: **v1** (53 samples: user recordings, EveryAyah reference, RetaSy crowdsourced) and **v2** (43 samples: RetaSy expanded + EveryAyah multi-verse).

Metrics: **Recall** = fraction of expected verses found. **Precision** = fraction of emitted verses that were expected. **SeqAcc** = emitted set exactly matches expected set.

ONNX inference is non-deterministic at **±3–6 samples per run** on v1 — streaming numbers below are medians over 3 runs (except the deferred-emission changelog entry, which was measured at 5 runs).

## Shipped model

`fastconformer-phoneme v4-tlog` (131 MB quantized ONNX), used in the browser and React Native:

| Mode | Corpus | Recall | Precision | SeqAcc | Correct |
|---|---|---|---|---|---|
| **Browser/RN streaming** (300ms chunks, `RecitationTracker`) | v1 | **78.6%** | **66.8%** | **47.2%** | 35–40/53 |
| **Browser/RN streaming** | v2 | **82.7%** | **63.7%** | **46.5%** | 34–36/43 |
| Non-streaming (full-file, single `matchVerse()`) | v1 | 84.1% | 84.9% | 81.1% | 43/53 |
| Non-streaming (full-file, single `matchVerse()`) | v2 | 78.1% | 79.1% | 74.4% | 32/43 |

### Streaming changelog

**2026-04-11 — deferred emission** (commit `63774dc`)  
Auto-advanced `verse_match` messages are now held as *pending* until fresh audio produces primary word alignment on the next verse; if tracking stales, the pending emission is silently dropped with full state rollback. This prevents cascades where verse N completing triggers emission of N+1, N+2, … without audio evidence.

v1: precision **53.8% → 66.8%** (+13.0pp), SeqAcc **26.4% → 47.2%** (+20.8pp), recall **78.9% → 78.6%** (−0.3pp). Same pattern on v2 blind check. 0 stable-pass → stable-fail regressions across 5 runs.

Measurement tool: `npx tsx web/frontend/test/stability-report.ts --repeats=5 [--corpus=test_corpus_v2]` produces per-sample stability classification + JSON.

**2026-04-03 — Phase A fixes**  
Short-utterance CTC rescue, span-aware commit, acoustic-dominant override. Also widened our understanding of variance: ONNX is ±3–6 samples/run on v1 (not ±2–3 as previously assumed). Earlier one-shot 45/53 and 50/53 figures sat at the high end of that distribution; the realistic pre-deferred-emission streaming baseline was 40–44/53.

## All experiments — streaming (Python, 3s chunks)

`StreamingPipeline` feeds 3s audio segments to each model, accumulates text into `VerseTracker` for progressive matching. Mirrors the browser pattern but with larger chunks.

| Experiment | Base model | FT | Type | Size | v1 Rec | v1 Prec | v1 Seq | v1 Lat | v2 Rec | v2 Prec | v2 Seq | v2 Lat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **tadabur-whisper-small** | FaisaI/tadabur-Whisper-Small | ✓ | arabic | 461 MB | **87%** | 58% | 42% | 3.3s | **84%** | 58% | 47% | 3.8s |
| **fastconformer-lm-fusion** | nvidia FastConformer | — | arabic | 115 MB | 82% | **66%** | **55%** | **0.8s** | 74% | **59%** | **53%** | **1.0s** |
| fastconformer-ctc-rescore | nvidia FastConformer | ✓ | arabic | 260 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| fastconformer-phoneme | nvidia FastConformer | ✓ | phoneme | 436 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| nvidia-fastconformer | nvidia FastConformer | — | arabic | 115 MB | 81% | 64% | 53% | 1.0s | 77% | 61% | 53% | 1.2s |
| fastconformer-nbest-bruteforce | nvidia FastConformer | — | arabic | 550 MB | 80% | 61% | 49% | 0.8s | 77% | 60% | 51% | 1.0s |
| rabah-pruned-ctc/8L-ft-fn | rabah wav2vec2-xlsr-quran | ✓ | arabic | 145 MB | 71% | 55% | 42% | 2.7s | 65% | 49% | 40% | 3.4s |
| whisper-lora | whisper-small + LoRA | ✓ | arabic | 485 MB | 64% | 40% | 19% | 5.6s | 72% | 49% | 37% | 6.3s |
| whisper-small | whisper-small | — | arabic | 461 MB | 63% | 42% | 26% | 3.8s | 53% | 33% | 21% | 6.0s |
| rabah-pruned-ctc/12L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 193 MB | 61% | 41% | 25% | 3.4s | 56% | 40% | 33% | 4.4s |
| two-stage | moonshine-tiny + wav2vec2 | ✓ | arabic | 463 MB | 47% | 23% | 13% | 3.7s | 38% | 24% | 19% | 5.8s |
| distilled-ctc | wav2vec2-base (distilled) | ✓ | arabic | 360 MB | 7% | 7% | 6% | 0.5s | 5% | 3% | 2% | 0.5s |

`tadabur-whisper-small` has the highest raw streaming recall but at 3–5× FastConformer latency. FastConformer variants dominate the speed/accuracy/size frontier. `w2v-phonemes` cannot stream — no chunked `transcribe()` path.

## All experiments — batch (Python, full-file)

Full-file transcription then single `matchVerse()` call.

| Experiment | Base model | FT | Type | Size | v1 Rec | v1 Prec | v1 Seq | v1 Lat | v2 Rec | v2 Prec | v2 Seq | v2 Lat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **w2v-phonemes/large** | hetchyy/r7 | — | phoneme | 970 MB | **100%** | **100%** | **100%** | 15.2s | **95%** | **95%** | **95%** | 30.4s |
| **fastconformer-lm-fusion** | nvidia FastConformer | — | arabic | 115 MB | 95% | 96% | **94%** | 7.2s | **95%** | **95%** | **95%** | 6.6s |
| **nvidia-fastconformer** | nvidia FastConformer | — | arabic | 115 MB | 95% | 95% | 92% | **0.7s** | 93% | 90% | 86% | **0.9s** |
| fastconformer-phoneme | nvidia FastConformer | ✓ | phoneme | 436 MB | 95% | 95% | 92% | 7.9s | 93% | 90% | 86% | 7.1s |
| fastconformer-ctc-rescore | nvidia FastConformer | ✓ | arabic | 260 MB | 95% | 95% | 92% | 7.3s | 93% | 90% | 86% | 6.7s |
| fastconformer-nbest-bruteforce | nvidia FastConformer | — | arabic | 550 MB | 95% | 95% | 92% | 0.6s | 93% | 90% | 86% | 0.9s |
| tadabur-whisper-small | FaisaI/tadabur-Whisper-Small | ✓ | arabic | 461 MB | 86% | 88% | 79% | 1.3s | 87% | 87% | 81% | 1.4s |
| whisper-lora | whisper-small + LoRA | ✓ | arabic | 485 MB | 82% | 86% | 77% | 2.3s | 81% | 84% | 79% | 2.1s |
| rabah-pruned-ctc/8L-ft-fn | rabah wav2vec2-xlsr-quran | ✓ | arabic | 145 MB | 75% | 75% | 74% | 3.7s | 77% | 77% | 77% | 3.9s |
| whisper-small | whisper-small | — | arabic | 461 MB | 73% | 76% | 68% | 1.0s | 50% | 50% | 47% | 1.1s |
| two-stage | moonshine-tiny + wav2vec2 | ✓ | arabic | 463 MB | 69% | 69% | 66% | 2.3s | 56% | 56% | 51% | 2.2s |
| rabah-pruned-ctc/12L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 193 MB | 63% | 63% | 60% | 5.3s | 67% | 67% | 67% | 5.2s |
| rabah-pruned-ctc/8L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 145 MB | 55% | 55% | 55% | 4.0s | 47% | 47% | 47% | 4.0s |
| rabah-pruned-ctc/6L-ft-es | rabah wav2vec2-xlsr-quran | ✓ | arabic | 121 MB | 54% | 54% | 51% | 3.3s | 56% | 56% | 56% | 3.1s |
| distilled-ctc | wav2vec2-base (distilled) | ✓ | arabic | 360 MB | 30% | 29% | 26% | 0.6s | 26% | 26% | 26% | 6.2s |

### Phoneme matcher: strategy comparison

Shipped ONNX phoneme model via Python `predict()`, swapping out the matching strategy:

| Matching strategy | v1 Recall | v1 SeqAcc | v2 Recall | v2 SeqAcc |
|---|---|---|---|---|
| Simple `ratio()` | 79% | 75% | 87% | 86% |
| **Multi-pass (fragment + span)** | **90%** | **87%** | **87%** | **84%** |

The multi-pass matcher (ported from the browser's `quran-db.ts` — fragment scoring, short-query boost, bismillah stripping, multi-verse spans) adds +11pp v1 recall at zero decode cost. Matching quality was the bottleneck, not decoding.

### 0% recall — broken or inapplicable

| Experiment | Base model | Type | Size | Reason |
|---|---|---|---|---|
| contrastive | HuBERT + AraBERT | embedding | 900 MB | English encoder → useless Arabic features |
| contrastive-v2 | HuBERT + AraBERT | embedding | 367 MB | Same fundamental issue as v1 |
| embedding-search | HuBERT + FAISS | embedding | 397 MB | HuBERT encodes speaker identity, not content |
| ctc-alignment | wav2vec2-xlsr-53-arabic | arabic | 1.2 GB | `transcribe()` path broken; runner uses it |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran | arabic | 290 MB | Model loading errors on all samples |
| streaming-asr | mlx-whisper base | arabic | 145 MB | Needs mlx-whisper (not installed) |
| two-stage-faster-whisper-pruned | faster-whisper + pruned CTC | arabic | — | Needs faster-whisper (not installed) |
| w2v-phonemes/base | hetchyy/r15_95m | phoneme | 116 MB | Model too small; phoneme output unusable |

## Deep dive: Rabah pruned CTC variants

Layer pruning + optional fine-tuning applied to `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final`.

| Variant | Layers | Pruning | FT | v1 Rec | v1 Seq | v2 Rec | v2 Seq | Lat | Size |
|---|---|---|---|---|---|---|---|---|---|
| 8L-ft-fn-int8 | 8 | first_n | ✓ | **75%** | **74%** | **77%** | **77%** | 3.7s | 145 MB |
| 12L-ft-es-int8 | 12 | evenly_spaced | ✓ | 63% | 60% | 67% | 67% | 5.3s | 193 MB |
| 12L-int8 | 12 | evenly_spaced | — | 62% | 62% | 51% | 51% | 5.5s | 193 MB |
| 8L-ft-es-int8 | 8 | evenly_spaced | ✓ | 55% | 55% | 47% | 47% | 4.0s | 145 MB |
| 6L-ft-es-int8 | 6 | evenly_spaced | ✓ | 54% | 51% | 56% | 56% | 3.3s | 121 MB |
| 8L-int8 | 8 | evenly_spaced | — | 2% | 2% | 0% | 0% | 4.0s | 145 MB |
| 6L-int8 | 6 | evenly_spaced | — | 0% | 0% | 0% | 0% | 3.2s | 121 MB |

`first_n` pruning (keep layers 0–7) beats `evenly_spaced` by ~20pp at the same layer count. Fine-tuning the CTC head is non-optional — unfinetuned pruned models score near 0%.

## Deep dive: TLOG data-mix fine-tunes

Fine-tuning the phoneme CTC head with varying amounts of TLOG (phone-recorded recitation).

| Model | TLOG | Filter | Streaming v1 | Streaming v2 | Notes |
|---|---|---|---|---|---|
| **v4-tlog** (shipped) | ~18K (5/verse) | 0.3 | **45/53 (85%)** † | **32/43 (74%)** † | best checkpoint |
| v5-robust-u6 | 0 (no TLOG) | — | 43/53 (81%) | 33/43 (77%) | removing TLOG also hurts |
| v4-tlog-heavy | ~53K (15/verse) | 0.3 | 36–38/53 (70%) | 25/43 (58%) | regression |
| v4-tlog-hq | ~74K (30/verse) | 0.5 | 29–31/53 (56%) | 23–24/43 (54%) | bigger regression |
| v6-augmented | ~29K (5/verse) | none | 26/53 (49%) NS | — | +MUSAN +teacher relabel, worst |

† v4-tlog figures are single-run; the post-Phase-A median was 40–44/53 v1.

**Takeaways:** ~18K TLOG at filter=0.3 is a genuine sweet spot. Scaling up volume regresses; removing TLOG also regresses; combining multiple data-side changes (v6) makes attribution impossible. **Rule: one data change per training run.**

**v6-augmented failure detail:** unfiltered TLOG (29K) + teacher pseudo-labels on 75% of samples + MUSAN noise aug, all together. Training metrics looked healthy (val_loss=58.39 at step 6500) but downstream accuracy collapsed. Unfiltered TLOG alone contains ~38% bad samples per the quality filter; the teacher relabeler added an unknown additional error rate on the rest. Streaming export also crashed with an ONNX mutex error (NeMo <2.7 compat).

## Per-experiment notes

**ctc-alignment** — CTC forced alignment with `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (1.2 GB). Scores verses directly against frame-level logits via the CTC forward algorithm, skipping greedy-decode information loss. Too large (6×) and too slow (5×) for on-device.

**nvidia-fastconformer** — `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`. Best speed/accuracy/size balance for streaming. A fine-tune sweep (v1, v2a, v2b, v3c) failed to beat the zero-shot baseline.

**fastconformer-ctc-rescore** — Two-stage: FastConformer ASR + CTC re-score top-50 candidates with the fine-tuned 8L Rabah head. Re-scoring doesn't recover failures — both models miss the same hard cases (short isolated letters, multi-verse).

**fastconformer-nbest-bruteforce** — N-best beam search + CTC brute-force. Regressed vs baseline: beam candidates without an LM are near-identical. A Quran-specific LM or constrained decode would be needed.

**fastconformer-lm-fusion** — FastConformer + pyctcdecode Quran LM. Best batch SeqAcc (94% v1, 95% v2) but too much added latency for streaming and awkward in-browser.

**fastconformer-phoneme** — Fine-tuned FastConformer CTC head on a 69-phoneme Buckwalter vocab. Shipped ONNX model (`fastconformer_phoneme_q8.onnx`, 131 MB). Trained on 71K Iqra + 55K TTS + 1.8K RetaSy + ~18K filtered TLOG.

**w2v-phonemes** — Phoneme CTC + Levenshtein matching. Large variant (970 MB) hits 100% batch on v1, proving the approach works; too large/slow for real-time. No streaming path.

**tadabur-whisper-small** — Best Whisper fine-tune we tested. Highest streaming recall (87% v1) at 3× FastConformer latency.

**rabah-pruned-ctc** — Layer-pruned Rabah CTC; see deep-dive above.

**two-stage** — Moonshine Tiny Arabic (103 MB) for fast ASR + CTC re-score on top 50 candidates, falling back to a large CTC. Blocked on the small CTC model.

**whisper-lora / whisper-small** — Whisper-small base + optional LoRA. LoRA helps vs base; both trail FastConformer, especially streaming.

**distilled-ctc (failed)** — wav2vec2-base knowledge-distilled from a large CTC teacher. English-only pretraining means no usable Arabic speech features.

**contrastive / contrastive-v2 / embedding-search (failed)** — All three failed for the same reason: English-pretrained audio encoders (HuBERT, wav2vec2-base) don't produce useful features for Arabic.

## Key findings

1. **FastConformer dominates for streaming.** Best speed/accuracy/size tradeoff across every viable experiment.
2. **CTC forced alignment is the most accurate batch approach**, but too large (1.2 GB) for on-device.
3. **ASR quality is the bottleneck.** All ASR-based approaches fail on the same samples.
4. **English-pretrained audio encoders fail on Arabic.** wav2vec2-base, HuBERT, Moonshine can't produce useful features.
5. **Pruning + fine-tuning works.** 24→8 layers with `first_n` pruning + CTC fine-tuning recovers most accuracy (75% at 145 MB).
6. **Short verses are hard across all approaches** — under 3–4 words doesn't give enough signal.
7. **Matching quality matters more than decode strategy.** Multi-pass phoneme matching takes Python batch from 79%→90% v1. pyctcdecode beam is worse than greedy for this model.
8. **Beam-candidate injection into the tracker regressed.** The verse/span trie (1.7M nodes, 2.2ms decode) works correctly, but beam-matched verses override correct greedy results. Surah-level expansion is the safer next step.
9. **TLOG: one quality-filtered bucket wins.** ~18K filtered at 0.3 is the sweet spot; more volume, lower filter, no TLOG, or combined data changes all regress.
10. **Streaming precision had a cascade bug.** Auto-advanced `verse_match` messages emitted without audio evidence. Deferred emission (2026-04-11) fixes it: +13pp precision, +20.8pp SeqAcc on v1.

## Methodology

- **Batch:** experiment's `transcribe()` processes the full audio file. `StreamingPipeline` matches transcript against all 6,236 verses via Levenshtein. Per-sample R/P/SeqAcc, averaged.
- **Python streaming:** 3s chunks, independent transcription per chunk, accumulated text fed to `VerseTracker` for progressive matching.
- **Browser/RN streaming:** `RecitationTracker` feeds 300ms chunks through ONNX with a 4s silence tail to flush discovery. Only the shipped ONNX phoneme model runs in this mode.
- **Latency:** wall-clock per sample, excluding first-sample warmup. Apple Silicon (CPU).
- **Variance:** ONNX inference is non-deterministic at ±3–6 samples/run on v1. Always report medians over 3 runs (max).

Raw JSON results live in `benchmark/results/`. Stability JSON from streaming runs lives in `web/frontend/test/*-stability.json`.

## Roadmap

Designs in `docs/plans/` for the work remaining between 78.6% streaming recall and the 95% target:

- **Curriculum / hard-example fine-tune (v7)** — start from v4-tlog, short low-LR second stage weighted by current failure buckets: short/noisy RetaSy, huruf-muqatta'at openers, clipped-start TLOG.
- **Streaming-like augmentation** — explicit start/end truncation, mild reverb, random short-window crops, adjacent-ayah concatenation. Current augmentor only has speed/gain/noise/shift/silence; the model never sees what streaming actually produces.
- **Phoneme n-gram anchoring in the browser matcher** — port rare-phoneme voting from `experiments/w2v-phonemes/` into `quran-db.ts` for surah-level expansion when `ratio()` is weak.
- **Teacher distillation (w2v-phonemes/large → FastConformer)** — use the 100%-batch teacher to generate soft labels. The earlier failed distillation used English wav2vec2-base as the student; that's what falsified, not the distillation idea.
- **Deferred A4 — gated trie beam candidate expansion** — expand the candidate surah set (don't inject direct candidates). Beam infrastructure already wired in `inference.ts`.

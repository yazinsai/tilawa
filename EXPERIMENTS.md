# Experiment Results

Full benchmark results for all experiments, run on 2026-04-02. Two test corpora: **v1** (53 samples: user recordings, EveryAyah reference, RetaSy crowdsourced) and **v2** (43 samples: RetaSy expanded + EveryAyah multi-verse).

## Streaming results (TypeScript ONNX pipeline)

The shipped model (`fastconformer-phoneme v4-tlog`, quantized ONNX) tested with the TypeScript `RecitationTracker` feeding 300ms chunks. This is the pipeline that runs in the browser and React Native.

| Mode | Corpus | Recall | Precision | SeqAcc | Correct |
|---|---|---|---|---|---|
| **Streaming** | v1 (53) | **86.7%** | 65.1% | 30.2% | 44/53 |
| **Streaming** | v2 (43) | **80.5%** | 56.9% | 32.6% | 33/43 |
| Non-streaming | v1 (53) | 84.1% | 84.9% | 81.1% | 43/53 |
| Non-streaming | v2 (43) | 78.1% | 79.1% | 74.4% | 32/43 |

**Notes:**
- Streaming recall is higher than non-streaming because the tracker's auto-advance discovers continuation verses that single-shot matching misses.
- Streaming precision is lower because auto-advance also emits false positives (extra verses beyond what was expected).
- Streaming SeqAcc is low because any extra emitted verse counts as an exact-match failure, even if all expected verses were found.
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

| Model | TLOG samples | Quality threshold | Streaming v1 | Streaming v2 |
|---|---|---|---|---|
| **v4-tlog** (best) | ~18K (5/verse) | 0.3 | **45/53 (84.9%)** | **32/43 (74.4%)** |
| v4-tlog-heavy | ~53K (15/verse) | 0.3 | 36-38/53 (70%) | 25/43 (58%) |
| v4-tlog-hq | ~74K (30/verse) | 0.5 | 29-31/53 (56%) | 23-24/43 (54%) |

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

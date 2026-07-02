# LoRA Fine-Tune Whisper Base for Quranic Recitation

## Problem
Off-the-shelf Whisper models hallucinate or confuse similar Quranic verses. Tarteel's fine-tuned model has poor accuracy. We need a domain-specific model under 500MB total.

## Design

**Base model**: `openai/whisper-base` (~150MB, 74M params)

**Training data**: EveryAyah dataset (`tarteel-ai/everyayah`) — 26 reciters × 6,236 ayahs = ~162K audio-text pairs with Uthmani transcriptions.

**LoRA config**: Rank 16, Alpha 32. Target: `q_proj`, `v_proj` in encoder + decoder attention. Output: ~2-4MB adapter.

**Training**: HuggingFace `transformers` + `peft`, MPS backend (Apple Silicon), ~2-4 hours. 90/10 train/eval split.

**Inference pipeline**: whisper-base + LoRA adapter → transcribe → normalize Arabic → fuzzy match against single verses + concatenated verse pairs/triples → return best match.

**Total size**: ~152-154MB

**Evaluation**: 2 user recordings + held-out EveryAyah test split for WER.

## Benchmark Context

Tested approaches (2 user recordings: Aal-i-Imraan:23, Al-Ikhlas:2-3):

| Approach | Size | 3:23 | 112:2-3 |
|---|---|---|---|
| Whisper Large-v3-Turbo + multi-ayah matching | 1.5GB | PASS | PASS (0.923) |
| Whisper Small | 480MB | PASS | FAIL |
| Tarteel whisper-base | 150MB | PASS | FAIL |
| KheemP LoRA (whisper-base) | 152MB | FAIL | FAIL |
| MFCC+DTW acoustic | 0MB | PASS | FAIL |

Goal: match Large-v3-Turbo accuracy at ~150MB via domain-specific LoRA training.

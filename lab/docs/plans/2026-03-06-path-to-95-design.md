# Path to 95% Accuracy — Design Document

**Date:** 2026-03-06
**Goal:** Raise combined accuracy from ~65% streaming / ~73% non-streaming to 95%+ across both test corpuses (96 total samples).

---

## Current State

| Corpus | Streaming | Non-Streaming |
|--------|-----------|---------------|
| v1 (53) | 36-39/53 (68-74%) | 38/53 (72%) |
| v2 (43) | 25/43 (58%) | 32/43 (74%) |
| **Combined** | **~62/96 (65%)** | **~70/96 (73%)** |

## Failure Analysis

### Gap Analysis (streaming vs non-streaming)

| Category | Count | Meaning |
|----------|-------|---------|
| Both pass | 56 | Working correctly |
| Both fail | 21 | Fundamental model/matching issue |
| Streaming-only fail | 11 | Streaming architecture problem (passes non-streaming) |
| Non-streaming-only fail | 8 | Auto-advance already helps these |

### Root Causes of 21 "Both Fail" Samples

**Model transcription quality (10 samples):** All retasy crowd-sourced recordings (1-9s). Model produces completely wrong phonemes — matches to wrong surahs entirely. Examples: retasy_009 (111:3→96:18), retasy_v2_003 (1:3→25:38), retasy_v2_008 (112:2→81:26).

**Long verse / window problem (5 samples):** Professional everyayah recordings (37-157s) where even non-streaming fails. The model struggles with very long verses — ea_004012 (157s, 645 phonemes), ea_049013 (49s), ea_017023 (61s).

**Multi-verse matching logic (4 samples):** multi_113/114 have a documented prefix-scoring fix (correct surah doesn't enter top-20). ea_multi_002/019 are hard segmentation problems.

**Model architectural limitation (2 samples):** Ya-Sin letter name (36:1) — unfixable with current model. Hallucination in multi_055.

### 11 Streaming-Only Failures (easiest wins)

Pass non-streaming, fail streaming:
- v2 retasy (5): Short clips where streaming thresholds/timing prevent match
- v2 everyayah (3): Long verses where chunked discovery loses correct verse
- v1 (3): ref_024035 (80s), ref_048029 (77s), retasy_010

---

## Architecture: Two Parallel Workstreams

### Workstream A: Algorithm Improvements (no model changes)

**A1. Close Streaming Gap (+11 samples)**
- Lower `FIRST_MATCH_THRESHOLD` for short audio clips
- Increase `DISCOVERY_REPEAT_CYCLES` from 2→3 for longer audio
- Better silence flush: for long audio, trigger intermediate discovery with accumulated audio
- Allow re-transcription of full window after extended silence (not just tail)

**A2. Multi-verse Prefix Scoring (+2 samples)**
- In Pass 1, for each verse: also compute `ratio(text_prefix, verse_phonemes)` where `text_prefix` = first N words (N = verse word count)
- This lets short verses score correctly against long multi-verse transcripts
- Documented fix in MULTI_VERSE_FAILURES.md

**A3. CTC Acoustic-Primary Selection for Short Clips (+3-5 samples)**
- Key hypothesis: even when greedy CTC decode produces wrong text, the CTC forward probability for the correct verse may still outscore alternatives
- For short clips (<6s, <15 phoneme words): broaden candidate set to top-100
- Use `scoreCtcSequence()` as the PRIMARY ranking signal (not just margin gating)
- Test on failing retasy samples to validate hypothesis before full implementation

**A4. Long Verse Progressive Matching (+2-3 samples)**
- For verses >30s: use sliding window with overlap
- Match each window independently, accumulate evidence across windows
- Emit verse_match when cumulative confidence exceeds threshold

**Expected yield from Workstream A: 78-85/96 (81-89%)**

### Workstream B: Model Fine-Tuning

**B1. Hypothesis Validation (1 day)**
- Run CTC rescore for all 10 failing retasy samples: does the correct verse rank higher acoustically even when text matching fails?
- If yes: CTC-primary selection (A3) handles it, lower priority for fine-tuning
- If no: model genuinely produces wrong acoustic posteriors, fine-tuning is essential

**B2. LoRA Fine-Tuning on Phoneme CTC Head**
- Architecture: LoRA rank 8-16 on the CTC decoder layers (NOT encoder — preserve acoustic features)
- Previous V1 failure (froze 10/18 encoder layers, 2K steps, 5e-5 LR, CTC-only) regressed by 4%.
- Key differences from V1:
  - LoRA instead of full fine-tuning (much less catastrophic forgetting risk)
  - Focus on CTC head only (encoder stays frozen via LoRA adapter)
  - Data augmentation: speed perturbation (0.9-1.1x), additive noise, random crop to 1-6s
  - Validation-based early stopping (not fixed step count)

**B3. Training Data Strategy**
- Primary: RetaSy corpus (crowd-sourced, diverse speakers, short clips)
- Secondary: EveryAyah (professional, clean, full verses)
- Augmentation:
  - Speed perturbation: 0.9x, 0.95x, 1.0x, 1.05x, 1.1x
  - Random noise injection at SNR 15-30dB
  - Random crop: extract 1-6s segments from longer recordings
  - Random trim: remove first/last 0.5-2s to simulate partial verse
- Teacher labels: Use Whisper Large-v3 as oracle for ambiguous samples

**B4. ONNX Export and Quantization**
- Export fine-tuned model to ONNX
- Quantize to uint8 (same as current production)
- Validate on both corpuses before deployment

**Expected yield from Workstream B: +5-10 samples on top of Workstream A**

---

## Implementation Order

Both workstreams run concurrently:

### Week 1-2
- **A1**: Close streaming gap (discovery timing, thresholds, flush)
- **A2**: Multi-verse prefix scoring
- **B1**: CTC rescore hypothesis validation on retasy failures

### Week 2-3
- **A3**: CTC acoustic-primary selection (if B1 validates hypothesis)
- **A4**: Long verse progressive matching
- **B2**: LoRA fine-tuning setup and first training runs

### Week 3-4
- **B2/B3**: Training iterations, validation, hyperparameter tuning
- **B4**: ONNX export, quantization, integration
- Edge case hardening, per-sample diagnosis of remaining failures

### Target
- After Workstream A: 82-85/96 (85-89%)
- After Workstream B: 91+/96 (95%+)

---

## Key Risks

1. **CTC hypothesis may be false**: If acoustic posteriors don't distinguish correct verse for retasy samples, algorithm improvements cap at ~85%. Fine-tuning becomes mandatory.
2. **LoRA fine-tuning may regress**: Previous V1 attempt regressed. Mitigation: LoRA is much lighter touch, plus validation-based early stopping.
3. **Long verse matching is architecturally hard**: Sliding window + evidence accumulation adds significant complexity. May need to accept some long-verse failures.
4. **ONNX non-determinism**: ±2-3 variance per run means we need 3x testing for every change. Slows iteration.

## Success Criteria

- Combined streaming accuracy ≥ 91/96 (95%) across 3x test runs (median)
- No regression in non-streaming accuracy
- Model size stays ≤ 150MB (current: 131MB)
- Inference latency stays ≤ 0.5s per chunk

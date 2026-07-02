# FastConformer Phoneme Robustness Training Design

**Date:** 2026-03-07
**Owner:** Codex
**Status:** Final design for the next training run

## Decision

The next training run should use the existing FastConformer phoneme CTC path in:

- `scripts/train_fastconformer_phoneme_modal.py`
- `scripts/export_phoneme_onnx_modal.py`
- `web/frontend/test/validate-streaming.ts`
- `web/frontend/test/validate-streaming-matrix.ts`

This supersedes the LoRA-only plan. The repo already has a real NeMo FastConformer phoneme training and export path; it does not have a production-ready LoRA adapter workflow for this model family.

## Why This Replaces the LoRA Plan

The older `2026-03-07-lora-finetuning-plan.md` was directionally right about needing model work, but it is not the right run design for this repository.

Main differences:

| Topic | Older LoRA plan | Final design |
|---|---|---|
| Training target | LoRA on FastConformer CTC head | Full CTC-only FastConformer phoneme fine-tune |
| Repo alignment | Assumed a LoRA path that does not exist here for NeMo FastConformer | Uses the actual scripts already in this repo |
| Export path | LoRA merge then ONNX | NeMo checkpoint -> ONNX -> Q8 export |
| Data plan | High-level RetaSy + EveryAyah mix | Explicit clean/noisy/long-form bands with mapping and augmentation |
| Promotion gate | General "no regressions" | Frozen browser validator and matrix gates |
| Prior evidence used | Mostly the failed text-model fine-tune | Failed text-model fine-tunes plus successful pruned CTC fine-tune plus streaming plateau evidence |

## Evidence Reviewed

### 1. The plain text FastConformer is still the strongest full-file model

`README.md` and `benchmark/results/*.json` show `nvidia-fastconformer` is still the best general full-file experiment in the main benchmark class:

- baseline around `85%` sequence accuracy
- low latency
- small enough deployment footprint

That means the architecture itself is not the problem.

### 2. The previous text-model FastConformer fine-tune sweep regressed

From:

- `docs/plans/2026-02-27-fastconformer-finetune-plan.md`
- `docs/plans/2026-02-27-fastconformer-finetune-research.md`
- `README.md`

Observed results:

- `v1` regressed to `81%`
- `v2a` regressed to `80%`
- `v2b` regressed to `81%`
- `v3c` regressed to `80%`

Likely causes:

- too little adaptation in some runs
- not enough steps
- wrong objective for the browser phoneme pipeline
- insufficient robustness data and augmentation

This is strong evidence against repeating the same text-transcript fine-tune path.

### 3. The browser path is now limited more by phoneme model robustness than tracker logic

From:

- `docs/plans/2026-03-06-path-to-95-plan.md`
- `docs/plans/2026-03-06-path-to-95-design.md`
- `tasks/streaming-accuracy-spec.md`
- `web/frontend/test/validate-streaming-matrix.ts`

What changed:

- several runtime and matching improvements already shipped
- combined streaming improved, but the system is still far from the holdout gate
- the remaining misses are concentrated in short noisy RetaSy clips, long ayahs, and a few multi-verse/opening cases

The runtime work helped. It did not remove the model bottleneck.

### 4. Fine-tuning does work in this repo when the data/task fit is correct

From:

- `scripts/train_pruned_ctc_modal.py`
- `README.md`

Important lesson:

- the pruned Rabah CTC line improved materially after fine-tuning on EveryAyah + RetaSy
- `first_n` pruning plus fine-tuning worked much better than naive alternatives

That matters because it shows domain-matched acoustic fine-tuning is not a dead end here. The failed text FastConformer sweep should not be misread as "all fine-tuning fails".

### 5. The current phoneme training path is real, but incomplete for the next push

The current `scripts/train_fastconformer_phoneme_modal.py` already provides:

- phoneme CTC head replacement
- CTC-only training step
- Iqra train/dev loading
- Iqra TTS blend
- Modal A100 training
- ONNX export and Q8 quantization path

But it is still missing or under-specifying the key robustness pieces for the next run:

- no RetaSy noisy-domain blend in the current checked-in phoneme script
- no explicit augmentation pipeline
- no checkpoint averaging
- no configurable variant-aware export path
- train loader currently clamps `train_ds.max_duration` to `15.0s` even though data prep allows up to `30.0s`

That duration mismatch is especially important because long-ayah robustness is one of the main reasons to run this training cycle.

## Final Training Objective

Train a phoneme-level FastConformer model for browser-like Quran recitation conditions:

- short noisy crowd recordings
- clipped starts and ends
- phone mic loudness variation
- room coloration and mild reverb
- long ayahs
- multi-ayah continuity windows

The objective is not "best clean phoneme transcription". The objective is "best frozen-validator behavior after export".

## Non-Goals

1. Do not fine-tune directly against `benchmark/test_corpus` or `benchmark/test_corpus_v2`.
2. Do not reopen Whisper LoRA, Moonshine, or contrastive retrieval as the primary training path.
3. Do not change the browser benchmark logic to make a candidate pass.
4. Do not treat full-file Python benchmark wins as a ship gate for the browser model.

## Run Design

### Training target

Base model:

- `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`

Training script:

- `scripts/train_fastconformer_phoneme_modal.py`

Export script:

- `scripts/export_phoneme_onnx_modal.py`

Ship gate:

- `web/frontend/test/validate-streaming-matrix.ts`

### Data bands

Use three explicit bands.

#### A. Clean phoneme-aligned base

Sources:

- `IqraEval/Iqra_train`
- `IqraEval/Iqra_TTS`

Purpose:

- preserve canonical phoneme coverage
- stabilize the model on correct phoneme sequences

#### B. Noisy domain blend

Source:

- `RetaSy/quranic_audio_dataset`

Requirements:

- filter invalid labels
- map each kept sample to canonical verse phonemes via Quran text normalization
- skip unmapped items and record counts

Purpose:

- train on the same short noisy conditions dominating holdout failures

#### C. Long-form continuity coverage

Sources:

- longest valid Iqra items already available
- synthetic concatenations of adjacent ayahs from phoneme-aligned clean data

Purpose:

- improve long-ayah stability
- reduce drift across multi-ayah windows

### Data weighting

Initial target weighting for the first serious run:

- clean base: `70-80%`
- RetaSy noisy blend: `15-20%`
- long-form synthetic continuity set: `5-10%`

This should be implemented as sampling/interleave balance, not by letting one source dominate raw row counts.

### Validation split

Keep validation cleaner than training, but not unrealistically clean.

Required validation slices:

1. existing Iqra validation/dev split
2. a clip-disjoint mapped RetaSy holdout slice

Why:

- clean-only validation will overstate progress
- RetaSy-only validation will understate regression risk on canonical phoneme transcription

## Augmentation Design

The next run must include augmentation. This is a requirement, not an optional follow-up.

Required families:

1. speed perturbation: `0.9x`, `0.95x`, `1.05x`, `1.1x`
2. gain jitter: mild loudness variation
3. additive noise: low-to-moderate SNR
4. mild reverb / room coloration
5. start truncation: approximately `50-400ms`
6. end truncation
7. trailing silence

Rules:

- do not augment every sample
- oversample short/noisy/clipped-start examples instead
- keep validation much cleaner than train
- preserve an unaugmented path for enough clean samples to avoid catastrophic drift

## Optimization Design

Keep the model family and CTC-only approach. Do not introduce LoRA as the primary path for this run.

Common settings:

- `train_batch_size=16`
- `grad_accum=2`
- `learning_rate=1e-4` to start
- `warmup_steps=500`
- early stopping on validation loss
- RNNT decoder/joint frozen
- preprocessor frozen

Primary run configuration:

- run name: `fastconformer-phoneme-v3-retasy-aug-u8`
- `freeze_encoder_layers=8`
- `max_steps=10000`

Fallback run if primary improves but plateaus below gate:

- run name: `fastconformer-phoneme-v3-retasy-aug-u6`
- `freeze_encoder_layers=6`
- `max_steps=12000`

Third run only if both still leave the short-noisy bucket flat:

- run name: `fastconformer-phoneme-v3-retasy-aug-u4`
- `freeze_encoder_layers=4`
- `max_steps=14000-15000`

Rationale:

- the old text-model FastConformer runs were too weak and too short
- the current phoneme task is closer to the browser runtime
- we still want a controlled sweep, not a hyperparameter explosion

## Required Code Changes Before Launch

These are design-critical and should be treated as pre-run work, not nice-to-haves.

### 1. Add RetaSy mapping into `train_fastconformer_phoneme_modal.py`

Needed because the current checked-in phoneme trainer only loads Iqra corpora.

Must record in metadata:

- total RetaSy seen
- total filtered
- total mapped
- total unmapped
- duration buckets

### 2. Fix duration mismatch in training config

Current issue:

- data prep allows up to `30.0s`
- train loader later sets `model.cfg.train_ds.max_duration = 15.0`

Decision:

- align training loader max duration with manifest policy
- if long-form synthetic examples exceed that limit, raise the effective ceiling for the long-form band only

### 3. Add augmentation support

Implement either:

- on-disk augmented rows during manifest building

or

- loader-level stochastic augmentation

Either approach is acceptable, but metadata must record which augmentation families were enabled.

### 4. Make export path run-specific

Current export script assumes fixed paths such as `fastconformer-phoneme-v1`.

Decision:

- export must take `output_name`
- export artifacts must include both `fp32` and `q8`
- export metadata must identify the training run that produced them

### 5. Add checkpoint selection discipline

Minimum acceptable:

- save regular checkpoints
- keep enough metadata to know what was selected

Preferred:

- retain top-k checkpoints by validation
- average top checkpoints before final export

This is not required to launch the first run if it delays training, but it is the first improvement to add if the first run shows noisy late-step behavior.

## Promotion Gate

The only real promotion gate is the frozen browser validator.

For each candidate:

1. export `fp32`
2. export `q8`
3. run streaming on `test_corpus` and `test_corpus_v2`
4. run non-streaming on `test_corpus` and `test_corpus_v2`
5. repeat enough times to measure variance; default target is the matrix runner's repeated mode

Primary gate:

- `test_corpus_v2` streaming median `>= 41/43`
- `test_corpus_v2` streaming minimum `>= 40/43`
- every v2 category `>= 90%`

Secondary gates:

- no non-streaming regression on either corpus
- `test_corpus` streaming returns to at least the current best recorded level
- `q8` must remain close to `fp32`

Q8 acceptance rule:

- ship `q8` only if it loses at most `1` sample on `test_corpus`
- and `0` samples on `test_corpus_v2`

## What To Measure Per Candidate

Track these explicitly:

1. overall streaming pass rate
2. overall non-streaming pass rate
3. short/noisy RetaSy bucket
4. long-ayah bucket
5. multi-verse bucket
6. opener bucket (`36:1`, `2:1`, similar short openers)
7. `fp32` vs `q8` delta
8. exact regressions against prior promoted model

## Risks

### Risk 1: RetaSy mapping coverage is lower than expected

Mitigation:

- record unmapped counts
- widen normalization only if it improves canonical mapping quality without false matches

### Risk 2: noise robustness hurts clean long recitation behavior

Mitigation:

- keep clean base dominant
- monitor category buckets, not just overall score

### Risk 3: Q8 erases gains from the new checkpoint

Mitigation:

- treat `fp32` as reference
- do not auto-ship the quantized export

### Risk 4: the current validator environment is brittle

There is already evidence of local environment instability around Node/esbuild architecture mismatches. The promotion protocol should therefore prefer reproducible scripted matrix runs in a known-good environment and should archive the produced logs with the checkpoint metadata.

## Immediate Next Actions

1. patch `scripts/train_fastconformer_phoneme_modal.py` to add RetaSy mapping, metadata, and duration alignment
2. patch augmentation support into the phoneme training path
3. patch `scripts/export_phoneme_onnx_modal.py` to export by `output_name`
4. launch `fastconformer-phoneme-v3-retasy-aug-u8`
5. export `fp32` and `q8`
6. run the full browser validation matrix
7. only then decide whether `u6` is needed

## Bottom Line

The correct design is not "LoRA because we need lighter fine-tuning". The correct design is "robust phoneme-task FastConformer fine-tuning on the actual browser model path, using noisy-domain data, augmentation, and frozen-validator promotion gates".

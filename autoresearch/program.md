# Autoresearch: FastConformer Phoneme CTC Optimization

## What You're Optimizing

A fine-tuned NVIDIA FastConformer (`stt_ar_fastconformer_hybrid_large_pcd_v1.0`) for Arabic
Quran phoneme recognition. The model uses CTC-only training (RNNT path disabled). The output
is a 70-token phoneme vocabulary (69 phonemes + CTC blank).

The model runs in-browser via ONNX Runtime. The real metric is **non-streaming verse recognition
accuracy**: given an audio clip of Quran recitation, can the exported ONNX model produce phoneme
sequences that fuzzy-match to the correct verse?

## Current Baselines

- Non-streaming accuracy (v4-tlog model): **44/53 (83%)**
- Streaming accuracy: **27/53 (51%)** (tracker bottleneck, not model)
- ~20/53 test samples have garbage CTC transcription (unfixable without model changes)
- Algorithm ceiling with current model: ~75-80% non-streaming

## How The Loop Works

1. You propose ONE change to `train.py`
2. Training runs for 5 minutes on A100-80GB
3. ONNX export + non-streaming accuracy eval runs on EVERY successful training
4. If accuracy improves → your change is committed and persisted
5. Otherwise → reverted

**Accuracy is the only promotion gate.** val_loss is logged for your analysis but never
used to filter or skip evaluation. Lower val_loss does NOT always mean better accuracy.

## What You Can Change in `train.py`

### Optimizer
- `LEARNING_RATE` (1e-5 to 1e-3, current: 1e-4)
- `WARMUP_STEPS` (0 to 500, current: 100)
- `WEIGHT_DECAY` (0 to 0.1)
- Scheduler type (in training logic section)

### Batching
- `TRAIN_BATCH_SIZE` (8 to 64, A100-80GB can handle 32+ for FastConformer)
- `GRAD_ACCUMULATION` (1 to 8)

### Architecture
- `FREEZE_ENCODER_LAYERS` (0 to 17 — the encoder has 18 Conformer layers)
- `FREEZE_PREPROCESSOR` (True/False — mel spectrogram frontend)
- `DROPOUT` (0 to 0.3)
- CTC decoder dimensions (in training logic section)

### Augmentation
- `AUGMENTATION` dict — speed perturbation, gain, white noise, shift, silence
- Probabilities, ranges, intensities
- You can add new augmentation types supported by NeMo

### Training Schedule
- `VAL_CHECK_INTERVAL` (how often to validate)
- `EARLY_STOPPING_PATIENCE`
- `TIME_BUDGET_SECONDS` (DO NOT change — fixed at 300)
- Gradient clipping, precision settings

## What You CANNOT Change

- Data manifests (fixed during prepare_data, not per-run)
- The phoneme vocabulary (69 tokens)
- The base model architecture (FastConformer)
- `prepare.py` (locked)
- `TIME_BUDGET_SECONDS` (must stay 300)

## Rules

1. **ONE change per iteration.** Isolate variables so we know what worked.
2. **Read results.csv first.** Don't repeat experiments that already failed.
3. **Explain your hypothesis** in a single sentence before the edit.
4. **Don't break the NeMo pipeline.** If your change causes a crash, it wastes 5 min of GPU time.
5. **Prioritize changes likely to affect exported-model quality**, not just training loss.

## Known Lessons (from prior experiments)

- Batch size should fill GPU memory (A100-80GB handles batch_size=32 for FastConformer, 16 only uses ~20GB)
- `cpu=8` / `NUM_WORKERS=8` is critical — default starves data loaders
- ONNX model has ±2-3 variance per accuracy run (non-determinism)
- Lower val_loss does NOT always mean better accuracy (confirmed across multiple experiments)
- The domain gap between clean Iqra/TTS data and noisy phone recordings is the main challenge

## Phase 1: CTC Ceiling Push

Focus on: learning rate schedules, freeze depth, augmentation intensity, batch size.
The hypothesis is that the current defaults (lr=1e-4, freeze=10, mild augmentation) leave
significant room for improvement.

## Phase 2: Advanced Augmentation & Export Optimization

When Phase 1 plateaus (10+ consecutive rejections), pivot to:
- Aggressive noise augmentation (simulate phone mic distortion)
- Room coloration / mild reverb
- SpecAugment-style masking
- Mixed precision export experiments
- Larger effective batch sizes via gradient accumulation

This repo does NOT have a production-ready LoRA workflow. Stay on the CTC-only path.

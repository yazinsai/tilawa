# LoRA Fine-Tuning Plan — Phoneme CTC Model

> **Goal:** Fine-tune `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` with LoRA to fix the ~20 model-quality failures that algorithms can't address. Target: 95%+ combined streaming accuracy (91/96).

## Why Fine-Tuning Is Required

After exhausting algorithm improvements (Workstream A), 18-20 of 96 test samples produce CTC phonemes that don't match the correct verse. These are:
- **RetaSy clips:** Short (1-10s), diverse speakers, background noise. Model produces completely wrong phonemes (e.g., expected "imra>atuhuu" → gets "sayaASHahh").
- **Long EveryAyah verses:** 60-157s recordings where the model's phoneme output drifts or misidentifies the verse.

## Previous Fine-Tuning Attempt (V1 — Failed)

- Froze 10/18 encoder layers, 2K steps, no augmentation
- Result: **regressed** from 85% → 81% SeqAcc
- Root cause: Too much frozen, too few steps, no data augmentation

## LoRA Approach (V2)

### Key Differences from V1
1. **LoRA adapters** on CTC decoder head only — encoder stays fully frozen
2. **Aggressive data augmentation** to prevent overfitting
3. **Mixed dataset** (RetaSy + EveryAyah + existing Quran recordings)
4. **Early stopping** on validation loss

### Prerequisites

| Item | Status | Notes |
|------|--------|-------|
| GPU machine | Needed | A100 or equivalent, ~8GB VRAM minimum |
| NeMo toolkit | Install | `pip install nemo_toolkit[asr] peft` |
| Base model checkpoint | Download | From NVIDIA NGC catalog |
| RetaSy dataset | Available | Short diverse-speaker clips in `benchmark/` |
| EveryAyah dataset | Available | Full verse recordings in `benchmark/` |

### Data Preparation

**Training data composition:**
1. **RetaSy-style clips** (high priority): Short, diverse-speaker recordings of individual verses
2. **EveryAyah excerpts**: Full verse recordings from multiple reciters
3. **Synthetic augmentation** of both:
   - Speed perturbation: 0.9x - 1.1x
   - Additive noise: SNR 15-30dB (background room noise, phone mic simulation)
   - Random crop: Extract 1-6s segments from longer recordings
   - Reverb: Mild room impulse responses

**Labels:** Phoneme sequences from `quran_phonemes.json` (ground truth for each verse)

**Split:** 90% train / 10% validation (stratify by surah to avoid data leakage)

### Training Configuration

```yaml
model:
  base: nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0

lora:
  target_modules: [ctc_decoder]  # Only CTC head, encoder frozen
  rank: 8
  alpha: 16
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 1e-4
  max_steps: 5000
  warmup_steps: 500
  optimizer: AdamW
  weight_decay: 0.01
  gradient_clip: 1.0

early_stopping:
  patience: 5
  monitor: val_loss

augmentation:
  speed_perturb: [0.9, 0.95, 1.0, 1.05, 1.1]
  noise_snr_db: [15, 20, 25, 30]
  random_crop_sec: [1, 2, 3, 4, 5, 6]
```

### Export Pipeline

1. Train LoRA adapter on base model
2. Merge LoRA weights into base model: `model.merge_and_unload()`
3. Export to ONNX: `model.export("fastconformer_phoneme_lora.onnx")`
4. Quantize to uint8: `python -m onnxruntime.quantization.quantize --quant_format QDQ`
5. Drop-in replace: `fastconformer_phoneme_q8.onnx`

### Validation Protocol

**CRITICAL: Must not regress on currently-passing samples.**

1. Run full streaming + non-streaming test suite on BOTH corpuses
2. Compare sample-by-sample against pre-fine-tuning results
3. If ANY currently-passing sample regresses:
   - Check if it's ONNX variance (run 3x)
   - If real regression: add the sample to training data and retrain
4. Target: 91+/96 streaming, 0 regressions on passing samples

### Expected Gains

| Category | Current Fails | Expected Fix Rate | Notes |
|----------|--------------|-------------------|-------|
| RetaSy model quality | 16 | 10-14 | Core target for LoRA |
| Long verse model drift | 5 | 2-3 | Augmentation with long clips |
| Multi-verse first verse | 4 | 1-2 | Better phoneme output helps discovery |
| Ambiguity / cascade | 4 | 0-1 | Algorithmic, not model |

**Conservative estimate:** 78 → 88-93/96 (92-97%)
**Optimistic estimate:** 78 → 93-95/96 (97-99%)

### File Structure

```
experiments/lora-finetune/
├── train_lora.py          # Training script
├── prepare_data.py        # Data preparation + augmentation
├── export_onnx.py         # Merge LoRA + export + quantize
├── config.yaml            # Training hyperparameters
└── README.md              # Setup instructions
```

### Execution Steps

1. **Set up GPU environment** — provision machine, install NeMo + PEFT
2. **Prepare training data** — extract audio + phoneme labels, apply augmentation
3. **Train LoRA adapter** — ~2-4 hours on A100
4. **Export to ONNX** — merge, export, quantize
5. **Validate** — full test suite, 3x runs, no regressions
6. **Iterate** — if regressions, adjust training data and retrain

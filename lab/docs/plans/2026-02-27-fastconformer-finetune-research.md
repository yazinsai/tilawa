# FastConformer Fine-Tuning Research & Trial Plan

Reference doc for interpreting V1 results and planning follow-up runs.

## Key Findings

### Tokenizer
**Keep the original.** The base model's SentencePiece Unigram (1,024 tokens) already covers Arabic with diacritical marks. Changing it randomizes the decoder weights and destroys pretrained knowledge. ([NeMo #8473](https://github.com/NVIDIA-NeMo/NeMo/discussions/8473))

### Diacritics
**Keep diacritics in training data.** The `pcd` model variant was trained to output diacritized Arabic. Stripping them creates a distribution mismatch. Our `normalize_arabic()` already strips diacritics at matching time.

### Learning Rate
NVIDIA rule of thumb: "at least 10x smaller than from-scratch" (from-scratch default: 1e-3).

| Parameter | Recommended Range | V1 Value | Notes |
|-----------|------------------|----------|-------|
| LR | 3e-5 to 1e-4 | 5e-5 | Good starting point |
| Min LR | 1e-6 | NeMo default (1e-4) | Lower min helps fine convergence |
| Warmup | 5-10% of total steps | 500 (5%) | Appropriate |
| Scheduler | CosineAnnealing | NeMo default | Already set |
| Weight decay | 1e-3 | NeMo default | Fine |

### Encoder Freezing
NeMo guidance: "Freezing the encoder entirely will often prevent learning." Instead, freeze lower layers and keep upper layers trainable (upper layers encode more domain-specific features).

| Strategy | Frozen Layers | Trainable Layers | When |
|----------|--------------|------------------|------|
| Conservative | 12/18 | 6 | First attempt |
| **Moderate (V1)** | **8/18** | **10** | **Current run** |
| Aggressive | 4/18 | 14 | If V1 improves but plateaus |
| Full unfreeze | 0/18 | 18 | Only with heavy augmentation + low LR |

Always freeze: preprocessor (no benefit), RNNT decoder (not used at inference).

### Training Duration
V1 does 10K steps x 32 effective batch = ~2.5 epochs over 126K samples. This is conservative. Domain adaptation typically needs 5-15 epochs. If V1 shows learning but hasn't converged, increase to 20-30K steps.

### SpecAugment
Base model defaults: 2 freq masks (width 27), 10 time masks (width 5%). With 126K samples this is reasonable. If overfitting, increase masks. If underfitting, reduce or disable.

### CTC vs RNNT
CTC is correct for our offline inference pipeline. In practice, RNNT training path required `libNVVM` (numba CUDA) and failed in our Modal image. Current run uses a patched CTC-only training step (`ctc_only_finetune=True`) to unblock training.

### Checkpoint Averaging
Georgian FastConformer blog highlights this as a key quality technique. Save checkpoints every 500-1K steps, then average top 3-5 by val WER.

---

## Trial Plan

### V1: Moderate Freeze (launched)

```
output_name    = nvidia-fastconformer-quran-ft-v1
batch_size     = 12, grad_accum = 2  (effective 24)
max_steps      = 2,000
lr             = 5e-5, warmup = 200
freeze_encoder = first 10/17 layers
freeze_rnnt    = True
ctc_only       = True
data           = 120K Quran + 6K RetaSy
```

**What to look for:**
- Val WER trending down? If flat after 2K steps, LR may be too low or too much frozen.
- Training loss decreasing? If yes but val loss increases, overfitting.
- SeqAcc improvement over baseline 85%? Even 1-2% is a positive signal.

**Observed outcome (2026-02-27):**
- Regressed on benchmark: 81% SeqAcc / 84% recall (`benchmark/results/2026-02-27_092540.json`)
- Two baseline-correct samples regressed (`retasy_025`, `multi_059_022_024`)
- No new recovered failures.

---

### V2: Larger Batch + More Epochs + Less Frozen

**Trigger:** V1 shows learning signal but hasn't converged.

```
output_name    = nvidia-fastconformer-quran-ft-v2
batch_size     = 32, grad_accum = 2  (effective 64)
max_steps      = 25,000  (~6 epochs)
lr             = 1e-4  (proportional to batch increase)
warmup         = 1,000
freeze_encoder = first 4/18 layers
freeze_rnnt    = True
val_interval   = 500
```

**Rationale:** Larger batch + proportionally higher LR for faster convergence. Fewer frozen layers to give the model more room to adapt.

---

### V3: Full Encoder + Hybrid Loss

**Trigger:** V2 plateaus; CTC-only gradient signal isn't enough.

```
output_name    = nvidia-fastconformer-quran-ft-v3
batch_size     = 32, grad_accum = 2  (effective 64)
max_steps      = 30,000  (~7.5 epochs)
lr             = 5e-5  (lower for full unfreeze)
warmup         = 1,500
freeze_encoder = 0 layers (full unfreeze)
freeze_rnnt    = False  (hybrid loss)
val_interval   = 500
```

**Rationale:** RNNT gradients through shared encoder may help CTC. Lower LR to prevent catastrophic forgetting with everything unfrozen.

---

### V4: Quran-Only (no RetaSy)

**Trigger:** Model performs well on clean EveryAyah but RetaSy noise hurts precision.

```
output_name    = nvidia-fastconformer-quran-ft-v4
include_retasy = False
max_train      = 120,000 (pure Quran corpus)
# Otherwise same as best-performing config
```

---

### V5: Heavy RetaSy (speaker diversity)

**Trigger:** Model works on EveryAyah test but fails on user recordings.

```
output_name    = nvidia-fastconformer-quran-ft-v5
max_retasy     = 15,000  (2.5x more crowd data)
# Otherwise same as best-performing config
```

---

### V6: SpecAugment Tuning

**Trigger:** Overfitting (train loss low, val loss diverging).

```
spec_freq_masks = 4   (up from 2)
spec_time_masks = 15  (up from 10)
# Add early stopping: patience=5 on val_wer
```

---

## Script Improvements to Consider

These features would help across all trials:

1. **SpecAugment control** -- expose `spec_freq_masks` / `spec_time_masks` as params
2. **Checkpoint top-K** -- save top 5 by val WER, not just last
3. **Checkpoint averaging** -- average top 3-5 checkpoints before saving final .nemo
4. **Early stopping** -- `EarlyStopping(monitor="val_wer", patience=5)`
5. **Min LR** -- expose `sched.min_lr` param (default 1e-6 instead of NeMo's 1e-4)

## Sources

- [NVIDIA NeMo ASR Transfer Learning Blog](https://developer.nvidia.com/blog/jump-start-training-for-speech-recognition-models-with-nemo/)
- [Georgian FastConformer Blog](https://developer.nvidia.com/blog/developing-robust-georgian-automatic-speech-recognition-with-fastconformer-hybrid-transducer-ctc-bpe/)
- [NeMo ASR CTC Language Fine-Tuning Tutorial](https://github.com/NVIDIA-NeMo/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb)
- [Japanese FastConformer Discussion #8473](https://github.com/NVIDIA-NeMo/NeMo/discussions/8473)
- [FastConformer Fine-Tune Discussion #8997](https://github.com/NVIDIA-NeMo/NeMo/discussions/8997)
- [Problem Fine-Tuning FastConformer Hybrid #8256](https://github.com/NVIDIA-NeMo/NeMo/issues/8256)
- [stt_ar_fastconformer_hybrid_large_pcd_v1.0](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0)
- [Tarteel Case Study](https://www.nvidia.com/en-us/case-studies/automating-real-time-arabic-speech-recognition/)

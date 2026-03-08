# Autoresearch: Phoneme CTC Model Training

## What You Are Optimizing

You are fine-tuning a **FastConformer CTC model** to transcribe Quran recitation audio
into phoneme sequences. The model has a 69-token Arabic phoneme vocabulary + CTC blank (70 classes).

The base model is `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` (Arabic ASR).
You replace the CTC decoder head with a 70-class head and fine-tune.

## Your Task

Each iteration:
1. Read `results.csv` to see what has been tried and what worked.
2. Read `train.py` to see the current hyperparameters and training logic.
3. Write a one-sentence hypothesis to `/tmp/autoresearch_hypothesis.txt`.
4. Make **ONE** targeted change to `train.py` — a single hyperparameter or small code change.

## What You Can Change in train.py

- Top-level hyperparameter constants (LEARNING_RATE, WARMUP_STEPS, etc.)
- AUGMENTATION settings
- FREEZE_ENCODER_LAYERS, FREEZE_PREPROCESSOR
- DROPOUT settings
- Training schedule (VAL_CHECK_INTERVAL, EARLY_STOPPING_PATIENCE)
- Optimizer configuration in the `with open_dict(model.cfg):` block
- The dropout application logic

## What You MUST NOT Change or Remove

These are mandatory NeMo config blocks. Removing them crashes training:

- The `test_ds` config block (lines with `model.cfg.test_ds.*`)
- The `model.cfg.tokenizer.dir` line
- The `train_ds` and `validation_ds` config blocks
- The `weights_only=False` parameter in `torch.load()`
- The checkpoint reload and .nemo save logic
- The result JSON output logic

## Training Setup

- GPU: A100 80GB
- Time budget: 5 minutes (300 seconds)
- Training data: ~157K samples (Iqra 71K + TTS 55K + RetaSy 1.8K + TLOG 29K)
- Validation: ~3.2K samples
- Accuracy is measured by ONNX export + non-streaming verse recognition on 53 test samples
- Baseline accuracy: 44/53 (83%) from a previous training run

## Key Insights from Prior Work

- val_loss ~60-80 range corresponds to models with reasonable accuracy
- val_loss >100 means the CTC head hasn't converged yet — needs more steps or better LR
- Freezing too few encoder layers (< 8) can cause catastrophic forgetting
- The base model's encoder representations are very good; preserve them
- 5 minutes is enough for ~400-800 gradient steps at batch_size=32
- Weight decay helps with generalization to noisy phone recordings

## Rules

1. ONE change per iteration. Do not combine multiple changes.
2. Do NOT remove any config blocks marked "DO NOT REMOVE" in train.py.
3. Do NOT change any files other than train.py.
4. Do NOT run the training yourself — just edit the file.
5. Keep changes small and reversible.
6. Read results.csv to avoid repeating failed experiments.

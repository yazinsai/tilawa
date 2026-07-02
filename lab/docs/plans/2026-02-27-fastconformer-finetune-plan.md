# FastConformer Quran Fine-Tune Plan (2026-02-27)

## Goal

Push `nvidia-fastconformer` recall toward 95% while staying under 200 MB and sub-second latency.

## Base model

- `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` (115 MB)
- Inference path in repo: `experiments/nvidia-fastconformer/run.py`

## Active detached runs

- Data prep app: `ap-GxOsmVZduIsO5L76OLntEr`
  - Function: `prepare_data`
  - Status: completed successfully (120k train + 8k val + 6k RetaSy blend)
- Training app: `ap-kPtWClU43VlUZqCVN3CCui`
  - Function: `train`
  - Status: completed, model saved to `/training/nvidia-fastconformer-quran-ft-v1/model/model.nemo`

Historical note:

- `ap-eUBSYAFi7VEeEkeztOqaYu` was stopped after waiting for manifests.
- `ap-5P5yCowc11lVbB1JeJHOiB` failed quickly due Lightning trainer/module mismatch;
  fixed by switching trainer import to `lightning.pytorch`.
- `ap-XtGTX7bjqiDARp0oNMskTG` failed due missing `libnvvm.so` from RNNT numba loss.
  Fixed by adding CTC-only training step patch (`ctc_only_finetune=True`).

Monitor:

```bash
modal app logs ap-GxOsmVZduIsO5L76OLntEr
modal app logs ap-kPtWClU43VlUZqCVN3CCui
modal app list
```

## V1 outcome

- Download app: `ap-1X9GUxM4v6gdHkf81U4iGu`
- Downloaded model: `data/nvidia-fastconformer-quran-ft-v1/model.nemo` (~459 MB)
- Benchmark file: `benchmark/results/2026-02-27_092540.json`
- Result:
  - SeqAcc: `81%` (down from baseline `85%`)
  - Recall: `84%` (down from baseline `87%`)
  - Precision: `85%` (down from baseline `89%`)
  - Latency: `0.35s` (baseline `0.33s`)
- Verdict: **regression**. Keep pretrained FastConformer as production baseline.

## V1 run config (launched)

- Output name: `nvidia-fastconformer-quran-ft-v1`
- Train data:
  - `rabah2026/Quran-Ayah-Corpus` train split (`max_train_samples=120000`)
  - + RetaSy blend (`max_retasy_samples=6000`)
- Validation data:
  - `rabah2026/Quran-Ayah-Corpus` validation split (`max_val_samples=8000`)
- Filters: `0.6s <= duration <= 30.0s`
- Training:
  - `max_steps=2000`
  - `train_batch_size=12`, `grad_accum=2`
  - `lr=5e-5`, `warmup_steps=200`
  - Freeze strategy:
    - preprocessor frozen
    - encoder layers `[0..9]` frozen (10 layers)
    - RNNT decoder/joint frozen (CTC-heavy adaptation)
  - Training mode:
    - `ctc_only_finetune=True` (patched NeMo training step to avoid RNNT/numba NVVM dependency)

## Evaluation protocol (after model download)

1. Download artifacts:

```bash
modal run scripts/train_nvidia_fastconformer_modal.py --download-only --output-name nvidia-fastconformer-quran-ft-v1
```

2. Benchmark full corpus:

```bash
NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR=data/nvidia-fastconformer-quran-ft-v1 \
  .venv/bin/python -m benchmark.runner --experiment nvidia-fastconformer
```

3. Compare against current best:

- Baseline: `benchmark/results/2026-02-26_230349.json`
- Check:
  - SeqAcc/Recall delta
  - Latency delta
  - model size on disk

## If V1 underperforms: prioritized follow-ups

1. `v2-unfreeze-more`:
   - `freeze_encoder_layers=4`, keep RNNT frozen.
2. `v3-hybrid-train`:
   - `freeze_rnnt_decoder=false`, `freeze_encoder_layers=4`.
3. `v4-longer-train`:
   - `max_steps=15000`, lower LR (`3e-5`) and warmup (`800`).
4. `v5-quran-only`:
   - disable RetaSy blend for a pure Quran-reciter domain run.

Each follow-up should keep the same evaluation protocol and produce a timestamped benchmark JSON.

# v7: FastConformer phoneme + streaming-like augmentation

## Goal

Close the streaming-vs-batch ASR gap on the shipped `fastconformer-phoneme`
model by training with an augmentation distribution that resembles what the
browser streaming pipeline actually feeds the model at inference time.

## Working hypothesis

The three 2026-04-21 matcher/tracker attempts (rare-phoneme n-gram
expansion always-on, same gated, short-text first-match gate) all landed
inside the ONNX ±3–6 sample variance envelope on v1. The nine stable-fail
samples (`retasy_024/025`, `multi_055_001_004`, `multi_113_001_005`, …)
share the same pattern: the CTC decoder emits phonemes that genuinely
match the wrong verse better than the right one under Levenshtein. This
is an ASR-quality floor, not a matcher bug.

The current v4-tlog augmentor only perturbs speed/gain and adds short
silence windows (≤0.4s start, ≤0.5s end). The shipped model runs at
300 ms chunks with a 4 s silence tail, and most sample audio from
RetaSy / user recordings has 0.5–2 s of pre/post silence with
non-trivial microphone noise. The model has never been trained on
inputs that look like the streaming distribution.

## Change

One data-side change, per AGENTS.md "one change per training run" rule:
a new `streaming_aug` flag that replaces the augmentor dict with
expanded ranges for the existing CTC-safe perturbations. Everything
else (data mix, freezing, optimizer, manifests) is held constant.

```python
"speed":       prob 0.3, ±10% rate              (unchanged)
"gain":        prob 0.4, -15 .. +8 dBFS         (was 0.3, -10..+5)
"white_noise": prob 0.5, -70 .. -40 dB          (was 0.3, -80..-50)
"shift":       prob 0.4, ±400 ms                (was 0.2, ±200 ms)
"silence":     prob 0.6, start 0..1.5s end 0..1.2s  (was 0.2, 0..0.4 / 0..0.5)
```

All perturbations leave the transcript unchanged, so CTC alignment is
not broken.

## Curriculum init

Rather than training from the NVIDIA base and re-learning the phoneme
vocab (8 000 steps), v7 starts from the v4-tlog checkpoint weights
(already on the `fastconformer-phoneme-training` Modal volume, profile
`ahmedbader801`). New `init_from_checkpoint` arg extracts the weights
tarball and loads the state_dict after the CTC head has been re-shaped.

## Data

Reuse v4-tlog's train and val manifests via the new
`train_manifest_override` / `val_manifest_override` flags. This skips
`prepare_data` entirely (hours of HF dataset download + RetaSy phoneme
remap + TLOG filter) and keeps the data identical to v4-tlog — so any
delta is attributable to the augmentation change alone.

## Pre-launch discipline checklist

- [x] Modal client installed; `ahmedbader801` profile has volume with
      v4-tlog checkpoint and manifests.
- [x] Script syntax-valid.
- [x] Changes additive: no existing behavior changed (all new args
      default to `""` / `False`).
- [x] One data-side change (augmentation config only).

## Launch

```bash
modal profile activate ahmedbader801
modal run --detach scripts/train_fastconformer_phoneme_modal.py \
  --output-name fastconformer-phoneme-v7-stream-aug \
  --train-manifest-override /training/fastconformer-phoneme-v4-tlog/manifests/train_manifest.jsonl \
  --val-manifest-override /training/fastconformer-phoneme-v4-tlog/manifests/val_manifest.jsonl \
  --init-from-checkpoint /training/fastconformer-phoneme-v4-tlog/model/model.nemo \
  --enable-streaming-aug \
  --max-steps 3000 \
  --learning-rate 2e-5 \
  --warmup-steps 200 \
  --val-check-interval 200 \
  --early-stopping-patience 6 \
  --train-only
```

- `--train-only` because we're reusing v4-tlog manifests — skip
  prepare_data.
- Low LR (2e-5, down from 1e-4) because we're doing a curriculum
  continuation from a pre-trained checkpoint, not a fresh fit.
- Short schedule (3 000 steps with val every 200) because we only need
  to adapt the encoder to the augmentation distribution — early stop
  will kick in if val_loss plateaus.

## Export + benchmark

On completion:

```bash
# download .nemo + metadata locally
modal run scripts/train_fastconformer_phoneme_modal.py \
  --download-only \
  --output-name fastconformer-phoneme-v7-stream-aug

# export to quantized ONNX
modal run scripts/export_phoneme_onnx_modal.py \
  --output-name fastconformer-phoneme-v7-stream-aug
```

Then drop the `.onnx` under `web/frontend/public/` (temporary rename or
symlink) and rerun the stability report:

```bash
cd web/frontend
npx tsx test/stability-report.ts --repeats=3 --json=test/v7-v1-stability.json
npx tsx test/stability-report.ts --repeats=3 --corpus=test_corpus_v2 --json=test/v7-v2-stability.json
```

Compare against `test/streaming-attempts-2026-04-21/baseline-main-v1.json`.

## Ship criteria

- v1 median recall or SeqAcc improves by ≥ 3pp (outside ONNX variance)
- v2 blind check does not regress
- Unit tests still pass
- If either gate fails, document the result in EXPERIMENTS.md and
  discard the checkpoint.

## Risk register

- v4-tlog weights may not load cleanly after CTC head replacement
  (missing/unexpected key count will signal). Script prints a warning
  above 50 of either.
- Expanded silence range might push audio past `max_duration=30.0`;
  NeMo clips but check val_loss behavior in first 400 steps.
- Curriculum-style fine-tunes can overshoot on the smaller
  distribution; early stopping patience is 6 evals (~1200 steps).

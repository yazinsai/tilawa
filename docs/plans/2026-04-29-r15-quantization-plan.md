# r15 quantization and evaluation plan

## Goal

Turn Ahmed's newly accessible `hetchyy/r15_95m` checkpoint into a deployable
artifact and decide whether it should become a verifier, a batch fallback, or a
distillation teacher.

## Current evidence

- `hetchyy/r15_95m` loads with Ahmed's HF read token.
- `hetchyy/r15_95m_onnx_int8` still returns 404.
- Full-file benchmark on the downloadable EveryAyah slice of v3:
  - 174 samples
  - 97.1% recall / 97.1% precision / 97.1% SeqAcc
  - 0.90s average CPU latency
  - 388 MB fp32 size
- The five failures are repeated/short-phrase collisions, not obvious acoustic
  failures.

Raw result: `benchmark/results/2026-04-29_091708.json`.

## Step 1: Export and quantize r15

Use Modal so the export does not depend on local CPU/RAM:

```bash
modal secret create hf-token HF_TOKEN=...
modal run --detach scripts/export_r15_onnx_modal.py --no-download
```

Expected outputs on Modal volume `w2v-phonemes-r15`:

```text
/vol/r15_95m_onnx/
  model_fp32.onnx
  model_int8.onnx
  preprocessor_config.json
  tokenizer_config.json
  vocab.json
  special_tokens_map.json
  export_metadata.json
```

Download after completion:

```bash
modal run scripts/export_r15_onnx_modal.py --download-only --output-dir data/r15-onnx
```

## Step 2: Add an ONNX-backed r15 model entry

If `model_int8.onnx` is sane:

1. Add a `base-local-int8` or `base-modal-int8` model entry to
   `experiments/w2v-phonemes/run.py`.
2. Point it at `data/r15-onnx/model_int8.onnx` by default, with an env override
   for local paths.
3. Keep `base` as the fp32 HF checkpoint for reproducibility.

## Step 3: Benchmark gates

Run:

```bash
.venv/bin/python -m benchmark.runner --experiment w2v-phonemes/base-local-int8 --corpus test_corpus_v3 --source everyayah
.venv/bin/python -m benchmark.runner --experiment w2v-phonemes/base-local-int8 --corpus test_corpus_v3
```

Promotion criteria:

- EveryAyah-v3 SeqAcc within 1pp of fp32 r15.
- Full v3 SeqAcc meaningfully above shipped FastConformer full-file and close to
  r7; target `>= 94%`.
- Average CPU latency no worse than fp32 r15; target `<= 1.0s` on the 174-sample
  EveryAyah slice.
- Model artifact around 100-130 MB. If it is much larger, try ORT external-data
  int8 or PyTorch dynamic quantization only as a server-side artifact.

## Step 4: Product decision

If int8 holds:

- short term: use r15 as server-side/batch verifier for current streaming output;
- medium term: use r15/r7 as teachers for a cache-aware streaming FastConformer
  RNNT/CTC student;
- do not treat r15 as a true streaming model until a real chunked/cache path
  exists. Naive 3s chunks do not use the phoneme matcher correctly and wav2vec2
  attention is not streaming-friendly.

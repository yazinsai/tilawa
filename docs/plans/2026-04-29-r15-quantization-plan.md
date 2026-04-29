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
- Modal export produced our own dynamic-int8 ONNX:
  - `data/r15-onnx/model_int8.onnx`
  - 123,183,175 bytes for the ONNX file; 123,731,968 bytes including tokenizer
    files
  - same 97.1% / 97.1% / 97.1% on the 174-sample EveryAyah-v3 slice
  - same five failures as fp32
  - 1.11s average CPU latency on this cloud VM
- Full v3 batch result for int8 ONNX:
  - 256 samples
  - 96.0% recall / 96.1% precision / 95.7% SeqAcc
  - 0.89s average CPU latency
- Naive phoneme-aware 3s chunk streaming for int8 ONNX:
  - 20.6% recall / 12.2% precision / 3.9% SeqAcc
  - This is a diagnostic baseline only; r15/wav2vec2 is not cache-aware or
    streaming-trained.

Raw results:

- fp32 HF checkpoint: `benchmark/results/2026-04-29_091708.json`
- int8 ONNX: `benchmark/results/2026-04-29_100633.json`
- int8 ONNX full v3: `benchmark/results/2026-04-29_103225.json`
- int8 ONNX naive 3s streaming v3: `benchmark/results/2026-04-29_103627.json`

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

Status: completed on 2026-04-29. Modal app runs:

- failed missing `onnxscript`: `ap-jLVHwCWXgaSGy2CS6RWrVp`
- successful export: `ap-on2dvsqGII2mZ9bXGUTNjj`
- download: `ap-efIh6Dud7nsepM46qfIe6A`

## Step 2: Add an ONNX-backed r15 model entry

Completed:

1. Added `base-local-int8` to `experiments/w2v-phonemes/run.py`.
2. It points at `data/r15-onnx/model_int8.onnx` by default, with
   `R15_ONNX_DIR` as an env override.
3. Kept `base` as the fp32 HF checkpoint for reproducibility.

## Step 3: Benchmark gates

Run:

```bash
.venv/bin/python -m benchmark.runner --experiment w2v-phonemes/base-local-int8 --corpus test_corpus_v3 --source everyayah
.venv/bin/python -m benchmark.runner --experiment w2v-phonemes/base-local-int8 --corpus test_corpus_v3
```

Promotion criteria status:

- EveryAyah-v3 SeqAcc within 1pp of fp32 r15: **passed** (identical 97.1%).
- Average CPU latency target `<= 1.0s`: **near miss** on this VM (1.11s vs
  0.90s fp32). Still acceptable for server-side verification; measure on target
  hardware before productizing.
- Model artifact around 100-130 MB: **passed** (123 MB ONNX).
- Full v3 including TLOG/user recordings: **passed as verifier** (95.7% SeqAcc).
- Streaming: **failed** for naive 3s chunks (3.9% SeqAcc). Do not ship r15 as a
  streaming model without a fundamentally different cache/segment design.

## Step 4: Product decision

If int8 holds:

- short term: use r15 as server-side/batch verifier for current streaming output;
- medium term: use r15/r7 as teachers for a cache-aware streaming FastConformer
  RNNT/CTC student;
- do not treat r15 as a true streaming model until a real chunked/cache path
  exists. Naive 3s chunks do not use the phoneme matcher correctly and wav2vec2
  attention is not streaming-friendly.

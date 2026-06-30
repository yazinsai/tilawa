# Tilawa

> Formerly called offline-tarteel.

[![Maintained by auto-maintainer](https://am.whhite.com/badge/yazinsai/tilawa)](https://am.whhite.com/stats/yazinsai/tilawa)

Offline Quran verse recognition. Record someone reciting, identify the surah and ayah -- no internet required.

**Winning entry:** Cyberistic's `c2c-direct-mixed-tta` -- **100% recall**, **100% sequence accuracy**, **88 MB**, **0.84s latency** on the v1 53-sample corpus. It uses a mixed int4+int8 FastConformer ONNX plus confidence-gated 0.9x/1.0x/1.1x test-time augmentation.

## Use in your app

The current model takes 16 kHz mono audio directly and returns a surah/ayah prediction. The pipeline has 4 steps:

1. **Audio** -- Record or load 16 kHz mono `Float32` samples
2. **ONNX inference** -- Run Cyberistic's mixed int4/int8 FastConformer export; preprocessing is embedded in the graph
3. **Decode** -- CTC-collapse 1025-token Arabic BPE output into normalized text
4. **Match + rerank** -- Retrieve Quran candidates, then CTC-rerank with precomputed verse token IDs

### Get the model

Download the Cyberistic full-mixed ONNX model (88 MB, int4 MatMul + int8 Conv/LayerNorm) and its runtime data from [GitHub Releases](https://github.com/yazinsai/tilawa/releases/tag/v0.2.0):

```bash
base=https://github.com/yazinsai/tilawa/releases/download/v0.2.0

curl -L -O "$base/fastconformer_full_mixed.onnx"
curl -L -O "$base/vocab.json"
curl -L -O "$base/quran_ctc_tokens.json"
curl -L -O "$base/tokenizer.model"
curl -L -O "$base/export_metadata.json"
```

You also need Quran text from this repo:

- [`web/frontend/public/quran.json`](web/frontend/public/quran.json) -- all 6,236 verses
- [`web/frontend/public/quran_ctc_tokens.json`](web/frontend/public/quran_ctc_tokens.json) -- precomputed candidate token IDs for CTC reranking

The old `fastconformer_ar_ctc_q8.onnx` and `fastconformer_phoneme_q8.onnx` assets are historical. New integrations should use `fastconformer_full_mixed.onnx`.

To regenerate the full-mixed export yourself, reproduce Cyberistic's `c2c-direct-mixed` path: export the CTC FastConformer with raw waveform input, then quantize MatMul weights to int4 and the remaining Conv/LayerNorm weights to int8. The runtime contract is captured in [`web/frontend/public/export_metadata.json`](web/frontend/public/export_metadata.json).

For local validation:

```bash
.venv/bin/python -m benchmark.runner --experiment c2c-direct-mixed
.venv/bin/python -m benchmark.runner --experiment c2c-direct-mixed-tta
```

### Web / React (ONNX Runtime Web)

Runs entirely in the browser using WebAssembly. See [`web/frontend/`](web/frontend/) for a complete working example.

```bash
npm install onnxruntime-web
```

```typescript
import * as ort from "onnxruntime-web/wasm";

// 1. Create session
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
const session = await ort.InferenceSession.create(modelBuffer, {
  executionProviders: ["wasm"],
});

// 2. Run raw 16 kHz mono audio through the full-mixed Cyberistic ONNX.
//    No JS mel-spectrogram step: preprocessing is inside the model graph.
const input = new ort.Tensor("float32", audioFloat32Array, [1, audioFloat32Array.length]);
const length = new ort.Tensor(
  "int64",
  BigInt64Array.from([BigInt(audioFloat32Array.length)]),
  [1],
);
const results = await session.run({
  audio_signal: input,
  length,
});
const logprobs = results[session.outputNames[0]];

// 3. Decode + match.
// See TextCTCDecoder, QuranDB.bestJoint03Match(), and the CTC rerank path in:
// web/frontend/src/worker/inference.ts
```

Key files to reference for a complete implementation:
- [`web/frontend/src/worker/session.ts`](web/frontend/src/worker/session.ts) -- ONNX Runtime Web session and raw-audio inputs
- [`web/frontend/src/worker/text-ctc-decode.ts`](web/frontend/src/worker/text-ctc-decode.ts) -- 1025-token BPE CTC greedy decoder
- [`web/frontend/src/worker/quran-text-adapter.ts`](web/frontend/src/worker/quran-text-adapter.ts) -- attaches precomputed CTC token IDs to Quran rows
- [`web/frontend/src/lib/quran-db.ts`](web/frontend/src/lib/quran-db.ts) -- candidate retrieval and Cyberistic-style reranking
- [`web/frontend/src/lib/normalizer.ts`](web/frontend/src/lib/normalizer.ts) -- Arabic text normalization

### React Native (ONNX Runtime Mobile)

Use [`onnxruntime-react-native`](https://www.npmjs.com/package/onnxruntime-react-native) which wraps the native ONNX Runtime for iOS/Android.

```bash
npm install onnxruntime-react-native
```

```typescript
import { InferenceSession, Tensor } from "onnxruntime-react-native";

// Bundle the model in your app assets, or download on first launch
const session = await InferenceSession.create("path/to/fastconformer_full_mixed.onnx");

// Same inference pattern as the web version:
// 1. Create input tensors: audio_signal [1, samples] + length [1]
// 2. session.run() -> CTC logprobs
// 3. Greedy BPE CTC decode + QuranDB candidate retrieval/CTC rerank
```

The decoder and QuranDB matching logic from [`web/frontend/src/`](web/frontend/src/) works directly in React Native -- it's pure TypeScript with no browser-specific APIs. You do not need to port `mel.ts` for the current model.

### Python

**Option A: ONNX Runtime (recommended for production)**

Use the repo experiment for the exact Cyberistic algorithm, including candidate retrieval and CTC rerank:

```bash
.venv/bin/python -m benchmark.runner --experiment c2c-direct-mixed
```

The ONNX contract is raw audio in, CTC logprobs out:

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf

session = ort.InferenceSession("fastconformer_full_mixed.onnx")
audio, sr = sf.read("recitation.wav", dtype="float32")
assert sr == 16000
if audio.ndim > 1:
    audio = audio.mean(axis=1)

logprobs = session.run(None, {
    "audio_signal": audio.reshape(1, -1).astype(np.float32),
    "length": np.array([audio.shape[0]], dtype=np.int64),
})[0]
```

**Option B: NeMo (full pipeline, heavier dependencies)**

```bash
pip install nemo_toolkit[asr]
```

```python
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)
model.change_decoding_strategy(decoder_type="ctc")
transcript = model.transcribe(["recitation.wav"])[0]
# Then match transcript against quran.json
```

**Option C: Use this repo directly**

```bash
git clone https://github.com/yazinsai/tilawa.git
cd tilawa
pip install -e ".[nemo]"
```

```python
import importlib.util
from pathlib import Path

run_py = Path("experiments/c2c-direct-mixed/run.py").resolve()
spec = importlib.util.spec_from_file_location("c2c_direct_mixed", run_py)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

result = mod.predict("recitation.wav")
# {"surah": 1, "ayah": 1, "ayah_end": 3, "score": 0.92, "transcript": "..."}
```

### Model details

| | Value |
|---|---|
| **Winning entry** | Cyberistic's `c2c-direct-mixed-tta` |
| **Base model** | `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` |
| **ONNX file** | `web/frontend/public/fastconformer_full_mixed.onnx` and `data/onnx_export/fastconformer_full_mixed.onnx` (88 MB, int4 MatMul + int8 Conv/LayerNorm) |
| **Input** | 16 kHz mono audio |
| **Output** | CTC logprobs over 1025-token Arabic BPE vocabulary, then CTC re-rank against Quran candidates |
| **Recall / Precision / SeqAcc** | 100% / 100% / 100% on the v1 53-sample benchmark, median across 3 reproduced runs |
| **Latency** | 0.84s average on Apple Silicon CPU |
| **License** | [CC-BY-4.0](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) (NVIDIA model) |

---

## Goal

Ship a model that runs on-device (phone or laptop) with **95%+ recall**, **sub-second latency**, and **under 200 MB** on disk. The current champion (`c2c-direct-mixed-tta`) reaches **100% recall / 100% precision / 100% sequence accuracy** at **88 MB** and **0.84s** latency on the v1 corpus. The browser worker now loads Cyberistic's non-TTA full-mixed text CTC model (`fastconformer_full_mixed.onnx`) directly from raw audio; the older phoneme ONNX path is kept for historical experiments.

## Design constraints

- **Offline-first.** No network calls at inference time. The model, index, and reference data all ship with the app.
- **Small models only.** Target < 200 MB total (model + any index). Phone storage is limited and download size matters.
- **Fast inference.** Under 1 second on Apple Silicon (MPS) or recent phone SoC. Users expect near-instant feedback after reciting.
- **Speaker-invariant.** Must work across accents, recording quality, and recitation styles -- not just professional studio audio from a single reciter.
- **Full Quran coverage.** All 6,236 verses, including short verses (3-4 words) that every approach currently struggles with.

## Results

Best reproduced batch result, plus the last fully measured browser streaming baseline before the Cyberistic worker swap:

| Mode | Corpus | Recall | Precision | SeqAcc |
|---|---|---|---|---|
| `c2c-direct-mixed-tta` full-file batch | v1 (53) | **100%** | **100%** | **100%** |
| `c2c-direct-mixed` full-file batch | v1 (53) | 98% | 98% | 98% |
| Browser/RN streaming, pre-Cyberistic phoneme path (300ms chunks) | v2 (43) | 87.9% | 68.9% | 55.8% |
| Browser/RN streaming, pre-Cyberistic phoneme path (300ms chunks) | v3 (256) | 89.3% | 73.4% | 58.2% |
| Pre-Cyberistic non-streaming phoneme path (full-file) | v1 (53) | 84.1% | 84.9% | 81.1% |
| Pre-Cyberistic non-streaming phoneme path (full-file) | v2 (43) | 78.1% | 79.1% | 74.4% |

The `c2c-direct-mixed-tta` row is the median of 3 reproduced benchmark runs from Cyberistic's imported champion. The browser now loads `fastconformer_full_mixed.onnx`; rerun the stability reports before replacing the pre-swap streaming rows. See **[EXPERIMENTS.md](EXPERIMENTS.md)** for per-change provenance.

Full matrix across 20 approaches (Whisper variants, Rabah pruned CTC, FastConformer sweep, contrastive/embedding failures), per-experiment notes, variant deep-dives, a changelog, and key findings live in **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Project structure

```
shared/                  # Common utilities used by all experiments
  audio.py               # load_audio(path, sr=16000) -> float32 numpy array
  normalizer.py          # normalize_arabic(text) - strip diacritics, normalize alef/taa
  quran_db.py            # QuranDB - 6,236 verses, fuzzy match, multi-ayah spans

experiments/             # Each approach gets its own directory
  ctc-alignment/         # CTC forced alignment (strong baseline, 81%)
  two-stage/             # Moonshine ASR + CTC re-score (72%, large model fallback)
  two-stage-faster-whisper-pruned/  # faster-whisper Quran + pruned CTC re-score
  distilled-ctc/         # wav2vec2-base knowledge-distilled (failed)
  rabah-pruned-ctc/      # Rabah Quran CTC (12/8/6 + fine-tuned int8 variants)
  nvidia-fastconformer/  # NeMo FastConformer Arabic benchmark
  fastconformer-ctc-rescore/  # FastConformer + CTC re-scoring fallback
  fastconformer-nbest-bruteforce/  # N-best beam search + CTC brute-force (worse than baseline)
  w2v-phonemes/          # wav2vec2 phoneme CTC + Levenshtein (100% large, 89% base)
  contrastive-v2/        # QuranCLAP v2 audio fingerprinting (failed)
  whisper-lora/          # Whisper-small + LoRA adapter
  tarteel-whisper-base/  # Tarteel's whisper-base-ar-quran
  embedding-search/      # HuBERT + FAISS nearest-neighbor
  contrastive/           # QuranCLAP v1 (proof of concept)
  streaming-asr/         # mlx-whisper chunked streaming
  new-models/            # Multi-model benchmark (8 ASR models)

benchmark/               # Evaluation framework
  runner.py              # CLI benchmark runner
  test_corpus/           # 54 audio samples + manifest.json
  results/               # Timestamped JSON results + latest.json

data/                    # Reference data
  quran.json             # 6,236 verses (uthmani + cleaned text)

web/                     # Live demo
  server.py              # FastAPI backend
  frontend/              # React frontend

scripts/                 # Training scripts (Modal A100-80GB GPU)
  train_fastconformer_phoneme_modal.py  # Phoneme CTC fine-tuning (best streaming model)
  export_phoneme_onnx_modal.py         # Historical phoneme ONNX export
  train_pruned_ctc_modal.py    # Fine-tune pruned Rabah CTC models (the key training script)
  quantize_pruned_models.py    # PyTorch/ONNX int8 quantization
  build_rabah_pruned_models.py # Build naive-pruned Rabah checkpoints
  train_nvidia_fastconformer_modal.py  # FastConformer Quran fine-tune (Modal)
  train_ctc_base_modal.py      # wav2vec2-base CTC fine-tuning (failed -- see "What we tried")
  train_distill_modal.py       # Knowledge distillation (blocked on CTC base)
  train_contrastive_v2_modal.py  # QuranCLAP v2 contrastive training
  train_modal.py               # LoRA training (whisper-lora experiment)
  train_lora.py                # Local LoRA training script (MPS/CUDA)

docs/plans/              # Design docs and experiment plans
REPORT.md                # Full experiment report with cross-comparison
RESEARCH-audio-to-verse.md  # Research notes on approaches
```

## Test corpus

`benchmark/test_corpus/manifest.json` contains 54 samples across three sources:

| Source | Count | Description |
|---|---|---|
| User recordings | 2 | Phone recordings (.m4a), ambient noise, non-professional |
| EveryAyah (Alafasy) | 23 | Professional studio recordings, includes long single-ayah and multi-ayah concatenated |
| RetaSy crowdsourced | 29 | Curated subset from 1,287 speakers across 81 countries |

**Categories:** short (17), medium (19), long (9), multi (9).

All audio files are tracked in the repository. To regenerate the EveryAyah long/multi samples from scratch (downloads from everyayah.com):

```bash
.venv/bin/python scripts/add_long_multi_samples.py
```

## Running benchmarks

```bash
# Activate the project venv
source .venv/bin/activate

# Run all registered experiments
.venv/bin/python -m benchmark.runner

# Run a single experiment
.venv/bin/python -m benchmark.runner --experiment ctc-alignment

# Run Rabah pruned variants (expanded via list_models)
.venv/bin/python -m benchmark.runner --experiment rabah-pruned-ctc
.venv/bin/python -m benchmark.runner --experiment rabah-pruned-ctc/8-layer-int8

# Run new two-stage faster-whisper pipeline
.venv/bin/python -m benchmark.runner --experiment two-stage-faster-whisper-pruned

# Run NVIDIA FastConformer benchmark (requires: pip install -e .[nemo])
.venv/bin/python -m benchmark.runner --experiment nvidia-fastconformer

# Filter by audio category
.venv/bin/python -m benchmark.runner --category short
```

Results are saved to `benchmark/results/<timestamp>.json`. The runner also maintains `benchmark/results/latest.json` with best results per scoped run (`mode`, `category`, and sample count).

To build local Rabah pruned checkpoints (naive, no fine-tuning):

```bash
.venv/bin/python scripts/build_rabah_pruned_models.py --layers 12 8 6 --save-source
```

To fine-tune pruned models on Modal A100 GPU:

```bash
modal run --detach scripts/train_pruned_ctc_modal.py --layers 8 --strategy first_n
modal run scripts/train_pruned_ctc_modal.py --layers 8 --strategy first_n --download-only
```

To fine-tune NVIDIA FastConformer on Modal A100 GPU:

```bash
# start detached (data prep + training)
modal run --detach scripts/train_nvidia_fastconformer_modal.py \
  --output-name nvidia-fastconformer-quran-ft-v1

# download artifacts once done
modal run scripts/train_nvidia_fastconformer_modal.py \
  --download-only \
  --output-name nvidia-fastconformer-quran-ft-v1

# benchmark with the fine-tuned checkpoint
NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR=data/nvidia-fastconformer-quran-ft-v1 \
  .venv/bin/python -m benchmark.runner --experiment nvidia-fastconformer
```

## Adding a new experiment

The end-to-end protocol — worktree setup, measurement discipline (3 runs, variance, v2 blind check), EXPERIMENTS.md update patterns, commit/merge rules, and a hard definition-of-done checklist — lives in **[AGENTS.md → Experiment workflow](AGENTS.md#experiment-workflow-end-to-end)**. Read that before starting. The steps below are just the minimum `run.py` contract.

1. Create `experiments/<name>/run.py` with two required functions:

```python
def predict(audio_path: str) -> dict:
    """Returns {"surah": int, "ayah": int, "ayah_end": int|None, "score": float, "transcript": str}"""

def model_size() -> int:
    """Total model size in bytes."""
```

2. Register it in `benchmark/runner.py`:

```python
EXPERIMENT_REGISTRY = {
    ...
    "your-experiment": EXPERIMENTS_DIR / "your-experiment" / "run.py",
}
```

3. Run the benchmark: `.venv/bin/python -m benchmark.runner --experiment your-experiment`

For multi-model experiments, also export `list_models()` and accept `model_name` as an optional parameter in `predict()` and `model_size()`.

## Shared utilities

All experiments use `from shared.X import ...`:

- `shared.audio.load_audio(path, sr=16000)` -- returns float32 numpy array at 16kHz
- `shared.normalizer.normalize_arabic(text)` -- strips diacritics, normalizes alef/taa marbuta/alef maqsura
- `shared.quran_db.QuranDB` -- loads `data/quran.json`, provides `match_verse(text)`, `search(text, top_k)`, `get_verse(surah, ayah)`

## Web demo

FastAPI backend + React frontend for live recitation and verse identification.

```bash
# Backend
.venv/bin/python web/server.py

# Frontend (separate terminal)
cd web/frontend && npm run dev

# Live websocket benchmark (full 53-sample corpus)
python scripts/benchmark_streaming_endpoint.py

# Multi-ayah subset
python scripts/benchmark_streaming_endpoint.py --category multi
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Some experiments have additional dependencies (faiss-cpu, moonshine, mlx-whisper). Check individual experiment READMEs.

## Further reading

- **[EXPERIMENTS.md](EXPERIMENTS.md)** — full benchmark tables, per-experiment writeups, changelog, and roadmap
- `REPORT.md` — detailed experiment report with per-sample breakdowns
- `RESEARCH-audio-to-verse.md` — research survey of approaches
- `docs/plans/` — design documents for individual experiments

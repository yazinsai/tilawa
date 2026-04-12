# Offline Tarteel

[![Maintained by auto-maintainer](https://am.whhite.com/badge/yazinsai/offline-tarteel)](https://am.whhite.com/stats/yazinsai/offline-tarteel)

Offline Quran verse recognition. Record someone reciting, identify the surah and ayah -- no internet required.

**Best model:** NVIDIA FastConformer -- **95% recall**, **115 MB**, **0.7s latency**. Available as a quantized ONNX file (131 MB) that runs in browsers, React Native, and Python.

## Use in your app

The model takes 16 kHz audio and returns a surah/ayah prediction. The pipeline has 4 steps:

1. **Audio** -- Record or load a `.wav` at 16 kHz mono
2. **Mel spectrogram** -- 80-bin NeMo-compatible features
3. **ONNX inference** -- Run the model, get CTC logprobs
4. **Decode + match** -- Greedy CTC decode, then fuzzy-match against all 6,236 Quran verses

### Get the model

Download the quantized ONNX model (131 MB, uint8) from [GitHub Releases](https://github.com/yazinsai/offline-tarteel/releases/tag/v0.1.0):

```bash
curl -L -o fastconformer_ar_ctc_q8.onnx \
  https://github.com/yazinsai/offline-tarteel/releases/download/v0.1.0/fastconformer_ar_ctc_q8.onnx
```

You also need two data files from this repo:

- [`data/vocab.json`](data/vocab.json) -- CTC vocabulary (token ID -> character mapping)
- [`data/quran.json`](data/quran.json) -- All 6,236 verses (for matching decoded text to surah:ayah)

Or generate the ONNX model yourself from the NeMo checkpoint:

```bash
pip install nemo_toolkit[asr]
python -c "
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
import torch, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model = EncDecHybridRNNTCTCBPEModel.from_pretrained('nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0')
model.change_decoding_strategy(decoder_type='ctc')
model.export('fastconformer_ar_ctc.onnx')
quantize_dynamic('fastconformer_ar_ctc.onnx', 'fastconformer_ar_ctc_q8.onnx', weight_type=QuantType.QUInt8)
"
```

### Web / React (ONNX Runtime Web)

Runs entirely in the browser using WebAssembly. See [`web/frontend/`](web/frontend/) for a complete working example.

```bash
npm install onnxruntime-web @huggingface/transformers
```

```typescript
import * as ort from "onnxruntime-web/wasm";

// 1. Create session
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
const session = await ort.InferenceSession.create(modelBuffer, {
  executionProviders: ["wasm"],
});

// 2. Compute mel spectrogram (80-bin, NeMo-compatible)
//    See web/frontend/src/worker/mel.ts for the full implementation
//    Uses @huggingface/transformers mel_filter_bank + spectrogram
const { features, timeFrames } = computeMelSpectrogram(audioFloat32Array);

// 3. Run inference
const input = new ort.Tensor("float32", features, [1, 80, timeFrames]);
const length = new ort.Tensor("int64", BigInt64Array.from([BigInt(timeFrames)]), [1]);
const results = await session.run({
  [session.inputNames[0]]: input,
  [session.inputNames[1]]: length,
});
const logprobs = results[session.outputNames[0]];

// 4. CTC greedy decode (see web/frontend/src/worker/ctc-decode.ts)
//    argmax per timestep, collapse repeats, remove blanks, join tokens

// 5. Match decoded text against QuranDB (see web/frontend/src/lib/quran-db.ts)
//    Levenshtein fuzzy match against all 6,236 verses
```

Key files to reference for a complete implementation:
- [`web/frontend/src/worker/mel.ts`](web/frontend/src/worker/mel.ts) -- Mel spectrogram (NeMo-compatible)
- [`web/frontend/src/worker/ctc-decode.ts`](web/frontend/src/worker/ctc-decode.ts) -- CTC greedy decoder
- [`web/frontend/src/lib/quran-db.ts`](web/frontend/src/lib/quran-db.ts) -- Verse matching with Levenshtein distance
- [`web/frontend/src/lib/normalizer.ts`](web/frontend/src/lib/normalizer.ts) -- Arabic text normalization

### React Native (ONNX Runtime Mobile)

Use [`onnxruntime-react-native`](https://www.npmjs.com/package/onnxruntime-react-native) which wraps the native ONNX Runtime for iOS/Android.

```bash
npm install onnxruntime-react-native
```

```typescript
import { InferenceSession, Tensor } from "onnxruntime-react-native";

// Bundle the model in your app assets, or download on first launch
const session = await InferenceSession.create("path/to/fastconformer_ar_ctc_q8.onnx");

// Same inference pattern as the web version:
// 1. Compute 80-bin mel spectrogram from 16kHz audio
// 2. Create input tensors: features [1, 80, T] + length [1]
// 3. session.run() -> CTC logprobs
// 4. Greedy decode + QuranDB match
```

The mel spectrogram, CTC decoder, and QuranDB matching logic from [`web/frontend/src/`](web/frontend/src/) works directly in React Native -- it's pure TypeScript with no browser-specific APIs.

### Python

**Option A: ONNX Runtime (recommended for production)**

```bash
pip install onnxruntime numpy soundfile librosa
```

```python
import numpy as np
import onnxruntime as ort
import json
import librosa

# Load model + vocab
session = ort.InferenceSession("fastconformer_ar_ctc_q8.onnx")
vocab = json.load(open("vocab.json"))
id_to_char = {int(k): v for k, v in vocab.items()}
blank_id = max(id_to_char.keys())

# Load audio at 16kHz
audio, sr = librosa.load("recitation.wav", sr=16000)

# Compute NeMo-compatible mel spectrogram
audio = audio + 1e-5 * np.random.randn(len(audio))  # dither
audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])  # preemphasis
mel = librosa.feature.melspectrogram(
    y=audio, sr=16000, n_fft=512, hop_length=160, win_length=400,
    n_mels=80, fmax=8000, htk=True, norm="slaney"
)
mel = np.log(mel + 1e-5)
# Per-feature normalization
mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-10)

# Run inference
features = mel.astype(np.float32)[np.newaxis]  # [1, 80, T]
length = np.array([mel.shape[1]], dtype=np.int64)
logprobs = session.run(None, {
    session.get_inputs()[0].name: features,
    session.get_inputs()[1].name: length,
})[0]  # [1, T, vocab_size]

# CTC greedy decode
ids = logprobs[0].argmax(axis=1)
prev, tokens = -1, []
for i in ids:
    if i != prev and i != blank_id:
        tokens.append(id_to_char.get(i, ""))
    prev = i
transcript = "".join(tokens).replace("\u2581", " ").strip()

print(f"Transcript: {transcript}")
# Then match against quran.json using Levenshtein distance
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
git clone https://github.com/yazinsai/offline-tarteel.git
cd offline-tarteel
pip install -e ".[nemo]"
```

```python
from experiments.nvidia_fastconformer.run import predict

result = predict("recitation.wav")
# {"surah": 1, "ayah": 1, "ayah_end": 3, "score": 0.92, "transcript": "..."}
```

### Model details

| | Value |
|---|---|
| **Model** | `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` |
| **ONNX file** | `fastconformer_ar_ctc_q8.onnx` (131 MB, uint8 quantized) |
| **Input** | 80-bin mel spectrogram, 16 kHz, mono |
| **Output** | CTC logprobs over 1025-token Arabic BPE vocabulary |
| **Recall** | 95% on 53-sample benchmark (user recordings, professional, crowdsourced) |
| **Latency** | 0.7s on Apple Silicon (CPU), ~0.5-1s in browser WASM |
| **License** | [CC-BY-4.0](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) (NVIDIA model) |

---

## Goal

Ship a model that runs on-device (phone or laptop) with **95%+ recall**, **sub-second latency**, and **under 200 MB** on disk. The current best approach (`nvidia-fastconformer`) reaches **95% recall** at **115 MB** and **0.7s** latency on the v1 corpus. The shipped ONNX phoneme model achieves **78.6% streaming recall / 66.8% precision** in the browser (v1). Everything in this repo exists to close the remaining streaming gap.

## Design constraints

- **Offline-first.** No network calls at inference time. The model, index, and reference data all ship with the app.
- **Small models only.** Target < 200 MB total (model + any index). Phone storage is limited and download size matters.
- **Fast inference.** Under 1 second on Apple Silicon (MPS) or recent phone SoC. Users expect near-instant feedback after reciting.
- **Speaker-invariant.** Must work across accents, recording quality, and recitation styles -- not just professional studio audio from a single reciter.
- **Full Quran coverage.** All 6,236 verses, including short verses (3-4 words) that every approach currently struggles with.

## Results

Shipped `fastconformer-phoneme v4-tlog` (131 MB quantized ONNX) on the v1 and v2 corpora:

| Mode | Corpus | Recall | Precision | SeqAcc |
|---|---|---|---|---|
| Browser/RN streaming (300ms chunks) | v1 (53) | 78.6% | 66.8% | 47.2% |
| Browser/RN streaming (300ms chunks) | v2 (43) | 82.7% | 63.7% | 46.5% |
| Non-streaming (full-file) | v1 (53) | 84.1% | 84.9% | 81.1% |
| Non-streaming (full-file) | v2 (43) | 78.1% | 79.1% | 74.4% |

Streaming metrics are medians across 5 runs on v1 and 3 runs on v2 (ONNX non-determinism is ±3-6 per run). Non-streaming runs the whole audio through ONNX once and does a single `matchVerse()`.

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
  export_phoneme_onnx_modal.py         # Export phoneme model to ONNX + uint8 quantization
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

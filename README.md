# Offline Tarteel

[![Maintained by auto-maintainer](https://am.whhite.com/badge/yazinsai/offline-tarteel)](https://am.whhite.com/stats/yazinsai/offline-tarteel)

Offline Quran verse recognition. Record someone reciting, identify the surah and ayah -- no internet required.

**Best model:** NVIDIA FastConformer -- **87% recall**, **115 MB**, **0.33s latency**. Available as a quantized ONNX file (131 MB) that runs in browsers, React Native, and Python.

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
| **Recall** | 87% on 54-sample benchmark (user recordings, professional, crowdsourced) |
| **Latency** | 0.33s on Apple Silicon, ~0.5-1s in browser WASM |
| **License** | [CC-BY-4.0](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) (NVIDIA model) |

---

## Goal

Ship a model that runs on-device (phone or laptop) with **95%+ recall**, **sub-second latency**, and **under 200 MB** on disk. The current best approach (`nvidia-fastconformer`) reaches **87% recall** at **115 MB** and **0.33s** latency, but still misses the 95% recall bar. Everything in this repo exists to close that final gap.

## Design constraints

- **Offline-first.** No network calls at inference time. The model, index, and reference data all ship with the app.
- **Small models only.** Target < 200 MB total (model + any index). Phone storage is limited and download size matters.
- **Fast inference.** Under 1 second on Apple Silicon (MPS) or recent phone SoC. Users expect near-instant feedback after reciting.
- **Speaker-invariant.** Must work across accents, recording quality, and recitation styles -- not just professional studio audio from a single reciter.
- **Full Quran coverage.** All 6,236 verses, including short verses (3-4 words) that every approach currently struggles with.

## Experiment summary

We're building for **real-time streaming** -- the user recites and sees verses identified live. Non-streaming (full-file) results are shown for reference but streaming is the metric that matters.

### Streaming results (primary)

Tested with the TypeScript streaming pipeline (`RecitationTracker` feeding 300ms chunks).

| Experiment | v1 (53) | v2 (43) | Size | Verdict |
|---|---|---|---|---|
| **fastconformer-phoneme v4-tlog** | **45/53 (84.9%)** | **32/43 (74.4%)** | 131 MB | Current shipped model; streaming tracker up from 51% → 85% on v1 |
| w2v-phonemes | — | — | 116-970 MB | Not yet tested in streaming mode |

### Live FastAPI `/ws` endpoint (current)

Benchmarked with `python scripts/benchmark_streaming_endpoint.py --category multi` against the live websocket endpoint. This is the 9-sample multi-ayah subset, not the full 54-sample corpus. The run was repeated twice with identical summary metrics.

| Endpoint | Corpus | Recall | Precision | SeqAcc | Notes |
|---|---|---|---|---|---|
| **FastAPI `/ws`** | multi (9 samples) | **0.759** | **0.833** | **0.667** | Span-aware `ayah_end` emissions, followup-only continuation fast path, lexical rerank for ambiguous close matches |

On the same corrected harness, `main` scores `0.687` recall / `0.639` precision / `0.111` SeqAcc on this subset, so the current websocket matcher is materially better on multi-ayah live recitation.

### Non-streaming results (full-file baseline)

These use the Python benchmark runner (`benchmark/runner.py`) which transcribes the entire audio file at once.

| Experiment | Recall | Latency | Size | Verdict |
|---|---|---|---|---|
| **w2v-phonemes/large** | 100% | ~12s | 970 MB | Perfect accuracy but too bulky/slow for real-time offline |
| fastconformer-phoneme v4-tlog | 83% | 0.7s | 131 MB | Best model for streaming — needs tracker improvements |
| **w2v-phonemes/base** | 89% | ~3-6s | 116 MB | Competitive with FastConformer but 10x slower |
| **nvidia-fastconformer** | 87% | 0.3s | 115 MB | Best non-streaming balance of accuracy, speed, and size |
| ctc-alignment | 83% | 3.2s | 1.2 GB | Strong baseline but 6x too large for mobile |
| rabah-pruned-ctc | 74% | 7.0s | 145 MB | Pruning + fine-tuning works but still trails top models |
| two-stage | 73% | 9.5s | 1.3 GB | Good idea (ASR→CTC rescore) but blocked on small CTC model |
| tadabur-whisper-small | 84% | 1.6s | 461 MB | Best Whisper fine-tune; still trails FastConformer on all axes |
| tarteel-whisper-base | 72% | ~3s | 290 MB | Decent Whisper fine-tune but not competitive |
| distilled-ctc | 28% | 3.4s | 360 MB | Blocked: wav2vec2-base can't learn Arabic |
| contrastive-v2 | 0% | 0.1s | 900 MB | Failed: English encoder produces useless Arabic features |
| embedding-search | 0% | 0.3s | 397 MB | 0% cross-speaker — HuBERT encodes speaker not content |

### fastconformer-phoneme v4-tlog detail

| Corpus | Streaming | Non-streaming |
|---|---|---|
| v1 (53 samples) | **45/53 (84.9%)** | 43/53 (81.1%) |
| v2 (43 samples) | **32/43 (74.4%)** | 33/43 (76.7%) |

Streaming now matches or exceeds non-streaming on both corpora, thanks to tracker improvements (discovery/tracking pipeline, auto-advance, anti-cascade guard, fragment scoring). The previous gap (51% streaming vs 83% non-streaming on v1) has been fully closed.

## Experiment status

- **w2v-phonemes/large** achieves 100% recall on v1 corpus but at 970 MB + ~12s latency -- proves phoneme-based matching works perfectly, but impractical for offline mobile use.
- **NVIDIA FastConformer** remains the best practical model: 85% SeqAcc, 115 MB, 0.33s latency.
- **FastConformer fine-tune sweep regressed across all tested variants (v1/v2a/v2b/v3c).**  
  v1 (Rabah+RetaSy, 2K steps): 81% SeqAcc / 84% recall (`benchmark/results/2026-02-27_092540.json`)  
  v2a (Rabah+RetaSy, LR=2e-5, 1K steps): 80% SeqAcc / 81% recall (`benchmark/results/2026-02-27_104327.json`)  
  v2b (Quran-only, LR=2e-5, 1K steps): 81% SeqAcc / 83% recall (`benchmark/results/2026-02-27_115055.json`)  
  v3c (Quran-only, LR=1e-5, 6K steps, freeze=12): 80% SeqAcc / 81% recall (`benchmark/results/2026-02-27_163433.json`)  
  All produced larger local checkpoints (~459 MB) and none beat pretrained baseline (85%/87%).
- **Rabah pruned+fine-tuned path now works.** Fine-tuning the CTC head on pruned representations recovered accuracy from 12% to 72% (8-layer first_n). The 8L int8 model is 145 MB -- well under the 200 MB target. The key insight: `first_n` pruning (keep layers 0-7) vastly outperforms `evenly_spaced` (72% vs 56%).
- **Two-stage faster-whisper path** now runs with int8 Stage 2 at 306 MB and 3.96s (down from 582 MB / 10s), but still trails on accuracy (70% SeqAcc).
- **Phoneme CTC fine-tuning is the current shipped model.** `v4-tlog` with 69-phoneme Buckwalter CTC head, runs in browser via ONNX. Streaming: **85% v1, 74% v2**. Non-streaming: 81% v1, 77% v2. Streaming now matches or beats non-streaming thanks to tracker improvements. TLOG data helps in small doses (5/verse) but scaling up (15-30/verse) regresses badly regardless of quality filtering.
- **The live FastAPI websocket matcher is now materially better on the full streaming harness, not just the multi subset.** On the corrected 53-sample `/ws` corpus it scores 83.6% recall / 79.3% precision / 71.7% SeqAcc. On the 9-sample multi subset it scores 92.6% recall / 93.7% precision / 66.7% SeqAcc. The older websocket baseline (`418a788`) scored 66.4% recall / 48.3% precision / 22.6% SeqAcc on the same full harness. The main gains came from fragment-aware long-ayah matching, sticky long-verse locks, residual trimming, followup-aware continuation commits, span-aware emissions, and lexical reranking.
- **Tadabur-Whisper-Small** (`FaisaI/tadabur-Whisper-Small`) is the best Whisper-family model tested: 84% recall, 77% SeqAcc, 1.6s latency at 461 MB. Beats tarteel-whisper-base (+12% recall, 2x faster) but still trails FastConformer on accuracy (93%), speed (0.6s), and size (115 MB). Streaming: 87% recall / 42% SeqAcc. Failures are mostly multi-verse truncation.
- **N-best + brute-force didn't help.** `fastconformer-nbest-bruteforce` (83% SeqAcc) is worse than plain FastConformer. CTC beam search without a language model produces near-identical hypotheses, and brute-forcing entire surahs just picks wrong candidates. CTC re-scoring can't recover failures caused by bad candidate retrieval.
- **v5-robust fine-tuning regressed.** Both u6 (freeze 6) and u8 (freeze 8) without TLOG performed worse than v4-tlog baseline. Removing TLOG entirely hurts — the 18K TLOG sweet spot provides essential phone-mic diversity. More encoder unfreezing also hurts (u8 worse than u6). See TLOG data mix table below for details.

### Next experiments to try

Ranked by expected ROI:

1. **Teacher-relabel TLOG/RetaSy** — Current training maps every clip to the full canonical verse phonemes, even if the clip only contains a fragment. Use `w2v-phonemes/large` (100% accuracy) as a teacher to force-align each clip and emit a trimmed phoneme span. This is likely why 5/verse works but more hurts: noisy clips labeled with wrong-length targets create conflicting gradients.

2. **Partial-window curriculum** — The model never sees what streaming actually produces: 1-6s clipped windows with missing starts/ends. Synthesize short windows from clean Iqra audio, trim phoneme labels to the spoken span, and add as a late low-LR robustness phase on top of v4-tlog.

3. **Phone-channel augmentation** — Current augmentation (speed/gain/white noise/shift/silence) doesn't simulate real phone recording conditions. Add: Opus/AAC codec encode-decode, band-limiting, clipping/AGC, room impulse responses, real noise beds.

4. **Phoneme n-gram candidate retrieval** — The browser matcher depends on full-string Levenshtein for candidate ranking. When CTC garbles part of the output, ratio() gets diluted. Add phoneme-word trigram or token 5-gram index for candidate retrieval (yawm-style), then CTC rescore top candidates.

5. **Constrained beam search** — Unconstrained N-best failed because hypotheses were near-identical. Prefix-constrained beam against a Quran phoneme trie or KenLM would force diverse, Quran-valid hypotheses.

6. **Knowledge distillation** — Distill from `w2v-phonemes/large` (100% accuracy, 970 MB) into the current FastConformer, not into wav2vec2-base. Frame-level KL on TLOG/RetaSy windows, or candidate-level distillation.

### Phoneme CTC Fine-Tuning (v4-tlog -- best streaming model)

Fine-tuned FastConformer's CTC head on a 69-phoneme Buckwalter vocabulary using Iqra, TTS, RetaSy, and TLOG datasets. Runs in the browser via ONNX Runtime Web with real-time streaming recognition.

- **Model:** `fastconformer_phoneme_q8.onnx` (131 MB, uint8 quantized)
- **Training:** Modal A100-80GB, NeMo, freeze 8/18 encoder layers, batch 32, grad accum 2
- **Best config (v4-tlog):** 71K Iqra + 55K TTS + 1.8K RetaSy + ~18K TLOG (5/verse, quality-filtered)
- **Test corpus v1** (53 samples): streaming **45/53 (84.9%)**, non-streaming 43/53 (81.1%)
- **Test corpus v2** (43 samples): streaming **32/43 (74.4%)**, non-streaming 33/43 (76.7%)

**TLOG data mix experiments:**

| Model | TLOG samples | Quality threshold | Streaming v1 | Streaming v2 |
|---|---|---|---|---|
| **v4-tlog** (best) | ~18K (5/verse) | 0.3 | **45/53 (84.9%)** | **32/43 (74.4%)** |
| v4-tlog-heavy | ~53K (15/verse) | 0.3 | 36-38/53 (70%) | 25/43 (58%) |
| v4-tlog-hq | ~74K (30/verse) | 0.5 | 29-31/53 (56%) | 23-24/43 (54%) |

**Robustness fine-tuning experiments (v5-robust):**

| Model | Freeze | TLOG | Streaming v1 | Streaming v2 | Combined |
|---|---|---|---|---|---|
| **v4-tlog** (best) | 8/18 | ~18K (5/verse) | **45/53 (84.9%)** | **32/43 (74.4%)** | **77/96** |
| v5-robust-u6 | 6/18 | 0 (skipped) | 43/53 (81.1%) | 33/43 (76.7%) | 76/96 |
| v5-robust-u8 | 8/18 | 0 (skipped) | 40/53 (75.5%) | 29/43 (67.4%) | 69/96 |

**Key findings:**
- TLOG volume is the bottleneck, not quality. Even high-quality-only TLOG (threshold 0.5) regresses when scaled up. The domain gap between phone mic recordings and clean studio data overwhelms quality filtering. Sweet spot: 5 samples/verse (~18K after filter).
- Removing TLOG entirely also regresses — the 18K TLOG sweet spot provides essential phone-mic diversity that Iqra/TTS alone can't match.
- More encoder unfreezing hurts: u8 (freeze 8, same as baseline) regressed worse than u6 (freeze 6). The pretrained Arabic encoder representations are valuable and should be preserved.
- RetaSy mapping yields only ~1.8K samples (30% hit rate on text normalization), insufficient to compensate for missing TLOG.

Scripts:
```bash
# Train phoneme CTC model
modal run --detach scripts/train_fastconformer_phoneme_modal.py \
  --output-name fastconformer-phoneme-v4-tlog --max-tlog-per-verse 5 --filter-tlog

# Export to ONNX
modal run scripts/export_phoneme_onnx_modal.py --output-name fastconformer-phoneme-v4-tlog

# Test accuracy (run 3x for reliable measurement due to ONNX non-determinism)
cd web/frontend && npx tsx test/validate-streaming.ts
npx tsx test/validate-streaming.ts --corpus=test_corpus_v2
```

### Two-Stage Retrieval (historical 72% setup; current pruned variant at 70%)
Moonshine Tiny Arabic (27M params, 103 MB) does fast ASR to get a rough transcript, then CTC forced-alignment re-scores only the top 50 verse candidates. This bounds the expensive CTC computation to 50 candidates instead of 6,236. Currently falls back to the large CTC model (1.2 GB) because the small CTC model failed to train (see "What we tried"). With a working small CTC re-scorer (~95 MB), this would hit the size target.

### Distilled CTC (failed)
wav2vec2-base (95M params) knowledge-distilled from the large CTC model. The base model can't learn Arabic CTC -- English-only pretraining means no Arabic speech representations to build on (see "What we tried"). The smallest multilingual alternatives (MMS-300M, XLS-R-300M) are ~300M params, barely smaller than the 317M large model, which defeats the purpose.

### Contrastive V2 (failed)
CLIP-style contrastive model mapping audio to a speaker-invariant 256-dim embedding. Trained for 8 epochs (6 frozen + 2 unfrozen) on 30k EveryAyah samples on A100-80GB. Validation accuracy stuck at ~9% (random chance = 3.1% with batch 32) -- the model memorizes training data but doesn't generalize. Same root cause: wav2vec2-base doesn't produce useful Arabic audio features. See "What we tried" below.

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

## All experiments

### ctc-alignment (strong baseline -- 81% accuracy, 1.2 GB)

CTC forced alignment using a pre-trained Arabic wav2vec2 model. Scores candidate verses directly against frame-level character logits using the CTC forward algorithm, bypassing the information loss of greedy decoding.

**Flow:** audio -> wav2vec2 frame logits -> greedy decode -> Levenshtein top-100 candidates -> CTC re-score -> multi-verse span scoring -> best score wins

- **Model:** `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (1.2 GB)
- **Gap to target:** Accurate enough to prove the approach works, but too large (6x) and too slow (5x) for on-device use.

### two-stage (72% SeqAcc -- large CTC fallback)

Moonshine Tiny Arabic (27M) for fast ASR, then CTC forced-alignment re-scoring on just the top 50 candidates. Bounds the expensive CTC computation from 6,236 verses to 50.

- **Stage 1:** Moonshine Tiny Arabic (103 MB) -> transcript -> QuranDB.search(top_k=50)
- **Stage 2:** CTC re-score 50 candidates (currently falls back to large model)
- **Result:** 72% SeqAcc, 78% recall, 78% precision with large CTC re-scorer
- **Blocker:** No working small CTC model. wav2vec2-base can't learn Arabic (see "What we tried").
- **Target size:** ~200 MB with a working small CTC model

### distilled-ctc (blocked)

wav2vec2-base (95M params) with knowledge distillation from the large CTC model. Same scoring approach as ctc-alignment but 3x smaller.

- **Teacher:** wav2vec2-large-xlsr-53-arabic (315M params, 1.2 GB)
- **Student:** wav2vec2-base + Arabic CTC head (95M params, ~380 MB, target ~95 MB int8)
- **Status:** Blocked. The student model (wav2vec2-base) can't learn Arabic CTC -- English-only pretraining means no Arabic speech representations to build on. See "What we tried" below.

### rabah-pruned-ctc (fine-tuned variants are the key result)

Rabah's Quran-specific wav2vec2-large checkpoint, layer-pruned and fine-tuned for small on-device CTC alignment.

- **Source model:** `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final` (24 layers, 1.2 GB)
- **Scoring:** Same CTC forced-alignment stack as `ctc-alignment`
- **Fine-tuned variants:** `8-layer-ft-fn-int8` (72%, 145 MB), `12-layer-ft-es-int8` (72%, 193 MB), `8-layer-ft-es-int8` (56%, 145 MB), `6-layer-ft-es-int8` (48%, 121 MB)
- **Naive-pruned baselines:** `12/8/6-layer-int8` (accuracy collapses without fine-tuning -- CTC head was trained on layer-24 representations)
- **Training:** `scripts/train_pruned_ctc_modal.py` on Modal A100-80GB. 5000 steps, LR 3e-5, EveryAyah + RetaSy, freeze CNN + lower half of transformer.
- **Key finding:** `first_n` pruning (keep layers 0-7) gets 72% vs 56% for `evenly_spaced` (keep layers 0,3,7,10,13,16,20,23). Contiguous early layers preserve better feature propagation.

### nvidia-fastconformer (new)

Arabic FastConformer hybrid model via NeMo:

- **Model:** `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
- **Pipeline:** audio -> FastConformer transcript -> QuranDB span-aware match
- **Dependency:** `nemo_toolkit[asr]` (optional extra `.[nemo]`)
- **Local override:** `NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR=/abs/path/to/fine_tuned_model`
- **Fine-tune sweep (2026-02-27):** CTC-only adaptation did not improve baseline.
  - `v1` (Rabah+RetaSy, 2K steps): 81% SeqAcc / 84% recall (`benchmark/results/2026-02-27_092540.json`)
  - `v2a` (Rabah+RetaSy, LR=2e-5, 1K steps): 80% SeqAcc / 81% recall (`benchmark/results/2026-02-27_104327.json`)
  - `v2b` (Quran-only, LR=2e-5, 1K steps): 81% SeqAcc / 83% recall (`benchmark/results/2026-02-27_115055.json`)
  - Best fine-tuned variant was `v2b`, but still below pretrained baseline (85% SeqAcc / 87% recall).

### fastconformer-ctc-rescore (new -- no accuracy gain over FastConformer alone)

Two-stage pipeline combining FastConformer ASR (best transcription) with CTC forced-alignment re-scoring (best alignment), using confidence-gated fallback to skip Stage 2 on easy samples.

- **Stage 1:** NVIDIA FastConformer (115 MB) -> transcript -> QuranDB span-aware match
- **Stage 2 (fallback):** CTC re-score top-50 candidates using fine-tuned 8L Rabah CTC (145 MB)
- **Gate:** If Stage 1 score >= threshold (default 0.7), return immediately without running Stage 2
- **Result:** 85% SeqAcc / 87% recall / 89% precision, 0.66s latency (threshold=0.7), 273 MB total
- **Finding:** CTC re-scoring does **not** recover any of the 8 samples FastConformer fails on, even with threshold=0.95 (forcing CTC on nearly everything). Both models fail on the same hard cases: short isolated letters (Ya-Sin, Al-Ikhlas) and multi-verse passages. The "complementary errors" hypothesis didn't hold -- the failures are in the candidate retrieval (bad transcript from FastConformer), not in the scoring.
- **Tuning:** `FASTCONFORMER_CTC_CONFIDENCE=0.7` and `FASTCONFORMER_CTC_STAGE2_DYNAMIC_INT8=1` (default). Threshold sweep: 0.7 → 85% / 0.66s, 0.95 → 85% / 2.43s. Higher threshold = more CTC runs = more latency, same accuracy.

### fastconformer-nbest-bruteforce (worse than baseline -- 83% SeqAcc, 500 MB)

N-best beam search on FastConformer CTC logits + CTC brute-force scoring of all verses in candidate surahs. The hypothesis: expanding the transcript candidate pool via beam search and brute-forcing CTC alignment across entire surahs would recover failures where the greedy transcript is wrong.

- **Stage 1:** FastConformer CTC logits → pyctcdecode beam search → 5-best hypotheses → QuranDB match each
- **Stage 2 (fallback):** If no match exceeds confidence threshold (0.7), collect top-10 surahs from all N-best matches, brute-force CTC score all verses + multi-verse spans (up to 6000 candidates)
- **Result:** 83% SeqAcc / 85% recall / 87% precision, 0.85s latency, ~500 MB total
- **Finding:** **Worse than baseline.** Introduced 2 new failures (`retasy_003` 1:2→37:182, `retasy_019` 3:2→102:6) where the N-best confidence gate incorrectly triggers brute-force, which picks wrong candidates. Did not recover any of the baseline's 8 failures -- the correct surah never appears in the candidate set because all N-best hypotheses produce similar wrong transcripts (low beam diversity without a language model). The brute-force path adds ~0.5s latency and 385 MB of Stage 2 model weight for no accuracy gain.
- **Key insight:** CTC beam search without a language model produces near-identical hypotheses (all 5 beams differ by only a few BPE tokens). A Quran-specific language model or constrained decoding would be needed to make N-best useful.

### two-stage-faster-whisper-pruned (new)

Two-stage retrieval variant using faster-whisper for ASR and fine-tuned pruned CTC for re-scoring:

- **Stage 1:** `OdyAsh/faster-whisper-base-ar-quran` (147 MB, CTranslate2 int8)
- **Stage 2:** CTC re-score top-50 candidates using fine-tuned 8L Rabah CTC with dynamic int8 (`TWO_STAGE_STAGE2_DYNAMIC_INT8=1`, default)
- **Result:** 70% SeqAcc / 72% recall / 71% precision, 3.96s latency, 306 MB total
- **Tradeoff:** Major speed/size gain vs prior fp32 run (10.06s, 582 MB) with small precision/recall drop

### w2v-phonemes (100% large, 89% base -- high latency)

Phoneme-based matching using wav2vec2 CTC models fine-tuned on Quranic recitations. Decodes audio into IPA phoneme sequences and matches against pre-phonemized Quran reference text via Levenshtein ratio + n-gram anchor voting.

- **Models:** `hetchyy/r7_onnx_int8` (large, 970 MB) and `hetchyy/r15_95m_onnx_int8` (base, 116 MB)
- **Pipeline:** audio → wav2vec2 phoneme CTC → IPA string → Pass 1 Levenshtein vs 6236 verses → Pass 1b n-gram voting → Pass 2 multi-verse spans → best match
- **Result (large):** 100% SeqAcc on v1 corpus (53 samples), 95% on v2 (43 samples), ~12s latency
- **Result (base):** 89% SeqAcc on v1, 88% on v2, ~3-6s latency
- **Tradeoff:** The large model achieves perfect accuracy but is too large (970 MB) and slow (~12s) for real-time offline use. The base model is competitive with FastConformer on accuracy at similar size but 10x slower.
- **Data files:** `data/phoneme_cache.pkl` (7.6 MB) + `data/phoneme_ngram_index_5.pkl` (6.0 MB) -- pre-computed phoneme reference for all verses

### contrastive-v2 (failed -- val accuracy stuck at 9%)

CLIP-style contrastive model (QuranCLAP v2). Maps audio to a speaker-invariant 256-dim embedding, matched against a pre-computed FAISS index of all 6,236 verses. One forward pass + nearest neighbor = verse ID. No ASR needed.

- **Audio encoder:** wav2vec2-base (95M params)
- **Text encoder:** AraBERT v02 (136M params)
- **Training:** 30k samples from EveryAyah, batch 32 (effective 128 via grad accum), two-phase (frozen -> unfreeze last 2 layers), A100-80GB
- **Result:** Val accuracy stuck at ~9% after 8 epochs. Model overfits (train acc 14%, val acc 9%). wav2vec2-base can't produce useful Arabic audio representations.
- **Target size:** ~367 MB (audio encoder + projection + FAISS index)

### tarteel-whisper-base (67% accuracy, 290 MB)

Tarteel's Whisper-base fine-tuned for Quranic Arabic. Transcribe-then-match pipeline.

### whisper-lora (58% accuracy, 485 MB)

Whisper-small + LoRA adapter fine-tuned on EveryAyah + RetaSy. Transcribe-then-match pipeline.

### embedding-search (not viable standalone)

HuBERT embeddings + FAISS nearest neighbor. 100% same-reciter, 0% cross-speaker. HuBERT encodes speaker identity more than linguistic content.

### contrastive v1 (proof of concept, ~1.6% accuracy)

First attempt at CLIP-style audio-text matching. Failed due to English HuBERT on Arabic audio, batch size 4, single reciter. Contrastive-v2 addresses all of these.

### streaming-asr (43% accuracy, 6x faster to first match)

mlx-whisper chunked streaming. Streaming is good for UX (0.19s to first match) but accuracy is limited by whisper-base quality.

### new-models (model comparison)

Head-to-head benchmark of 8 ASR models. Key finding: Moonshine Tiny Arabic (103 MB) matches Whisper Large-v3-Turbo (3.1 GB) at 30x smaller. This is why the two-stage experiment uses Moonshine as Stage 1.

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

## Key findings

1. **CTC forced alignment is the most accurate approach** -- scoring candidates directly against frame logits avoids the information loss of greedy decoding, giving 81% accuracy. But the model is too large (1.2 GB) for on-device deployment.

2. **Two-stage retrieval works.** Using ASR only for candidate retrieval (top 50) then CTC for the final decision gets 72% SeqAcc -- only 9 points behind scoring all 6,236 verses. The retrieval step is fast and the accuracy gap comes from candidate recall, not re-scoring quality.

3. **ASR quality is the bottleneck for transcribe-then-match approaches.** All ASR-based approaches fail on the same samples. Two-stage sidesteps this by using ASR only for candidate retrieval, not the final decision.

4. **Layer pruning + fine-tuning works for CTC size reduction.** While there are no small Arabic wav2vec2 models pretrained from scratch, pruning a large Quran-specific model (24→8 layers) and fine-tuning the CTC head recovers most accuracy (72% vs 81% full model). The `first_n` strategy (keep contiguous early layers) outperforms `evenly_spaced` by 16 percentage points, suggesting early transformer layers carry the most transferable features.

5. **Contrastive audio-text matching needs a multilingual encoder.** Both embedding search (HuBERT, 0% cross-speaker) and contrastive-v2 (wav2vec2-base, 9% val accuracy) failed because English-pretrained audio encoders don't produce useful features for Arabic speech. Multi-reciter training and deeper projection heads aren't enough to overcome bad audio representations.

6. **Small ASR models can match large ones.** Moonshine Tiny Arabic (103 MB) matches Whisper Large-v3-Turbo (3.1 GB) on our benchmark. But fine-tuning Moonshine degrades it (LLaMA tokenizer with character-level Arabic tokens is fragile).

7. **Short verses are hard across all approaches.** Verses under 3-4 words don't provide enough signal. May need minimum-length gating or surah-context bias in the final product.

## What we tried (and didn't work)

### Moonshine Tiny AR LoRA fine-tuning

Attempted to LoRA fine-tune [UsefulSensors/moonshine-tiny-ar](https://huggingface.co/UsefulSensors/moonshine-tiny-ar) (27M params, 103 MB) on Quran audio to improve its transcription for verse matching.

**What we tried:**
- LoRA r=8 on all attention (q/k/v/o_proj), lr=5e-4, 3000 steps
- LoRA r=4 encoder-only (decoder frozen), lr=1e-4, 2000 steps
- LoRA r=8 decoder-only, various learning rates
- Full fine-tuning (all 27M params), lr=2e-5, 3000 steps
- Diacritics stripped + EOS token appended (fixing data distribution mismatch)

Every configuration degraded the model. Base Moonshine scores 56% SeqAcc; the best fine-tuned variant dropped to 35-38%.

**Root cause:** Moonshine uses a LLaMA-derived tokenizer with only ~54 Arabic character tokens out of 32K vocab. Every Arabic character is its own token, so the decoder does character-by-character generation. Any weight perturbation (even LoRA r=4 at low LR) corrupts the character sequences, producing garbled output like `ب س الهر حا الحي` instead of `بسم الله الرحمن الرحيم`.

**Conclusion:** The base model (56% SeqAcc, 103 MB) is useful as-is for the two-stage pipeline's Stage 1 (rough transcript for candidate retrieval), but shouldn't be fine-tuned.

### wav2vec2-base Arabic CTC fine-tuning

Attempted to fine-tune `facebook/wav2vec2-base` (95M params, English SSL pretrained) with an Arabic CTC head, using the vocabulary from `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`. Trained on EveryAyah + RetaSy datasets (85/15 interleave) on Modal A100-80GB.

**What we tried:**
- LR 3e-4, 5000 steps, batch 32 x grad_accum 2 → model collapsed at step ~1400, outputting only token 46 (a diacritic)
- LR 1e-4, 5000 steps → loss plateaued at 3.2-3.3 through all 5000 steps, model outputs only token 0 (pad)
- Frozen CNN feature extractor, only training transformer layers + CTC head

**Root cause:** wav2vec2-base was pretrained on English-only LibriSpeech. Its SSL representations encode English phonemes, not Arabic ones. Fine-tuning the transformer layers and CTC head alone can't bridge this gap -- the CNN feature extractor (which is frozen) produces features that don't represent Arabic speech sounds. The loss never drops below 3.2 because the model fundamentally can't distinguish Arabic characters from the audio features it receives.

**Alternatives considered:**
- `facebook/mms-300m` (multilingual, knows Arabic) -- 300M params, barely smaller than the 317M large model
- `facebook/wav2vec2-xls-r-300m` (multilingual) -- same size issue
- `DistilHuBERT` (23.5M params) -- English-only, same problem as wav2vec2-base

**Conclusion:** There is no existing small (<150M params) wav2vec2-family model with Arabic speech representations. Getting a small CTC model requires either (a) distilling from a multilingual model that already works, or (b) a different architecture entirely. This blocks both the distilled-ctc and two-stage experiments' path to the size target.

### Contrastive v2 (QuranCLAP v2) training

Attempted CLIP-style contrastive learning with wav2vec2-base (audio) + AraBERT v02 (text). Trained on 30k EveryAyah samples on Modal A100-80GB, batch 32 with gradient accumulation 4 (effective batch 128).

**Training log:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| P1 E1 (frozen) | 2.88 | 9.7% | 3.05 | 9.0% |
| P1 E6 (frozen) | 2.56 | 13.9% | 3.12 | 8.8% |
| P2 E1 (unfrozen) | 2.77 | 11.9% | 3.03 | 9.0% |
| P2 E2 (unfrozen) | 2.56 | 14.1% | 3.07 | 9.4% |

Random chance with batch 32 = 3.1%. Val accuracy plateaued at ~9% across all 8 epochs while train accuracy climbed to 14% -- classic overfitting. Unfreezing the last 2 encoder layers in Phase 2 didn't help.

**Root cause:** Same as CTC: wav2vec2-base produces English speech features, not Arabic ones. The projection heads can memorize training pairs but can't learn generalizable audio-text alignment because the audio features don't encode Arabic phonetic content. The AraBERT text encoder works fine -- the bottleneck is entirely on the audio side.

**Conclusion:** Any approach using wav2vec2-base as an Arabic audio encoder will fail. Future contrastive attempts need a multilingual audio encoder (e.g., XLS-R-300M, MMS-1B) or a completely different architecture.

## Further reading

- `REPORT.md` -- Full experiment report with per-sample breakdowns, failure analysis, and recommendations
- `RESEARCH-audio-to-verse.md` -- Research survey of approaches (WavLink, Moonshine v2, WhisperKit, contrastive learning, audio fingerprinting)
- `docs/plans/` -- Design documents for individual experiments
- Individual experiment `README.md` files for reproduction instructions

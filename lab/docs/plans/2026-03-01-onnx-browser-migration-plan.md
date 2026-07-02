# Browser-Only ONNX Runtime Web Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Python FastAPI backend with fully in-browser inference using ONNX Runtime Web, producing a PWA that works offline after first load.

**Architecture:** Web Worker runs the full pipeline (mel spectrogram → ONNX inference → CTC decode → QuranDB matching → discovery/tracking state machine). Main thread handles audio capture and UI. Service Worker caches app shell for offline.

**Tech Stack:** TypeScript, Vite, onnxruntime-web (WASM/WebGPU), @huggingface/transformers (audio utils only), vite-plugin-pwa

**Design doc:** `docs/plans/2026-03-01-onnx-browser-migration-design.md`

---

## Task 1: ONNX Export & Validation (Python)

Export the FastConformer model to ONNX and validate that ONNX Runtime produces identical outputs to NeMo native. Also extract the BPE vocabulary for the CTC decoder.

**Files:**
- Create: `scripts/export_onnx.py`
- Output: `data/fastconformer_ar_ctc.onnx`, `data/vocab.json`

**Step 1: Write the export script**

```python
"""Export FastConformer to ONNX (CTC-only) and extract BPE vocabulary."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_onnx():
    import nemo.collections.asr as nemo_asr

    model_id = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    out_dir = PROJECT_ROOT / "data"
    onnx_path = out_dir / "fastconformer_ar_ctc.onnx"
    vocab_path = out_dir / "vocab.json"

    print(f"Loading model: {model_id}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    model.eval()

    # Export CTC-only (single ONNX file)
    model.set_export_config({"decoder_type": "ctc"})
    print(f"Exporting to: {onnx_path}")
    model.export(str(onnx_path))
    print(f"ONNX export complete: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Extract BPE vocabulary
    tokenizer = model.tokenizer
    vocab = {}
    for i in range(tokenizer.vocab_size):
        vocab[str(i)] = tokenizer.ids_to_tokens([i])[0]
    # Add blank token (last index for CTC)
    vocab[str(tokenizer.vocab_size)] = "<blank>"

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved: {vocab_path} ({len(vocab)} tokens)")


def validate_onnx():
    """Validate ONNX model produces same output as NeMo native."""
    import numpy as np
    import onnxruntime as ort
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    import tempfile

    model_id = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    onnx_path = PROJECT_ROOT / "data" / "fastconformer_ar_ctc.onnx"

    # Load NeMo model
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)
    model.eval()
    model.change_decoding_strategy(decoder_type="ctc")

    # Use a test audio file
    test_files = sorted((PROJECT_ROOT / "benchmark" / "test_corpus").glob("*.wav"))
    if not test_files:
        print("No test audio files found. Skipping validation.")
        return
    test_audio = test_files[0]
    print(f"Validating with: {test_audio.name}")

    # NeMo native transcription
    outputs_nemo = model.transcribe(
        audio=[str(test_audio)], batch_size=1, return_hypotheses=False, verbose=False
    )
    if isinstance(outputs_nemo, tuple):
        outputs_nemo = outputs_nemo[0]
    nemo_text = outputs_nemo[0] if isinstance(outputs_nemo, list) else str(outputs_nemo)
    print(f"NeMo output:  {nemo_text[:100]}")

    # ONNX inference with NeMo's own preprocessor
    import torch
    audio_data, sr = sf.read(str(test_audio), dtype="float32")
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
    audio_length = torch.tensor([audio_tensor.shape[1]])

    # Use NeMo's preprocessor
    processed, proc_len = model.preprocessor(
        input_signal=audio_tensor, length=audio_length
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    print(f"ONNX inputs:  {[(n, sess.get_inputs()[i].shape) for i, n in enumerate(input_names)]}")
    print(f"ONNX outputs: {[(n, sess.get_outputs()[i].shape) for i, n in enumerate(output_names)]}")

    ort_inputs = {
        input_names[0]: processed.detach().numpy(),
        input_names[1]: proc_len.detach().numpy(),
    }
    logprobs = sess.run(None, ort_inputs)[0]

    # CTC greedy decode
    import json
    vocab_path = PROJECT_ROOT / "data" / "vocab.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    blank_id = max(int(k) for k in vocab.keys())

    predicted_ids = np.argmax(logprobs[0], axis=-1)
    tokens = []
    prev = -1
    for idx in predicted_ids:
        if idx != prev and idx != blank_id:
            tokens.append(vocab.get(str(idx), "?"))
        prev = idx
    onnx_text = "".join(tokens).replace("▁", " ").strip()
    print(f"ONNX output:  {onnx_text[:100]}")

    # Compare
    if nemo_text.strip() == onnx_text.strip():
        print("MATCH: ONNX output matches NeMo native.")
    else:
        print("MISMATCH: Outputs differ. Investigate preprocessing.")
        print(f"  NeMo: {nemo_text}")
        print(f"  ONNX: {onnx_text}")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate_onnx()
    else:
        export_onnx()
        if "--skip-validate" not in sys.argv:
            validate_onnx()
```

**Step 2: Run the export**

Run: `cd /Users/rock/ai/projects/offline-tarteel && python scripts/export_onnx.py`
Expected: `data/fastconformer_ar_ctc.onnx` (~115 MB) and `data/vocab.json` created. Validation prints "MATCH".

**Step 3: Extract preprocessor config**

After export, print the exact preprocessor parameters to verify against design doc assumptions:

```python
# Add to export script or run interactively
print(model.preprocessor._cfg)
```

Confirm: sample_rate=16000, features=80, n_fft=512, window_size=0.025, window_stride=0.01, window="hann", normalize="per_feature", dither=1e-5, preemph_coef=0.97

**Step 4: Commit**

```bash
git add scripts/export_onnx.py data/vocab.json
git commit -m "feat: ONNX export script and BPE vocabulary for browser inference"
```

Note: `data/fastconformer_ar_ctc.onnx` is gitignored (*.onnx). It will be hosted separately for the PWA to download.

---

## Task 2: Shared Types & Constants

Create the TypeScript types and constants that multiple modules share.

**Files:**
- Create: `web/frontend/src/lib/types.ts`

**Step 1: Write types and constants**

```typescript
// ---------------------------------------------------------------------------
// Message protocol (Worker ↔ Main Thread)
// ---------------------------------------------------------------------------
export interface VerseMatchMessage {
  type: "verse_match";
  surah: number;
  ayah: number;
  verse_text: string;
  surah_name: string;
  confidence: number;
  surrounding_verses: SurroundingVerse[];
}

export interface WordProgressMessage {
  type: "word_progress";
  surah: number;
  ayah: number;
  word_index: number;
  total_words: number;
  matched_indices: number[];
}

export interface RawTranscriptMessage {
  type: "raw_transcript";
  text: string;
  confidence: number;
}

export interface SurroundingVerse {
  surah: number;
  ayah: number;
  text: string;
  is_current: boolean;
}

// Main → Worker
export type WorkerInbound =
  | { type: "init" }
  | { type: "audio"; samples: Float32Array };

// Worker → Main
export type WorkerOutbound =
  | { type: "loading"; percent: number }
  | { type: "ready" }
  | VerseMatchMessage
  | WordProgressMessage
  | RawTranscriptMessage;

// ---------------------------------------------------------------------------
// Quran data (from quran.json)
// ---------------------------------------------------------------------------
export interface QuranVerse {
  surah: number;
  ayah: number;
  text_uthmani: string;
  text_clean: string;
  surah_name: string;
  surah_name_en: string;
  text_clean_no_bsm?: string | null;
}

export interface SurahData {
  surah: number;
  surah_name: string;
  surah_name_en: string;
  verses: { ayah: number; text_uthmani: string }[];
}

// ---------------------------------------------------------------------------
// Constants (matching server.py exactly)
// ---------------------------------------------------------------------------
export const SAMPLE_RATE = 16000;
export const TRIGGER_SECONDS = 2.0;
export const TRIGGER_SAMPLES = SAMPLE_RATE * TRIGGER_SECONDS;
export const MAX_WINDOW_SECONDS = 10.0;
export const MAX_WINDOW_SAMPLES = SAMPLE_RATE * MAX_WINDOW_SECONDS;
export const SILENCE_RMS_THRESHOLD = 0.005;

export const VERSE_MATCH_THRESHOLD = 0.45;
export const FIRST_MATCH_THRESHOLD = 0.75;
export const RAW_TRANSCRIPT_THRESHOLD = 0.25;
export const SURROUNDING_CONTEXT = 2;

export const TRACKING_TRIGGER_SECONDS = 0.5;
export const TRACKING_TRIGGER_SAMPLES = SAMPLE_RATE * TRACKING_TRIGGER_SECONDS;
export const TRACKING_SILENCE_TIMEOUT = 4.0;
export const TRACKING_SILENCE_SAMPLES = SAMPLE_RATE * TRACKING_SILENCE_TIMEOUT;
export const TRACKING_MAX_WINDOW_SECONDS = 5.0;
export const TRACKING_MAX_WINDOW_SAMPLES = SAMPLE_RATE * TRACKING_MAX_WINDOW_SECONDS;
export const STALE_CYCLE_LIMIT = 4;
export const LOOKAHEAD = 5;
```

**Step 2: Commit**

```bash
git add web/frontend/src/lib/types.ts
git commit -m "feat: shared types and constants for browser inference"
```

---

## Task 3: Arabic Normalizer (TypeScript port)

Port `shared/normalizer.py` to TypeScript. This is a direct transliteration.

**Files:**
- Create: `web/frontend/src/lib/normalizer.ts`

**Reference:** `shared/normalizer.py`

**Step 1: Write the normalizer**

```typescript
const DIACRITICS_RE =
  /[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]/g;

const NORM_MAP: Record<string, string> = {
  "\u0623": "\u0627", // أ -> ا
  "\u0625": "\u0627", // إ -> ا
  "\u0622": "\u0627", // آ -> ا
  "\u0671": "\u0627", // ٱ -> ا
  "\u0629": "\u0647", // ة -> ه
  "\u0649": "\u064A", // ى -> ي
};

export function normalizeArabic(text: string): string {
  text = text.replace(DIACRITICS_RE, "");
  text = text.replace(/./g, (ch) => NORM_MAP[ch] ?? ch);
  text = text.split(/\s+/).filter(Boolean).join(" ");
  return text;
}
```

**Step 2: Verify against Python with a quick test**

Pick a verse from quran.json and normalize it with both the Python and TypeScript versions. The results must be identical. You can verify by running the Python normalizer on a few strings and comparing manually.

**Step 3: Commit**

```bash
git add web/frontend/src/lib/normalizer.ts
git commit -m "feat: Arabic normalizer (TypeScript port of normalizer.py)"
```

---

## Task 4: Levenshtein Distance

Implement Levenshtein edit distance and ratio in TypeScript. No external dependency.

**Files:**
- Create: `web/frontend/src/lib/levenshtein.ts`

**Step 1: Write the implementation**

```typescript
/**
 * Levenshtein edit distance between two strings.
 * Uses a single-row DP approach for O(min(m,n)) space.
 */
export function distance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  // Ensure a is the shorter string for space efficiency
  if (a.length > b.length) [a, b] = [b, a];

  const m = a.length;
  const n = b.length;
  let prev = new Uint16Array(m + 1);
  let curr = new Uint16Array(m + 1);

  for (let i = 0; i <= m; i++) prev[i] = i;

  for (let j = 1; j <= n; j++) {
    curr[0] = j;
    for (let i = 1; i <= m; i++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[i] = Math.min(
        prev[i] + 1,      // deletion
        curr[i - 1] + 1,  // insertion
        prev[i - 1] + cost // substitution
      );
    }
    [prev, curr] = [curr, prev];
  }

  return prev[m];
}

/**
 * Normalized Levenshtein similarity ratio.
 * Returns 1.0 for identical strings, 0.0 for completely different.
 * Matches python-Levenshtein's `ratio()` behavior.
 */
export function ratio(a: string, b: string): number {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1.0;
  return 1 - distance(a, b) / maxLen;
}
```

**Step 2: Verify against Python**

Test with the same string pairs used by QuranDB matching. For example:
- `ratio("بسم الله الرحمن الرحيم", "بسم الله الرحمن الرحيم")` → 1.0
- `ratio("بسم الله", "بسم الله الرحمن")` → should match Python's `Levenshtein.ratio()`

**Step 3: Commit**

```bash
git add web/frontend/src/lib/levenshtein.ts
git commit -m "feat: Levenshtein distance and ratio (pure TypeScript)"
```

---

## Task 5: QuranDB (TypeScript port)

Port `shared/quran_db.py` to TypeScript. This is the largest single port — includes searching, multi-verse span matching, continuation bonuses, suffix-prefix scoring, and partial ratio.

**Files:**
- Create: `web/frontend/src/lib/quran-db.ts`
- Copy: `data/quran.json` → `web/frontend/public/quran.json`

**Reference:** `shared/quran_db.py` (234 lines)

**Step 1: Copy quran.json to public/**

```bash
cp data/quran.json web/frontend/public/quran.json
```

**Step 2: Write QuranDB**

Port every method from `quran_db.py` exactly. Key details:
- `partialRatio()` — sliding window of `ratio()` calls
- `QuranDB` class constructor — indexes by ref and surah, pre-computes bismillah-stripped text
- `_continuationBonuses()` — bonus values `[0.22, 0.12, 0.06]`
- `_suffixPrefixScore()` — trim leading words, match against verse prefix
- `matchVerse()` — two-pass: single verse scoring then multi-verse span scoring
- `search()`, `getVerse()`, `getSurah()`, `getNextVerse()`

Import `ratio` from `./levenshtein.ts` and `normalizeArabic` from `./normalizer.ts`.

The `QuranDB` constructor takes the parsed JSON array directly (not a file path — in the browser we fetch and parse it).

**Step 3: Verify matching behavior**

Pick 3-4 test cases from the benchmark corpus manifest and manually verify that `matchVerse()` returns the same surah:ayah as the Python version for those inputs.

**Step 4: Commit**

```bash
git add web/frontend/src/lib/quran-db.ts web/frontend/public/quran.json
git commit -m "feat: QuranDB TypeScript port with full matching algorithm"
```

---

## Task 6: Recitation Tracker (State Machine)

Port the discovery/tracking state machine from `server.py` to TypeScript.

**Files:**
- Create: `web/frontend/src/lib/tracker.ts`

**Reference:** `web/server.py` lines 358-703 (the WebSocket handler)

**Step 1: Write the tracker**

The `RecitationTracker` class encapsulates all per-session state. It has a synchronous `feed(samples: Float32Array)` method that the worker calls with each audio chunk. It returns an array of messages to emit.

The tracker needs a `transcribe` callback (injected via constructor) since it doesn't own the ONNX model directly:

```typescript
import type { QuranDB } from "./quran-db";
import type { WorkerOutbound } from "./types";

type TranscribeFn = (audio: Float32Array) => Promise<string>;

export class RecitationTracker {
  constructor(
    private db: QuranDB,
    private transcribe: TranscribeFn,
  ) {}

  async feed(samples: Float32Array): Promise<WorkerOutbound[]> {
    // ... state machine logic from server.py
  }
}
```

Port every piece of logic:
- Audio buffer append + trim (use Float32Array concatenation)
- Tracking mode: silence detection, stale cycles, word alignment, verse completion, auto-advance
- Discovery mode: trigger interval, silence skip, residual check, match_verse call, threshold selection
- `_enterTracking()` / `_exitTracking()` state transitions
- `_isSilence()` — RMS computation
- `_getSurroundingVerses()` — context window
- `_wordsMatch()` — fuzzy word comparison
- `_alignPosition()` — greedy forward alignment

All constants from `types.ts`.

**Step 2: Commit**

```bash
git add web/frontend/src/lib/tracker.ts
git commit -m "feat: recitation tracker (discovery/tracking state machine)"
```

---

## Task 7: Model Cache (IndexedDB)

Load the ONNX model with download progress, cache in IndexedDB.

**Files:**
- Create: `web/frontend/src/worker/model-cache.ts`

**Step 1: Write the model cache**

```typescript
const DB_NAME = "tarteel-models";
const STORE_NAME = "models";
const MODEL_KEY = "fastconformer-ar-ctc-v1";

export async function loadModel(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  // Try IndexedDB cache first
  const cached = await getFromCache(MODEL_KEY);
  if (cached) return cached;

  // Download with progress
  const response = await fetch(url);
  const total = parseInt(response.headers.get("content-length") || "0");
  const reader = response.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  await saveToCache(MODEL_KEY, buffer.buffer);
  return buffer.buffer;
}
```

Include the IndexedDB open/get/put helpers (openDB, getFromCache, saveToCache).

**Step 2: Commit**

```bash
git add web/frontend/src/worker/model-cache.ts
git commit -m "feat: ONNX model cache with IndexedDB and download progress"
```

---

## Task 8: Mel Spectrogram Extraction

Implement NeMo-compatible mel spectrogram extraction in TypeScript using @huggingface/transformers audio utilities.

**Files:**
- Create: `web/frontend/src/worker/mel.ts`

**Step 1: Install dependency**

```bash
cd web/frontend && npm install @huggingface/transformers
```

**Step 2: Write mel spectrogram extraction**

This must exactly replicate NeMo's `AudioToMelSpectrogramPreprocessor` with these parameters:
- sample_rate=16000, n_fft=512, hop_length=160, win_length=400
- window="hann", n_mels=80, fmin=0, fmax=8000
- mel_norm="slaney", power=2.0
- pre-emphasis=0.97, dither=1e-5
- log with guard (add 1e-5)
- per-feature normalization (mean/std per mel bin across time)

```typescript
import { mel_filter_bank, spectrogram, window_function } from "@huggingface/transformers";

const SAMPLE_RATE = 16000;
const N_FFT = 512;
const HOP_LENGTH = 160;
const WIN_LENGTH = 400;
const N_MELS = 80;
const PREEMPH = 0.97;
const DITHER = 1e-5;
const LOG_GUARD = 1e-5;

// Pre-compute mel filterbank and window (immutable, reuse across calls)
let _melFilters: Float64Array[] | null = null;
let _window: Float64Array | null = null;

function getMelFilters() { ... }
function getWindow() { ... }

export function computeMelSpectrogram(audio: Float32Array): Float32Array {
  // 1. Dither
  // 2. Pre-emphasis
  // 3. STFT via spectrogram() with mel_filters
  // 4. Log (add guard)
  // 5. Per-feature normalization
  // Returns flat Float32Array in [n_mels, time] layout
}
```

**Important:** The transformers.js `spectrogram()` function may handle some of these steps (power spectrum, mel application). Read its API carefully to avoid double-applying. The pre-emphasis step and per-feature normalization must be done manually.

**Step 3: Validate against Python**

Create a small validation script that:
1. Takes a test .wav file
2. Computes mel features with NeMo's preprocessor in Python, saves as numpy array
3. Computes mel features with this TypeScript implementation
4. Compares the two — max absolute difference should be <1e-3

This is the most critical validation step. If mel features don't match, the model will produce garbage.

**Step 4: Commit**

```bash
git add web/frontend/src/worker/mel.ts
git commit -m "feat: NeMo-compatible mel spectrogram extraction for browser"
```

---

## Task 9: CTC Decoder

Greedy CTC decode: argmax → collapse duplicates → remove blanks → BPE detokenize.

**Files:**
- Create: `web/frontend/src/worker/ctc-decode.ts`

**Step 1: Write the decoder**

```typescript
export class CTCDecoder {
  private vocab: Map<number, string>;
  private blankId: number;

  constructor(vocabJson: Record<string, string>) {
    this.vocab = new Map();
    let maxId = 0;
    for (const [id, token] of Object.entries(vocabJson)) {
      const numId = parseInt(id);
      this.vocab.set(numId, token);
      if (numId > maxId) maxId = numId;
    }
    this.blankId = maxId; // blank is the last token
  }

  decode(logprobs: Float32Array, timeSteps: number, vocabSize: number): string {
    // argmax per timestep
    const ids: number[] = [];
    for (let t = 0; t < timeSteps; t++) {
      let maxIdx = 0;
      let maxVal = logprobs[t * vocabSize];
      for (let v = 1; v < vocabSize; v++) {
        const val = logprobs[t * vocabSize + v];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = v;
        }
      }
      ids.push(maxIdx);
    }

    // Collapse consecutive duplicates, remove blanks
    const tokens: string[] = [];
    let prev = -1;
    for (const id of ids) {
      if (id !== prev && id !== this.blankId) {
        const token = this.vocab.get(id) ?? "";
        tokens.push(token);
      }
      prev = id;
    }

    // BPE detokenize: join tokens, replace ▁ (sentencepiece) with space
    return tokens.join("").replace(/▁/g, " ").trim();
  }
}
```

**Step 2: Commit**

```bash
git add web/frontend/src/worker/ctc-decode.ts
git commit -m "feat: CTC greedy decoder with BPE detokenization"
```

---

## Task 10: ONNX Session Manager

Create/manage the onnxruntime-web InferenceSession.

**Files:**
- Create: `web/frontend/src/worker/session.ts`

**Step 1: Install dependency**

```bash
cd web/frontend && npm install onnxruntime-web
```

**Step 2: Write session manager**

```typescript
import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

export async function createSession(modelBuffer: ArrayBuffer): Promise<void> {
  // Try WebGPU first, fall back to WASM
  const providers: string[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (adapter) providers.push("webgpu");
    } catch {}
  }
  providers.push("wasm");

  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;

  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: providers,
  });
}

export async function runInference(
  melFeatures: Float32Array,
  numMels: number,
  timeFrames: number,
): Promise<{ logprobs: Float32Array; timeSteps: number; vocabSize: number }> {
  if (!session) throw new Error("Session not initialized");

  const inputTensor = new ort.Tensor("float32", melFeatures, [1, numMels, timeFrames]);
  const lengthTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(timeFrames)]), [1]);

  const inputNames = session.inputNames;
  const feeds: Record<string, ort.Tensor> = {
    [inputNames[0]]: inputTensor,
    [inputNames[1]]: lengthTensor,
  };

  const results = await session.run(feeds);
  const outputTensor = results[session.outputNames[0]];
  const [_batch, timeSteps, vocabSize] = outputTensor.dims as number[];

  return {
    logprobs: outputTensor.data as Float32Array,
    timeSteps,
    vocabSize,
  };
}
```

**Note:** The actual input names and length tensor dtype may differ. Verify against the exported ONNX model's metadata from Task 1 (the validation step prints input names and shapes). Adjust accordingly.

**Step 3: Commit**

```bash
git add web/frontend/src/worker/session.ts
git commit -m "feat: ONNX Runtime Web session manager with WebGPU/WASM fallback"
```

---

## Task 11: Inference Worker

The Web Worker entry point that ties everything together: loads the model, receives audio chunks, runs the full pipeline, sends messages back.

**Files:**
- Create: `web/frontend/src/worker/inference.ts`

**Step 1: Write the worker**

```typescript
import { loadModel } from "./model-cache";
import { computeMelSpectrogram } from "./mel";
import { CTCDecoder } from "./ctc-decode";
import { createSession, runInference } from "./session";
import { QuranDB } from "../lib/quran-db";
import { RecitationTracker } from "../lib/tracker";
import { normalizeArabic } from "../lib/normalizer";
import type { WorkerInbound, WorkerOutbound } from "../lib/types";

let tracker: RecitationTracker | null = null;
let decoder: CTCDecoder | null = null;

function post(msg: WorkerOutbound) {
  self.postMessage(msg);
}

async function transcribe(audio: Float32Array): Promise<string> {
  const mel = computeMelSpectrogram(audio);
  const numMels = 80;
  const timeFrames = mel.length / numMels;
  const { logprobs, timeSteps, vocabSize } = await runInference(mel, numMels, timeFrames);
  const text = decoder!.decode(logprobs, timeSteps, vocabSize);
  return normalizeArabic(text);
}

async function init() {
  // Load vocab
  const vocabRes = await fetch("/vocab.json");
  const vocabJson = await vocabRes.json();
  decoder = new CTCDecoder(vocabJson);

  // Load ONNX model
  const modelBuffer = await loadModel("/fastconformer_ar_ctc.onnx", (loaded, total) => {
    post({ type: "loading", percent: total ? Math.round((loaded / total) * 100) : 0 });
  });
  await createSession(modelBuffer);

  // Load QuranDB
  const quranRes = await fetch("/quran.json");
  const quranData = await quranRes.json();
  const db = new QuranDB(quranData);

  // Create tracker
  tracker = new RecitationTracker(db, transcribe);
  post({ type: "ready" });
}

self.onmessage = async (e: MessageEvent<WorkerInbound>) => {
  const msg = e.data;
  if (msg.type === "init") {
    await init();
  } else if (msg.type === "audio") {
    if (!tracker) return;
    const messages = await tracker.feed(msg.samples);
    for (const m of messages) {
      post(m);
    }
  }
};
```

**Step 2: Copy vocab.json to public/**

```bash
cp data/vocab.json web/frontend/public/vocab.json
```

**Step 3: Commit**

```bash
git add web/frontend/src/worker/inference.ts web/frontend/public/vocab.json
git commit -m "feat: inference worker tying together full browser pipeline"
```

---

## Task 12: Adapt Frontend (main.ts)

Replace WebSocket communication with Worker messaging. Keep all UI rendering code intact.

**Files:**
- Modify: `web/frontend/src/main.ts`

**Step 1: Replace WebSocket with Worker**

Changes needed:
1. Remove `connectWebSocket()` function
2. Remove `ws` from state, add `worker: Worker | null`
3. Create worker: `new Worker(new URL("./worker/inference.ts", import.meta.url), { type: "module" })`
4. Replace `ws.send(chunk)` with `worker.postMessage({ type: "audio", samples }, [samples.buffer])` where `samples` is the Float32Array from the AudioWorklet
5. Replace `ws.onmessage` handler with `worker.onmessage` — the message parsing changes from `JSON.parse(e.data)` to just `e.data` (worker messages are already objects)
6. Replace `pollModelStatus()` with handling `loading` and `ready` messages from the worker
7. Replace `fetchSurah()` — load surah data from the QuranDB in the worker. Change the surah data flow: the worker already sends `surrounding_verses` in `verse_match` messages. For full surah data (needed for rendering all verses), the worker should include it in the `verse_match` message. Add a `surah_data` field to the `verse_match` message with the full verses array.
8. Send `{ type: "init" }` to the worker on page load instead of connecting WebSocket
9. Remove the proxy config from `vite.config.ts` (no more server)

**Step 2: Update vite.config.ts**

Remove the proxy configuration (no Python server). Add COOP/COEP headers for dev server:

```typescript
import { defineConfig } from "vite";

export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
```

**Step 3: Test locally**

```bash
cd web/frontend && npm run dev
```

Open browser, grant mic permission, verify:
- Model download progress shows
- After model loads, audio is captured and sent to worker
- Verse matches appear in the UI

**Step 4: Commit**

```bash
git add web/frontend/src/main.ts web/frontend/vite.config.ts
git commit -m "feat: replace WebSocket with Worker for browser-only inference"
```

---

## Task 13: PWA Setup

Add service worker and manifest for offline support.

**Files:**
- Modify: `web/frontend/vite.config.ts`
- Modify: `web/frontend/index.html`
- Create: `web/frontend/public/manifest.json`

**Step 1: Install vite-plugin-pwa**

```bash
cd web/frontend && npm install -D vite-plugin-pwa
```

**Step 2: Configure PWA plugin in vite.config.ts**

```typescript
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: "autoUpdate",
      workbox: {
        globPatterns: ["**/*.{js,css,html,woff,woff2,json}"],
        // Don't precache the ONNX model — it's handled by IndexedDB
        globIgnores: ["**/*.onnx"],
        maximumFileSizeToCacheInBytes: 5 * 1024 * 1024, // 5 MB for quran.json
      },
      manifest: {
        name: "Offline Tarteel",
        short_name: "Tarteel",
        description: "Offline Quran verse recognition in your browser",
        theme_color: "#2c2416",
        background_color: "#faf8f3",
        display: "standalone",
        icons: [
          { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
        ],
      },
    }),
  ],
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
```

**Step 3: Add manifest link to index.html**

Add to `<head>`:
```html
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#2c2416">
```

**Step 4: Create placeholder icons**

Create simple placeholder icons (can be replaced with real ones later).

**Step 5: Commit**

```bash
git add web/frontend/vite.config.ts web/frontend/index.html web/frontend/public/manifest.json
git commit -m "feat: PWA setup with service worker for offline app shell"
```

---

## Task 14: Vite Build Configuration

Configure Vite to handle the WASM files, worker bundling, and COOP/COEP headers for production.

**Files:**
- Modify: `web/frontend/vite.config.ts`
- Modify: `web/frontend/package.json`

**Step 1: Install vite-plugin-wasm (if needed)**

onnxruntime-web ships its own WASM files. Vite needs to know how to handle them. Check if onnxruntime-web's WASM files are loaded via URL (they typically are — `ort.env.wasm.wasmPaths` can point to the right location).

If the WASM files need to be copied to the output:
```bash
cd web/frontend && npm install -D vite-plugin-static-copy
```

Configure to copy `node_modules/onnxruntime-web/dist/*.wasm` to the build output.

Alternatively, onnxruntime-web can load WASM from a CDN — set `ort.env.wasm.wasmPaths` in `session.ts` to point to the CDN URL matching the installed version.

**Step 2: Test production build**

```bash
cd web/frontend && npm run build && npm run preview
```

Verify the app loads, model downloads, and inference works in the preview server.

**Step 3: Commit**

```bash
git add web/frontend/vite.config.ts web/frontend/package.json
git commit -m "feat: Vite build config for WASM and worker bundling"
```

---

## Task 15: End-to-End Testing & Latency Measurement

Run the full browser pipeline end-to-end and measure latency.

**Files:**
- No new files

**Step 1: Test with real recitation**

1. Open the app in Chrome (desktop)
2. Grant mic permission
3. Recite Al-Fatiha
4. Verify:
   - Verse 1 is recognized and highlighted
   - Word-level tracking works
   - Auto-advance to verse 2 works
   - Reciting through the whole surah completes successfully

**Step 2: Measure inference latency**

Add `console.time`/`console.timeEnd` calls around:
- `computeMelSpectrogram()` — target <50ms
- `runInference()` — target <500ms (WASM), <200ms (WebGPU)
- `decoder.decode()` — target <5ms
- `db.matchVerse()` — target <50ms

Log total per-cycle time. If exceeding 1s on desktop, investigate:
- Is SIMD enabled? (check `ort.env.wasm.simd`)
- Is multi-threading working? (check `crossOriginIsolated` in console)
- Try WebGPU if available

**Step 3: Test offline**

1. Load the app (model downloads and caches)
2. Disconnect from internet (airplane mode / disable wifi)
3. Reload the page
4. Verify everything still works — app loads from service worker, model from IndexedDB

**Step 4: Commit any fixes**

```bash
git add -p  # stage specific fixes
git commit -m "fix: end-to-end testing fixes for browser inference"
```

---

## Task 16: Model Hosting

The ONNX model (~115 MB) needs to be hosted somewhere the PWA can download it from. Since it's too large for git, it needs a separate hosting solution.

**Options (pick one):**
- **GitHub Releases**: Attach as a release asset. Free, simple, CDN-backed.
- **Hugging Face Hub**: Upload to a model repo. Free, built for ML models.
- **Static hosting**: Upload to any CDN/S3 bucket.

**Step 1: Upload the model**

Upload `data/fastconformer_ar_ctc.onnx` to the chosen host.

**Step 2: Update the model URL**

In `web/frontend/src/worker/inference.ts`, update the `loadModel()` URL to point to the hosted model. Make sure the hosting server sets appropriate CORS headers (`Access-Control-Allow-Origin: *`).

**Step 3: Commit**

```bash
git add web/frontend/src/worker/inference.ts
git commit -m "feat: configure model download URL for hosted ONNX model"
```

---

## Execution Order & Dependencies

```
Task 1 (ONNX Export) ──────────────────────────────────────┐
                                                            │
Task 2 (Types) ─────┐                                      │
Task 3 (Normalizer) ─┤                                     │
Task 4 (Levenshtein) ─┼── Task 5 (QuranDB) ─┐              │
                      │                      │              │
                      └──────────────────────┼── Task 6 (Tracker) ──┐
                                             │                      │
Task 7 (Model Cache) ───────────────────────────┐                   │
Task 8 (Mel Spectrogram) ───────────────────────┤                   │
Task 9 (CTC Decoder) ──────────────────────────┤                   │
Task 10 (Session Manager) ─────────────────────┼── Task 11 (Worker) │
                                                │                   │
                                                └───────────────────┼── Task 12 (Frontend)
                                                                    │
                                                                    ├── Task 13 (PWA)
                                                                    ├── Task 14 (Build Config)
                                                                    └── Task 15 (E2E Testing)

Task 16 (Model Hosting) — can be done anytime after Task 1
```

Tasks 2-4 and 7-10 can be done in parallel. Task 5 depends on 2-4. Task 6 depends on 5. Task 11 depends on 6 and 7-10. Task 12 depends on 11. Tasks 13-15 depend on 12.

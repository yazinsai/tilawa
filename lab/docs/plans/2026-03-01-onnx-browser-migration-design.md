# Design: Browser-Only Offline Tarteel via ONNX Runtime Web

**Date**: 2026-03-01
**Status**: Approved
**Goal**: Replace the Python FastAPI backend with fully in-browser inference using ONNX Runtime Web (WASM/WebGPU). The result is a public PWA that works offline after first load — no server, no Python, verifiably offline.

## Constraints

- **Must use FastConformer** (`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`) — 85% SeqAcc is non-negotiable
- **Full replacement** of Python backend — not a fallback, the only way to run it
- **Under 1 second** inference latency target
- **Desktop primary**, mobile best-effort (WASM fallback for browsers without WebGPU)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Main Thread                                         │
│  ┌──────────────────────────────────────────────┐   │
│  │  UI (existing Vite + TS app, adapted)         │   │
│  │  - Verse rendering, word highlighting         │   │
│  │  - Audio capture (AudioWorklet → 16kHz)       │   │
│  │  - Model download progress bar                │   │
│  └──────────────┬───────────────────────────────┘   │
│                  │ postMessage (float32 chunks)       │
│                  ▼                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Inference Worker (Web Worker)                 │   │
│  │  ┌────────────────────────────────────────┐   │   │
│  │  │  1. Audio buffer management            │   │   │
│  │  │  2. Silence detection                  │   │   │
│  │  │  3. Mel spectrogram extraction (JS)    │   │   │
│  │  │  4. ONNX Runtime inference (WASM)      │   │   │
│  │  │  5. CTC greedy decode                  │   │   │
│  │  │  6. QuranDB matching                   │   │   │
│  │  │  7. Discovery/Tracking state machine   │   │   │
│  │  └────────────────────────────────────────┘   │   │
│  └──────────────┬───────────────────────────────┘   │
│                  │ postMessage (verse_match,          │
│                  │   word_progress, raw_transcript)   │
│                  ▼                                    │
│  ┌──────────────────────────────────────────────┐   │
│  │  Service Worker (PWA)                         │   │
│  │  - Caches app shell (HTML/CSS/JS/WASM)        │   │
│  │  - Offline-first after first load             │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  IndexedDB: ONNX model (115 MB), quran.json (3 MB)  │
└─────────────────────────────────────────────────────┘
```

## ONNX Export

One-time Python script. The ONNX model does NOT include the preprocessor — only encoder + CTC decoder.

```python
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)
model.set_export_config({"decoder_type": "ctc"})
model.export("fastconformer_ar_ctc.onnx")
```

**Inputs**: `audio_signal` (mel features, `[1, 80, T]`), `length` (`[T]`)
**Outputs**: `logprobs` (`[1, T/8, vocab_size]`)

### Validation strategy

Before building the browser version, validate in Python:
1. Export to ONNX
2. Run with `onnxruntime` using NeMo's own preprocessor — verify identical outputs to NeMo native
3. Run with manual numpy preprocessing — verify outputs match
4. Only then port preprocessing to JavaScript

## Mel Spectrogram (JavaScript)

NeMo's preprocessor parameters for this model:

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Window | 25ms (400 samples), Hann |
| Hop | 10ms (160 samples) |
| FFT size | 512 |
| Mel bins | 80 |
| Mel range | 0–8000 Hz |
| Mel norm | slaney |
| Pre-emphasis | 0.97 |
| Dither | 1e-5 |
| Log guard | add 1e-5 before log |
| Normalize | per-feature (mean/std per mel bin across time) |

Pipeline:
1. Dither (add tiny random noise)
2. Pre-emphasis (`x[n] = x[n] - 0.97 * x[n-1]`)
3. STFT (512-point FFT, 400-sample window, 160-sample hop)
4. Power spectrum (`|FFT|^2`)
5. Mel filterbank (80 bins, slaney norm)
6. Log (`ln(mel + 1e-5)`)
7. Per-feature normalization (subtract mean, divide by std per bin)

Implementation uses `spectrogram()` and `mel_filter_bank()` from `@huggingface/transformers` audio utils.

## CTC Decode

Argmax per timestep, collapse consecutive duplicates, remove blank token. ~20 lines of TypeScript. Need to extract the BPE vocabulary from the NeMo model and bundle it as a JSON file.

## File Structure

```
web/frontend/
├── src/
│   ├── main.ts              # Existing UI (adapted: WebSocket → worker)
│   ├── style.css             # Unchanged
│   ├── worker/
│   │   ├── inference.ts      # Web Worker entry — orchestrates everything
│   │   ├── mel.ts            # Mel spectrogram extraction
│   │   ├── ctc-decode.ts     # CTC greedy decoder + BPE token → text
│   │   ├── model-cache.ts    # IndexedDB load/store for ONNX model
│   │   └── session.ts        # onnxruntime-web session management
│   ├── lib/
│   │   ├── quran-db.ts       # Full port of quran_db.py
│   │   ├── normalizer.ts     # Port of normalizer.py
│   │   ├── levenshtein.ts    # Levenshtein ratio (~15 lines)
│   │   ├── tracker.ts        # Discovery/tracking state machine
│   │   └── types.ts          # Shared message types, interfaces
│   └── sw.ts                 # Service Worker (PWA offline)
├── public/
│   ├── audio-processor.js    # Unchanged
│   └── quran.json            # Moved from data/
└── index.html                # Add PWA manifest link
```

## Worker ↔ Main Thread Protocol

```typescript
// Main → Worker
type WorkerInbound =
  | { type: 'audio'; samples: Float32Array }
  | { type: 'init' }

// Worker → Main
type WorkerOutbound =
  | { type: 'loading'; percent: number }
  | { type: 'ready' }
  | { type: 'verse_match'; surah; ayah; verse_text; surah_name; confidence; surrounding_verses }
  | { type: 'word_progress'; surah; ayah; word_index; total_words; matched_indices }
  | { type: 'raw_transcript'; text; confidence }
```

Same 3 recognition message types the UI already handles. Frontend changes are minimal: `ws.send(chunk)` → `worker.postMessage(...)`, `ws.onmessage` → `worker.onmessage`.

## State Machine (tracker.ts)

Direct port of server.py. Exposes `feed(samples: Float32Array): Message[]` called by the worker on each audio message. All constants preserved exactly:

| Constant | Value |
|----------|-------|
| TRIGGER_SECONDS | 2.0 |
| MAX_WINDOW_SECONDS | 10.0 |
| SILENCE_RMS_THRESHOLD | 0.005 |
| VERSE_MATCH_THRESHOLD | 0.45 |
| FIRST_MATCH_THRESHOLD | 0.75 |
| RAW_TRANSCRIPT_THRESHOLD | 0.25 |
| TRACKING_TRIGGER_SECONDS | 0.5 |
| TRACKING_SILENCE_TIMEOUT | 4.0 |
| TRACKING_MAX_WINDOW_SECONDS | 5.0 |
| STALE_CYCLE_LIMIT | 4 |
| LOOKAHEAD | 5 |

## QuranDB Port

Faithful TypeScript port of `shared/quran_db.py`:
- Levenshtein `ratio()` — implemented from scratch
- `partial_ratio()` — sliding window of ratio calls
- `QuranDB` class with all methods including multi-verse span matching
- Continuation bonuses: `[0.22, 0.12, 0.06]`
- Word alignment: greedy forward with lookahead of 5

## PWA & Offline

- `vite-plugin-pwa` for service worker generation
- App shell: cache-first, precached at install
- ONNX model: IndexedDB with custom download + progress bar (not SW — too large)
- `quran.json`: precached with app shell

### COOP/COEP Headers

Required for WASM multi-threading (`SharedArrayBuffer`):
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them, falls back to single-threaded WASM — slower but functional.

## Performance Budget

| Operation | Target | Fallback (single-thread) |
|-----------|--------|--------------------------|
| Model load (cached) | <1s | — |
| Mel spectrogram (2s audio) | <50ms | — |
| ONNX inference (WASM+SIMD+threads) | <500ms | <1.5s |
| ONNX inference (WebGPU) | <200ms | — |
| CTC decode | <5ms | — |
| QuranDB match | <50ms | — |
| **Total per cycle** | **<600ms** | **<1.6s** |

If tracking mode inference exceeds 500ms, increase tracking interval to 1s.

## Dependencies

```
onnxruntime-web            # ONNX inference
@huggingface/transformers  # Audio utils only (mel spectrogram)
vite-plugin-pwa            # Service worker generation
```

Levenshtein implemented from scratch. Everything else vanilla TypeScript.

## What Changes vs. Current App

| Aspect | Before (Python) | After (Browser) |
|--------|-----------------|-----------------|
| Server | FastAPI + NeMo | None |
| Inference | PyTorch CUDA/CPU | onnxruntime-web WASM/WebGPU |
| Audio transport | WebSocket binary | Worker postMessage (transferable) |
| QuranDB | Python, server-side | TypeScript, in worker |
| Surah data | REST API `/api/surah/n` | Static quran.json in worker |
| Offline | Only with server running | True offline after first load |
| UI code | ~90% unchanged | ~90% unchanged |

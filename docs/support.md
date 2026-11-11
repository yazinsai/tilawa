# Support matrix

What `@tilawa/core` 1.0.0-rc (Preview) runs on, and what it needs.

Measured results below are from desktop CPU only. Phone and browser performance has not been measured yet.

## Platforms

| Platform | Import | Status | Runtime |
|---|---|---|---|
| Browsers | `@tilawa/core/web` | Supported | `onnxruntime-web` ≥ 1.19 (peer). See the [version note](#onnx-runtime-versions). |
| Node.js ≥ 20.12 | `@tilawa/core/node` | Supported | `onnxruntime-node` ≥ 1.20 (peer), CPU |
| React Native | `@tilawa/core/react-native` | **Experimental, not yet validated on devices** | `onnxruntime-react-native` ≥ 1.20 (peer) |

The package is ESM only. Its JavaScript uses JSON import attributes (`import … with { type: "json" }`), so your bundler must support them. The tilawa.dev demo builds with Vite 7.

## Browsers

### Requirements

| Needs | Why | Without it |
|---|---|---|
| A secure context (https or localhost) | The microphone | `listen()` rejects with `MicrophoneError("insecure-context")` |
| `getUserMedia` and `AudioWorklet` | Microphone capture in `listen()` | `MicrophoneError("unavailable")`. You can still push audio yourself. |
| WebAssembly with SIMD | The WASM backend (onnxruntime-web's `ort-wasm-simd-threaded` build) | No backend can load: `TilawaLoadError` (`stage: "model"`) |
| `BigInt64Array` | The model's int64 state | No backend can load |
| Cache API (`caches`) | Keeping the model between visits | Falls back to memory: the model downloads on every page load |
| WebCrypto (`crypto.subtle`) | sha256 check of downloads | Downloads are checked by length only (for example on plain-http LAN pages) |
| WebGPU (`navigator.gpu`) | Optional, faster backend | Uses WASM |

### Tested browsers

| Browser | OS | Result |
|---|---|---|
| Chrome | Android | Not yet tested on a device |
| Safari | iOS | Not yet tested on a device |
| Chrome | macOS / Windows | Works (macOS, headless Chrome 2026-11-11: demo and /help follow a recitation, WASM backend, cross-origin isolated) |
| Safari | macOS | Not yet tested |
| Firefox | desktop | Not yet tested |
| Edge | Windows | Not yet tested |

iOS Safari only starts audio created during a user gesture. Use `prepareMicrophone()` in the tap handler ([quickstart](quickstart.md#ios-safari-open-the-microphone-inside-the-tap)).

### WebGPU and WASM

`createTilawa` tries WebGPU first, then WASM. A backend that fails to load, or fails a warm-up run on silence, falls through to the next. `tilawa.model.backend` says which one runs (for example `onnxruntime-web/wasm`), and `tilawa.model.fallbacks` says why earlier ones were skipped. To force one, pass `providers: ["wasm"]`.

### Threads and cross-origin isolation

The default is one WASM thread. That works on any page, with no special headers.

More threads (`threads: 2` or more) need `SharedArrayBuffer`, so the page must be cross-origin isolated:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without isolation, the SDK uses one thread whatever you pass. With isolation, every cross-origin file the page loads must allow it. `https://tilawa.dev/v1/` sends `Cross-Origin-Resource-Policy: cross-origin` and `Access-Control-Allow-Origin: *`, so the default asset host works on an isolated page. A mirror must do the same ([self-hosting](self-hosting.md#headers)).

Whether more threads help on a given device has not been measured yet.

### ONNX Runtime versions

By default, onnxruntime-web loads its `.wasm`/`.mjs` files from `<assetBase>runtime/ort/<version>/`, where `<version>` is the onnxruntime-web version your app bundles. The runtime folders `https://tilawa.dev/v1/` serves are listed in [`/v1/manifest.json`](https://tilawa.dev/v1/manifest.json) as `runtime/ort/<version>/…` (1.30.0 at the time of writing; the SDK's own tests use 1.30.0).

If you install a different onnxruntime-web version, either serve its files yourself and pass `wasmPaths`, or self-host the whole `/v1/` tree with your version ([self-hosting](self-hosting.md)). Otherwise loading fails with `TilawaLoadError` (`stage: "model"`).

## Node.js

- Node 20.12 or newer (`engines` in `package.json`).
- `onnxruntime-node` 1.20 or newer, on any platform it ships binaries for. Inference runs on the CPU (`threads` sets intra-op threads, default 1).
- The reference ships in the package; only the model is downloaded, into `~/.cache/tilawa` (or `cacheDir`, or `$TILAWA_CACHE_DIR`).
- No microphone helper: push PCM with `session.pushAudio(samples, sampleRate)`. `readWav` reads PCM WAV files.
- `tier: "auto"` always picks `full` on Node.

## React Native

**Experimental, not yet validated on devices.** It is shipped so you can try it. The API may change in a 1.x minor release, and nothing about performance or correctness on phones is claimed yet.

- Needs `onnxruntime-react-native` ≥ 1.20 and `BigInt64Array`.
- No microphone helper. Record with your own module, then call `session.pushAudio(samples, sampleRate)` at the rate you record.
- Pass a persistent `storage` (and `pathOf`, so ONNX Runtime can load from a file path). The default is in memory, so the model downloads on every launch.
- Without WebCrypto, downloads are checked by length only.
- `tier: "auto"` always picks `lite`.

## Models, sizes and memory

| | Size | Notes |
|---|---|---|
| `full` model, `spliced-clean@1-int8` | 67,759,744 bytes (67.8 MB) | Default |
| `lite` model, `spliced-clean-lite@1-int8` | 24,650,054 bytes (24.7 MB) | Picked by `tier: "auto"` when `navigator.deviceMemory` is under 4, on mobile browsers without WebGPU, and on React Native |
| Reference, `ref-v2.json` | 4,749,281 bytes (4.7 MB) | Downloaded on web and React Native; bundled on Node |
| Storage budget | 200 MiB | All model tiers cached on one device together (`MODEL_BUDGET_BYTES`). Over it, loading fails with `ModelBudgetError` before downloading. |
| Report recorder, clip mode | about 1.4 MB | Only with `createRecorder`: (10 s before + 5 s after + 30 s slack) of 16 kHz PCM16 |
| Report recorder, take mode | up to 9.6 MB | Only with `createRecorder({ retain: { take: true } })`: 5 minutes of 16 kHz PCM16 |

Both tiers check the same four error classes. See the [model card](model-card.md).

Runtime memory and speed, per tier and backend:

| Device | Tier | Backend | Peak memory | Real-time factor | Mistake latency |
|---|---|---|---|---|---|
| Desktop (Apple M4), Node | full | onnxruntime-node CPU | not measured | 0.055 per session (listener 0.026) | first warning p95 1,079 ms; word highlight p95 468 ms |
| Desktop (Apple M4), Node | lite | onnxruntime-node CPU | not measured | 0.037 per session (listener 0.018) | not measured |
| Phones, browsers | full / lite | WebGPU / WASM | not measured | not measured | not measured |

## Device floor

Not yet measured. The target floor is a 2020 mid-range Android phone (Snapdragon 7xx, 4 GB) and iPhone XR/11; until it is checked, treat phone performance as unknown.

## What 1.0 checks

| | Supported |
|---|---|
| Following the reciter, word by word, anywhere in the Quran | Yes |
| Wrong, omitted and added words | Yes |
| Skipped passages | Yes |
| Pronunciation and tajweed (vowels, letters, shadda, madd) | **No.** These classes are listed in `capabilities.unsupported` and never emitted. |
| Qira'at other than Hafs | No |
| Children's voices | Not evaluated |

## Getting help

File issues at [github.com/yazinsai/tilawa/issues](https://github.com/yazinsai/tilawa/issues). Include `tilawa.model` (id, version, tier, backend, fallbacks), `SDK_VERSION`, the browser or Node version, and whether the page is cross-origin isolated.

# Experimental Swift engine

Native Swift feature extraction, streaming CTC decoding, Quran position tracking, and memorization review prompts for Tilawa model/reference assets. Supports iOS 16+ and macOS 13+. There are no SwiftUI, UIKit, WebKit, React Native, network, microphone, analytics, or model-download dependencies in the package.

**This is a new experimental tracker/checker, not a complete port of the TypeScript engine.** It does not inherit the SDK's published precision/recall, confirmation policies, event schema, or pronunciation capabilities. Do not describe it as production-ready or use the SDK's accuracy claims for it. The existing TypeScript engine and SDK are unchanged.

## Package and model runtime

The repository-root `Package.swift` exposes the `Tilawa` product. Add this repository as a Swift package dependency and select that product. The package is inference-runtime neutral: the app supplies an `InferenceBackend` that executes one model window and retains its recurrent tensors. Each engine must exclusively own its backend instance.

`Adapters/ONNXRuntime/` contains a small optional iOS integration using Microsoft's native ONNX Runtime. It is source-only and deliberately outside the dependency-free Swift target; see its README. Model weights, the full reference asset, and runtime binaries are not duplicated or committed here. The reference remains available in `packages/tilawa-core/assets/ref-v2.json`; use the matching model manifest and verified 1.x weights.

```swift
import Tilawa

let engine = try TilawaEngine(
    reference: referenceJSON,
    manifest: modelManifestJSON,
    backend: yourNativeBackend
)

try await engine.startMemorization(
    MemorizationPassage(surah: 18, fromAyah: 60, toAyah: 70)
)
// Or: try await engine.startFreeRecitation()

let update = try await engine.pushAudio(monoSamples, sampleRate: 16000)
if update.tracking == .locked, let word = update.position {
    // word.surah, word.ayah, word.word are all 1-based; render your own Quran text.
}
let final = try await engine.end()
// final.reviewPrompts contains tentative memorization review results, not tajweed grading.
```

Initialization indexes the reference and can be expensive; prepare the engine away from the UI thread. Supply calls in capture order. Actor isolation protects engine state; it does not reorder independently created audio tasks. Keep sample rate constant during a take. Input is finite, mono Float32 PCM in `[-1, 1]`; rates from 8–192 kHz are accepted and resampled to 16 kHz. The host owns capture permissions, interruptions, cancellation, and audio backpressure.

An engine processes one take at a time. Starting during an active take throws; `end()` drains and is idempotent. Audio after the take ends throws. A failed inference marks the take failed; starting a new take resets the backend and all streaming state.

## Implementation boundaries

- **Reference ports:** Kaldi-compatible log-mel features, windowed-sinc resampling, weighted phoneme substring alignment, and the streaming model/CTC contract.
- **New native implementation:** n-gram acquisition, bounded tracking, conservative lexical/omission/insertion review prompts, and actor lifecycle.
- **Not implemented:** full TypeScript checker parity, posterior-based mistake confirmation, all self-correction/parallel-passage policies, tajweed, model downloads, capture helpers, and reporting APIs.

The current checker requires stable, confident alignment and repeated evidence before a prompt. Shared vowels must not conceal consonant substitutions; confusable sound pairs alone do not satisfy the lexical replacement threshold. A confirmed prompt is not resolved merely because the rolling evidence window cuts through its word.

## Tests

```sh
swift test
```

Tests are offline and use synthetic signals, small public reference excerpts, a blank inference backend, and fixtures generated from `@tilawa/core@1.0.0-rc.1`. They check feature/resampler/alignment parity, edited phoneme sequences, validation, session lifecycle, failure recovery, and the rolling-window replacement regression. They do not require model or runtime downloads.

`VALIDATION.md` describes the additional native ONNX/iOS simulator recording checks used during development. Those checks are evidence of a working integration, not a speaker-held-out accuracy evaluation or real-device certification.

## Attribution

MIT, under this repository's license. DSP and alignment are ported from Tilawa's pinned 1.x implementation; the tracker and lifecycle are new Swift implementations. The small test reference excerpt derives from the same Quran reference as the SDK. Quran text derives from [Tanzil](https://tanzil.net). See the repository's `NOTICE.md` for model, data, and training contributors; 1.x model weights are CC BY 4.0. No weights, third-party runtime binary, or recording ships in this contribution.

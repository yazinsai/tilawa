# @tilawa/sdk

Offline streaming Quran recitation checking for the web, Node and React Native. The listener (a streaming Zipformer phoneme-CTC model) runs on onnxruntime; the tracker and checker come from `@tilawa/core`.

## Use

```ts
import { createEngine, loadWebModel, startMicrophone } from "@tilawa/sdk/web";
import { loadReferenceIndex } from "@tilawa/core";

const model = await loadWebModel({ allowOverBudget: true }); // WebGPU, falling back to WASM SIMD
const engine = createEngine({ reference: loadReferenceIndex(refAsset), phrases, model });

const session = engine.startMemorization({ surah: 18, fromAyah: 60, toAyah: 70 }, { countSelfCorrections: false });
session.subscribe((e) => {
  // e.type: "capabilities" (first), "position", "checker"
});
const mic = await startMicrophone(session);
// ...
await mic.stop();
session.mistakes();
```

- `startMemorization({ surah, fromAyah, toAyah })` declares the passage: the checker aligns against it and uses the tracker's lock only to follow a deliberate jump. `startFreeRecitation()` follows the tracker's lock.
- Policy: `countSelfCorrections` (default false: a self-correction goes `suspected` then `resolved` and isn't a mistake), `confirm` (`{ after: "delay", ms }` or `{ after: "next-word" }`), `minConfidence`.
- The first event of every session is `capabilities`. Every class the model's metadata doesn't list is in `unsupported`; the v1 model supports the four memorization classes only.
- Node: `loadNodeModel(dir)` or `loadNodeTier({ tier, cacheDir })` from `@tilawa/sdk/node`, then `session.pushAudio(samples, sampleRate)` with WAV data from `readWav`. React Native: `loadNativeModel({ storage })` from `@tilawa/sdk/react-native`; the app supplies storage and microphone PCM.

## Model tiers

Models are downloaded on first use, versioned separately from the SDK (`src/models-v1.json`), cached (Cache API on web, a directory on Node, app storage on React Native), length- and SHA-256-checked, and refused when the tiers held exceed the 200 MB budget. A new version replaces the cached one.

| Tier | Model | Size | Status |
|---|---|---|---|
| `full` | `spliced-clean` (ticket 08), fp32 | 257.7 MB | over budget until ticket 10 quantises it; load with `allowOverBudget` |
| `lite` | — | — | none shipped yet |

The export is `modal/student_export.py`: icefall's streaming CTC graph, 45 fbank frames per call, advancing 32 (320 ms), 8 CTC frames out at 25 Hz, 128 frames of left context. Its decode matches the PyTorch model on the help clips (`docs/benchmarks/onnx-parity-spliced-clean-v1.json`, `sdk-help-spliced-clean-v1.json`).

## Sample app

`packages/sample`: see its README for running it on a phone.

## Credits

The listener is trained and evaluated on:

- **tlog** (`tarteel-ai/tlog`), by Tarteel, licensed CC-BY.
- **Quranic Universal Ayahs** and the QUD alignment benchmark (`QUD-Technologies/quranic-universal-ayahs`), by QUD Technologies, licensed CC-BY.

It is distilled from `obadx/muaalem-model-v3_2` (MIT) and uses the `quran-transcript` phoneme inventory (MIT). The streaming Zipformer and its export follow k2-fsa icefall (Apache-2.0); the full tier is initialised from icefall's LibriSpeech streaming Zipformer.

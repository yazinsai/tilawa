# @tilawa/sdk

Offline streaming Quran recitation checking for the web, Node and React Native. The listener (a streaming Zipformer phoneme-CTC model) runs on onnxruntime; the tracker and checker come from `@tilawa/core`.

## Use

```ts
import { createEngine, loadWebModel, startMicrophone } from "@tilawa/sdk/web";
import { loadReferenceIndex } from "@tilawa/core";

const model = await loadWebModel({ tier: "full" }); // WebGPU, falling back to WASM SIMD; tier "lite" for slow phones
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
- Policy: `countSelfCorrections` (default false: a self-correction goes `suspected` then `resolved` and isn't a mistake), `confirm` (`{ after: "delay", ms }` or `{ after: "next-word" }`), `minConfidence`, `evidenceMs`.
- Events don't depend on how you chunk audio or evidence: the session cuts the listener's output into `evidenceMs` cells on a fixed grid (default one 40 ms CTC frame), and the tracker searches for its first lock once per second of evidence on its own grid. Push whatever your audio callback delivers.
- The first event of every session is `capabilities`. Every class the model's metadata doesn't list is in `unsupported`; the v1 model supports the four memorization classes only.
- Node: `loadNodeModel(dir)` or `loadNodeTier({ tier, cacheDir })` from `@tilawa/sdk/node`, then `session.pushAudio(samples, sampleRate)` with WAV data from `readWav`. React Native: `loadNativeModel({ storage })` from `@tilawa/sdk/react-native`; the app supplies storage and microphone PCM.

## Model tiers

Models are downloaded on first use, versioned separately from the SDK (`src/models-v1.json`), cached (Cache API on web, a directory on Node, app storage on React Native), length- and SHA-256-checked, and refused when the tiers held exceed the 200 MB budget. A new version replaces the cached one.

| Tier | Model | Size | Gold PER | Help declared P / R | Help free P / R |
|---|---|---|---|---|---|
| `full` | `spliced-clean` (ticket 08), int8 | 67.8 MB | 4.82% | 82.4% / 87.5% | 77.8% / 43.8% |
| `lite` | `spliced-clean-lite`, int8 | see ticket 10 | | | |

int8 is dynamic quantisation of the MatMul weights (`tools/quantize_onnx.py`; icefall's recipe, and `ConvInteger` is missing from onnxruntime-web). It costs 0.02 points of gold PER against the fp32 export (4.80%). fp16 (129.6 MB, 4.79%) is a little faster in desktop Chrome WASM (RTF 0.096 vs 0.123) but peaks at 659 MB of page memory against int8's 252 MB, too much for a 4 GB phone, so no tier ships it. Weight-only 8-bit (`MatMulNBits`, 75 MB) runs at half int8's speed in WASM and was dropped. Comparison: `docs/benchmarks/scoreboard.md`, rows `spliced-clean-{onnx,int8,fp16}`.

The export is `modal/student_export.py`: icefall's streaming CTC graph, 45 fbank frames per call, advancing 32 (320 ms), 8 CTC frames out at 25 Hz, 128 frames of left context. Its decode matches the PyTorch model on the help clips (`docs/benchmarks/onnx-parity-spliced-clean-v1.json`, `sdk-help-spliced-clean-v1.json`).

## Latency

`measureLatency(session, samples, rate, pushMs)` replays audio on a real-time clock from measured compute: push `i` arrives at its audio time, waits for the previous push, and delivers its events when it finishes. Word highlight is measured from the end of the phoneme that moved the tracker to the word; `suspected` from the onset of the first word after the error. `pnpm bench:help` reports both over the help clips, with 20 ms pushes like the web microphone. Desktop numbers are in ticket 10; device runs use the sample's `device.html`.

## Sample app

`packages/sample`: see its README for running it on a phone, and `device.html` for RTF, memory, battery and latency measurements.

## Credits

The listener is trained and evaluated on:

- **tlog** (`tarteel-ai/tlog`), by Tarteel, licensed CC-BY.
- **Quranic Universal Ayahs** and the QUD alignment benchmark (`QUD-Technologies/quranic-universal-ayahs`), by QUD Technologies, licensed CC-BY.

It is distilled from `obadx/muaalem-model-v3_2` (MIT) and uses the `quran-transcript` phoneme inventory (MIT). The streaming Zipformer and its export follow k2-fsa icefall (Apache-2.0); the full tier is initialised from icefall's LibriSpeech streaming Zipformer.

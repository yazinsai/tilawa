# Tilawa

Offline, streaming Quran recitation engine. It follows a reciter word by word and flags memorization mistakes (wrong, omitted and added words, and skipped passages) as they happen, entirely on-device: in the browser, Node, and (soon) React Native. Audio never leaves the device.

`@tilawa/core` is TypeScript with no runtime dependencies. You install the ONNX Runtime for your platform as a peer dependency.

> **Upgrading from 0.x?** 1.0 is a new engine with a new API. See the [migration guide](https://github.com/yazinsai/tilawa/blob/main/packages/core/CHANGELOG.md#migrating-from-04).

## Install

```bash
npm i @tilawa/core@next onnxruntime-web    # browser
npm i @tilawa/core@next onnxruntime-node   # Node
```

`onnxruntime-web` ≥ 1.19, `onnxruntime-node` ≥ 1.20. Node ≥ 20.12.

## Browser

```ts
import { createTilawa } from "@tilawa/core/web";

// Downloads the model and the Quran reference once, then serves them from the Cache API.
const tilawa = await createTilawa({ onProgress: (p) => (bar.value = p.fraction) });

const session = tilawa.startFreeRecitation();
session.subscribe((e) => {
  if (e.type === "position" && e.observed[0]) highlight(e.observed[0].word); // { surah, ayah, word }
  if (e.type === "checker" && e.state === "confirmed") mark(e.location, e.errorClass);
});

const mic = await tilawa.listen(session); // microphone → session
// …
await mic.stop();                         // stops the tracks, then ends the session
console.log(session.mistakes());
tilawa.dispose();                         // releases the ONNX session
```

`listen()` needs a secure context (https or localhost). If it can't open the microphone it rejects with a `MicrophoneError` whose `reason` is `"denied"`, `"unavailable"` or `"insecure-context"`.

### A declared passage

When the app knows what the reciter is about to read, for example a page on screen, start a memorization session for it:

```ts
const session = tilawa.startMemorization(
  { surah: 18, fromAyah: 60, toAyah: 70 },
  { confirm: { after: "next-word" } }, // confirm a mistake once the next word starts (default: 2.5 s after it)
);
```

The tracker then stays inside the passage, and `position` events carry the word the reciter should be on (`expected`).

## Node

```ts
import { createTilawa, readWav } from "@tilawa/core/node";

const tilawa = await createTilawa(); // model cached in ~/.cache/tilawa; the reference ships in the package
const session = tilawa.startMemorization({ surah: 1, fromAyah: 1, toAyah: 7 });
session.subscribe((e) => {
  if (e.type === "checker" && e.state === "confirmed") console.log(e.errorClass, e.location);
});

const { samples, sampleRate } = await readWav("take.wav");
for (let i = 0; i < samples.length; i += 4800) session.pushAudio(samples.subarray(i, i + 4800), sampleRate);
await session.end();
console.log(session.mistakes());
await tilawa.dispose();
```

`pushAudio` takes audio at any sample rate. Calls run one at a time in call order, so a capture callback doesn't need to await them. Use `modelDir` to load a model directory you already have, or `cacheDir` to put downloads somewhere else.

## Events

The event schema is versioned (`v: 1`). Every session sends three kinds of event:

- **`capabilities`** comes first. It lists which error classes the model supports and which it doesn't, and carries `modelVersion`, `referenceVersion`, `sdkVersion`, `modelSha256`, `tier` and `backend`. V1 supports the four memorization classes `wrong_word`, `omitted_word`, `added_word` and `skipped_passage`.
- **`position`**: where the reciter is. `observed` lists candidates, best first, each with a `word` (`{surah, ayah, word}`, 1-based word index), a `span` and a `confidence`. `state` is `acquiring`, `locked`, `ambiguous` or `lost`. In memorization sessions `expected` is the passage word the reciter should be on.
- **`checker`**: a possible mistake, with an `id` that stays the same across its states. Each one goes `suspected` → `confirmed`, or `resolved` (a self-correction), or `uncertain` (the audio was too unclear to judge). It carries `errorClass`, `location`, and the `expected`/`observed` phonemes.

`timeMs` on every event is the session audio clock: ms of 16 kHz audio delivered since the first sample. `session.now()` reads the same clock. `ayahsRecited(events)` gives the ayahs recited, in order.

## Sessions

- A session is one take. `end()` drains the audio and decides every open mistake. It is idempotent, and after it starts `pushAudio` rejects with `SessionEndedError`. To start again, start a new session.
- `dispose()` drops a session's subscribers and buffers. `tilawa.dispose()` disposes every session and releases the model.
- `session.id` is a uuid. `session.startedAt` is the wall-clock time of the first sample, and `session.policy` is the policy with its defaults filled in.
- Policy: `confirm` (`{ after: "delay", ms }` or `{ after: "next-word" }`), `minConfidence` (default 0.6), and `countSelfCorrections` (default false: resolved mistakes aren't in `mistakes()`).

## Models and hosting

| Tier | Model | Size | Picked when |
|---|---|---|---|
| `full` | `spliced-clean@1-int8` | 68 MB | default |
| `lite` | `spliced-clean-lite@1-int8` | 25 MB | `tier: "auto"` on a device reporting under 4 GB of memory, or on mobile without WebGPU |

By default the browser entry loads from `https://tilawa.dev/v1/`: models from `models/<id>/<version>/model.onnx`, the reference from `reference/ref-v2.json`, and the onnxruntime-web WASM files from `runtime/ort/<version>/`. These files are immutable; a new model or reference gets a new path. Downloads are checked against the sha256 in the bundled registry.

- **Self-hosting**: mirror that layout and pass `assetBase`.
- **Offline**: pass `reference` (import `@tilawa/core/assets/ref-v2.json`) and a `model` you loaded with `loadWebModel` / `loadNodeModel`. If you serve the WASM files yourself, also pass `wasmPaths`.

Load failures reject `createTilawa` with `TilawaLoadError` (`stage`: `model`, `reference` or `warmup`), `ModelIntegrityError` or `ModelBudgetError`. Pass `signal` to cancel a download.

## Packages

| Import | What |
|---|---|
| `@tilawa/core` | Runtime-neutral engine: `createEngine`, event types, `loadReferenceIndex`, the model registry, `ayahsRecited`, errors |
| `@tilawa/core/web` | `createTilawa`, `loadWebModel`, `startMicrophone`, `cacheStorage`, `decodeWav` |
| `@tilawa/core/node` | `createTilawa`, `loadNodeModel`, `loadNodeTier`, `fsStorage`, `readWav` |
| `@tilawa/core/react-native` | `createTilawa`, `loadNativeModel`. Experimental: not yet validated on devices |
| `@tilawa/core/report` | Opt-in problem reports (no audio is kept or sent unless the app asks) |
| `@tilawa/core/assets/*.json` | `ref-v2.json`, `phrases-v1.json` |

## Licence and credits

The code is MIT. The listener model is trained and evaluated on tlog (Tarteel, CC-BY) and Quranic Universal Ayahs / the QUD alignment benchmark (QUD Technologies, CC-BY). It is distilled from `obadx/muaalem-model-v3_2` (MIT) and uses the `quran-transcript` phoneme inventory (MIT). The architecture and export follow k2-fsa icefall (Apache-2.0). The Quran text derives from Tanzil. Full notices: [NOTICE.md](https://github.com/yazinsai/tilawa/blob/main/NOTICE.md).

## This repository

- `packages/core/`: the SDK. `src/engine/` and `src/runtime/` are vendored from the research engine by `scripts/sync-engine.mjs`. Don't edit them by hand.
- `web/`: the demo at tilawa.dev.
- `lab/`: the earlier Python research harness.

# Tilawa

> Formerly called offline-tarteel.

[![Maintained by auto-maintainer](https://am.whhite.com/badge/yazinsai/tilawa)](https://am.whhite.com/stats/yazinsai/tilawa)

Offline Quran recognition. Give it 16 kHz mono audio, get back `surah:ayah`. Fully on-device — web, mobile, or node, no network at inference time.

`@tilawa/core` is pure TypeScript with **zero native dependencies**. You inject the ONNX runtime.

**Licence split:** package code is MIT. The Zipformer model and phoneme corpus are **NPL-1.2** (non-commercial, share-alike). FastConformer assets are MIT / NVIDIA CC-BY-4.0. Details: [NOTICE.md](https://github.com/yazinsai/tilawa/blob/main/NOTICE.md).

Two engines ship in the box. The default is **Zipformer** — streaming Zipformer2-CTC over a 251-token tajweed-phoneme vocabulary. **FastConformer** (text CTC) is still there under its original API.

## Install

```bash
npm i @tilawa/core
# plus the onnxruntime for your platform (you own this dep):
npm i onnxruntime-web            # browser / WASM
npm i onnxruntime-node           # node
npm i onnxruntime-react-native   # React Native
```

Default-engine assets: the model from [release zipformer-a0w-ep1-a0.5](https://github.com/yazinsai/tilawa/releases/tag/zipformer-a0w-ep1-a0.5), the corpus from [v0.3.0](https://github.com/yazinsai/tilawa/releases/tag/v0.3.0):

```bash
curl -L -O https://github.com/yazinsai/tilawa/releases/download/zipformer-a0w-ep1-a0.5/zipformer_a0w_ep1_a05.int8.onnx  # 66 MB, NPL-1.2
curl -L -O https://github.com/yazinsai/tilawa/releases/download/v0.3.0/zipformer_quran.json                            # 5.5 MB, NPL-1.2
# optional — Arabic text on verse_match events
curl -L -O https://github.com/yazinsai/tilawa/releases/download/v0.2.0/quran.json
```

The model's I/O manifest is bundled (`DEFAULT_ZIPFORMER_IO`). The previous default, `zipformer_interp_gentle_a05.int8.onnx` (v0.3.0), still works with the same manifest. `quran.json` is display text only; matching works without it.

## Browser

```ts
import * as ort from "onnxruntime-web";
// `onnxruntime-web/wasm` works too
import { createRecognitionSession } from "@tilawa/core";

// Vite copies `*.wasm` into the bundle by default. A raw <script type=module>
// or a bundler that doesn't should set:
// ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/";

const session = await createRecognitionSession({
  ort,
  model: () => fetch("/zipformer_a0w_ep1_a05.int8.onnx").then((r) => r.arrayBuffer()),
  corpus: () => fetch("/zipformer_quran.json").then((r) => r.json()),
  quran: () => fetch("/quran.json").then((r) => r.json()), // optional
  onEvent: (msg) => {
    if (msg.type === "verse_match") console.log(`${msg.surah}:${msg.ayah}`, msg.verse_text);
  },
});

for await (const chunk of micChunks) await session.feed(chunk);
const final = await session.stop();
session.reset();
```

Passing `ort` + `model` picks the provider automatically: `["wasm"]` under onnxruntime-web, `["cpu"]` under onnxruntime-node. Override with `executionProviders`. On web we also set `ort.env.wasm.numThreads = 1` unless you already set it — pthread init hangs in workers without COOP/COEP; single-thread is what the demo ships.

`createZipformerSession(opts)` is the same thing without the engine switch, and returns the richer `ZipformerSession` (`transcript`, `verses`, `engineState`, …).

## Node

```ts
import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { createRecognitionSession } from "@tilawa/core";

const session = await createRecognitionSession({
  ort,
  model: () => readFile("zipformer_a0w_ep1_a05.int8.onnx"),
  corpus: async () => JSON.parse(await readFile("zipformer_quran.json", "utf8")),
  quran: async () => JSON.parse(await readFile("quran.json", "utf8")),
  onEvent: (msg) => {
    if (msg.type === "verse_match") console.log(`${msg.surah}:${msg.ayah}`);
    if (msg.type === "raw_transcript") console.log(msg.text);
  },
});

const CHUNK = Math.round(0.3 * 16000); // 300 ms
for (let i = 0; i < pcm16k.length; i += CHUNK) {
  await session.feed(pcm16k.subarray(i, i + CHUNK));
}
const final = await session.stop();
session.reset();
```

## React Native

RN can't hand the model to ORT as an `ArrayBuffer` — bundle the `.onnx` as an asset, copy it to the documents dir, create the session from the **path**, and pass that session in with the runtime's `Tensor`:

```ts
import * as ort from "onnxruntime-react-native";
import { createZipformerSession } from "@tilawa/core";

const session = await createZipformerSession({
  session: await ort.InferenceSession.create(modelPath),
  Tensor: ort.Tensor,
  corpus: () => loadJsonAsset("zipformer_quran.json"),
  quran: () => loadJsonAsset("quran.json"),
});
```

Walkthrough: [examples/react-native.md](https://github.com/yazinsai/tilawa/blob/main/packages/core/examples/react-native.md). Copy-paste runners: [examples/](https://github.com/yazinsai/tilawa/tree/main/packages/core/examples).

## Alternate engine: FastConformer

Pick it for one-shot `transcribe()`, a raw Arabic transcript, or MIT-only assets. Download from [release v0.2.0](https://github.com/yazinsai/tilawa/releases/tag/v0.2.0):

```bash
base=https://github.com/yazinsai/tilawa/releases/download/v0.2.0
curl -L -O "$base/fastconformer_full_mixed.onnx"
curl -L -O "$base/vocab.json"
curl -L -O "$base/quran_ctc_tokens.json"
```

Write a `SessionRunner` that owns `ort`, then hand it to `createTilawaSession` with `{ vocab, quranCtcTokens, quran }`. Missing keys throw `Error("fastconformer engine requires assets: vocab, ctcTokens, quran ...")` before anything is read.

```ts
import * as ort from "onnxruntime-web";
import { createTilawaSession, type SessionRunner } from "@tilawa/core";

async function createWebSessionRunner(modelBuffer: ArrayBuffer): Promise<SessionRunner> {
  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ["wasm"],
  });
  return {
    async run(audio) {
      const input = new ort.Tensor("float32", audio, [1, audio.length]);
      const length = new ort.Tensor("int64", BigInt64Array.from([BigInt(audio.length)]), [1]);
      const results = await session.run({ audio_signal: input, length });
      const output = results[session.outputNames[0]];
      const [, timeSteps, vocabSize] = output.dims as number[];
      return { logprobs: output.data as Float32Array, timeSteps, vocabSize };
    },
  };
}

const session = createTilawaSession(await createWebSessionRunner(modelBuffer), {
  vocab,
  quranCtcTokens,
  quran,
});
const pred = await session.transcribe(audioFloat32);
// { surah: 1, ayah: 1, ayah_end: 3, score: 0.92, transcript: "..." }
```

Same runner shape on node (`onnxruntime-node`) and RN (create from a file path). FastConformer has no `stop()` — it finalizes on trailing silence. Wrap it in `createRecognitionSession({ engine: "fastconformer", runner, assets })` for the uniform `feed()` / `stop()` / `reset()` surface.

## Verse events

Both engines emit the same `WorkerOutbound` union — via `onEvent` / `onOutput`, and as the return value of `feed()` / `stop()`:

| `msg.type` | Meaning | Key fields |
|---|---|---|
| `verse_match` | Confident match for the current verse | `surah`, `ayah`, `verse_text`, `surah_name`, `confidence`, `surrounding_verses` |
| `verse_candidate` | Ranked candidates before lock-in | `candidates[]`, `stable`, `final_flush` |
| `word_progress` | Word-level alignment within a verse | `surah`, `ayah`, `word_index`, `total_words`, `matched_indices` |
| `raw_transcript` | Accumulated transcript so far (and again on `stop()`) | `text`, `confidence` |
| `final_sequence` | Full ordered sequence when recitation ends | `verses[]`, `confidence` |

## API

### `createRecognitionSession(options)`

`engine` defaults to `"zipformer"` (`DEFAULT_ENGINE`). Pass `engine: "fastconformer"` with `{ runner, assets }`. Returns `feed()`, `stop()` / `flush()`, `reset()`, plus `zipformer` / `fastconformer` (the other is `null`).

### `createZipformerSession(options)` → `ZipformerSession`

- `{ ort, model }` — runtime namespace + model bytes (or a loader). Providers: `["wasm"]` under onnxruntime-web, `["cpu"]` under onnxruntime-node.
- `{ session, Tensor }` — an `InferenceSession` you created plus that runtime's `Tensor`. RN shape.

| Option | Default | Purpose |
|---|---|---|
| `corpus` | *required* | Parsed `zipformer_quran.json`, or a loader |
| `quran` | empty | Arabic text for `verse_match` |
| `io` | `DEFAULT_ZIPFORMER_IO` | Override only for your own export |
| `executionProviders` | auto | See above |
| `onEvent` | — | Verse events, same order `feed()` / `stop()` return them |
| `minWordFraction` | `0.5` | Fraction of an ayah's words that must land |
| `enableFallback` | `true` | Whole-ayah search when nothing locked |
| `tailSeconds` | `2.0` | Silence `stop()` appends to flush the CTC tail |

Also: `transcript`, `tallies` / `verses`, `engineState`, `config`.

#### Structural correction rules (on by default in correction mode)

Correction mode (`setMode("correction")`) adds two text-only rules over the free decode, validated with the a0w model. They are on by default (`DEFAULT_STRUCTURAL`). Pass `structural: false` to turn them off, or `structural: {...}` / `await session.setStructural({...})` to choose; the look-alike index (56 KB gzipped) loads when the session is created. Tracking mode never runs them.

| Rule | Raises | Needs `setExpected`? |
|---|---|---|
| `ayahOrder: true` (`"guarded"` for the previous interp model) | `possible_skipped_ayah` — a whole ayah skipped between two read ones, found by matching pause-delimited segments to a local ayah window | No. Without a passage the window is the first located ayah −1 .. +4; with one it is the passage |
| `similarVerse: true` | `possible_substitution` (a word from a look-alike ayah), `possible_omission` (a dropped word) | For substitutions, yes: without a passage a swapped-in look-alike word reads as the other ayah. Drops work either way |

Issues they raise carry `source: "ayah_order" | "similar_verse"`. By default (`timing: "stop"`) they run once over the whole take at `stop()`, which is how they were validated; each `correct()` that closes an issue raises the next one. `timing: "pause"` also runs them at each pause so a flag can interrupt mid-recitation (more false flags). Similar-verse flags on or next to an ayah flagged as skipped are dropped.

### `createTilawaSession(runner, assets, options?)` → `TilawaSession`

`assets`: `{ vocab, quranCtcTokens, quran, blankId? }`. `transcribe()` / `transcribeRaw()` / `feed()` / `reset()` / `setConfig()` / `getConfig()`. Streaming presets: `"conservative"` / `"balanced"` / `"aggressiveAdvance"`.

`audio` on `SessionRunner.run` is **borrowed, not owned** — treat it as read-only.

## Models

| | Zipformer (default) | FastConformer |
|---|---|---|
| **File** | `zipformer_a0w_ep1_a05.int8.onnx` (66 MB; previous: `zipformer_interp_gentle_a05.int8.onnx`) | `fastconformer_full_mixed.onnx` (88 MB) |
| **Input** | 16 kHz mono `Float32Array`, streamed | same, preprocessing in-graph |
| **Recall / Precision / SeqAcc** | 100% / 100% / 100% on v1 (53/53); held-out multi-ayah windows 56/58 | 100% / 100% / 100% on v1 (53/53) |
| **Phoneme error rate** | 3.59% on the 600-clip q-lab benchmark (Quran-Lab v3: 4.45%) | — |
| **Licence** | **NPL-1.2** non-commercial share-alike | NVIDIA [CC-BY-4.0](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0) |

## This repository

Live demo: [web/frontend](https://github.com/yazinsai/tilawa/tree/main/web/frontend) (Zipformer only). Bake-off writeups: [lab/EXPERIMENTS.md](https://github.com/yazinsai/tilawa/blob/main/lab/EXPERIMENTS.md).

Zipformer models, vocabulary, and the phoneme corpus derive from [Quran-Lab/zipformer_p-arabic-v3](https://huggingface.co/Quran-Lab/zipformer_p-arabic-v3) and alketab's [ملقّن القرآن](https://prompter.alketab.app/). They are **NPL-1.2** and are not covered by this repo's MIT licence. [NOTICE.md](https://github.com/yazinsai/tilawa/blob/main/NOTICE.md).

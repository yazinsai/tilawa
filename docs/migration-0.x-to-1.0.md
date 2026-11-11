# Migrating from 0.x to 1.0

1.0 is a new engine with a new API. It replaces the 0.x verse matcher with a streaming engine that follows the reciter word by word and flags memorization mistakes. The API, events, model and assets are all new, so there is no compatibility layer. Most apps get shorter.

This guide covers 0.4.0. Earlier 0.x versions map the same way.

The 0.4 snippets are abridged from the 0.4.0 README. The 1.0 snippets are typechecked in [`examples/migration.ts`](examples/migration.ts).

## What you gain and what goes away

- **Gained:** word-level positions, a mistake checker with clear states, zero-config loading, a session lifecycle, and opt-in problem reports.
- **Gone:** display text from the engine, the FastConformer engine, the `correct()` retry/dismiss state machine, harakah (vowel) and repetition flags, and every tuning knob.
- **Licence:** the 1.x model and assets are not derived from Quran-Lab, so the 0.x NPL-1.2 restrictions don't apply to them. The model weight licence is {{PENDING: decision 7}}. See [NOTICE.md](../NOTICE.md) and the [model card](model-card.md).

## Steps

1. Install 1.0 and a runtime: `npm i @tilawa/core@next onnxruntime-web` (or `onnxruntime-node`).
2. Delete the model, corpus and `quran.json` downloads, and the code that fetches them. `createTilawa()` loads everything.
3. Replace `createRecognitionSession` / `createZipformerSession` with `createTilawa()` and a session per take.
4. Replace your verse-event handling with `position` and `checker` events.
5. Render the Quran text yourself from `{ surah, ayah, word }`.
6. Optionally, delete the old model cache (below).

`@tilawa/core@next` is the 1.0 preview. `latest` stays on 0.4.0 until 1.0.0 is released.

## Setup

**0.4** (abridged):

```ts
import * as ort from "onnxruntime-web";
import { createRecognitionSession } from "@tilawa/core";

const session = await createRecognitionSession({
  ort,
  model: () => fetch("/zipformer_a0w_ep1_a05.int8.onnx").then((r) => r.arrayBuffer()),
  corpus: () => fetch("/zipformer_quran.json").then((r) => r.json()),
  quran: () => fetch("/quran.json").then((r) => r.json()),
  onEvent: (msg) => {
    if (msg.type === "verse_match") console.log(`${msg.surah}:${msg.ayah}`, msg.verse_text);
  },
});
```

**1.0:**

```ts
import { createTilawa } from "@tilawa/core/web"; // or "@tilawa/core/node"

const tilawa = await createTilawa(); // model + reference, downloaded once and cached
const session = tilawa.startFreeRecitation();
session.subscribe((e) => {
  if (e.type === "position" && e.state === "locked" && e.observed[0]) {
    const { surah, ayah, word } = e.observed[0].word;
    console.log(`${surah}:${ayah}, word ${word}`);
  }
});
```

You no longer pass `ort`. Import the subpath for your platform and install its ONNX Runtime as a peer dependency.

## Feeding audio and stopping

**0.4:**

```ts
for await (const chunk of micChunks) await session.feed(chunk); // 16 kHz only
const final = await session.stop();
session.reset();
```

**1.0:**

```ts
// Browser: let the SDK run the microphone.
const mic = await tilawa.listen(session);
await mic.stop(); // stops the microphone, then ends the session

// Or push audio yourself, at any sample rate:
for (const chunk of chunks) session.pushAudio(chunk, sampleRate); // no need to await each one
await session.end();

// Next take: a new session. Sessions are one-shot.
session.dispose();
const next = tilawa.startFreeRecitation();
```

## A declared passage (correction mode)

**0.4:**

```ts
const zs = await createZipformerSession({ ort, model, corpus });
zs.setMode("correction");
zs.setExpected({ surah: 18, ayah: 60, ayahEnd: 70 });
```

**1.0:**

```ts
const session = tilawa.startMemorization({ surah: 18, fromAyah: 60, toAyah: 70 });
```

A memorization session is its own mode. Its `position` events carry `expected`, the passage word the reader should be on.

## Mistakes

**0.4:**

```ts
onEvent: (msg) => {
  if (msg.type === "correction") showCorrectionPrompt(msg);    // then session.correct("retry" | "dismiss" | ...)
  if (msg.type === "correction_note") showNote(msg.issue);
}
```

**1.0:**

```ts
session.subscribe((e) => {
  if (e.type !== "checker") return;
  if (e.state === "suspected") showPending(e.id, e.location);
  if (e.state === "confirmed") mark(e.location, e.errorClass);
  if (e.state === "resolved" || e.state === "uncertain") clearPending(e.id);
});
// After the take:
const mistakes = session.mistakes(); // the confirmed ones
```

Each mistake has an `id` that stays the same as it moves from `suspected` to `confirmed`, `resolved` (self-correction) or `uncertain`. What your app does with it (retry, dismiss, review later) is your UI. The engine no longer holds a correction state machine.

## The final sequence

**0.4:** the `final_sequence` event.

**1.0:**

```ts
import { ayahsRecited } from "@tilawa/core";

const events: EngineEvent[] = [];
session.subscribe((e) => events.push(e));
// ...after the take:
const recited = ayahsRecited(events); // [{ surah, ayah }, ...] in order
```

## Full mapping

| 0.4 | 1.0 |
|---|---|
| `createRecognitionSession({ ort, model, corpus, quran })` | `const tilawa = await createTilawa()` (`@tilawa/core/web` or `/node`), then `tilawa.startFreeRecitation()` |
| `createZipformerSession`, `ZipformerSession` | Same as above |
| `setExpected({ surah, ayah, ayahEnd })` + `setMode("correction")` | `tilawa.startMemorization({ surah, fromAyah, toAyah })` |
| `feed(samples16k)` | `session.pushAudio(samples, sampleRate)`, any rate |
| `stop()` / `flush()` | `await session.end()` (or `mic.stop()`) |
| `reset()` | `session.dispose()`, then start a new session |
| `onEvent` / `onOutput` / returned `WorkerOutbound[]` | `session.subscribe(fn)`. `pushAudio`/`end` also return the same events. |
| `verse_match {surah, ayah, confidence}` / `verse_candidate` | `position`: `observed[0].word`, `observed[0].confidence`, `state` |
| `word_progress {surah, ayah, word_index}` | `position.observed[0].word`, plus `expected` in memorization sessions |
| `final_sequence` | `ayahsRecited(events)` |
| `raw_transcript`, `debug` | `createTilawa({ recordEvidence: true })`, then `session.evidence()` |
| `correction` / `correction_note` | `checker` `{ id, state, errorClass, location, expected, observed }`; `session.mistakes()` |
| `possible_substitution` | `wrong_word` |
| `possible_omission` | `omitted_word` |
| `possible_skipped_ayah` | `skipped_passage` |
| (none) | `added_word` (new) |
| `possible_vowel`, `possible_repetition`, `unclear_ayah` | Not in 1.0. Listed in `capabilities.unsupported`. Low-clarity checks end `uncertain`. |
| `correct(action)` retry/dismiss state machine | Your app's UI |
| `setSlipHead`, `setStructural`, `StreamingConfig` and its presets, `DEFAULT_STRUCTURAL` | `SessionPolicy`: `confirm`, `minConfidence`, `countSelfCorrections` |
| `DEFAULT_ZIPFORMER_IO`, `io` | Gone. Models carry their manifest in the registry. |
| `executionProviders` | `providers` |
| `ort.env.wasm.numThreads`, `ort.env.wasm.wasmPaths` | `threads`, `wasmPaths` |
| FastConformer: `createTilawaSession`, `SessionRunner`, `QuranDB`, `TextCTCDecoder`, `RecitationTracker` | Removed. No replacement. |
| `verse_text`, `surah_name`, `quran.json` | Not part of the engine. Render text from `WordRef` with your own Quran text. |
| GitHub release model and corpus URLs | `https://tilawa.dev/v1/` (or your mirror). The old release URLs stay up for 0.x users. |
| IndexedDB database `tarteel-models` | Cache API cache `tilawa-models` |
| Microphone plumbing | `await tilawa.listen(session)` (web). VAD is off unless `{ vad: true }`. |

## Display text

The 0.x engine returned `verse_text`. 1.0 returns positions only. A `WordRef` is `{ surah, ayah, word }`, all 1-based, where `word` indexes the words of the Uthmani ayah. Use any Uthmani (Hafs) text whose word split matches, or read the words from `@tilawa/core/assets/ref-v2.json`.

## Clean up the old model cache

The 0.x demo and many apps stored the 66 MB model in an IndexedDB database called `tarteel-models`. 1.0 doesn't read it. If your app used that name, delete it once:

```ts
indexedDB.deleteDatabase("tarteel-models");
```

Don't delete the Cache API cache `tilawa-models`: 1.0 keeps its model there.

## Staying on 0.x

`@tilawa/core@0.4.0` stays on npm, and its model and corpus URLs on GitHub releases stay up. It gets no further changes. Its model is NPL-1.2 (non-commercial, share-alike); see [NOTICE.md](../NOTICE.md).

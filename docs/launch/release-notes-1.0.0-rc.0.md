# Tilawa 1.0.0-rc.0 (preview)

**Draft.** For the GitHub release and the npm `next` tag. Labelled **Preview**.

This is a preview of Tilawa 1.0. It is published as `@tilawa/core@next`; `latest` stays on 0.4.0 until 1.0.0 is released.

## What's new

1.0 replaces the 0.x verse matcher with a new streaming engine.

- **Word by word.** Tilawa follows the reader through the Quran word by word. `position` events give `{ surah, ayah, word }`.
- **Memorization feedback.** `checker` events flag wrong, omitted and added words, and skipped passages, as they happen. Each mistake moves from `suspected` to `confirmed`, `resolved` (the reader corrected themselves) or `uncertain` (too unclear to judge).
- **Recite from anywhere, or declare a passage.** `startFreeRecitation()` finds the reader's place. `startMemorization({ surah, fromAyah, toAyah })` stays inside a passage the app knows about.
- **Zero configuration.** `createTilawa()` loads the model and the Quran reference from `https://tilawa.dev/v1/`, caches them, and checks them against sha256. Self-hosting and fully offline setups are documented.
- **On the device.** Inference runs in the browser (WebGPU, or WebAssembly) or in Node. Audio is not uploaded.
- **Opt-in problem reports.** `@tilawa/core/report` lets readers report a wrong warning, a missed mistake or a wrong position. It produces a portable `report.json` + `audio.wav` artifact. Nothing is kept or sent unless the app asks.
- **Two model sizes.** `full` (67.8 MB) and `lite` (24.7 MB). `tier: "auto"` picks `lite` on low-memory devices and on mobile browsers without WebGPU.
- **A real session lifecycle.** `end()`, `dispose()`, clear errors, cancellation and load progress.

## Get started

```bash
npm i @tilawa/core@next onnxruntime-web
```

```ts
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa();
const session = tilawa.startFreeRecitation();
session.subscribe((e) => {
  if (e.type === "position" && e.state === "locked" && e.observed[0]) highlight(e.observed[0].word);
  if (e.type === "checker" && e.state === "confirmed") mark(e.location, e.errorClass);
});
const mic = await tilawa.listen(session);
```

- [Quickstart](https://github.com/yazinsai/tilawa/blob/main/docs/quickstart.md)
- [API reference](https://github.com/yazinsai/tilawa/blob/main/docs/api.md)
- [Migrating from 0.x](https://github.com/yazinsai/tilawa/blob/main/docs/migration-0.x-to-1.0.md)
- [Support matrix](https://github.com/yazinsai/tilawa/blob/main/docs/support.md)
- [Model card](https://github.com/yazinsai/tilawa/blob/main/docs/model-card.md)
- Live demo: [tilawa.dev](https://tilawa.dev/)

## Help it learn

Recite for a few minutes at [tilawa.dev/help](https://tilawa.dev/help/) and tell us where Tilawa got it wrong. Recordings are stored privately, and you can delete yours with the code you're given. [Privacy](https://github.com/yazinsai/tilawa/blob/main/docs/privacy.md).

## Breaking changes

Everything. 1.0 is a new API with new events, a new model and new assets. The [migration guide](https://github.com/yazinsai/tilawa/blob/main/docs/migration-0.x-to-1.0.md) maps every 0.4 call and event. In short:

- `createRecognitionSession`, `createZipformerSession` and the FastConformer engine are removed. Use `createTilawa()`.
- The `ort` namespace is no longer passed in. Install `onnxruntime-web` or `onnxruntime-node` and import the matching subpath.
- `verse_match`, `word_progress`, `final_sequence`, `correction` and the other 0.x events are replaced by `capabilities`, `position` and `checker` (schema `v: 1`).
- The engine no longer returns display text.
- Harakah (vowel) and repetition flags are not in 1.0.
- The model cache moves from IndexedDB `tarteel-models` to the Cache API cache `tilawa-models`.

## What 1.0 doesn't do

- **Pronunciation and tajweed are not supported.** Those error classes are listed in `capabilities.unsupported` and never emitted.
- Hafs only. Children's voices have not been evaluated.
- **React Native is experimental, and not yet validated on devices.**

## Evaluation

Preview quality. In our tests on long recitations recorded through a simulated phone, it caught about 9 in 10 planted mistakes, with about 0.7 false warnings per 5 minutes. On recordings from real users, we measured between 0.4 and 3.5 false warnings per 5 minutes, depending on the recording set; learners' phone recordings were at the high end. 7 of our 13 release bars pass; the full table is in the [model card](../model-card.md#evaluation). Phone and browser performance has not been measured yet. Please report wrong feedback at tilawa.dev/help so we can fix it.

## Known issues

- In browsers, the default `wasmPaths` points at `https://tilawa.dev/v1/runtime/ort/<your onnxruntime-web version>/`. Only the versions listed in `/v1/manifest.json` are there (1.30.0 at the time of writing). With another version, pass `wasmPaths` or self-host.
- `capabilities.modelVersion` holds the model id (`spliced-clean`), so reports record `engine.modelVersion` as the id too. Pass `engine: { modelVersion: tilawa.model.version ?? undefined }` to `createRecorder` until it is fixed.

## Licences

The code is MIT. The 1.x model and assets are not derived from Quran-Lab, so the 0.x NPL-1.2 terms don't apply to them. The model weights are CC BY 4.0. Credits: tlog (Tarteel, CC-BY), Quranic Universal Ayahs and the QUD alignment benchmark (QUD Technologies, CC-BY), muaalem and quran-transcript (MIT), k2-fsa icefall (Apache-2.0), and Tanzil for the Quran text. See [NOTICE.md](https://github.com/yazinsai/tilawa/blob/main/NOTICE.md).

## Artifacts

| | Identity |
|---|---|
| npm | `@tilawa/core@1.0.0-rc.0` (tag `next`) |
| Source | `yazinsai/tilawa` @ {{PENDING: release commit}} |
| Engine source | `packages/core/src/engine/SOURCE.json` |
| Model `full` | `spliced-clean@1-int8`, sha256 `714e98addee73b1b8781ad7d99b6d3c745acd9116638baf4e12891f53a15af1b` |
| Model `lite` | `spliced-clean-lite@1-int8`, sha256 `4974de78d3c35d2542e7a24d99b747e1160d6e31e366fb68492b74a789890f1b` |
| Reference | `ref-v2`, sha256 `f236384ea91e81ad554ba2777bc4073c3ff41e34120343a8a474571c577b7727` |
| Event schema | `v: 1` |
| Report schema | `tilawa.report/1` |

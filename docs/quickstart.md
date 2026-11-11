# Quickstart

`@tilawa/core` follows someone reciting the Quran, word by word, and flags memorization mistakes as they happen. It runs on the device. Audio is not uploaded.

This is a **Preview** release (`1.0.0-rc`, npm tag `next`). It does not yet meet all of Tilawa's own release bars; see the [model card](model-card.md#evaluation). `latest` stays on 0.4.0 until 1.0.0.

Every snippet on this page is also in [`docs/examples/`](examples/) and typechecked against the built package.

## Install

```bash
npm i @tilawa/core@next onnxruntime-web     # browser
npm i @tilawa/core@next onnxruntime-node    # Node 20.12 or newer
```

ONNX Runtime runs the model. You install the one for your platform; `@tilawa/core` has no other dependencies.

> **Browser apps:** by default the SDK loads onnxruntime-web's `.wasm` files from `https://tilawa.dev/v1/runtime/ort/<version>/`, where `<version>` is the onnxruntime-web version in your bundle. The runtime folders tilawa.dev hosts are listed in [`/v1/manifest.json`](https://tilawa.dev/v1/manifest.json) (`runtime/ort/<version>/…`; 1.30.0 at the time of writing). Install that version (`onnxruntime-web@1.30.0`), or serve the files yourself and pass `wasmPaths` ([self-hosting](self-hosting.md)).

## Web in ten lines

```ts
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa(); // downloads the model and the Quran reference once, then caches them
const session = tilawa.startFreeRecitation();
session.subscribe((e) => {
  if (e.type === "position" && e.state === "locked" && e.observed[0]) highlight(e.observed[0].word);
  if (e.type === "checker" && e.state === "confirmed") mark(e.location, e.errorClass);
});
const mic = await tilawa.listen(session); // asks for the microphone, then streams it into the session

stopButton.onclick = async () => {
  await mic.stop(); // stops the microphone and ends the session
  await tilawa.dispose();
};
```

`highlight` and `mark` are your UI. The engine gives you positions, not text: `{ surah, ayah, word }`, where `word` is the 1-based index of the word in the Uthmani ayah. Your app renders the Quran text.

What the reader sees:

- They can start anywhere in the Quran. The tracker finds the place, then follows it. `state` is `acquiring` until then.
- Move your highlight only on `locked` positions. `ambiguous` and `lost` positions are the tracker unsure, and moving the highlight on them looks jumpy.
- A mistake first arrives as `suspected`, then becomes `confirmed`, `resolved` (the reader corrected themselves) or `uncertain` (the audio was too unclear to judge). Show `confirmed` ones.

The page needs a secure context (https, or localhost) for the microphone.

### Show download progress

The first visit downloads the model: 67.8 MB for the `full` tier, 24.7 MB for `lite`. Later visits load it from the browser's Cache API.

```ts
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa({
  onProgress: (p) => {
    // p.stage is "model", "reference" or "warmup"; p.fraction is 0..1 within that stage.
    if (p.stage === "model") bar.value = p.fraction;
  },
});
console.log(`Loaded ${tilawa.model.id}@${tilawa.model.version} (${tilawa.model.tier}) on ${tilawa.model.backend}`);
```

`p.fromCache` is true when the bytes came from the cache.

## iOS Safari: open the microphone inside the tap

iOS only starts audio that was created during a user gesture. On the first visit, the download can take long enough for that gesture to expire. Call `prepareMicrophone()` synchronously in the tap handler, before any `await`, and hand the result to `listen()`.

```ts
import { createTilawa, prepareMicrophone } from "@tilawa/core/web";

startButton.onclick = async () => {
  // Synchronously, before any await: iOS only starts audio created during the tap.
  const prepared = prepareMicrophone();
  try {
    const tilawa = await createTilawa(); // may take a while on the first visit
    const session = tilawa.startFreeRecitation();
    session.subscribe((e) => {
      if (e.type === "position" && e.state === "locked" && e.observed[0]) highlight(e.observed[0].word);
    });
    const mic = await tilawa.listen(session, { prepared }); // listen() takes ownership of `prepared`
    stopButton.onclick = () => void mic.stop();
  } catch (e) {
    prepared.release(); // safe to call twice
    throw e;
  }
};
```

It does no harm on other browsers, so you can always do it.

## A declared passage

When your app knows what the reader is about to recite, for example a page on screen in a memorization app, start a memorization session for it:

```ts
import { createTilawa } from "@tilawa/core/web";

const tilawa = await createTilawa();
// The reader is about to recite Al-Kahf 18:60-70 from memory.
const session = tilawa.startMemorization(
  { surah: 18, fromAyah: 60, toAyah: 70 },
  { confirm: { after: "next-word" } }, // confirm a mistake once the next word starts (default: 2.5 s after it)
);
session.subscribe((e) => {
  // In a memorization session, `expected` is the passage word the reader should be on.
  if (e.type === "position" && e.expected) highlight(e.expected);
  if (e.type === "checker" && e.state === "confirmed") mark(e.location, e.errorClass);
});
const mic = await tilawa.listen(session);
stopButton.onclick = () => void mic.stop();
```

The tracker then stays inside the passage. Mistakes are judged against the passage, so a wrong word that happens to form another ayah is still wrong.

## Node

On Node you push audio yourself, for example from a file:

```ts
import { createTilawa, readWav } from "@tilawa/core/node";

const tilawa = await createTilawa(); // model cached in ~/.cache/tilawa; the reference ships in the package
const session = tilawa.startMemorization({ surah: 1, fromAyah: 1, toAyah: 7 });
session.subscribe((e) => {
  if (e.type === "checker" && e.state === "confirmed") console.log(e.errorClass, e.location);
});

const { samples, sampleRate } = await readWav("take.wav");
const chunk = Math.round(sampleRate * 0.3); // 300 ms at a time, like a live stream
for (let i = 0; i < samples.length; i += chunk) session.pushAudio(samples.subarray(i, i + chunk), sampleRate);
await session.end(); // waits for every queued push, then decides every open mistake

console.log(session.mistakes());
await tilawa.dispose();
```

- `pushAudio` takes mono float samples in [-1, 1] at any sample rate. The SDK resamples to 16 kHz.
- Pushes run one at a time, in call order. You don't need to await each one.
- `readWav` reads 8/16/24/32-bit integer or 32-bit float PCM WAV.
- `cacheDir` (or the `TILAWA_CACHE_DIR` environment variable) moves the model cache. `modelDir` loads a model directory you already have.

## Stopping and cleaning up

```ts
import { createTilawa } from "@tilawa/core/web";

// Cancel a slow first download, for example when the reader leaves the page.
const abort = new AbortController();
window.addEventListener("pagehide", () => abort.abort(), { once: true });
const tilawa = await createTilawa({ signal: abort.signal }); // rejects with an AbortError if cancelled

// One session is one take.
const session = tilawa.startFreeRecitation();
const unsubscribe = session.subscribe((e) => console.log(e.type));
const mic = await tilawa.listen(session);

// Stop: the microphone stops, then the session ends and its last events arrive.
await mic.stop();
const mistakes = session.mistakes(); // read results before disposing
unsubscribe();
session.dispose(); // drops subscribers and buffers; the model stays loaded

// Another take: start a new session. Sessions can't be restarted.
const next = tilawa.startFreeRecitation();
next.dispose();

// Done with Tilawa altogether: disposes any open sessions and releases the model.
await tilawa.dispose();
console.log(mistakes.length);
```

The rules:

- **`session.end()`** finishes a take. It drains the audio and decides every open mistake, so the last `checker` events arrive here. It is idempotent: every call returns the same promise. `mic.stop()` calls it for you.
- After `end()` starts, `pushAudio` rejects with `SessionEndedError`.
- **`session.dispose()`** drops the session's subscribers, audio taps and buffers. It does not release the model, which other sessions share.
- **`tilawa.dispose()`** disposes every open session and releases the ONNX model. Create a new instance to start again.
- Keep one `Tilawa` instance for the life of the page. Loading is the slow part; sessions are cheap.

## Errors

```ts
import { MicrophoneError, ModelBudgetError, ModelIntegrityError, SessionEndedError, TilawaLoadError } from "@tilawa/core";
import { createTilawa } from "@tilawa/core/web";
import type { Tilawa } from "@tilawa/core/web";

async function load(): Promise<Tilawa | null> {
  try {
    return await createTilawa();
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") return null; // you cancelled it with `signal`
    if (e instanceof ModelIntegrityError) showMessage("The download was damaged. Please try again.");
    else if (e instanceof ModelBudgetError) showMessage("Not enough room to store the model on this device.");
    else if (e instanceof TilawaLoadError) showMessage(`Couldn't load the ${e.stage}. Check the connection and try again.`);
    else throw e;
    return null;
  }
}

const tilawa = await load();
if (tilawa) {
  const session = tilawa.startFreeRecitation();
  try {
    await tilawa.listen(session);
  } catch (e) {
    session.dispose();
    if (e instanceof MicrophoneError) {
      if (e.reason === "denied") showMessage("Allow the microphone in your browser settings, then try again.");
      else if (e.reason === "insecure-context") showMessage("The microphone needs https (or localhost).");
      else showMessage("No microphone was found.");
    } else if (!(e instanceof SessionEndedError)) throw e;
  }
}
```

The error classes are exported from the root, `@tilawa/core`.

| Error | When |
|---|---|
| `TilawaLoadError` | `createTilawa` couldn't load the model or the reference. `stage` says which (`"model"` covers the download, the runtime and the warm-up run); `cause` is the underlying error. |
| `ModelIntegrityError` | A download was truncated, too large, or failed its sha256 check. |
| `ModelBudgetError` | The model, plus other tiers already cached, would exceed the 200 MiB storage budget. |
| `AbortError` | You aborted `signal` during `createTilawa`. |
| `MicrophoneError` | `listen()` couldn't open the microphone. `reason` is `denied`, `unavailable` or `insecure-context`. |
| `SessionEndedError` | `pushAudio` (or `listen`) on a session that has ended or been disposed. |

If inference fails mid-take, the `pushAudio` call rejects with the runtime's error and the session ends. `listen()` logs these by default; pass `onError` to handle them.

## Next

- [API reference](api.md): every export, event field and option.
- [Support matrix](support.md): browsers, Node, React Native, model sizes.
- [Self-hosting and offline use](self-hosting.md).
- [Reporting](reporting.md): let readers report a wrong warning or a missed mistake.
- [Migrating from 0.x](migration-0.x-to-1.0.md).
- [Model card](model-card.md): what the model can and can't do.

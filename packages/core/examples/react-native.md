# React Native

`@tilawa/core` has no DOM, `Worker`, or `fetch` dependency — the Zipformer engine
is plain TypeScript over typed arrays, so it runs on Hermes as-is. What differs on
React Native is how the model and the JSON assets get to it.

```bash
npm i @tilawa/core onnxruntime-react-native
```

## 1. Create the session

`onnxruntime-react-native` loads models from a **file path**, not bytes. So instead
of `{ ort, model }`, hand the SDK a session you already created plus that runtime's
`Tensor` constructor:

```ts
import * as ort from "onnxruntime-react-native";
import { createZipformerSession, type ZipformerSession } from "@tilawa/core";

export async function loadRecognizer(
  modelPath: string,
  corpusPath: string,
  quranPath: string,
): Promise<ZipformerSession> {
  return createZipformerSession({
    session: await ort.InferenceSession.create(modelPath),
    Tensor: ort.Tensor,
    corpus: () => readJson(corpusPath),
    quran: () => readJson(quranPath),
    onEvent: (msg) => {
      if (msg.type === "verse_match") {
        console.log(`${msg.surah}:${msg.ayah}`, msg.verse_text);
      }
    },
  });
}
```

Both loaders are lazy — the SDK calls them during `create()` and awaits them, so
you can stream the 5.5 MB corpus off disk without blocking module init.

`createRecognitionSession({ session, Tensor, corpus })` works identically; it just
wraps the result in the engine-agnostic `RecognitionSession` surface.

## 2. Get the assets onto the device

Three files: the 66 MB `.onnx`, the phoneme corpus, and (optionally) `quran.json`
for Arabic verse text in `verse_match` events.

```bash
curl -L -O https://github.com/yazinsai/tilawa/releases/download/zipformer-a0w-ep1-a0.5/zipformer_a0w_ep1_a05.int8.onnx
curl -L -O https://github.com/yazinsai/tilawa/releases/download/v0.3.0/zipformer_quran.json
```

Bundling a 66 MB binary in the app is usually the wrong trade — download it on
first launch and cache it in the documents dir:

```ts
import RNFS from "react-native-fs";

const MODEL_URL =
  "https://github.com/yazinsai/tilawa/releases/download/zipformer-a0w-ep1-a0.5/zipformer_a0w_ep1_a05.int8.onnx";

async function ensureModel(): Promise<string> {
  const dest = `${RNFS.DocumentDirectoryPath}/zipformer_a0w_ep1_a05.int8.onnx`;
  if (await RNFS.exists(dest)) return dest;
  await RNFS.downloadFile({ fromUrl: MODEL_URL, toFile: dest }).promise;
  return dest;
}

async function readJson(path: string): Promise<unknown> {
  return JSON.parse(await RNFS.readFile(path, "utf8"));
}
```

If you'd rather ship the JSON in the JS bundle, `require("./zipformer_quran.json")`
also works — Metro inlines it, which costs startup time but removes the file I/O.
The model can't go that route; it has to be a path ORT can open.

## 3. Feed the microphone

The engine wants mono **16 kHz** `Float32Array` PCM, any chunk size (≈480 ms is a
good cadence). Native mic libraries hand you base64 16-bit PCM, so convert:

```ts
function pcm16ToFloat32(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

const session = await loadRecognizer(modelPath, corpusPath, quranPath);

// e.g. react-native-live-audio-stream, configured { sampleRate: 16000, channels: 1, bitsPerSample: 16 }
LiveAudioStream.on("data", async (base64) => {
  await session.feed(pcm16ToFloat32(base64ToBytes(base64)));
});

// when the user stops recording
const final = await session.stop(); // ends with final_sequence
session.reset();                    // next recitation, model stays loaded
```

`feed()` is async and must not be re-entered — serialize the calls (a promise chain
or a small queue) if your audio callback can fire while inference is in flight.

## Hermes notes

- **BigInt is required.** The Zipformer graph takes `int64` cache states, which ORT
  represents as `BigInt64Array`. Hermes has had BigInt since 0.12 (RN 0.70-era), so
  this is a non-issue on any current RN; on a runtime without it, session setup
  throws an explicit "BigInt64Array is unavailable" error instead of failing deep
  inside state init.
- **No top-level `await`** anywhere in the package — safe for Metro.
- **Typed arrays only.** No `TextDecoder`, `Buffer`, `Blob`, or `structuredClone`,
  so nothing to polyfill.
- **Licensing.** The Zipformer model and phoneme corpus are NPL-1.2
  (non-commercial, share-alike) — see `NOTICE.md`. For MIT-only assets use
  `engine: "fastconformer"` with `session-react-native.ts` in this directory.

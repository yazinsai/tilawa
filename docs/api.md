# API reference

`@tilawa/core` 1.0.0-rc.0 (Preview, npm tag `next`).

Start with the [quickstart](quickstart.md). This page lists every public export, by import path.

| Import | What it holds | Needs |
|---|---|---|
| [`@tilawa/core`](#tilawacore) | The runtime-neutral engine, event types, errors, model registry | nothing |
| [`@tilawa/core/web`](#tilawacoreweb) | `createTilawa` for browsers, microphone capture | `onnxruntime-web` ≥ 1.19 |
| [`@tilawa/core/node`](#tilawacorenode) | `createTilawa` for Node, WAV input | `onnxruntime-node` ≥ 1.20, Node ≥ 20.12 |
| [`@tilawa/core/react-native`](#tilawacorereact-native) | `createTilawa` for React Native. **Experimental, not yet validated on devices** | `onnxruntime-react-native` ≥ 1.20 |
| [`@tilawa/core/report`](#tilawacorereport) | Opt-in problem reports | nothing |
| [`@tilawa/core/assets/*`](#tilawacoreassets) | `ref-v2.json`, `phrases-v1.json` | nothing |

The package is ESM only. Every subpath ships its own type declarations.

Contents: [createTilawa](#createtilawa) · [Tilawa](#tilawa) · [Sessions](#sessions) · [Events](#events-schema-v1) · [Errors](#errors) · [Version policy](#version-policy)

---

## `createTilawa`

```ts
function createTilawa(options?: TilawaOptions): Promise<Tilawa>
```

Exported by `/web`, `/node` and `/react-native`. It loads the model and the Quran reference, warms the model up, and resolves with a ready [`Tilawa`](#tilawa). The model and the reference load in parallel.

### `TilawaOptions` (all platforms)

| Option | Type | Default | Meaning |
|---|---|---|---|
| `tier` | `"full" \| "lite" \| "auto"` | `"auto"` | Which model. See [tiers](#model-tiers). |
| `assetBase` | `string` | `"https://tilawa.dev/v1/"` | Where models, the reference and (web) the ONNX Runtime files are fetched from. Point it at a mirror to [self-host](self-hosting.md). A trailing `/` is added if missing. |
| `reference` | `ReferenceAsset` | fetched (web, RN) or bundled (Node) | The reference, for offline use: import `@tilawa/core/assets/ref-v2.json`. |
| `phrases` | `PhraseAsset` | the bundled `phrases-v1` | Non-Quran phrases said around a recitation (isti'adha, takbir, amin…). |
| `model` | `LoadedModel` | loaded from the registry | A model you loaded yourself (`loadWebModel`, `loadNodeModel`, `loadNativeModel`). You keep ownership: `tilawa.dispose()` won't release it. |
| `storage` | `ModelStorage` | Cache API (web), `fsStorage(cacheDir)` (Node), in memory (RN) | Where downloaded model bytes (and, on web and RN, the reference) are kept. |
| `providers` | `string[]` | web `["webgpu", "wasm"]`; RN `["cpu"]` | ONNX execution providers, tried in order on web. Ignored on Node (CPU). |
| `threads` | `number` | `1` | ONNX intra-op threads. On web, more than 1 only takes effect when the page is cross-origin isolated. |
| `wasmPaths` | `string` | `<assetBase>runtime/ort/<onnxruntime-web version>/` | Web only. Where onnxruntime-web's `.wasm`/`.mjs` files are served. Ignored when you pass `model`; give it to `loadWebModel` instead. |
| `onProgress` | `(p: LoadProgress) => void` | none | Load progress, see below. |
| `signal` | `AbortSignal` | none | Cancels loading. `createTilawa` then rejects with an `AbortError`. |
| `recordEvidence` | `boolean` | `false` | Keep each session's phoneme evidence for `session.evidence()`. Use it for bug reports and benchmarks. It also keeps the session's 16 kHz audio in memory, and at `end()` re-decodes gaps, so `evidence()` after `end()` can differ from what streamed. |

`LoadProgress` is `{ stage: "model" | "reference" | "warmup", loaded: number, total: number, fraction: number, fromCache: boolean }`. `fraction` is 0..1 within the stage. Byte counts are bytes for `model` and `reference`; `warmup` goes 0 → 1.

### Platform options

| Option | Platform | Meaning |
|---|---|---|
| `providers?: WebProvider[]` | web | `"webgpu"` and/or `"wasm"`. A provider that fails to load or run its first window falls through to the next. |
| `cacheDir?: string` | Node | Model cache directory. Default `$TILAWA_CACHE_DIR`, else `~/.cache/tilawa`. |
| `modelDir?: string` | Node | Load `manifest.json` and its `.onnx` file from this directory instead of the registry. Mirrors `/v1/models/<id>/<version>/`. |
| `pathOf?: (key: string) => string` | RN | The file path where your `storage` keeps a key. onnxruntime-react-native loads a file path faster than bytes across the bridge. |

The option types are exported as `WebTilawaOptions`, `NodeTilawaOptions` and `NativeTilawaOptions`.

### Model tiers

| Tier | Model | Download | `tier: "auto"` picks it |
|---|---|---|---|
| `full` | `spliced-clean@1-int8` | 67,759,744 bytes (67.8 MB) | Otherwise. Always on Node. |
| `lite` | `spliced-clean-lite@1-int8` | 24,650,054 bytes (24.7 MB) | Web: `navigator.deviceMemory` under 4, or a mobile browser without WebGPU (`navigator.gpu`). React Native: always. |

Both detect the same four error classes. See the [model card](model-card.md).

---

## `Tilawa`

```ts
interface Tilawa {
  startFreeRecitation(policy?: SessionPolicy): TilawaSession;
  startMemorization(passage: MemorizationPassage, policy?: SessionPolicy): TilawaSession;
  listen(session: TilawaSession, options?: MicrophoneOptions): Promise<Microphone>;
  readonly capabilities: CapabilityEvent;
  readonly model: TilawaModelInfo;
  dispose(): Promise<void>;
}
```

- **`startFreeRecitation(policy?)`**: a session for recitation from anywhere in the Quran. The tracker finds the place.
- **`startMemorization({ surah, fromAyah, toAyah }, policy?)`**: a session for a declared passage. The tracker stays in it, `position.expected` is the word the reader should be on, and mistakes are judged against the passage. Throws a `RangeError` for an empty or reversed passage.
- **`listen(session, options?)`**: web only. Opens the microphone and streams it into the session. On Node and React Native it rejects with `MicrophoneError("unavailable")`: push audio yourself.
- **`capabilities`**: the `capabilities` event every session sends first.
- **`model`**: `TilawaModelInfo` `{ id, version, sha256, tier, backend, fallbacks }`. `version`, `sha256` and `tier` are null for a passed-in model that isn't in the registry. `backend` is, for example, `onnxruntime-web/webgpu`. `fallbacks` says why each earlier web provider was skipped.
- **`dispose()`**: disposes every open session and releases the ONNX model (unless you passed `model`). Idempotent. Starting a session afterwards throws.

### `SessionPolicy`

| Field | Default | Meaning |
|---|---|---|
| `confirm` | `{ after: "delay", ms: 2500 }` | When a suspected mistake is confirmed: a fixed delay after it, or `{ after: "next-word" }`, once the next word has begun. |
| `minConfidence` | `0.6` | Below this mean decoder confidence over a mistake's phonemes, it ends `uncertain` instead of `confirmed`. |
| `countSelfCorrections` | `false` | Whether `resolved` mistakes (self-corrections, recovered skips) count in `session.mistakes()`. |
| `evidenceMs` | one model frame (40 ms) | The tracker and checker read evidence in cells of this much audio on a fixed grid, so events don't depend on how you chunk audio. Leave it alone unless you know why. |

`session.policy` is the resolved policy, with every default filled in (`ResolvedSessionPolicy`).

### `MicrophoneOptions` and `Microphone` (web)

| Option | Meaning |
|---|---|
| `vad?: boolean \| VadOptions` | Gate the audio with an energy voice-activity detector. **Default off.** With it on, silence is dropped before the model, so the session clock runs behind the wall clock. `VadOptions`: `marginDb` (default 12 dB above the learnt noise floor), `hangMs`, `preRollMs`. Keep the hangover longer than the confirm delay. |
| `onLevel?: (db) => void` | Raw input level in dBFS, about every 20 ms, for a meter. |
| `onEvents?: (events) => void` | The events each push produced. `subscribe` is usually simpler. |
| `onError?: (e) => void` | A push failed (an inference error). Default: logged to the console, except `SessionEndedError`. |
| `prepared?: PreparedMicrophone` | From `prepareMicrophone()`, see below. `listen()` takes ownership. |

The microphone is opened mono, with echo cancellation, noise suppression and auto gain control **off**. It runs at the device's rate and is resampled to 16 kHz.

`Microphone` is `{ sampleRate, speaking, stop() }`. `sampleRate` is the device rate. `speaking` is true while the VAD hears speech (always true without the VAD). `stop()` stops the tracks, closes the audio context, waits for queued audio, then calls `session.end()` and resolves with its events. It is idempotent.

`listen()` rejects with `MicrophoneError` (`denied`, `unavailable`, `insecure-context`), or with `SessionEndedError` if the session isn't `active`.

---

## Sessions

`TilawaSession` (alias `Session`) is one take.

```ts
interface TilawaSession {
  readonly id: string;                      // uuid v4
  readonly mode: "memorization" | "free";
  readonly passage: DeclaredPassage | null; // {surah, fromAyah, toAyah, from: WordRef, to: WordRef}
  readonly policy: Readonly<ResolvedSessionPolicy>;
  readonly capabilities: CapabilityEvent;
  readonly state: SessionState;             // "active" | "ending" | "ended" | "disposed"
  readonly startedAt: string | null;        // ISO wall-clock time of the first sample
  readonly vad: boolean;                    // true while listen() gates this session with the VAD
  now(): number;                            // session clock, ms
  subscribe(listener: (event: EngineEvent) => void): () => void;
  onAudio(fn: AudioTap): () => void;
  pushAudio(samples: Float32Array, sampleRate: number): Promise<EngineEvent[]>;
  end(): Promise<EngineEvent[]>;
  dispose(): void;
  idle(): Promise<void>;
  onEnded(fn: () => void): () => void;
  pushEvidence(evidence: PhonemeEvidence): EngineEvent[];
  endEvidence(): EngineEvent[];
  mistakes(): CheckerEvent[];
  evidence(): PhonemeEvidence[];
}
```

### Lifecycle

```
active ──end()──▶ ending ──(drained)──▶ ended
   └──────────────── dispose() from any state ───────────▶ disposed
```

- **`pushAudio(samples, sampleRate)`**: mono float samples at any rate. Calls run one at a time in call order; you don't need to await them. Resolves with the events this audio produced (subscribers get the same events; the return value is a convenience for tests). Rejects with `SessionEndedError` once `end()` has started or after `dispose()`.
- **`end()`**: drains the audio still in the model and decides every open mistake, then the state becomes `ended`. Idempotent: every call returns the same promise. Rejects with `SessionEndedError` after `dispose()`.
- **`dispose()`**: drops subscribers, audio taps and buffers, and rejects queued pushes with `SessionEndedError`. It doesn't release the model. `mistakes()` and `evidence()` return `[]` afterwards.
- **Inference failure**: the failing `pushAudio` rejects with the runtime's error, and the session moves to `ended`. Later queued pushes reject with `SessionEndedError`.
- **`idle()`**: resolves once every queued `pushAudio`/`end` has run, so all events for the audio received so far are out.
- **`onEnded(fn)`**: calls `fn` once when the session ends or is disposed, or at once if that already happened. Returns an unsubscribe function.
- **`subscribe(fn)`**: calls `fn` with the `capabilities` event at once, then with every event. Returns an unsubscribe function. A subscriber that throws doesn't stop the others; its error is rethrown asynchronously. Subscribing to a disposed session does nothing.
- **`onAudio(fn)`**: taps the 16 kHz samples the model receives, with the session-clock time of the first one: `(pcm16k: Float32Array, startMs: number) => void`. It never changes them. The reporting recorder uses it. Don't mutate the array.
- **`mistakes()`**: the `confirmed` checker events, plus `resolved` ones when `countSelfCorrections` is on.
- **`evidence()`**: the phoneme evidence so far, when `createTilawa({ recordEvidence: true })`; otherwise `[]`.
- **`pushEvidence(evidence)` / `endEvidence()`**: feed evidence from another listener, bypassing the model (tests, replays). Don't mix with `pushAudio`. Synchronous; throw `SessionEndedError` when the session isn't active.

### The session clock

`timeMs` on `position` and `checker` events, `session.now()`, and the report recorder all use one clock: **milliseconds of 16 kHz audio delivered to the session since its first sample**. It is not the wall clock.

- It advances only as audio is pushed. A paused stream pauses the clock.
- With the VAD on, dropped silence doesn't count, so the clock runs behind the wall clock.
- Inference can lag behind capture. An event's `timeMs` is where it happened in the audio, not when it was emitted.
- `session.startedAt` is the wall-clock time of the first sample, for display.

---

## Events (schema v1)

```ts
type EngineEvent = CapabilityEvent | PositionEvent | CheckerEvent;
```

Every event has `type` and `v: 1` (`EVENT_SCHEMA_VERSION`). Events are plain JSON-serialisable objects.

### `capabilities`

Sent first to every subscriber.

| Field | Type | Meaning |
|---|---|---|
| `modelVersion` | `string` | The model's id, for example `spliced-clean` (name kept from schema v1). |
| `modelRelease` | `string?` | The model's registry version, for example `1-int8`. Absent for an injected model outside the registry. |
| `referenceVersion` | `string` | The reference, for example `ref-v2`. |
| `supported` | `ErrorClass[]` | Error classes this model checks. In 1.0: `wrong_word`, `omitted_word`, `added_word`, `skipped_passage`. |
| `unsupported` | `ErrorClass[]` | Every other class. Never emitted. |
| `sdkVersion?` | `string` | The SDK version, `SDK_VERSION`. Added in 1.0. |
| `modelSha256?` | `string` | The model's sha256. Added in 1.0. |
| `tier?` | `string` | `full` or `lite`. Added in 1.0. |
| `backend?` | `string` | The ONNX backend, for example `onnxruntime-web/wasm`. Added in 1.0. |

### `position`

Where the reader is.

| Field | Type | Meaning |
|---|---|---|
| `timeMs` | `number` | Session clock. |
| `referenceVersion` | `string` | |
| `state` | `TrackerState` | `acquiring` (no place yet), `locked` (following), `ambiguous` (several places fit, for example repeated ayahs), `lost` (the place no longer fits). |
| `observed` | `PositionCandidate[]` | Candidate places, best first, each a distinct location. Can be empty. |
| `expected` | `WordRef \| null` | Memorization sessions: the passage word the reader should be on. Null in free recitation. |
| `hold?` | `"start"` | Set on the locked event that starts a misfit hold: the heard words stopped fitting and the tracker is waiting before it moves. |
| `relocated?` | `{ from: WordRef; to: WordRef }` | A passage that appears in near-identical copies resolved to one copy. `from` equal to `to` means the current copy was confirmed. |
| `spanHold?` | `true` | The first lock is inside such a passage and is still waiting for a word that tells the copies apart. |
| `trace?` | `TrackerTrace` | Tracker internals. Not set by `createTilawa` sessions. |

`PositionCandidate`:

| Field | Meaning |
|---|---|
| `word: WordRef` | The most recently recited word. |
| `span: WordSpan` | `{ start, end }`: the words covered by the evidence so far. It can cross ayahs. |
| `confidence: number` | 0..1. |
| `cost: number` | Weighted alignment cost per phoneme, for ranking. 0 is an exact match. |
| `per: number` | Phoneme error rate against the span. |
| `prefix?: "basmala"` | A leading basmala was recited before the span. |

`WordRef` is `{ surah, ayah, word }`. All three are 1-based. `word` indexes the words of the ayah as the reference lists them (Uthmani text). The engine returns no display text.

`ayahsRecited(events)` turns a list of events into the ayahs recited, in order: the best word of every `locked` position, with consecutive repeats of one ayah collapsed.

### `checker`

A possible mistake.

| Field | Type | Meaning |
|---|---|---|
| `id` | `string` | Stable across the mistake's states. |
| `timeMs` | `number` | Session clock. |
| `state` | `CheckerState` | `suspected`, then one of `confirmed`, `resolved` (the reader corrected themselves, or a skip was recovered) or `uncertain` (too unclear to judge). |
| `errorClass` | `ErrorClass` | One of the `supported` classes. |
| `location` | `WordRef` | The word the mistake is at. For `skipped_passage`, the first skipped word. |
| `expected?` | `string` | The phonemes expected there. |
| `observed?` | `string` | The phonemes heard. |
| `confidence` | `number` | Mean decoder confidence over the mistake's phonemes. |

Error classes (`ErrorClass` = `MemorizationErrorClass | PronunciationErrorClass`):

- **Supported in 1.0** (`MEMORIZATION_CLASSES`): `wrong_word`, `omitted_word`, `added_word`, `skipped_passage`.
- **Not supported**: every `PronunciationErrorClass` (`short_vowel`, `missing_shadda`, `madd_natural` and the `consonant_*` pairs). They are in the type so the schema can grow without a breaking change. 1.0 never emits them, and it does not judge pronunciation or tajweed.

`ALL_CLASSES` lists every class.

---

## Errors

All exported from `@tilawa/core`.

| Class | Fields | Thrown by |
|---|---|---|
| `TilawaLoadError` | `stage: "model" \| "reference" \| "warmup"`, `cause` | `createTilawa` |
| `ModelIntegrityError` | | `createTilawa`, `fetchModel`, loaders: truncated, oversized or sha256 mismatch |
| `ModelBudgetError` | | `createTilawa`, `fetchModel`: over the 200 MiB budget (`MODEL_BUDGET_BYTES`) |
| `MicrophoneError` | `reason: "denied" \| "unavailable" \| "insecure-context"` | `listen`, `prepareMicrophone`, `startMicrophone` |
| `SessionEndedError` | | `pushAudio`, `end`, `pushEvidence`, `endEvidence`, `listen` on an ended or disposed session |

`createTilawa` also rejects with an `AbortError` (a `DOMException`, or `signal.reason` if that is an Error) when you abort `signal`. `LoadStage` and `MicrophoneErrorReason` are exported as types.

---

## `@tilawa/core`

Runtime-neutral: no Node built-ins, no ONNX Runtime import. Use it for types, or to build an engine around your own runtime.

**Engine**

- `createEngine(options: EngineOptions): Engine`: sessions from a loaded reference and model, without `createTilawa`'s loading. `EngineOptions` is `{ reference: ReferenceIndex, model: LoadedModel, phrases?, listener?: ListenerOptions, recordEvidence?, info?: { tier?, backend?, modelSha256? } }`. `Engine` has `capabilities`, `startFreeRecitation` and `startMemorization`.
- Types: `Engine`, `EngineOptions`, `Session`, `TilawaSession`, `SessionState`, `SessionPolicy`, `ResolvedSessionPolicy`, `MemorizationPassage`, `DeclaredPassage`, `AudioTap`, `CapabilityEvent`, `EngineEvent`, `ListenerOptions`, `VadOptions`, and the `createTilawa` types `Tilawa`, `TilawaOptions`, `TilawaModelInfo`, `LoadProgress`, `Microphone`, `MicrophoneOptions`.

**Events**: `EVENT_SCHEMA_VERSION`, `ALL_CLASSES`, `MEMORIZATION_CLASSES`, `ayahsRecited`; types `CheckerEvent`, `CheckerState`, `ErrorClass`, `MemorizationErrorClass`, `PronunciationErrorClass`, `PositionEvent`, `PositionCandidate`, `TrackerState`, `WordRef`, `WordSpan`.

**Evidence** (types): `PhonemeEvidence` `{ inventory, frameMs, phonemes: DecodedPhoneme[], posteriors?: TopKPosteriors, decodedTo? }`, `DecodedPhoneme`, `TopKPosteriors`.

**Reference**: `loadReferenceIndex(asset: ReferenceAsset): ReferenceIndex`; types `ReferenceAsset`, `ReferenceIndex`, `PhraseAsset`, `AyahRef` (`{ surah, ayah }`).

**Models**

- `MODEL_REGISTRY: ModelRegistry`: the shipped tiers, with relative `base` paths, sizes, sha256 and manifests.
- `withBase(registry, base)`: the registry with every relative `base` resolved against `base`.
- `fetchModel(options: FetchModelOptions): Promise<FetchedModel>`: downloads (or reads from `storage`) one tier's bytes. Checks length and sha256 on download; a cache hit is only length-checked. Deletes older versions of the same model id from `storage` once the new one is stored. Throws `ModelBudgetError` before downloading when the tier plus other cached tiers exceed `budgetBytes` (default `MODEL_BUDGET_BYTES`, 200 MiB), unless `allowOverBudget`.
- `memoryStorage(): ModelStorage`: an in-memory store.
- `ortBackend(ort, model, name, sessionOptions?)`: wraps any ONNX Runtime module as an `InferenceBackend`.
- `MODEL_BUDGET_BYTES`.
- Types: `ModelRegistry`, `ModelTier`, `TierName` (`"full" | "lite"`), `ModelStorage` (`get`, `put`, `delete`, `keys`, all async), `FetchModelOptions`, `FetchedModel`, `LoadedModel`, `ModelManifest`, `InferenceBackend`, `Tensor`, `StateSpec`, `OrtModule`.

**Constants**: `DEFAULTS` (the defaults in the tables above, frozen), `REFERENCE` (`{ version: "ref-v2", path, bytes, sha256 }`), `SDK_VERSION`.

**Errors**: see [Errors](#errors).

## `@tilawa/core/web`

- `createTilawa(options?: WebTilawaOptions)`.
- `prepareMicrophone(): PreparedMicrophone`: call synchronously inside a tap handler. It creates and resumes an `AudioContext` and requests the microphone while the user gesture is fresh, which iOS Safari needs. Pass the result to `listen(session, { prepared })`. `PreparedMicrophone` is `{ context, stream, release() }`; call `release()` if you never call `listen()`. Throws `MicrophoneError` synchronously when there is no microphone API.
- `startMicrophone(session, options?)`: what `tilawa.listen` calls.
- `loadWebModel(options?: WebModelOptions)`: loads a tier with onnxruntime-web and warms it up. Options: `tier` (default `"full"`), `registry` (default `MODEL_REGISTRY`, whose relative paths resolve against the page; use `withBase`), `storage` (default `cacheStorage()`), `providers`, `threads`, `wasmPaths`, `allowOverBudget`, `onProgress(loaded, total)`. Resolves with the `LoadedModel` plus `fromCache` and `fallbacks`.
- `cacheStorage(name = "tilawa-models"): ModelStorage`: the Cache API store `createTilawa` uses.
- `decodeWav(bytes: Uint8Array): { sampleRate, samples }`: PCM WAV (8/16/24/32-bit integer or 32-bit float) to mono floats.
- Types: `WebTilawaOptions`, `WebModelOptions`, `WebProvider`, `PreparedMicrophone`, and the shared `Tilawa`, `TilawaOptions`, `TilawaModelInfo`, `LoadProgress`, `Microphone`, `MicrophoneOptions`.

## `@tilawa/core/node`

- `createTilawa(options?: NodeTilawaOptions)`. The reference is read from the package; only the model is downloaded.
- `readWav(path): Promise<{ sampleRate, samples }>` and `decodeWav(bytes)`.
- `loadNodeModel(dir, { threads? }): Promise<LoadedModel>`: loads `manifest.json` and its ONNX file from a directory.
- `loadNodeTier({ tier, cacheDir, registry?, threads?, ... })`: downloads a registry tier into `cacheDir` on first use, then loads it.
- `fsStorage(dir): ModelStorage`: files under `dir`, one per versioned key (`<id>@<version>/model.onnx`).
- Types: `NodeTilawaOptions`, and the shared `Tilawa`, `TilawaOptions`, `TilawaModelInfo`, `LoadProgress`, `Microphone`, `MicrophoneOptions`.

## `@tilawa/core/react-native`

**Experimental: not yet validated on devices.** The API may change in a 1.x minor release.

- `createTilawa(options?: NativeTilawaOptions)`. There is no microphone helper: record with your own module and call `session.pushAudio(samples, sampleRate)`. The default storage is in memory, so pass a persistent `storage` (and `pathOf`) or the model downloads on every launch. Without WebCrypto, downloads are length-checked only.
- `loadNativeModel(options: NativeModelOptions)`: `{ storage, tier?, registry?, pathOf?, providers?, threads?, allowOverBudget?, onProgress? }`. `providers` defaults to `["cpu"]`; for example `["coreml", "cpu"]` on iOS or `["nnapi", "cpu"]` on Android.
- Types: `NativeTilawaOptions`, `NativeModelOptions`, and the shared `Tilawa`, `TilawaOptions`, `TilawaModelInfo`, `LoadProgress`.

## `@tilawa/core/report`

Opt-in report capture. Nothing is kept without a recorder, and nothing is sent without `sendReport`. The full guide is [reporting.md](reporting.md).

- `createRecorder(session, options): Recorder`.
- `Recorder`: `state`, `error`, `mark(input)`, `report(input)`, `contribution(input?)`, `incidents()`, `stats()`, `dispose()`.
- `encodeReport(report): { json, wav }`, `decodeReport(json, wav)`, `reportAudio(report)`, `validateReport(json, wav)`, `encodeWav(pcm: Int16Array)`.
- `sendReport(report, options): Promise<ReportReceipt>`.
- Errors: `ConsentError`, `RecorderError` (`reason`: `disposed`, `expired`, `failed`, `empty`, `limit`, `misuse`), `ReportValidationError` (`errors: string[]`), `ReportSendError` (`status`, `body`).
- Constants: `REPORT_SCHEMA` (`"tilawa.report/1"`), `REPORT_LIMITS`.
- Types: `TilawaReport`, `Incident`, `IncidentKind`, `IncidentClaim`, `ClaimAnswer`, `ReportAudio`, `ReportConsent`, `ReportEngine`, `ReportPassage`, `RecorderOptions`, `RetainOptions`, `MarkInput`, `ReportInput`, `ContributionInput`, `RecorderState`, `RecorderStats`, `RecordableSession`, `SessionState`, `EncodedReport`, `ReportValidation`, `ReportReceipt`, `SendReportOptions`.

## `@tilawa/core/assets/*`

Plain JSON files, for bundling and self-hosting.

| File | Size | sha256 |
|---|---|---|
| `@tilawa/core/assets/ref-v2.json` | 4,749,281 bytes | `f236384ea91e81ad554ba2777bc4073c3ff41e34120343a8a474571c577b7727` |
| `@tilawa/core/assets/phrases-v1.json` | 1,452 bytes | `0277d8dd78d6042795c1332b5a322d08eb71e262a79ce8887b6d66b08c53f688` |

`ref-v2.json` is the Quran reference (Hafs, Uthmani text from Tanzil, phonemized with quran-transcript 0.6.4). `phrases-v1.json` is bundled into the SDK already.

---

## Version policy

Each part is versioned on its own:

| Thing | Versioned by | Rule |
|---|---|---|
| SDK | npm semver | 1.x keeps this API. Changes are additive only. |
| Model | `<id>@<version>` plus sha256 in the bundled registry | The default model changes only in an SDK minor release, with a changelog line. Old model URLs stay up. |
| Reference | `ref-vN` | A new reference is a new file name. A model's phoneme inventory must match its reference. |
| Events | the `v` field (`EVENT_SCHEMA_VERSION`) | Stays `1` for all of 1.x. New optional fields may be added. |
| Report | `schema: "tilawa.report/1"` | Readers reject an unknown major. New optional fields may be added (`tilawa.report/1.<minor>` is accepted). |
| Asset URLs | the `/v1/` path prefix | Files are immutable and never overwritten: `Cache-Control: public, max-age=31536000, immutable`. |

What counts as additive: new exports, new optional options, new optional event or report fields, new values in `unsupported`. Handle unknown fields and unknown future error classes gracefully.

Exceptions: `@tilawa/core/react-native` is experimental until it is validated on devices. Pre-release versions (`1.0.0-rc.x`) may still change before 1.0.0.

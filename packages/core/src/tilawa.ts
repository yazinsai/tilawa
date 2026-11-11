/** Shared `createTilawa` plumbing (contract §2); each platform entry supplies its loaders. Runtime-neutral. */
import { loadReferenceIndex } from "./engine/index.js";
import type { PhraseAsset, ReferenceAsset } from "./engine/index.js";
import phrasesV1 from "./engine/phrases-v1.json" with { type: "json" };
import type { VadOptions } from "./runtime/audio.js";
import type { LoadedModel } from "./runtime/model.js";
import { MODEL_REGISTRY, withBase } from "./runtime/registry.js";
import type { ModelRegistry, ModelStorage, TierName } from "./runtime/models.js";
import { DEFAULTS, REFERENCE } from "./defaults.js";
import { ModelBudgetError, ModelIntegrityError, TilawaLoadError, abortError, isAbortError } from "./errors.js";
import type { LoadStage } from "./errors.js";
import { createEngine, internals } from "./session.js";
import type { CapabilityEvent, EngineEvent, MemorizationPassage, SessionPolicy, TilawaSession } from "./session.js";
import { sha256Hex } from "./util.js";

export const BUNDLED_PHRASES = phrasesV1 as PhraseAsset;

export interface LoadProgress {
  stage: LoadStage;
  loaded: number;
  total: number;
  /** 0..1 within the stage. */
  fraction: number;
  fromCache: boolean;
}

export interface TilawaOptions {
  /** Default "auto": `full`, unless the device reports under 4 GB of memory, or is mobile without WebGPU (`lite`). */
  tier?: TierName | "auto";
  /** Mirror of `https://tilawa.dev/v1/` (the default). */
  assetBase?: string;
  /** The reference asset (`@tilawa/core/assets/ref-v2.json`), for offline use; otherwise fetched from `assetBase`. */
  reference?: ReferenceAsset;
  /** Default: the bundled `phrases-v1`. */
  phrases?: PhraseAsset;
  /** A model you loaded yourself. The app keeps ownership: `dispose()` won't release it. */
  model?: LoadedModel;
  /** Where model (and, on web, reference) bytes are cached. */
  storage?: ModelStorage;
  providers?: string[];
  threads?: number;
  /** Where onnxruntime-web's `.wasm`/`.mjs` files are served. Default `<assetBase>runtime/ort/<ortVersion>/`. */
  wasmPaths?: string;
  onProgress?: (p: LoadProgress) => void;
  /** Cancels the download; `createTilawa` then rejects with an `AbortError`. */
  signal?: AbortSignal;
  /** Keep every session's evidence for `session.evidence()` (bug reports, benchmarks). Default false. */
  recordEvidence?: boolean;
}

export interface TilawaModelInfo {
  id: string;
  /** Null for a model passed in that isn't in the registry. */
  version: string | null;
  sha256: string | null;
  tier: TierName | null;
  backend: string;
  /** Why each earlier execution provider was skipped (web). */
  fallbacks: string[];
}

export interface MicrophoneOptions {
  /** Gate the audio with the energy VAD. Default off: the session clock then matches the wall clock. */
  vad?: VadOptions | boolean;
  /** Raw input level in dBFS, about every 20 ms; for a meter. */
  onLevel?: (db: number) => void;
  onEvents?: (events: EngineEvent[]) => void;
  /** A failed push (inference error). Default: logged, except `SessionEndedError`. */
  onError?: (e: unknown) => void;
  /**
   * Web: a context + stream acquired by `prepareMicrophone()` inside the user's tap. iOS Safari only starts audio
   * created or resumed during a user gesture, so call `prepareMicrophone()` synchronously in the click handler when
   * anything async (such as the first model download) happens before `listen()`. `listen()` takes ownership.
   */
  prepared?: PreparedMicrophone;
}

/** Web only: see `prepareMicrophone()` in `@tilawa/core/web`. */
export interface PreparedMicrophone {
  readonly context: AudioContext;
  readonly stream: Promise<MediaStream>;
  /** Releases the context and tracks if `listen()` is never called. */
  release(): void;
}

export interface Microphone {
  readonly sampleRate: number;
  /** True while the VAD hears speech (or its hangover); always true without the VAD. */
  readonly speaking: boolean;
  /** Stops the tracks, then ends the session: its final events. */
  stop(): Promise<EngineEvent[]>;
}

export interface Tilawa {
  startFreeRecitation(policy?: SessionPolicy): TilawaSession;
  startMemorization(passage: MemorizationPassage, policy?: SessionPolicy): TilawaSession;
  /** Microphone → session (web). Rejects with `MicrophoneError`. */
  listen(session: TilawaSession, options?: MicrophoneOptions): Promise<Microphone>;
  readonly capabilities: CapabilityEvent;
  readonly model: TilawaModelInfo;
  /** Disposes every open session and releases the ONNX session. */
  dispose(): Promise<void>;
}

export interface LoadContext {
  assetBase: string;
  registry: ModelRegistry;
  tier: TierName;
  signal?: AbortSignal;
  fetch: (url: string) => Promise<Response>;
  progress(stage: LoadStage, loaded: number, total: number, fromCache: boolean): void;
}

export interface Platform<O extends TilawaOptions> {
  autoTier(): TierName;
  loadModel(o: O, ctx: LoadContext): Promise<{ model: LoadedModel; info: TilawaModelInfo }>;
  loadReference(o: O, ctx: LoadContext): Promise<ReferenceAsset>;
  listen(session: TilawaSession, options?: MicrophoneOptions): Promise<Microphone>;
}

const passThrough = (e: unknown) =>
  isAbortError(e) || e instanceof ModelIntegrityError || e instanceof ModelBudgetError || e instanceof TilawaLoadError;
const asLoadError = (stage: LoadStage, e: unknown) => (passThrough(e) ? e : new TilawaLoadError(stage, e));

/** The registry entry a model came from, when it came from one. */
export function infoFor(model: LoadedModel, registry: ModelRegistry, fallbacks: string[] = []): TilawaModelInfo {
  const t = registry.tiers.find((x) => x.id === model.manifest.id && x.sha256 === model.manifest.sha256);
  return {
    id: model.manifest.id,
    version: t?.version ?? null,
    sha256: model.manifest.sha256 ?? null,
    tier: t?.tier ?? null,
    backend: model.backend.name,
    fallbacks,
  };
}

export async function createTilawaWith<O extends TilawaOptions>(platform: Platform<O>, o: O): Promise<Tilawa> {
  const signal = o.signal;
  if (signal?.aborted) throw abortError(signal);
  const assetBase = (o.assetBase ?? DEFAULTS.assetBase).replace(/\/?$/, "/");
  const requested = o.tier ?? DEFAULTS.tier;
  const ctx: LoadContext = {
    assetBase,
    registry: withBase(MODEL_REGISTRY, assetBase),
    tier: requested === "auto" ? platform.autoTier() : requested,
    signal,
    fetch: (url) => fetch(url, signal ? { signal } : undefined),
    progress: (stage, loaded, total, fromCache) =>
      o.onProgress?.({ stage, loaded, total, fraction: total > 0 ? Math.min(1, loaded / total) : 1, fromCache }),
  };
  const stage = async <T>(name: LoadStage, f: () => Promise<T>) => {
    try {
      return await f();
    } catch (e) {
      throw asLoadError(name, e);
    }
  };
  const [m, r] = await Promise.allSettled([
    o.model ? Promise.resolve({ model: o.model, info: infoFor(o.model, ctx.registry) }) : stage("model", () => platform.loadModel(o, ctx)),
    o.reference ? Promise.resolve(o.reference) : stage("reference", () => platform.loadReference(o, ctx)),
  ]);
  const owned = !o.model && m.status === "fulfilled" ? m.value.model : null;
  const fail = async (e: unknown) => {
    await owned?.backend.release().catch(() => {});
    throw e;
  };
  if (m.status === "rejected") return fail(m.reason);
  if (r.status === "rejected") return fail(r.reason);
  if (signal?.aborted) return fail(abortError(signal));
  const { model, info } = m.value;

  let engine;
  try {
    engine = createEngine({
      reference: loadReferenceIndex(r.value),
      phrases: o.phrases ?? BUNDLED_PHRASES,
      model,
      recordEvidence: o.recordEvidence,
      info: { backend: info.backend, ...(info.tier && { tier: info.tier }), ...(info.sha256 && { modelSha256: info.sha256 }) },
    });
  } catch (e) {
    return fail(new TilawaLoadError("reference", e));
  }

  const sessions = new Set<TilawaSession>();
  let disposed = false;
  const track = (s: TilawaSession) => {
    sessions.add(s);
    internals.get(s)!.onClose(() => sessions.delete(s));
    return s;
  };
  const live = () => {
    if (disposed) throw new Error("This Tilawa instance was disposed");
  };
  return {
    startFreeRecitation: (policy) => (live(), track(engine.startFreeRecitation(policy))),
    startMemorization: (passage, policy) => (live(), track(engine.startMemorization(passage, policy))),
    listen: (session, options) => platform.listen(session, options),
    get capabilities() {
      return engine.capabilities;
    },
    model: info,
    async dispose() {
      if (disposed) return;
      disposed = true;
      for (const s of [...sessions]) s.dispose();
      if (owned) await owned.backend.release();
    },
  };
}

/** The reference from `<assetBase>reference/ref-v2.json`, kept in `storage` once verified. */
export async function fetchReference(ctx: LoadContext, storage: ModelStorage): Promise<ReferenceAsset> {
  const key = REFERENCE.path;
  let bytes = await storage.get(key);
  let fromCache = true;
  if (!bytes || bytes.length !== REFERENCE.bytes) {
    if (bytes) await storage.delete(key);
    fromCache = false;
    const url = ctx.assetBase + REFERENCE.path;
    const res = await ctx.fetch(url);
    if (!res.ok) throw new Error(`${res.status} fetching ${url}`);
    bytes = await readBody(res, REFERENCE.bytes, (n) => ctx.progress("reference", n, REFERENCE.bytes, false));
    if (bytes.length !== REFERENCE.bytes) throw new ModelIntegrityError(`Truncated ${key}: ${bytes.length} of ${REFERENCE.bytes} bytes`);
    const digest = await sha256Hex(bytes);
    if (digest !== null && digest !== REFERENCE.sha256) throw new ModelIntegrityError(`Checksum mismatch for ${key}`);
    await storage.put(key, bytes);
  }
  ctx.progress("reference", bytes.length, bytes.length, fromCache);
  return JSON.parse(new TextDecoder().decode(bytes)) as ReferenceAsset;
}

async function readBody(res: Response, total: number, onProgress: (n: number) => void): Promise<Uint8Array> {
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const parts: Uint8Array[] = [];
  let n = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    n += value.length;
    if (n > total) throw new ModelIntegrityError(`Download larger than the expected ${total} bytes`);
    onProgress(n);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) out.set(p, o), (o += p.length);
  return out;
}

/** A storage that already holds `bytes` at `key`, so a research loader takes the cached path. */
export function holding(key: string, bytes: Uint8Array): ModelStorage {
  return {
    get: async (k) => (k === key ? bytes : null),
    put: async () => {},
    delete: async () => {},
    keys: async () => [key],
  };
}

export const tierKey = (t: ModelRegistry["tiers"][number]) => `${t.id}@${t.version}/${t.manifest.file}`;

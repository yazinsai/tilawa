/** `@tilawa/core/web`: onnxruntime-web (WebGPU, then WASM), Cache API storage, microphone capture. */
import * as ort from "onnxruntime-web/all";
import { fetchModel, memoryStorage } from "./runtime/models.js";
import type { ModelStorage, TierName } from "./runtime/models.js";
import { cacheStorage, loadWebModel, startMicrophone as startResearchMicrophone } from "./runtime/web.js";
import type { WebProvider } from "./runtime/web.js";
import type { Session as ResearchSession } from "./runtime/session.js";
import { MicrophoneError, SessionEndedError } from "./errors.js";
import { createTilawaWith, fetchReference, holding, tierKey } from "./tilawa.js";
import type { Microphone, MicrophoneOptions, Platform, Tilawa, TilawaOptions } from "./tilawa.js";
import { internals } from "./session.js";
import type { TilawaSession } from "./session.js";
import { DEFAULTS } from "./defaults.js";

export { cacheStorage, loadWebModel };
export type { WebModelOptions, WebProvider } from "./runtime/web.js";
export { decodeWav } from "./runtime/wav.js";
export type { Microphone, MicrophoneOptions, Tilawa, TilawaOptions, LoadProgress, TilawaModelInfo } from "./tilawa.js";

export interface WebTilawaOptions extends TilawaOptions {
  providers?: WebProvider[];
}

/** Model + reference (cached in the Cache API), then a ready `Tilawa`. */
export function createTilawa(options: WebTilawaOptions = {}): Promise<Tilawa> {
  return createTilawaWith(web, options);
}

const defaultStorage = (): ModelStorage =>
  typeof caches !== "undefined" ? cacheStorage(DEFAULTS.cacheName) : memoryStorage();

const web: Platform<WebTilawaOptions> = {
  autoTier(): TierName {
    const nav = (globalThis.navigator ?? {}) as Navigator & { deviceMemory?: number; userAgentData?: { mobile?: boolean } };
    if (typeof nav.deviceMemory === "number" && nav.deviceMemory < 4) return "lite";
    const mobile = nav.userAgentData?.mobile ?? /Android|iPhone|iPad|iPod|Mobile/i.test(nav.userAgent ?? "");
    return mobile && !("gpu" in nav) ? "lite" : "full";
  },
  async loadModel(o, ctx) {
    const storage = o.storage ?? defaultStorage();
    const fetched = await fetchModel({
      tier: ctx.tier,
      registry: ctx.registry,
      storage,
      fetch: ctx.fetch,
      onProgress: (n, total) => ctx.progress("model", n, total, false),
    });
    ctx.progress("model", fetched.bytes.length, fetched.bytes.length, fetched.fromCache);
    ctx.progress("warmup", 0, 1, fetched.fromCache);
    const version = (ort.env as { versions?: { web?: string } }).versions?.web;
    const model = await loadWebModel({
      tier: ctx.tier,
      registry: ctx.registry,
      storage: holding(tierKey(fetched.tier), fetched.bytes),
      providers: o.providers,
      threads: o.threads,
      wasmPaths: o.wasmPaths ?? (version ? `${ctx.assetBase}runtime/ort/${version}/` : undefined),
    });
    ctx.progress("warmup", 1, 1, fetched.fromCache);
    const t = fetched.tier;
    return {
      model,
      info: { id: t.id, version: t.version, sha256: t.sha256, tier: t.tier, backend: model.backend.name, fallbacks: model.fallbacks },
    };
  },
  loadReference: (o, ctx) => fetchReference(ctx, o.storage ?? defaultStorage()),
  listen: startMicrophone,
};

/**
 * Microphone → session: native rate, resampled to 16 kHz, optionally VAD-gated (off by default). Rejects with
 * `MicrophoneError` (`denied`, `unavailable`, `insecure-context`), or `SessionEndedError` for a session that has ended.
 */
export async function startMicrophone(session: TilawaSession, o: MicrophoneOptions = {}): Promise<Microphone> {
  if (session.state !== "active") throw new SessionEndedError("listen() on a session that has ended");
  const nav = globalThis.navigator;
  if (!nav?.mediaDevices?.getUserMedia) {
    if (globalThis.isSecureContext === false)
      throw new MicrophoneError("insecure-context", "The microphone needs a secure context (https or localhost)");
    throw new MicrophoneError("unavailable", "No microphone API in this environment");
  }
  const vad = o.vad === true ? {} : o.vad ?? DEFAULTS.vad;
  // The research capture loop only calls pushAudio and end.
  const target = {
    pushAudio: (samples: Float32Array, rate: number) => session.pushAudio(samples, rate),
    end: () => (session.state === "disposed" ? Promise.resolve([]) : session.end()),
  } as unknown as ResearchSession;
  let mic;
  try {
    mic = await startResearchMicrophone(target, {
      vad,
      onLevel: o.onLevel,
      onEvents: o.onEvents,
      onError: o.onError ?? ((e) => {
        if (!(e instanceof SessionEndedError)) console.error("@tilawa/core: audio push failed", e);
      }),
    });
  } catch (e) {
    throw microphoneError(e);
  }
  internals.get(session)?.setVad(vad !== false);
  return mic;
}

function microphoneError(e: unknown): MicrophoneError {
  const name = e instanceof Error ? e.name : "";
  const message = e instanceof Error ? e.message : String(e);
  if (name === "NotAllowedError" || name === "PermissionDeniedError") return new MicrophoneError("denied", message, { cause: e });
  if (name === "SecurityError")
    return new MicrophoneError(globalThis.isSecureContext === false ? "insecure-context" : "denied", message, { cause: e });
  return new MicrophoneError("unavailable", message, { cause: e });
}

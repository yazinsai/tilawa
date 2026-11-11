/** `@tilawa/core/web`: onnxruntime-web (WebGPU, then WASM), Cache API storage, microphone capture. */
import * as ort from "onnxruntime-web/all";
import { fetchModel, memoryStorage } from "./runtime/models.js";
import type { ModelStorage, TierName } from "./runtime/models.js";
import { cacheStorage, loadWebModel } from "./runtime/web.js";
import { Resampler, Vad } from "./runtime/audio.js";
import type { WebProvider } from "./runtime/web.js";
import { MicrophoneError, SessionEndedError } from "./errors.js";
import { createTilawaWith, fetchReference, holding, tierKey } from "./tilawa.js";
import type { Microphone, MicrophoneOptions, Platform, PreparedMicrophone, Tilawa, TilawaOptions } from "./tilawa.js";
import { internals } from "./session.js";
import type { EngineEvent, TilawaSession } from "./session.js";
import { DEFAULTS } from "./defaults.js";

export { cacheStorage, loadWebModel };
export type { WebModelOptions, WebProvider } from "./runtime/web.js";
export { decodeWav } from "./runtime/wav.js";
export type { Microphone, MicrophoneOptions, PreparedMicrophone, Tilawa, TilawaOptions, LoadProgress, TilawaModelInfo } from "./tilawa.js";

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

const CONSTRAINTS: MediaStreamConstraints = {
  audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
};

// Same tap as the research capture loop (runtime/web.ts): copies each 128-frame input block to the main thread.
const WORKLET = `registerProcessor("tilawa-tap", class extends AudioWorkletProcessor {
  process(inputs) { const c = inputs[0] && inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
});`;

function micApi() {
  const nav = globalThis.navigator;
  if (!nav?.mediaDevices?.getUserMedia) {
    if (globalThis.isSecureContext === false)
      throw new MicrophoneError("insecure-context", "The microphone needs a secure context (https or localhost)");
    throw new MicrophoneError("unavailable", "No microphone API in this environment");
  }
  return nav.mediaDevices;
}

/**
 * Call synchronously inside a click/tap handler: creates and resumes the AudioContext and asks for the microphone
 * while the gesture is still fresh (needed on iOS Safari), so `listen(session, { prepared })` works after awaits.
 */
export function prepareMicrophone(): PreparedMicrophone {
  const devices = micApi();
  const context = new AudioContext();
  void context.resume().catch(() => {});
  const stream = devices.getUserMedia(CONSTRAINTS);
  stream.catch(() => {});
  let released = false;
  return {
    context,
    stream,
    release() {
      if (released) return;
      released = true;
      void stream.then((s) => s.getTracks().forEach((t) => t.stop()), () => {});
      void context.close().catch(() => {});
    },
  };
}

/**
 * Microphone → session: native rate, resampled to 16 kHz, optionally VAD-gated (off by default). Rejects with
 * `MicrophoneError` (`denied`, `unavailable`, `insecure-context`), or `SessionEndedError` for a session that has ended.
 * The loop mirrors the research `startMicrophone`; only the context and stream may be supplied by the caller.
 */
export async function startMicrophone(session: TilawaSession, o: MicrophoneOptions = {}): Promise<Microphone> {
  if (session.state !== "active") {
    o.prepared?.release();
    throw new SessionEndedError("listen() on a session that has ended");
  }
  const vadOptions = o.vad === true ? {} : o.vad ?? DEFAULTS.vad;
  const onError = o.onError ?? ((e: unknown) => {
    if (!(e instanceof SessionEndedError)) console.error("@tilawa/core: audio push failed", e);
  });
  let stream: MediaStream;
  let ctx: AudioContext;
  try {
    if (o.prepared) {
      ctx = o.prepared.context;
      stream = await o.prepared.stream;
    } else {
      const devices = micApi();
      stream = await devices.getUserMedia(CONSTRAINTS);
      ctx = new AudioContext();
    }
    if (ctx.state === "suspended") await ctx.resume().catch(() => {});
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
  } catch (e) {
    o.prepared?.release();
    if (e instanceof MicrophoneError) throw e;
    throw microphoneError(e);
  }
  const source = ctx.createMediaStreamSource(stream);
  const tap = new AudioWorkletNode(ctx, "tilawa-tap");
  const resampler = new Resampler(ctx.sampleRate, 16000);
  const vad = vadOptions === false ? null : new Vad(vadOptions);
  let pending: Promise<unknown> = Promise.resolve();
  let levelAcc = 0, levelN = 0;
  tap.port.onmessage = (ev: MessageEvent<Float32Array>) => {
    const raw = ev.data;
    for (let i = 0; i < raw.length; i++) levelAcc += raw[i] * raw[i];
    levelN += raw.length;
    if (levelN >= ctx.sampleRate / 50) {
      o.onLevel?.(10 * Math.log10(levelAcc / levelN + 1e-12));
      levelAcc = levelN = 0;
    }
    const pcm = resampler.push(raw);
    const gated = vad ? vad.push(pcm) : pcm;
    if (!gated.length) return;
    pending = session.pushAudio(gated, 16000).then((events) => events.length && o.onEvents?.(events), onError);
  };
  source.connect(tap);
  // Some browsers only pull a worklet that reaches the destination; a muted gain keeps it silent.
  const mute = ctx.createGain();
  mute.gain.value = 0;
  tap.connect(mute).connect(ctx.destination);
  internals.get(session)?.setVad(vad !== null);
  let stopping: Promise<EngineEvent[]> | null = null;
  return {
    sampleRate: ctx.sampleRate,
    get speaking() {
      return vad?.active ?? true;
    },
    stop() {
      stopping ??= (async () => {
        tap.port.onmessage = null;
        stream.getTracks().forEach((t) => t.stop());
        await ctx.close().catch(() => {});
        await pending;
        const events = session.state === "disposed" ? [] : await session.end();
        if (events.length) o.onEvents?.(events);
        return events;
      })();
      return stopping;
    },
  };
}

function microphoneError(e: unknown): MicrophoneError {
  const name = e instanceof Error ? e.name : "";
  const message = e instanceof Error ? e.message : String(e);
  if (name === "NotAllowedError" || name === "PermissionDeniedError") return new MicrophoneError("denied", message, { cause: e });
  if (name === "SecurityError")
    return new MicrophoneError(globalThis.isSecureContext === false ? "insecure-context" : "denied", message, { cause: e });
  return new MicrophoneError("unavailable", message, { cause: e });
}

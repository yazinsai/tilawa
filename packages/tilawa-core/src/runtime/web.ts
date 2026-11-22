/** Browser entry: onnxruntime-web (WebGPU, falling back to WASM SIMD), Cache API model storage, microphone capture. */
import * as ort from "onnxruntime-web/all";
import type { EngineEvent } from "../engine/index.js";
import { Resampler, Vad } from "./audio.js";
import type { VadOptions } from "./audio.js";
import type { InferenceBackend, LoadedModel } from "./model.js";
import { fetchModel } from "./models.js";
import type { ModelRegistry, ModelStorage, TierName } from "./models.js";
import type { OrtModule } from "./ort.js";
import { ortBackend } from "./ort.js";
import { MODEL_REGISTRY } from "./registry.js";
import type { Session } from "./session.js";

export * from "./index.js";

export function cacheStorage(name = "tilawa-models"): ModelStorage {
  const url = (k: string) => `https://tilawa.models/${k}`;
  return {
    async get(k) {
      const r = await (await caches.open(name)).match(url(k));
      return r ? new Uint8Array(await r.arrayBuffer()) : null;
    },
    async put(k, b) {
      await (await caches.open(name)).put(url(k), new Response(b as Uint8Array<ArrayBuffer>, { headers: { "content-length": String(b.length) } }));
    },
    async delete(k) {
      await (await caches.open(name)).delete(url(k));
    },
    async keys() {
      return (await (await caches.open(name)).keys()).map((r) => new URL(r.url).pathname.slice(1));
    },
  };
}

export type WebProvider = "webgpu" | "wasm";

export interface WebModelOptions {
  tier?: TierName;
  registry?: ModelRegistry;
  storage?: ModelStorage;
  /** Tried in order; a provider that fails to load or run a first window falls through to the next. */
  providers?: WebProvider[];
  /** WASM threads; above 1 needs cross-origin isolation. Ticket 02: 1 thread is as fast for a model this size. */
  threads?: number;
  /** Where the onnxruntime-web `.wasm`/`.mjs` files are served; self-host them for offline use. */
  wasmPaths?: string;
  allowOverBudget?: boolean;
  onProgress?: (loaded: number, total: number) => void;
}

/** `fallbacks`: why each earlier provider was skipped. */
export async function loadWebModel(o: WebModelOptions = {}): Promise<LoadedModel & { fromCache: boolean; fallbacks: string[] }> {
  const { tier, bytes, fromCache } = await fetchModel({
    tier: o.tier ?? "full",
    registry: o.registry ?? MODEL_REGISTRY,
    storage: o.storage ?? cacheStorage(),
    allowOverBudget: o.allowOverBudget,
    onProgress: o.onProgress,
  });
  if (o.wasmPaths) ort.env.wasm.wasmPaths = o.wasmPaths;
  ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? (o.threads ?? 1) : 1;
  const providers = o.providers ?? ["webgpu", "wasm"];
  const errors: string[] = [];
  for (const p of providers) {
    if (p === "webgpu" && !("gpu" in navigator)) {
      errors.push("webgpu: navigator.gpu missing");
      continue;
    }
    try {
      const backend = await ortBackend(ort as unknown as OrtModule, bytes, `onnxruntime-web/${p}`, { executionProviders: [p] });
      await warmup(backend, tier.manifest);
      return { manifest: tier.manifest, backend, supported: tier.supported, fromCache, fallbacks: errors };
    } catch (e) {
      errors.push(`${p}: ${(e as Error).message}`);
    }
  }
  throw new Error(`No onnxruntime-web provider could run the model (${errors.join("; ")})`);
}

/** One silent window: surfaces kernel failures (ticket 02 hit a WebGPU Concat bug) before the session starts. */
async function warmup(backend: InferenceBackend, m: LoadedModel["manifest"]) {
  const feeds: Parameters<InferenceBackend["run"]>[0] = {
    x: { type: "float32", data: new Float32Array(m.window * m.fbank.mels).fill(m.fbank.padValue), dims: [1, m.window, m.fbank.mels] },
  };
  for (const s of m.states) {
    const n = s.shape.reduce((a, b) => a * b, 1);
    feeds[s.name] = { type: s.dtype, data: s.dtype === "int64" ? new BigInt64Array(n) : new Float32Array(n), dims: s.shape };
  }
  await backend.run(feeds);
}

const WORKLET = `registerProcessor("tilawa-tap", class extends AudioWorkletProcessor {
  process(inputs) { const c = inputs[0] && inputs[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; }
});`;

export interface Microphone {
  readonly sampleRate: number;
  /** True while the VAD hears speech (or its hangover). */
  readonly speaking: boolean;
  /** Stops capture and ends the session: the final events. */
  stop(): Promise<EngineEvent[]>;
}

export interface MicrophoneOptions {
  vad?: VadOptions | false;
  /** Raw input level in dBFS, about every 20 ms; for a meter. */
  onLevel?: (db: number) => void;
  onEvents?: (events: EngineEvent[]) => void;
  onError?: (e: unknown) => void;
}

/**
 * Captures the microphone into a session: native rate, resampled to 16 kHz, gated by the VAD. Browser echo
 * cancellation, noise suppression and AGC are off; they distort recitation more than they help a phoneme model.
 */
export async function startMicrophone(session: Session, o: MicrophoneOptions = {}): Promise<Microphone> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const ctx = new AudioContext();
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);
  const source = ctx.createMediaStreamSource(stream);
  const tap = new AudioWorkletNode(ctx, "tilawa-tap");
  const resampler = new Resampler(ctx.sampleRate, 16000);
  const vad = o.vad === false ? null : new Vad(o.vad);
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
    pending = session.pushAudio(gated, 16000).then((events) => events.length && o.onEvents?.(events), o.onError);
  };
  source.connect(tap);
  // Some browsers only pull a worklet that reaches the destination; a muted gain keeps it silent.
  const mute = ctx.createGain();
  mute.gain.value = 0;
  tap.connect(mute).connect(ctx.destination);
  return {
    sampleRate: ctx.sampleRate,
    get speaking() {
      return vad?.active ?? true;
    },
    async stop() {
      tap.port.onmessage = null;
      stream.getTracks().forEach((t) => t.stop());
      await ctx.close();
      await pending;
      const events = await session.end();
      if (events.length) o.onEvents?.(events);
      return events;
    },
  };
}

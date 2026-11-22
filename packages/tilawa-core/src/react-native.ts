/**
 * `@tilawa/core/react-native`: onnxruntime-react-native. Not advertised until validated on devices (issue 03).
 * The app pushes PCM from its own recorder with `session.pushAudio(samples, sampleRate)`.
 */
import { fetchModel, memoryStorage } from "./runtime/models.js";
import { loadNativeModel } from "./runtime/react-native.js";
import { MicrophoneError } from "./errors.js";
import { createTilawaWith, fetchReference, holding, infoFor, tierKey } from "./tilawa.js";
import type { Platform, Tilawa, TilawaOptions } from "./tilawa.js";

export { loadNativeModel };
export type { NativeModelOptions } from "./runtime/react-native.js";
export type { Tilawa, TilawaOptions, LoadProgress, TilawaModelInfo } from "./tilawa.js";

export interface NativeTilawaOptions extends TilawaOptions {
  /** Path the storage keeps a key at: onnxruntime-react-native loads files faster than bytes across the bridge. */
  pathOf?: (key: string) => string;
}

/** Model + reference (cached in `storage`; in memory if none is given), then a ready `Tilawa`. */
export function createTilawa(options: NativeTilawaOptions = {}): Promise<Tilawa> {
  return createTilawaWith(native, options);
}

const shared = memoryStorage();

const native: Platform<NativeTilawaOptions> = {
  // Contract §2: mobile without WebGPU runs the lite tier.
  autoTier: () => "lite",
  async loadModel(o, ctx) {
    const storage = o.storage ?? shared;
    const fetched = await fetchModel({
      tier: ctx.tier,
      registry: ctx.registry,
      storage,
      fetch: ctx.fetch,
      onProgress: (n, total) => ctx.progress("model", n, total, false),
    });
    ctx.progress("model", fetched.bytes.length, fetched.bytes.length, fetched.fromCache);
    const model = await loadNativeModel({
      tier: ctx.tier,
      registry: ctx.registry,
      storage: holding(tierKey(fetched.tier), fetched.bytes),
      pathOf: o.pathOf,
      providers: o.providers,
      threads: o.threads,
    });
    return { model, info: infoFor(model, ctx.registry) };
  },
  loadReference: (o, ctx) => fetchReference(ctx, o.storage ?? shared),
  async listen() {
    throw new MicrophoneError("unavailable", "No microphone helper on React Native: push PCM with session.pushAudio(samples, sampleRate)");
  },
};

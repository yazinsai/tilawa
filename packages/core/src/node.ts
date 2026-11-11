/** `@tilawa/core/node`: onnxruntime-node, models cached on disk, WAV input, the bundled reference. */
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import { fetchModel } from "./runtime/models.js";
import { fsStorage, loadNodeModel, loadNodeTier, readWav } from "./runtime/node.js";
import { ortBackend } from "./runtime/ort.js";
import type { OrtModule } from "./runtime/ort.js";
import type { ReferenceAsset } from "./engine/index.js";
import { MicrophoneError } from "./errors.js";
import { createTilawaWith, infoFor } from "./tilawa.js";
import type { Platform, Tilawa, TilawaOptions } from "./tilawa.js";

export { fsStorage, loadNodeModel, loadNodeTier, readWav };
export { decodeWav } from "./runtime/wav.js";
export type { Microphone, MicrophoneOptions, Tilawa, TilawaOptions, LoadProgress, TilawaModelInfo } from "./tilawa.js";

export interface NodeTilawaOptions extends TilawaOptions {
  /** Downloaded models live here. Default `$TILAWA_CACHE_DIR`, else `~/.cache/tilawa`. */
  cacheDir?: string;
  /** Load `manifest.json` + its ONNX file from this directory instead of the registry. */
  modelDir?: string;
}

/** Model (downloaded once into `cacheDir`) + the reference shipped in this package, then a ready `Tilawa`. */
export function createTilawa(options: NodeTilawaOptions = {}): Promise<Tilawa> {
  return createTilawaWith(node, options);
}

const node: Platform<NodeTilawaOptions> = {
  autoTier: () => "full",
  async loadModel(o, ctx) {
    if (o.modelDir) {
      const model = await loadNodeModel(o.modelDir, { threads: o.threads });
      ctx.progress("model", 1, 1, true);
      return { model, info: infoFor(model, ctx.registry) };
    }
    const onProgress = (n: number, total: number) => ctx.progress("model", n, total, false);
    if (o.storage) {
      // A custom store: onnxruntime-node takes the bytes.
      const { tier, bytes, fromCache } = await fetchModel({ tier: ctx.tier, registry: ctx.registry, storage: o.storage, fetch: ctx.fetch, onProgress });
      ctx.progress("model", bytes.length, bytes.length, fromCache);
      const backend = await ortBackend(ort as unknown as OrtModule, bytes, "onnxruntime-node", {
        executionProviders: ["cpu"],
        intraOpNumThreads: o.threads ?? 1,
      });
      const model = { manifest: tier.manifest, backend, supported: tier.supported };
      return { model, info: infoFor(model, ctx.registry) };
    }
    const cacheDir = o.cacheDir ?? process.env.TILAWA_CACHE_DIR ?? join(homedir(), ".cache", "tilawa");
    const { bytes, fromCache } = await fetchModel({ tier: ctx.tier, registry: ctx.registry, storage: fsStorage(cacheDir), fetch: ctx.fetch, onProgress });
    ctx.progress("model", bytes.length, bytes.length, fromCache);
    const model = await loadNodeTier({ tier: ctx.tier, registry: ctx.registry, cacheDir, threads: o.threads });
    return { model, info: infoFor(model, ctx.registry) };
  },
  async loadReference(_o, ctx) {
    const bytes = await readFile(new URL("../assets/ref-v2.json", import.meta.url));
    ctx.progress("reference", bytes.length, bytes.length, true);
    return JSON.parse(bytes.toString("utf8")) as ReferenceAsset;
  },
  async listen() {
    throw new MicrophoneError("unavailable", "No microphone helper on Node: push PCM with session.pushAudio(samples, sampleRate)");
  },
};

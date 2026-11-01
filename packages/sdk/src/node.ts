/** Node entry: onnxruntime-node, models from a directory, WAV input. */
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import * as ort from "onnxruntime-node";
import { MEMORIZATION_CLASSES } from "./capabilities.js";
import type { LoadedModel, ModelManifest } from "./model.js";
import { fetchModel } from "./models.js";
import type { FetchModelOptions, ModelRegistry, ModelStorage } from "./models.js";
import { MODEL_REGISTRY } from "./registry.js";
import type { OrtModule } from "./ort.js";
import { ortBackend } from "./ort.js";
import { decodeWav } from "./wav.js";
import { createHash } from "node:crypto";
import { createWordEmbedding } from "./word-embedding.js";
import type { WordEmbeddingAsset } from "./word-embedding.js";

export * from "./index.js";
export { decodeWav };

/** Loads `manifest.json` and its ONNX file from `dir`. */
export async function loadNodeModel(dir: string, options: { threads?: number } = {}): Promise<LoadedModel> {
  const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8")) as ModelManifest & { supported?: LoadedModel["supported"] };
  const backend = await ortBackend(ort as unknown as OrtModule, join(dir, manifest.file), "onnxruntime-node", {
    executionProviders: ["cpu"],
    intraOpNumThreads: options.threads ?? 1,
  });
  let wordEmbedding: LoadedModel["wordEmbedding"];
  if (manifest.wordEmbedding) {
    const readChecked = async (spec: { file: string; sha256: string }) => {
      const bytes = await readFile(join(dir, spec.file));
      if (createHash("sha256").update(bytes).digest("hex") !== spec.sha256) throw new Error(`Word embedding hash mismatch: ${spec.file}`);
      return bytes;
    };
    try {
      const asset = JSON.parse((await readChecked(manifest.wordEmbedding)).toString("utf8")) as WordEmbeddingAsset;
      const [encoder, words, prefixes, keys] = await Promise.all([
        readChecked(asset.encoder), readChecked(asset.words), readChecked(asset.prefixes), readChecked(asset.prefixKeys),
      ]);
      const wordBackend = await ortBackend(ort as unknown as OrtModule, encoder, "onnxruntime-node", {
        executionProviders: ["cpu"], intraOpNumThreads: options.threads ?? 1,
      });
      const buffer = (b: Uint8Array) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
      try {
        wordEmbedding = createWordEmbedding(asset, wordBackend, new Float32Array(buffer(words)), new Float32Array(buffer(prefixes)), new Uint32Array(buffer(keys)));
      } catch (error) { await wordBackend.release(); throw error; }
    } catch (error) { await backend.release(); throw error; }
    const release = backend.release.bind(backend);
    backend.release = async () => { await wordEmbedding!.release(); await release(); };
  }
  return { manifest, backend, supported: manifest.supported ?? MEMORIZATION_CLASSES, ...(wordEmbedding && { wordEmbedding }) };
}

export async function readWav(path: string) {
  return decodeWav(new Uint8Array(await readFile(path)));
}

/** Model files under `dir`, one path per versioned key. */
export function fsStorage(dir: string): ModelStorage {
  const path = (k: string) => join(dir, k);
  return {
    async get(k) {
      try {
        return new Uint8Array(await readFile(path(k)));
      } catch {
        return null;
      }
    },
    async put(k, b) {
      await mkdir(dirname(path(k)), { recursive: true });
      await writeFile(path(k), b);
    },
    async delete(k) {
      await rm(path(k), { force: true });
    },
    async keys() {
      const out: string[] = [];
      for (const e of await readdir(dir, { recursive: true, withFileTypes: true }).catch(() => [])) {
        if (e.isFile()) out.push(relative(dir, join(e.parentPath, e.name)));
      }
      return out;
    },
  };
}

/** Downloads a registry tier into `cacheDir` on first use, then loads it from there. */
export async function loadNodeTier(o: Omit<FetchModelOptions, "storage" | "registry"> & { registry?: ModelRegistry; cacheDir: string; threads?: number }): Promise<LoadedModel> {
  const storage = fsStorage(o.cacheDir);
  const { tier } = await fetchModel({ ...o, registry: o.registry ?? MODEL_REGISTRY, storage });
  const backend = await ortBackend(ort as unknown as OrtModule, join(o.cacheDir, `${tier.id}@${tier.version}`, tier.manifest.file), "onnxruntime-node", {
    executionProviders: ["cpu"],
    intraOpNumThreads: o.threads ?? 1,
  });
  return { manifest: tier.manifest, backend, supported: tier.supported };
}

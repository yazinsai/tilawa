/** Node entry: onnxruntime-node, models from a directory, WAV input. */
import { access, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
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
import { createInsertionTeacher } from "./insertion-teacher.js";
import { createWordDecoder, type WordTokenizer } from "./word-decoder.js";
import { createSslEar, type SslEarGeometry } from "./ssl-ear.js";

export * from "./index.js";
export { decodeWav };

/** Loads `manifest.json` and its ONNX file from `dir`. */
export async function loadNodeModel(dir: string, options: { threads?: number } = {}): Promise<LoadedModel> {
  const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8")) as ModelManifest & { supported?: LoadedModel["supported"] };
  const backend = await ortBackend(ort as unknown as OrtModule, join(dir, manifest.file), "onnxruntime-node", {
    executionProviders: ["cpu"],
    intraOpNumThreads: options.threads ?? 1,
  });
  return attachWordSslEars(await attachInsertionTeacher({ manifest, backend, supported: manifest.supported ?? MEMORIZATION_CLASSES },
    dir, options.threads), dir, options.threads);
}

/** The student's normal directory loader resolves declared auxiliary assets from the same bundle. */
async function attachInsertionTeacher(model: LoadedModel, dir: string, threads = 1): Promise<LoadedModel> {
  const asset = MODEL_REGISTRY.auxiliary?.find(a => a.purpose === "seam-insertion");
  if (!asset) return model;
  const installed = await access(join(dir, asset.file)).then(() => true,
    e => { if (e.code === "ENOENT") return false; throw e; });
  if (!installed) return model;
  if (asset.inventory !== model.manifest.inventory) throw new Error("Insertion teacher inventory differs from the primary model");
  for (const data of asset.externalData) await access(join(dir, data.file));
  let auxiliary: Promise<ReturnType<typeof createInsertionTeacher>> | undefined;
  let teacherBackend: import("./model.js").InferenceBackend | undefined;
  const release = model.backend.release.bind(model.backend);
  model.backend.release = async () => { if (auxiliary) await auxiliary; await teacherBackend?.release(); await release(); };
  model.insertionTeacher = { async decode(samples, k) {
    auxiliary ??= ortBackend(ort as unknown as OrtModule, join(dir, asset.file), "onnxruntime-node", {
      executionProviders: ["cpu"], intraOpNumThreads: threads, graphOptimizationLevel: "disabled",
    }).then(b => { teacherBackend = b; return createInsertionTeacher(b, model.manifest.tokens, model.manifest.inventory); });
    return (await auxiliary).decode(samples, k);
  } };
  return model;
}

export async function readWav(path: string) {
  return decodeWav(new Uint8Array(await readFile(path)));
}

/** Install the two independently trained seam ears together. Both are lazy;
 * their inference and release never replace the primary streaming backend. */
async function attachWordSslEars(model: LoadedModel, dir: string, threads = 1): Promise<LoadedModel> {
  const word = MODEL_REGISTRY.wordEar, ssl = MODEL_REGISTRY.sslEar;
  if (!word || !ssl) return model;
  const installed = async (file: string) => access(join(dir, file)).then(() => true,
    e => { if (e.code === "ENOENT") return false; throw e; });
  if (!await installed(word.encoder.file) || !await installed(ssl.model.file)) return model;
  for (const file of [word.decoder, word.tokenizer, ssl.spec, ...ssl.externalData]) await access(join(dir, file.file));
  const tokenizer = JSON.parse(await readFile(join(dir, word.tokenizer.file), "utf8")) as WordTokenizer;
  const spec = JSON.parse(await readFile(join(dir, ssl.spec.file), "utf8")) as {
    tokens: string; inventory: string; geometry: SslEarGeometry;
  };
  if (ssl.inventory !== model.manifest.inventory || spec.inventory !== model.manifest.inventory || spec.tokens !== model.manifest.tokens)
    throw new Error("SSL ear inventory differs from the primary model");
  let wordDecoder: Promise<ReturnType<typeof createWordDecoder>> | undefined;
  let sslDecoder: Promise<ReturnType<typeof createSslEar>> | undefined;
  const backends: import("./model.js").InferenceBackend[] = [];
  const open = async (file: string) => {
    const b = await ortBackend(ort as unknown as OrtModule, join(dir, file), "onnxruntime-node", {
      executionProviders: ["cpu"], intraOpNumThreads: threads,
    });
    backends.push(b);
    return b;
  };
  const release = model.backend.release.bind(model.backend);
  model.backend.release = async () => {
    try { await Promise.all([wordDecoder, sslDecoder]); }
    finally {
      try { await Promise.all(backends.map(b => b.release())); }
      finally { await release(); }
    }
  };
  model.wordEar = { async decode(samples) {
    wordDecoder ??= (async () => createWordDecoder(await open(word.encoder.file), await open(word.decoder.file), tokenizer))();
    return (await wordDecoder).decode(samples);
  } };
  model.sslEar = { async decodeSeams(samples, spans) {
    sslDecoder ??= open(ssl.model.file).then(b => createSslEar(b, spec.tokens, spec.inventory, spec.geometry));
    return (await sslDecoder).decodeSeams(samples, spans);
  } };
  return model;
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
  const dir = join(o.cacheDir, `${tier.id}@${tier.version}`);
  return attachWordSslEars(await attachInsertionTeacher({ manifest: tier.manifest, backend, supported: tier.supported },
    dir, o.threads), dir, o.threads);
}

/** Node entry: onnxruntime-node, models from a directory, WAV input. */
import { createHash } from "node:crypto";
import { validateSparseFactorModel } from "./sparse-factor.js";
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

export * from "./index.js";
export { decodeWav };

/** Loads `manifest.json` and its ONNX file from `dir`. */
export async function loadNodeModel(dir: string, options: { threads?: number } = {}): Promise<LoadedModel> {
  const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8")) as ModelManifest & { supported?: LoadedModel["supported"] };
  const backend = await ortBackend(ort as unknown as OrtModule, join(dir, manifest.file), "onnxruntime-node", {
    executionProviders: ["cpu"],
    intraOpNumThreads: options.threads ?? 1,
  });
  return attachSparseFactorModel(await attachInsertionTeacher({ manifest, backend, supported: manifest.supported ?? MEMORIZATION_CLASSES }, dir, options.threads), dir, options.threads);
}

/** A declared side bundle must be complete, hashed and tied to this exact primary graph. */
async function attachSparseFactorModel(model: LoadedModel, dir: string, threads = 1): Promise<LoadedModel> {
  try {
    const descriptor = await readFile(join(dir, "sparse-factor-asset.json"), "utf8").catch(e => {
      if (e.code === "ENOENT") return null;
      throw e;
    });
    if (descriptor === null) return model;
    const asset = JSON.parse(descriptor);
    if (asset.schema !== 1 || !Array.isArray(asset.files) || asset.files.length !== 2 ||
      asset.files[0].file !== "sparse-factor-readouts.json" || asset.files[1].file !== "sparse-factor.onnx")
      throw new Error("Invalid sparse factor asset descriptor");
    const files: Buffer[] = [];
    for (const entry of asset.files) {
      if (!Number.isSafeInteger(entry.bytes) || entry.bytes <= 0 || !/^[0-9a-f]{64}$/.test(entry.sha256))
        throw new Error("Invalid sparse factor file declaration");
      const bytes = await readFile(join(dir, entry.file));
      if (bytes.length !== entry.bytes || createHash("sha256").update(bytes).digest("hex") !== entry.sha256)
        throw new Error("Sparse factor asset checksum mismatch");
      files.push(bytes);
    }
    if (model.manifest.sampleRate !== 16000 || model.manifest.frameMs !== 40)
      throw new Error("Sparse factor requires 16 kHz / 40 ms geometry");
    const primaryBytes = await readFile(join(dir, model.manifest.file));
    if (createHash("sha256").update(primaryBytes).digest("hex") !== model.manifest.sha256)
      throw new Error("Sparse factor primary model checksum mismatch");
    const fitted = validateSparseFactorModel(JSON.parse(files[0].toString("utf8")), model.manifest);
    const extractor = await ortBackend(ort as unknown as OrtModule, join(dir, "sparse-factor.onnx"), "onnxruntime-node", {
      executionProviders: ["cpu"], intraOpNumThreads: threads,
    });
    const release = model.backend.release.bind(model.backend);
    model.backend.release = async () => { await extractor.release(); await release(); };
    model.sparseFactor = { model: fitted, backend: extractor };
    return model;
  } catch (e) { await model.backend.release(); throw e; }
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
  return attachSparseFactorModel(await attachInsertionTeacher({ manifest: tier.manifest, backend, supported: tier.supported },
    join(o.cacheDir, `${tier.id}@${tier.version}`), o.threads), join(o.cacheDir, `${tier.id}@${tier.version}`), o.threads);
}

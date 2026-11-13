/** Node entry: onnxruntime-node, models from a directory, WAV input. */
import { access, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import { createHash } from "node:crypto";
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
import { validateBasinReadouts } from "./connected-basin.js";

export * from "./index.js";
export { decodeWav };

/** Loads `manifest.json` and its ONNX file from `dir`. */
export async function loadNodeModel(dir: string, options: { threads?: number } = {}): Promise<LoadedModel> {
  const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8")) as ModelManifest & { supported?: LoadedModel["supported"] };
  const backend = await ortBackend(ort as unknown as OrtModule, join(dir, manifest.file), "onnxruntime-node", {
    executionProviders: ["cpu"],
    intraOpNumThreads: options.threads ?? 1,
  });
  return attachBasinReadout(await attachInsertionTeacher({ manifest, backend, supported: manifest.supported ?? MEMORIZATION_CLASSES }, dir, options.threads), dir, options.threads);
}

/** An installed descriptor is a declaration: missing or corrupt side files fail loading. */
async function attachBasinReadout(model: LoadedModel, dir: string, threads = 1): Promise<LoadedModel> {
  const raw = await readFile(join(dir, "connected-basin-asset.json"), "utf8").catch(e => {
    if (e.code === "ENOENT") return null;
    throw e;
  });
  if (raw === null) return model;
  const asset = JSON.parse(raw) as {schema: number; primarySha256: string; files: Record<string, {bytes: number; sha256: string}>};
  if (asset.schema !== 1 || asset.primarySha256 !== model.manifest.sha256 || !asset.files ||
      Object.keys(asset.files).sort().join(",") !== "encoder-features.onnx,readouts.json")
    throw new Error("Invalid connected-basin asset declaration");
  const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");
  if (sha(await readFile(join(dir, model.manifest.file))) !== model.manifest.sha256)
    throw new Error("Connected-basin primary checksum mismatch");
  let readouts: unknown;
  for (const file of ["encoder-features.onnx", "readouts.json"]) {
    const spec = asset.files[file], bytes = await readFile(join(dir, file));
    if (!Number.isSafeInteger(spec.bytes) || spec.bytes <= 0 || bytes.length !== spec.bytes || sha(bytes) !== spec.sha256)
      throw new Error("Connected-basin asset checksum mismatch: "+file);
    if (file === "readouts.json") readouts = JSON.parse(bytes.toString("utf8"));
  }
  const readout = validateBasinReadouts(readouts, model.manifest);
  const backend = await ortBackend(ort as unknown as OrtModule, join(dir, "encoder-features.onnx"), "onnxruntime-node", {
    executionProviders: ["cpu"], intraOpNumThreads: threads,
  });
  model.connectedBasin = {model: readout, backend};
  const release = model.backend.release.bind(model.backend);
  model.backend.release = async () => { await backend.release(); await release(); };
  return model;
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
  const dir = join(o.cacheDir, `${tier.id}@${tier.version}`);
  return attachBasinReadout(await attachInsertionTeacher({ manifest: tier.manifest, backend, supported: tier.supported }, dir, o.threads), dir, o.threads);
}

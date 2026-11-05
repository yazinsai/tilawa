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

export * from "./index.js";
export { decodeWav };

/** Loads `manifest.json` and its ONNX file from `dir`. */
export async function loadNodeModel(dir: string, options: { threads?: number } = {}): Promise<LoadedModel> {
  const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8")) as ModelManifest & { supported?: LoadedModel["supported"] };
  const backend = await ortBackend(ort as unknown as OrtModule, join(dir, manifest.file), "onnxruntime-node", {
    executionProviders: ["cpu"],
    intraOpNumThreads: options.threads ?? 1,
  });
  return attachInsertionTeacher({ manifest, backend, supported: manifest.supported ?? MEMORIZATION_CLASSES }, dir, options.threads);
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
  return attachInsertionTeacher({ manifest: tier.manifest, backend, supported: tier.supported },
    join(o.cacheDir, `${tier.id}@${tier.version}`), o.threads);
}

/** Node entry: onnxruntime-node, models from a directory, WAV input. */
import { access, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import { createHash } from "node:crypto";
import * as ort from "onnxruntime-node";
import { MEMORIZATION_CLASSES } from "./capabilities.js";
import type { InferenceBackend, LoadedModel, ModelManifest } from "./model.js";
import { fetchModel } from "./models.js";
import type { FetchModelOptions, ModelRegistry, ModelStorage } from "./models.js";
import { MODEL_REGISTRY } from "./registry.js";
import type { OrtModule } from "./ort.js";
import { ortBackend } from "./ort.js";
import { decodeWav } from "./wav.js";
import { createInsertionTeacher } from "./insertion-teacher.js";
import { encodeFrozenWords } from "./edit-localization.js";

export * from "./index.js";
export { decodeWav };

/** Loads `manifest.json` and its ONNX file from `dir`. */
export async function loadNodeModel(dir: string, options: { threads?: number } = {}): Promise<LoadedModel> {
  const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8")) as ModelManifest & { supported?: LoadedModel["supported"] };
  const backend = await ortBackend(ort as unknown as OrtModule, join(dir, manifest.file), "onnxruntime-node", {
    executionProviders: ["cpu"],
    intraOpNumThreads: options.threads ?? 1,
  });
  try {
    return await attachEditLocalization(await attachInsertionTeacher({ manifest, backend, supported: manifest.supported ?? MEMORIZATION_CLASSES }, dir, options.threads), dir, options.threads);
  } catch (error) {
    await backend.release?.();
    throw error;
  }
}

/** A declaration is a promise of a complete trained bundle, never an optional
 * missing-file fallback. Models without a declaration retain the ordinary path. */
async function attachEditLocalization(model: LoadedModel, dir: string, threads = 1): Promise<LoadedModel> {
  const declaration = model.manifest.editLocalization;
  if (!declaration) return model;
  const probabilities = [declaration.preservation, declaration.independentPreservation];
  if (probabilities.some(p => !Number.isFinite(p) || p <= 0 || p > 1) ||
    !Number.isInteger(declaration.featureDim) || declaration.featureDim <= 0)
    throw new Error("Invalid edit-localization declaration");
  for (const asset of [declaration.feature,declaration.head]) {
    if (!asset.file || /^(\/|\\)|(^|[\/\\])\.\.([\/\\]|$)/.test(asset.file)) throw new Error("Localization asset must be inside its bundle");
    const bytes = await readFile(join(dir,asset.file));
    if (createHash("sha256").update(bytes).digest("hex") !== asset.sha256) throw new Error(`Localization asset hash mismatch: ${asset.file}`);
  }
  let feature: InferenceBackend | undefined, head: InferenceBackend | undefined;
  let loading: Promise<void> | undefined;
  const load = () => loading ??= (async () => {
    feature = await ortBackend(ort as unknown as OrtModule,join(dir,declaration.feature.file),"onnxruntime-node", {
      executionProviders:["cpu"],intraOpNumThreads:threads });
    head = await ortBackend(ort as unknown as OrtModule,join(dir,declaration.head.file),"onnxruntime-node", {
      executionProviders:["cpu"],intraOpNumThreads:threads });
  })();
  const release = model.backend.release.bind(model.backend);
  model.backend.release = async () => {
    try { if (loading) await loading; } finally { await head?.release(); await feature?.release(); await release(); }
  };
  model.editLocalization = {
    preservation:declaration.preservation, independentPreservation:declaration.independentPreservation,
    async encode(audio) { await load(); return encodeFrozenWords(feature!,model.manifest,audio); },
    async score(features,words) {
      if (features.length !== words*declaration.featureDim || !words) throw new Error("Localization feature shape differs from trained head");
      await load();
      const out = await head!.run({ x:{type:"float32",data:features,dims:[words,declaration.featureDim]} });
      const contextual = Array.from(out.contextual.data as Float32Array), independent = Array.from(out.independent.data as Float32Array);
      if (contextual.length !== words || independent.length !== words) throw new Error("Localization output differs from word count");
      return {contextual,independent};
    },
  };
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
  return attachEditLocalization(await attachInsertionTeacher({ manifest: tier.manifest, backend, supported: tier.supported },
    dir, o.threads), dir, o.threads);
}

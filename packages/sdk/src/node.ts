/** Node entry: onnxruntime-node, models from a directory, WAV input. */
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import { MEMORIZATION_CLASSES } from "./capabilities.js";
import type { LoadedModel, ModelManifest } from "./model.js";
import type { OrtModule } from "./ort.js";
import { ortBackend } from "./ort.js";
import { decodeWav } from "./wav.js";

export * from "./index.js";
export { decodeWav };

/** Loads `manifest.json` and its ONNX file from `dir`. */
export async function loadNodeModel(dir: string, options: { threads?: number } = {}): Promise<LoadedModel> {
  const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8")) as ModelManifest & { supported?: LoadedModel["supported"] };
  const backend = await ortBackend(ort as unknown as OrtModule, join(dir, manifest.file), "onnxruntime-node", {
    executionProviders: ["cpu"],
    intraOpNumThreads: options.threads ?? 1,
  });
  return { manifest, backend, supported: manifest.supported ?? MEMORIZATION_CLASSES };
}

export async function readWav(path: string) {
  return decodeWav(new Uint8Array(await readFile(path)));
}

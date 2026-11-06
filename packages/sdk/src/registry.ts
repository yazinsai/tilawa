import type { ModelRegistry } from "./models.js";
import registry from "./models-v1.json" with { type: "json" };

/** The shipped tiers. `base` is relative; point it at wherever the app hosts the files with `withBase`. */
export const MODEL_REGISTRY: ModelRegistry = { ...registry as ModelRegistry, pauseMask: {
  id: "pause-mask-trained", precision: "int8", inventory: "muaalem-v3_2-phonemes",
  file: "pause-mask/model.onnx", manifestFile: "pause-mask/manifest.json", bytes: 72403248,
  sha256: "22c8e197f2e60955f0b7794a8198db4d5ae9ee57aa679d3fc163075092fd8c58",
}, auxiliary: [{
  id: "muaalem-v3_2-insertion", purpose: "seam-insertion", precision: "float32",
  inventory: "muaalem-v3_2-phonemes", file: "teacher.onnx", bytes: 11342825,
  sha256: "5ee50c2b14d7e8e4808ee4adc33e6114f73481b607291136992a5d45444a1e18",
  externalData: [{ file: "teacher.onnx.data", bytes: 2422885632,
    sha256: "d3f24efba836219f0c556ad25e3a1046f9caf36b2ed639cd283f795c78da58b4" }],
}] };

export function withBase(r: ModelRegistry, base: string): ModelRegistry {
  const root = base.endsWith("/") ? base : `${base}/`;
  return { ...r, tiers: r.tiers.map((t) => ({ ...t, base: /^[a-z]+:/.test(t.base) ? t.base : root + t.base })) };
}

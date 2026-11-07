import type { ModelRegistry } from "./models.js";
import registry from "./models-v1.json" with { type: "json" };

/** The shipped tiers. `base` is relative; point it at wherever the app hosts the files with `withBase`. */
export const MODEL_REGISTRY: ModelRegistry = { ...registry as ModelRegistry, auxiliary: [{
  id: "muaalem-v3_2-insertion", purpose: "seam-insertion", precision: "float32",
  inventory: "muaalem-v3_2-phonemes", file: "teacher.onnx", bytes: 11342825,
  sha256: "5ee50c2b14d7e8e4808ee4adc33e6114f73481b607291136992a5d45444a1e18",
  externalData: [{ file: "teacher.onnx.data", bytes: 2422885632,
    sha256: "d3f24efba836219f0c556ad25e3a1046f9caf36b2ed639cd283f795c78da58b4" }],
}], scorers: [{
  id: "within-band-residual-v1", purpose: "wrong-word-spectral", inventory: "muaalem-v3_2-phonemes",
  file: "spectral-residual.json", bytes: 1165426,
  sha256: "36d97e045ac13fd3afc145c15848c6effa7993f347efabbb896a3d6d2d068a8e",
}] };

export function withBase(r: ModelRegistry, base: string): ModelRegistry {
  const root = base.endsWith("/") ? base : `${base}/`;
  return { ...r, tiers: r.tiers.map((t) => ({ ...t, base: /^[a-z]+:/.test(t.base) ? t.base : root + t.base })) };
}

import type { ModelRegistry } from "./models.js";
import registry from "./models-v1.json" with { type: "json" };

/** The shipped tiers. `base` is relative; point it at wherever the app hosts the files with `withBase`. */
export const MODEL_REGISTRY: ModelRegistry = { ...registry as ModelRegistry, auxiliary: [{
  id: "muaalem-v3_2-insertion", purpose: "seam-insertion", precision: "float32",
  inventory: "muaalem-v3_2-phonemes", file: "teacher.onnx", bytes: 11342825,
  sha256: "5ee50c2b14d7e8e4808ee4adc33e6114f73481b607291136992a5d45444a1e18",
  externalData: [{ file: "teacher.onnx.data", bytes: 2422885632,
    sha256: "d3f24efba836219f0c556ad25e3a1046f9caf36b2ed639cd283f795c78da58b4" }],
}], wordEar: {
  id: "short-crop-whisper-small-word-ear", purpose: "seam-word-candidates", precision: "int8",
  encoder: { file: "word-encoder.onnx", bytes: 99236529,
    sha256: "ad4ab2311b9a43c74cc322ed3956001630a2a11a22d3de747e7a29e0741c6319" },
  decoder: { file: "word-decoder.onnx", bytes: 315826781,
    sha256: "d03c3d61272ffc7b009bcf8b9ddd85ee357d14c9eaacb8d096aa17330dc7bbd5" },
  tokenizer: { file: "word-tokenizer.json", bytes: 627413,
    sha256: "28887b7ba1457da64400f0479ad28d2b4c6cfa56e7fe5355349479296d60774a" },
}, sslEar: {
  id: "undistilled-ssl-plain-ctc-v4", purpose: "seam-word-agreement", precision: "int8",
  inventory: "muaalem-v3_2-phonemes", spec: { file: "ssl-export.json" },
  model: { file: "ssl-ear.onnx", bytes: 1599190,
    sha256: "326d3816a70aaaae7747b18ae148e665cd6bddf1c062e9bf29970d1e9d862578" },
  externalData: [{ file: "ssl-ear.onnx.data", bytes: 811752584,
    sha256: "88972ee65dc049e25218e037e1e8c69763c63cca84147f4f057b07fcc51391db" }],
} };

export function withBase(r: ModelRegistry, base: string): ModelRegistry {
  const root = base.endsWith("/") ? base : `${base}/`;
  return { ...r, tiers: r.tiers.map((t) => ({ ...t, base: /^[a-z]+:/.test(t.base) ? t.base : root + t.base })) };
}

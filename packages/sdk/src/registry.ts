import type { ModelRegistry } from "./models.js";
import registry from "./models-v1.json" with { type: "json" };

/** The shipped tiers. `base` is relative; point it at wherever the app hosts the files with `withBase`. */
export const MODEL_REGISTRY = registry as ModelRegistry;

export function withBase(r: ModelRegistry, base: string): ModelRegistry {
  const root = base.endsWith("/") ? base : `${base}/`;
  return { ...r, tiers: r.tiers.map((t) => ({ ...t, base: /^[a-z]+:/.test(t.base) ? t.base : root + t.base })) };
}

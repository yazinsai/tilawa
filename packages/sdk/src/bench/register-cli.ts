/**
 * `pnpm register --tier full --model <dir> --version 1-int8 --note '...'`: writes a model export's manifest into the
 * shipped registry (`src/models-v1.json`) as that tier, replacing the tier's previous version.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import type { ModelRegistry, ModelTier, TierName } from "../models.js";

const { values } = parseArgs({ options: { tier: { type: "string" }, model: { type: "string" }, version: { type: "string" }, note: { type: "string" } } });
if (!values.tier || !values.model || !values.version) throw new Error("--tier, --model and --version are required");
const path = new URL("../models-v1.json", import.meta.url);
const registry = JSON.parse(readFileSync(path, "utf8")) as ModelRegistry;
const manifest = JSON.parse(readFileSync(resolve(values.model, "manifest.json"), "utf8"));
const tier: ModelTier = {
  tier: values.tier as TierName,
  id: manifest.id,
  version: values.version,
  base: `models/${manifest.id}/${values.version}/`,
  bytes: manifest.bytes,
  sha256: manifest.sha256,
  supported: manifest.supported,
  manifest,
  ...(values.note ? { note: values.note } : {}),
};
const order: TierName[] = ["lite", "full"];
registry.tiers = [...registry.tiers.filter((t) => t.tier !== tier.tier), tier].sort((a, b) => order.indexOf(a.tier) - order.indexOf(b.tier));
writeFileSync(path, JSON.stringify(registry, null, 1) + "\n");
console.log(`${tier.tier}: ${tier.id}@${tier.version}, ${(tier.bytes / 1e6).toFixed(1)} MB`);

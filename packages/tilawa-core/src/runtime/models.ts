import type { ErrorClass } from "../engine/index.js";
import type { ModelManifest } from "./model.js";

/** PLAN.md: every tier a device holds, together, within 200 MB. */
export const MODEL_BUDGET_BYTES = 200 * 1024 * 1024;

export type TierName = "lite" | "full";

export interface ModelTier {
  tier: TierName;
  id: string;
  /** Versioned separately from the SDK; a new version replaces the cached one. */
  version: string;
  /** URL prefix of the model files; `manifest.file` is fetched from here. */
  base: string;
  bytes: number;
  sha256: string;
  supported: ErrorClass[];
  manifest: ModelManifest;
  /** Shown to developers; e.g. that the tier is over budget until it is quantised. */
  note?: string;
}

export interface ModelRegistry {
  schema: 1;
  tiers: ModelTier[];
  auxiliary?: {
    id: string; purpose: "seam-insertion"; precision: "float32"; inventory: string;
    file: string; bytes: number; sha256: string;
    externalData: { file: string; bytes: number; sha256: string }[];
  }[];
}

/** Persistent bytes by key: Cache API on web, a directory on Node, an app-provided store on React Native. */
export interface ModelStorage {
  get(key: string): Promise<Uint8Array | null>;
  put(key: string, bytes: Uint8Array): Promise<void>;
  delete(key: string): Promise<void>;
  keys(): Promise<string[]>;
}

export class ModelBudgetError extends Error {}
export class ModelIntegrityError extends Error {}

export interface FetchModelOptions {
  tier: TierName;
  registry: ModelRegistry;
  storage: ModelStorage;
  fetch?: (url: string) => Promise<Response>;
  onProgress?: (loaded: number, total: number) => void;
  /** Load a tier that doesn't fit the budget (the fp32 export before ticket 10 quantises it). Default false. */
  allowOverBudget?: boolean;
  budgetBytes?: number;
}

export interface FetchedModel {
  tier: ModelTier;
  bytes: Uint8Array;
  fromCache: boolean;
}

const keyOf = (t: ModelTier) => `${t.id}@${t.version}/${t.manifest.file}`;

export async function fetchModel(o: FetchModelOptions): Promise<FetchedModel> {
  const tier = o.registry.tiers.find((t) => t.tier === o.tier);
  if (!tier) throw new Error(`No ${o.tier} tier in the model registry`);
  const key = keyOf(tier);
  const budget = o.budgetBytes ?? MODEL_BUDGET_BYTES;
  const others = o.registry.tiers.filter((t) => t.tier !== o.tier);
  const held = (await o.storage.keys()).filter((k) => k !== key);
  const heldOther = others.filter((t) => held.includes(keyOf(t))).reduce((n, t) => n + t.bytes, 0);
  if (!o.allowOverBudget && tier.bytes + heldOther > budget)
    throw new ModelBudgetError(`${tier.id}@${tier.version} (${mb(tier.bytes)}) plus cached tiers (${mb(heldOther)}) exceeds the ${mb(budget)} model budget`);

  const cached = await o.storage.get(key);
  if (cached && cached.length === tier.bytes) return { tier, bytes: cached, fromCache: true };
  if (cached) await o.storage.delete(key);

  const fetchFn = o.fetch ?? ((u: string) => fetch(u));
  const res = await fetchFn(tier.base + tier.manifest.file);
  if (!res.ok) throw new Error(`${res.status} fetching ${tier.base + tier.manifest.file}`);
  const bytes = await readAll(res, tier.bytes, o.onProgress);
  if (bytes.length !== tier.bytes) throw new ModelIntegrityError(`Truncated ${key}: ${bytes.length} of ${tier.bytes} bytes`);
  const digest = await sha256(bytes);
  if (digest !== null && digest !== tier.sha256) throw new ModelIntegrityError(`Checksum mismatch for ${key}`);

  // Other versions of this model are stale once the new one is in.
  for (const k of held) if (k.startsWith(`${tier.id}@`)) await o.storage.delete(k);
  await o.storage.put(key, bytes);
  return { tier, bytes, fromCache: false };
}

const mb = (n: number) => `${(n / 1024 / 1024).toFixed(1)} MB`;

async function readAll(res: Response, total: number, onProgress?: (loaded: number, total: number) => void): Promise<Uint8Array> {
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const out = new Uint8Array(total);
  let n = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (n + value.length > total) throw new ModelIntegrityError(`Download larger than the expected ${total} bytes`);
    out.set(value, n);
    n += value.length;
    onProgress(n, total);
  }
  return out.subarray(0, n);
}

/** Null where WebCrypto is missing (plain-http LAN pages, React Native); the length check still applies. */
async function sha256(bytes: Uint8Array): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const d = new Uint8Array(await subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

export function memoryStorage(): ModelStorage {
  const m = new Map<string, Uint8Array>();
  return {
    get: async (k) => m.get(k) ?? null,
    put: async (k, b) => void m.set(k, b),
    delete: async (k) => void m.delete(k),
    keys: async () => [...m.keys()],
  };
}

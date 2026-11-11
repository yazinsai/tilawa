import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { MODEL_BUDGET_BYTES, ModelBudgetError, ModelIntegrityError, fetchModel, memoryStorage } from "../../src/runtime/models.js";
import type { ModelRegistry } from "../../src/runtime/models.js";

const bytes = (n: number, seed = 1) => Uint8Array.from({ length: n }, (_, i) => (i * seed) & 255);
const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");

function server(files: Record<string, Uint8Array>) {
  const calls: string[] = [];
  const fetch = async (url: string) => {
    calls.push(url);
    const body = files[url];
    return body ? new Response(body, { headers: { "content-length": String(body.length) } }) : new Response("missing", { status: 404 });
  };
  return { fetch, calls };
}

function registry(model: Uint8Array, version = "1", over: Partial<ModelRegistry["tiers"][number]> = {}): ModelRegistry {
  const manifest = { id: "m", file: "model.onnx", bytes: model.length, sha256: sha(model) };
  return {
    schema: 1,
    tiers: [{ tier: "lite", id: "m", version, base: `https://cdn/m/${version}/`, bytes: model.length, sha256: sha(model), supported: ["omitted_word"], manifest, ...over }],
  } as unknown as ModelRegistry;
}

describe("model tiers", () => {
  it("download on first use and come from the cache afterwards", async () => {
    const model = bytes(1000);
    const reg = registry(model);
    const { fetch, calls } = server({ "https://cdn/m/1/model.onnx": model });
    const storage = memoryStorage();
    const a = await fetchModel({ tier: "lite", registry: reg, storage, fetch });
    const b = await fetchModel({ tier: "lite", registry: reg, storage, fetch });
    expect(a.bytes).toEqual(model);
    expect(b.bytes).toEqual(model);
    expect(calls).toEqual(["https://cdn/m/1/model.onnx"]);
    expect(b.fromCache).toBe(true);
  });

  it("refuse a model above the 200 MB budget unless explicitly allowed", async () => {
    const model = bytes(10);
    const reg = registry(model, "1", { bytes: MODEL_BUDGET_BYTES + 1 });
    const { fetch, calls } = server({ "https://cdn/m/1/model.onnx": model });
    await expect(fetchModel({ tier: "lite", registry: reg, storage: memoryStorage(), fetch })).rejects.toBeInstanceOf(ModelBudgetError);
    expect(calls).toEqual([]);
  });

  it("reject a corrupt download and don't cache it", async () => {
    const model = bytes(1000);
    const reg = registry(model);
    const storage = memoryStorage();
    const bad = server({ "https://cdn/m/1/model.onnx": bytes(1000, 3) });
    await expect(fetchModel({ tier: "lite", registry: reg, storage, fetch: bad.fetch })).rejects.toBeInstanceOf(ModelIntegrityError);
    expect(await storage.keys()).toEqual([]);
  });

  it("are versioned: a new version replaces the cached old one", async () => {
    const v1 = bytes(1000);
    const v2 = bytes(1200, 7);
    const storage = memoryStorage();
    const { fetch } = server({ "https://cdn/m/1/model.onnx": v1, "https://cdn/m/2/model.onnx": v2 });
    await fetchModel({ tier: "lite", registry: registry(v1, "1"), storage, fetch });
    const b = await fetchModel({ tier: "lite", registry: registry(v2, "2"), storage, fetch });
    expect(b.bytes).toEqual(v2);
    expect(b.fromCache).toBe(false);
    expect(await storage.keys()).toEqual(["m@2/model.onnx"]);
  });
});

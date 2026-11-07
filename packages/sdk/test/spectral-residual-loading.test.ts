import { createHash } from "node:crypto";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadNodeModel } from "../src/node.js";
import { fetchModel, memoryStorage, ModelIntegrityError } from "../src/models.js";
import type { ModelRegistry } from "../src/models.js";
import type { ModelManifest } from "../src/model.js";

const fixture = new URL("./fixtures/tiny/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", fixture), "utf8")) as ModelManifest;
const sha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");
function parameters() {
  return { schema: 1, tokens: manifest.tokens, projection: [Array.from({ length: 256 }, () => 0)],
    predictor: Array.from({ length: 81 }, () => [0]), scale: [1], means: Array.from({ length: manifest.tokens.length + 1 }, () => [0]),
    variance: [1], covered: Array.from({ length: manifest.tokens.length + 1 }, () => true) };
}
function registry(bytes: Uint8Array): ModelRegistry {
  return { schema: 1, tiers: [{ tier: "lite", id: "test", version: "1", base: "https://test/", bytes: 1,
    sha256: sha(new Uint8Array([1])), manifest: { ...manifest, file: "model.onnx" }, supported: [] }],
    scorers: [{ id: "residual-test", purpose: "wrong-word-spectral", inventory: manifest.inventory,
      file: "spectral-residual.json", bytes: bytes.length, sha256: sha(bytes) }] };
}

describe("declared spectral residual assets", () => {
  it("loads a verified side asset through the ordinary Node bundle without changing the primary model", async () => {
    const dir = mkdtempSync(join(tmpdir(), "tilawa-residual-"));
    try {
      copyFileSync(new URL("model.onnx", fixture), join(dir, manifest.file));
      writeFileSync(join(dir, "manifest.json"), JSON.stringify(manifest));
      const bytes = Buffer.from(JSON.stringify(parameters())), reg = registry(bytes);
      writeFileSync(join(dir, "spectral-residual.json"), bytes);
      const model = await loadNodeModel(dir, { registry: reg });
      expect(model.manifest).toEqual(manifest);
      expect(model.spectralResidual!.parameters).toEqual(parameters());
      await model.backend.release();
      writeFileSync(join(dir, "spectral-residual.json"), Buffer.from("corrupt"));
      await expect(loadNodeModel(dir, { registry: reg })).rejects.toThrow("integrity");
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });

  it("fetches and caches the declared scorer even when the primary graph is already cached", async () => {
    const bytes = new TextEncoder().encode(JSON.stringify(parameters())), reg = registry(bytes), storage = memoryStorage();
    const calls: string[] = [];
    const fetch = async (url: string) => {
      calls.push(url);
      return new Response(url.endsWith("model.onnx") ? new Uint8Array([1]) : bytes);
    };
    await fetchModel({ tier: "lite", registry: reg, storage, fetch });
    const cached = await fetchModel({ tier: "lite", registry: reg, storage, fetch });
    expect(cached.fromCache).toBe(true);
    expect(cached.spectralResidual).toEqual(bytes);
    expect(calls).toEqual(["https://test/model.onnx", "https://test/spectral-residual.json"]);
    await storage.put("test@1/spectral-residual.json", new Uint8Array(bytes.length));
    await fetchModel({ tier: "lite", registry: reg, storage, fetch });
    expect(calls.at(-1)).toBe("https://test/spectral-residual.json");
    expect(calls).toHaveLength(3);
  });

  it("rejects a missing declared scorer instead of evaluating the primary graph alone", async () => {
    const bytes = new TextEncoder().encode(JSON.stringify(parameters())), reg = registry(bytes), storage = memoryStorage();
    await storage.put("test@1/model.onnx", new Uint8Array([1]));
    await expect(fetchModel({ tier: "lite", registry: reg, storage, fetch: async () => new Response("missing", { status: 404 }) }))
      .rejects.toBeInstanceOf(ModelIntegrityError);
  });
});

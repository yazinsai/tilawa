import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createListener } from "../../src/runtime/listener.js";
import type { ModelManifest } from "../../src/runtime/model.js";
import type { ModelRegistry } from "../../src/runtime/models.js";
import { loadNodeTier, readWav } from "../../src/runtime/node.js";

const dir = new URL("../fixtures/tiny/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", dir), "utf8")) as ModelManifest;
const onnx = readFileSync(new URL("model.onnx", dir));
const registry: ModelRegistry = {
  schema: 1,
  tiers: [{ tier: "lite", id: manifest.id, version: "t1", base: "https://cdn/tiny/", bytes: manifest.bytes, sha256: manifest.sha256, supported: [], manifest }],
};

describe("node tier loading", () => {
  it("downloads into the cache directory once, then runs from disk", async () => {
    const cacheDir = mkdtempSync(join(tmpdir(), "tilawa-models-"));
    let fetches = 0;
    const fetch = async () => (fetches++, new Response(onnx));
    const a = await loadNodeTier({ tier: "lite", registry, cacheDir, fetch });
    await loadNodeTier({ tier: "lite", registry, cacheDir, fetch });
    expect(fetches).toBe(1);
    const wav = await readWav(new URL("../fixtures/tlog-clean-00010-630.wav", import.meta.url).pathname);
    const l = createListener(a);
    await l.push(wav.samples, wav.sampleRate);
    const e = await l.end();
    expect(e?.inventory).toBe(manifest.inventory);
  });
});

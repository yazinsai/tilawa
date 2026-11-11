import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import * as root from "../src/index.js";
import { DEFAULTS, MODEL_REGISTRY, REFERENCE, SDK_VERSION, TilawaLoadError, ayahsRecited, memoryStorage } from "../src/index.js";
import type { EngineEvent, PositionEvent } from "../src/index.js";
import { createTilawa } from "../src/node.js";
import { tinyDir } from "./helpers.js";

const pkgFile = (p: string) => new URL(`../${p}`, import.meta.url);
const pkg = JSON.parse(readFileSync(pkgFile("package.json"), "utf8"));
const sha = (b: Buffer) => createHash("sha256").update(b).digest("hex");

describe("package", () => {
  it("reports its own version", () => {
    expect(SDK_VERSION).toBe(pkg.version);
  });

  it("vendors files unedited since sync-engine wrote them (src/engine/SOURCE.json)", () => {
    const source = JSON.parse(readFileSync(pkgFile("src/engine/SOURCE.json"), "utf8")) as { commit: string; files: Record<string, string> };
    expect(source.commit).toMatch(/^[0-9a-f]{40}$/);
    expect(Object.keys(source.files).length).toBeGreaterThan(40);
    for (const [path, digest] of Object.entries(source.files)) expect(sha(readFileSync(pkgFile(path))), path).toBe(digest);
  });

  it("ships the reference REFERENCE describes, and the registry's models are built on it", () => {
    const bytes = readFileSync(pkgFile("assets/ref-v2.json"));
    expect(bytes.length).toBe(REFERENCE.bytes);
    expect(sha(bytes)).toBe(REFERENCE.sha256);
    const ref = JSON.parse(bytes.toString("utf8"));
    expect(ref.version).toBe(REFERENCE.version);
    for (const t of MODEL_REGISTRY.tiers) expect(t.manifest.inventory).toBe(ref.inventory.id);
  });

  it("exports a curated, runtime-neutral root", () => {
    expect(Object.keys(root).sort()).toEqual([
      "ALL_CLASSES", "DEFAULTS", "EVENT_SCHEMA_VERSION", "MEMORIZATION_CLASSES", "MODEL_BUDGET_BYTES", "MODEL_REGISTRY",
      "MicrophoneError", "ModelBudgetError", "ModelIntegrityError", "REFERENCE", "SDK_VERSION", "SessionEndedError",
      "TilawaLoadError", "ayahsRecited", "createEngine", "fetchModel", "loadReferenceIndex", "memoryStorage", "ortBackend", "withBase",
    ]);
    for (const f of ["index.ts", "session.ts", "tilawa.ts", "recited.ts", "defaults.ts", "errors.ts"])
      expect(readFileSync(pkgFile(`src/${f}`), "utf8"), f).not.toMatch(/from "(node:|onnxruntime)/);
  });
});

describe("createTilawa (node)", () => {
  it("loads a model directory and the bundled reference, reporting progress", async () => {
    const stages: string[] = [];
    const t = await createTilawa({ modelDir: tinyDir, onProgress: (p) => stages.push(`${p.stage}:${p.fraction}`) });
    expect(stages).toContain("reference:1");
    expect(stages).toContain("model:1");
    expect(t.model).toMatchObject({ id: "tiny-random", backend: "onnxruntime-node", version: null, tier: null, fallbacks: [] });
    await t.dispose();
  });

  it("downloads a registry tier once into cacheDir, and wraps load failures in TilawaLoadError", async () => {
    const { mkdtempSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const cacheDir = mkdtempSync(join(tmpdir(), "tilawa-cache-"));
    await expect(createTilawa({ cacheDir, assetBase: "http://127.0.0.1:9/" })).rejects.toBeInstanceOf(TilawaLoadError);
    await expect(createTilawa({ cacheDir, assetBase: "http://127.0.0.1:9/" })).rejects.toMatchObject({ stage: "model" });
  });

  it("rejects with AbortError when the signal is already aborted", async () => {
    const c = new AbortController();
    c.abort();
    await expect(createTilawa({ modelDir: tinyDir, signal: c.signal })).rejects.toMatchObject({ name: "AbortError" });
  });

  it("listen() rejects with MicrophoneError on Node", async () => {
    const t = await createTilawa({ modelDir: tinyDir });
    await expect(t.listen(t.startFreeRecitation())).rejects.toMatchObject({ name: "MicrophoneError", reason: "unavailable" });
    await t.dispose();
  });

  it("defaults to the tilawa.dev v1 layout", () => {
    expect(DEFAULTS.assetBase).toBe("https://tilawa.dev/v1/");
    expect(REFERENCE.path).toBe("reference/ref-v2.json");
    expect(MODEL_REGISTRY.tiers.map((t) => `${t.base}${t.manifest.file}`)).toEqual(
      MODEL_REGISTRY.tiers.map((t) => `models/${t.id}/${t.version}/model.onnx`));
    void memoryStorage;
  });
});

describe("ayahsRecited", () => {
  const pos = (surah: number, ayah: number, state: PositionEvent["state"] = "locked"): EngineEvent => ({
    type: "position", v: 1, timeMs: 0, referenceVersion: "ref-v2", expected: null, state,
    observed: [{ word: { surah, ayah, word: 1 }, span: { start: { surah, ayah, word: 1 }, end: { surah, ayah, word: 1 } }, confidence: 1, cost: 0, per: 0 }],
  });
  it("lists locked ayahs in order, collapsing consecutive repeats", () => {
    expect(ayahsRecited([pos(1, 1), pos(1, 1), pos(1, 2, "acquiring"), pos(1, 2), pos(1, 3), pos(1, 2)])).toEqual([
      { surah: 1, ayah: 1 }, { surah: 1, ayah: 2 }, { surah: 1, ayah: 3 }, { surah: 1, ayah: 2 },
    ]);
    expect(ayahsRecited([])).toEqual([]);
  });
});

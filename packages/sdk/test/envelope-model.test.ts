import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { EnvelopeModel } from "../src/temporal-envelope.js";

const mocks = vi.hoisted(() => ({ access: vi.fn(), readFile: vi.fn(), backend: vi.fn() }));
vi.mock("node:fs/promises", async importOriginal => ({ ...await importOriginal<typeof import("node:fs/promises")>(), access: mocks.access, readFile: mocks.readFile }));
vi.mock("../src/ort.js", () => ({ ortBackend: mocks.backend }));
import { loadNodeModel } from "../src/node.js";

const manifest = { file: "primary.onnx", tokens: "بن", inventory: "toy", sampleRate: 16000, frameMs: 40, supported: ["wrong_word"] };
const primary = { name: "primary", run: vi.fn(), release: vi.fn() };
const model = (): EnvelopeModel => ({ schema: 1, inventory: "toy", tokens: "بن", frameMs: 40, order: 2, ridge: .001, maxFrames: 4096,
  phoneCounts: [10, 10], envelopeScale: Array(19).fill(1), controlScale: Array(19).fill(1),
  envelope: Array.from({ length: 3 }, () => ({ mean: Array(19).fill(0), variance: Array(19).fill(1) })),
  control: Array.from({ length: 3 }, () => ({ mean: Array(19).fill(0), variance: Array(19).fill(1) })) });

beforeEach(() => {
  vi.resetAllMocks();
  mocks.backend.mockResolvedValue(primary);
  mocks.access.mockRejectedValue(Object.assign(new Error("absent"), { code: "ENOENT" }));
});
const install = (asset: EnvelopeModel | null, corrupt = false) => {
  const bytes = Buffer.from(JSON.stringify(asset));
  const descriptor = { schema: 1, file: "envelope-emissions.json", bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
  mocks.readFile.mockImplementation(async (path: string) => {
    if (path.endsWith("manifest.json")) return JSON.stringify(manifest);
    if (path.endsWith("envelope-asset.json")) return JSON.stringify(descriptor);
    if (asset && path.endsWith("envelope-emissions.json")) return corrupt ? Buffer.from("corrupt") : bytes;
    throw Object.assign(new Error("missing emissions"), { code: "ENOENT" });
  });
};

describe("declared temporal-envelope bundle", () => {
  it("loads a verified side asset while retaining the primary model and manifest", async () => {
    install(model());
    const loaded = await loadNodeModel("/bundle");
    expect(loaded.temporalEnvelopes).toEqual(model());
    expect(loaded.backend).toBe(primary);
    expect(loaded.manifest).toEqual(manifest);
    expect(mocks.backend).toHaveBeenCalledOnce();
    expect(primary.run).not.toHaveBeenCalled();
  });
  it("allows ordinary primary-only models when no side asset is declared", async () => {
    mocks.readFile.mockImplementation(async (path: string) => {
      if (path.endsWith("manifest.json")) return JSON.stringify(manifest);
      throw Object.assign(new Error("absent"), { code: "ENOENT" });
    });
    expect((await loadNodeModel("/bundle")).temporalEnvelopes).toBeUndefined();
  });
  it.each(["missing", "checksum", "inventory"])("rejects %s side data and releases the allocated primary", async failure => {
    const asset = model();
    if (failure === "inventory") asset.inventory = "other";
    install(failure === "missing" ? null : asset, failure === "checksum");
    await expect(loadNodeModel("/bundle")).rejects.toThrow();
    expect(primary.release).toHaveBeenCalledOnce();
  });
});

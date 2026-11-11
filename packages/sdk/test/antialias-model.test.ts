import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ access: vi.fn(), readFile: vi.fn(), ortBackend: vi.fn() }));
vi.mock("node:fs/promises", async original => ({
  ...await original<typeof import("node:fs/promises")>(), access: mocks.access, readFile: mocks.readFile,
}));
vi.mock("../src/ort.js", () => ({ ortBackend: mocks.ortBackend }));
import { loadNodeModel } from "../src/node.js";

const manifest = { file: "primary.onnx", sha256: "primary-digest", inventory: "test" };
const bytes = Buffer.from("independent-graph");
const declaration = { recipe: "centered-binomial-strides-v1", sourceSha256: manifest.sha256,
  sha256: createHash("sha256").update(bytes).digest("hex") };
let files: Map<string, string | Buffer>;
let primary: { name: string; run: ReturnType<typeof vi.fn>; release: ReturnType<typeof vi.fn> };

beforeEach(() => {
  vi.resetAllMocks();
  files = new Map([["manifest.json", JSON.stringify(manifest)]]);
  mocks.readFile.mockImplementation(async (path: string) => {
    const value = files.get(path.split("/").at(-1)!);
    if (value === undefined) throw Object.assign(new Error("absent"), { code: "ENOENT" });
    return value;
  });
  mocks.access.mockRejectedValue(Object.assign(new Error("absent"), { code: "ENOENT" }));
  primary = { name: "primary", run: vi.fn(), release: vi.fn() };
  mocks.ortBackend.mockResolvedValueOnce(primary);
});

describe("Node antialias graph loading", () => {
  it("leaves the original model unchanged without a declaration", async () => {
    const model = await loadNodeModel("/models/caller");
    expect(model.backend).toBe(primary);
    expect(model.antialiasModel).toBeUndefined();
    expect(mocks.ortBackend).toHaveBeenCalledTimes(1);
    expect(primary.release).not.toHaveBeenCalled();
  });

  it("loads the verified independent graph with the original state geometry and releases both", async () => {
    files.set("antialias.json", JSON.stringify(declaration));
    files.set("antialias.onnx", bytes);
    const side = { name: "side", run: vi.fn(), release: vi.fn() };
    const primaryRelease = primary.release;
    mocks.ortBackend.mockResolvedValueOnce(side);
    const model = await loadNodeModel("/models/caller", { threads: 2 });
    expect(model.backend).toBe(primary);
    expect(model.antialiasModel?.manifest).toBe(model.manifest);
    expect(model.antialiasModel?.backend).toBe(side);
    expect(mocks.ortBackend.mock.calls[1]).toEqual([expect.anything(), bytes, "onnxruntime-node",
      { executionProviders: ["cpu"], intraOpNumThreads: 2 }]);
    expect(primary.run).not.toHaveBeenCalled();
    await model.backend.release();
    expect(side.release).toHaveBeenCalledOnce();
    expect(primaryRelease).toHaveBeenCalledOnce();
  });

  it("rejects a missing declared graph and frees the primary backend", async () => {
    files.set("antialias.json", JSON.stringify(declaration));
    await expect(loadNodeModel("/models/caller")).rejects.toMatchObject({ code: "ENOENT" });
    expect(primary.release).toHaveBeenCalledOnce();
    expect(mocks.ortBackend).toHaveBeenCalledTimes(1);
  });

  it("rejects a different parent, recipe or graph checksum", async () => {
    for (const replacement of [ { sourceSha256: "other" }, { recipe: "other" }, { sha256: "other" } ]) {
      files.set("antialias.json", JSON.stringify({ ...declaration, ...replacement }));
      files.set("antialias.onnx", bytes);
      mocks.ortBackend.mockResolvedValueOnce(primary);
      await expect(loadNodeModel("/models/caller")).rejects.toThrow(/Antialias graph/);
    }
    expect(primary.release).toHaveBeenCalledTimes(3);
  });
});

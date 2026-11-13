import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({readFile: vi.fn(), access: vi.fn(), backend: vi.fn()}));
vi.mock("node:fs/promises", async original => ({...await original<typeof import("node:fs/promises")>(), readFile: mocks.readFile, access: mocks.access}));
vi.mock("../src/ort.js", () => ({ortBackend: mocks.backend}));
import { loadNodeModel } from "../src/node.js";
import { BASIN_ARMS } from "../src/connected-basin.js";

const hash = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");
const primary = Buffer.from("primary graph"), extractor = Buffer.from("same graph plus output");
const manifest = {file: "model.onnx", inventory: "test", tokens: "ab", sha256: hash(primary)};
const readouts = Buffer.from(JSON.stringify({schema: 1, inventory: "test", tokens: "ab", primarySha256: hash(primary),
  width: 1, hidden: 1, featureOutput: "h", center: [0], scale: [1], heads: Object.fromEntries(BASIN_ARMS.map(arm => [arm,
    {"0.weight": [[1]], "0.bias": [0], "2.weight": [[0], [1], [-1]], "2.bias": [0, 0, 0]}]))}));
const descriptor = JSON.stringify({schema: 1, primarySha256: hash(primary), files: {
  "encoder-features.onnx": {bytes: extractor.length, sha256: hash(extractor)},
  "readouts.json": {bytes: readouts.length, sha256: hash(readouts)},
}});
const files: Record<string, string | Buffer> = {"manifest.json": JSON.stringify(manifest), "model.onnx": primary,
  "encoder-features.onnx": extractor, "readouts.json": readouts, "connected-basin-asset.json": descriptor};
beforeEach(() => {
  vi.resetAllMocks();
  mocks.access.mockRejectedValue(Object.assign(new Error("missing teacher"), {code: "ENOENT"}));
  mocks.readFile.mockImplementation(async (path: string) => files[path.split("/").at(-1)!]);
});
describe("declared connected-basin assets", () => {
  it("loads a verified side backend without replacing the primary backend, and releases both", async () => {
    const a = {release: vi.fn()}, b = {release: vi.fn()};
    mocks.backend.mockResolvedValueOnce(a).mockResolvedValueOnce(b);
    const model = await loadNodeModel("/bundle");
    expect(model.backend).toBe(a); expect(model.connectedBasin?.backend).toBe(b);
    await model.backend.release();
    expect(b.release).toHaveBeenCalledOnce();
  });
  it("refuses missing declared files and a corrupt extractor instead of running the main model", async () => {
    mocks.backend.mockResolvedValue({release: vi.fn()});
    mocks.readFile.mockImplementation(async (path: string) => {
      if (path.endsWith("encoder-features.onnx")) throw Object.assign(new Error("missing side asset"), {code: "ENOENT"});
      return files[path.split("/").at(-1)!];
    });
    await expect(loadNodeModel("/bundle")).rejects.toMatchObject({code: "ENOENT"});
    mocks.readFile.mockImplementation(async (path: string) => path.endsWith("encoder-features.onnx") ? Buffer.from("corrupt") : files[path.split("/").at(-1)!]);
    await expect(loadNodeModel("/bundle")).rejects.toThrow("checksum mismatch");
  });
});

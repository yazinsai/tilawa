import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ access: vi.fn(), readFile: vi.fn(), ortBackend: vi.fn() }));
vi.mock("node:fs/promises", async original => ({ ...await original<typeof import("node:fs/promises")>(),
  access: mocks.access, readFile: mocks.readFile }));
vi.mock("../src/ort.js", () => ({ ortBackend: mocks.ortBackend }));
import { loadNodeModel } from "../src/node.js";
import { zeroModel } from "./fixtures/moving-zero.js";

const manifest = { file: "model.onnx", inventory: "test", tokens: "ab", frameMs: 40, sampleRate: 16000 };
const asset = Buffer.from(JSON.stringify(zeroModel()));
const descriptor = { schema: 1, file: "moving-zero-emissions.json", bytes: asset.length,
  sha256: createHash("sha256").update(asset).digest("hex") };

beforeEach(() => {
  vi.resetAllMocks();
  mocks.access.mockRejectedValue(Object.assign(new Error("no teacher"), {code:"ENOENT"}));
  mocks.ortBackend.mockResolvedValue({ run: vi.fn(), release: vi.fn() });
  mocks.readFile.mockImplementation((path: string) => Promise.resolve(path.endsWith("moving-zero-asset.json")
    ? JSON.stringify(descriptor) : path.endsWith("moving-zero-emissions.json") ? asset : JSON.stringify(manifest)));
});

describe("complete moving zero side asset loading", () => {
  it("loads a verified side scorer without replacing the primary backend", async () => {
    const m = await loadNodeModel("/bundle");
    expect(m.movingZero?.stationaryVariance).toEqual([.0036,.01]);
    expect(m.backend).toBe(await mocks.ortBackend.mock.results[0].value);
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
  it("rejects a declared missing asset and releases the backend", async () => {
    mocks.readFile.mockImplementation((path: string) => path.endsWith("moving-zero-emissions.json")
      ? Promise.reject(Object.assign(new Error("missing asset"), {code:"ENOENT"}))
      : Promise.resolve(JSON.stringify(path.endsWith("moving-zero-asset.json") ? descriptor : manifest)));
    await expect(loadNodeModel("/bundle")).rejects.toThrow("missing asset");
    expect((await mocks.ortBackend.mock.results[0].value).release).toHaveBeenCalledOnce();
  });
  it("rejects corruption rather than silently running without the scorer", async () => {
    mocks.readFile.mockImplementation((path: string) => Promise.resolve(path.endsWith("moving-zero-asset.json")
      ? JSON.stringify({...descriptor, sha256:"0".repeat(64)}) : path.endsWith("moving-zero-emissions.json") ? asset : JSON.stringify(manifest)));
    await expect(loadNodeModel("/bundle")).rejects.toThrow("checksum");
    expect((await mocks.ortBackend.mock.results[0].value).release).toHaveBeenCalledOnce();
  });
});

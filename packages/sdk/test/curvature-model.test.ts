import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ access: vi.fn(), readFile: vi.fn(), ortBackend: vi.fn() }));
vi.mock("node:fs/promises", async original => ({ ...await original<typeof import("node:fs/promises")>(),
  access: mocks.access, readFile: mocks.readFile }));
vi.mock("../src/ort.js", () => ({ ortBackend: mocks.ortBackend }));
import { loadNodeModel } from "../src/node.js";

const manifest = { file: "model.onnx", inventory: "test", tokens: "ab", frameMs: 40, sampleRate: 16000 };
const arm = { center: Array(14).fill(0), scale: Array(14).fill(1),
  emissions: [0,1,2].map(() => ({ mean: Array(14).fill(0), variance: Array(14).fill(1) })) };
const asset = Buffer.from(JSON.stringify({ schema: 1, inventory: "test", tokens: "ab", frameMs: 40,
  resolution: 16, phoneCounts: [5,5], stay: .8, maxFrames: 2048, control: arm, curvature: arm }));
const descriptor = { schema: 1, file: "curvature-emissions.json", bytes: asset.length,
  sha256: createHash("sha256").update(asset).digest("hex") };

beforeEach(() => {
  vi.resetAllMocks();
  mocks.access.mockRejectedValue(Object.assign(new Error("no teacher"), {code:"ENOENT"}));
  mocks.ortBackend.mockResolvedValue({ run: vi.fn(), release: vi.fn() });
  mocks.readFile.mockImplementation((path: string) => Promise.resolve(path.endsWith("curvature-asset.json")
    ? JSON.stringify(descriptor) : path.endsWith("curvature-emissions.json") ? asset : JSON.stringify(manifest)));
});

describe("complete curvature side asset loading", () => {
  it("loads a verified side scorer without replacing the primary backend", async () => {
    const m = await loadNodeModel("/bundle");
    expect(m.spectralCurvature?.resolution).toBe(16);
    expect(m.backend).toBe(await mocks.ortBackend.mock.results[0].value);
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
  });
  it("rejects a declared missing asset and releases the backend", async () => {
    mocks.readFile.mockImplementation((path: string) => path.endsWith("curvature-emissions.json")
      ? Promise.reject(Object.assign(new Error("missing asset"), {code:"ENOENT"}))
      : Promise.resolve(JSON.stringify(path.endsWith("curvature-asset.json") ? descriptor : manifest)));
    await expect(loadNodeModel("/bundle")).rejects.toThrow("missing asset");
    expect((await mocks.ortBackend.mock.results[0].value).release).toHaveBeenCalledOnce();
  });
  it("rejects corruption rather than silently running without the scorer", async () => {
    mocks.readFile.mockImplementation((path: string) => Promise.resolve(path.endsWith("curvature-asset.json")
      ? JSON.stringify({...descriptor, sha256:"0".repeat(64)}) : path.endsWith("curvature-emissions.json") ? asset : JSON.stringify(manifest)));
    await expect(loadNodeModel("/bundle")).rejects.toThrow("checksum");
    expect((await mocks.ortBackend.mock.results[0].value).release).toHaveBeenCalledOnce();
  });
});

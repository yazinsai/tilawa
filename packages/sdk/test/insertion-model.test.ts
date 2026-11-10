import { beforeEach, describe, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";

const mocks = vi.hoisted(() => ({ access: vi.fn(), readFile: vi.fn(), ortBackend: vi.fn() }));
vi.mock("node:fs/promises", async importOriginal => ({
  ...await importOriginal<typeof import("node:fs/promises")>(),
  access: mocks.access,
  readFile: mocks.readFile,
}));
vi.mock("../src/ort.js", () => ({ ortBackend: mocks.ortBackend }));

import { loadNodeModel } from "../src/node.js";

const manifest = { file: "primary.onnx", tokens: "بت", inventory: "muaalem-v3_2-phonemes", sampleRate: 16000, frameMs: 40, supported: ["added_word"] };

beforeEach(() => {
  vi.resetAllMocks();
  mocks.readFile.mockImplementation(async path => {
    if (String(path).endsWith("/manifest.json")) return JSON.stringify(manifest);
    throw Object.assign(new Error("absent"), {code:"ENOENT"});
  });
});

describe("Node synchronous cue scorer bundle loading", () => {
  const scorer = {schema:1,inventory:manifest.inventory,tokens:manifest.tokens,frameMs:40,observationMs:10,dimensions:20,
    center:Array(20).fill(0),scale:Array(20).fill(1),counts:[0,1,1],classes:[[0,0,0],[1,1,1],[2,2,2]],stay:.9,maxFrames:4096,
    means:Array.from({length:3},()=>Array.from({length:3},()=>Array(20).fill(0))),variances:Array.from({length:3},()=>Array.from({length:3},()=>Array(20).fill(1)))};
  function install(asset: Buffer | null, descriptorBytes?: Buffer) {
    const bytes=Buffer.from(JSON.stringify(scorer));
    const descriptor={schema:1,file:"cue-emissions.json",bytes:bytes.length,
      sha256:createHash("sha256").update(bytes).digest("hex")};
    mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
    mocks.readFile.mockImplementation(async path => {
      if (String(path).endsWith("/manifest.json")) return JSON.stringify(manifest);
      if (String(path).endsWith("/cue-asset.json")) return (descriptorBytes ?? Buffer.from(JSON.stringify(descriptor))).toString("utf8");
      if (String(path).endsWith("/cue-emissions.json") && asset) return asset;
      throw Object.assign(new Error("absent"),{code:"ENOENT"});
    });
  }
  it("loads a verified scorer while preserving the primary model", async () => {
    install(Buffer.from(JSON.stringify(scorer)));
    const primary={name:"primary",run:vi.fn(),release:vi.fn()};
    mocks.ortBackend.mockResolvedValue(primary);
    const loaded=await loadNodeModel("/models/caller");
    expect(loaded.backend).toBe(primary);
    expect(loaded.synchronousCues).toEqual(scorer);
    expect(primary.run).not.toHaveBeenCalled();
  });
  it("fails and releases the primary backend when a declared scorer is missing", async () => {
    install(null);
    const primary={name:"primary",release:vi.fn()};
    mocks.ortBackend.mockResolvedValue(primary);
    await expect(loadNodeModel("/models/caller")).rejects.toMatchObject({code:"ENOENT"});
    expect(primary.release).toHaveBeenCalledOnce();
  });
  it("rejects corrupted assets and invalid descriptors", async () => {
    const primary={name:"primary",release:vi.fn()};
    mocks.ortBackend.mockResolvedValue(primary);
    install(Buffer.from("{}"));
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("checksum mismatch");
    install(Buffer.from(JSON.stringify(scorer)),Buffer.from("{}"));
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("Invalid synchronous cue asset descriptor");
    expect(primary.release).toHaveBeenCalledTimes(2);
  });
});

describe("Node insertion teacher loading", () => {
  it("retains the caller's primary model and loads the adjacent teacher once for repeated crops", async () => {
    const primary = { name: "primary", run: vi.fn(), release: vi.fn() };
    const rows = Float32Array.from([.9, .05, .05], Math.log);
    const teacher = { name: "teacher", run: vi.fn().mockResolvedValue({
      log_probs: { type: "float32", data: rows, dims: [1, 1, 3] },
    }), release: vi.fn() };
    mocks.access.mockResolvedValue(undefined);
    mocks.ortBackend.mockResolvedValueOnce(primary).mockResolvedValueOnce(teacher);

    const model = await loadNodeModel("/models/caller", { threads: 2 });
    expect(model.backend).toBe(primary);
    expect(model.manifest).toEqual(manifest);
    expect(mocks.ortBackend).toHaveBeenCalledTimes(1);
    const wave = Float32Array.from({ length: 1600 }, (_, i) => Math.sin(i * .1));
    await model.insertionTeacher!.decode(wave, 3);
    await model.insertionTeacher!.decode(wave, 3);
    expect(teacher.run).toHaveBeenCalledTimes(2);
    expect(primary.run).not.toHaveBeenCalled();
    expect(mocks.ortBackend).toHaveBeenCalledTimes(2);
    expect(mocks.ortBackend.mock.calls[1][3]).toMatchObject({executionProviders:["cpu"],intraOpNumThreads:2,graphOptimizationLevel:"disabled"});
    await model.backend.release();
    expect(teacher.release).toHaveBeenCalledOnce();
  });

  it("uses the ordinary primary model when no adjacent teacher is installed", async () => {
    const primary = { name: "primary" };
    mocks.access.mockRejectedValue(Object.assign(new Error("absent"), { code: "ENOENT" }));
    mocks.ortBackend.mockResolvedValue(primary);
    const model = await loadNodeModel("/models/caller");
    expect(model.backend).toBe(primary);
    expect(model.insertionTeacher).toBeUndefined();
    expect(mocks.ortBackend).toHaveBeenCalledTimes(1);
  });

  it("reports an inaccessible installed teacher instead of silently dropping its evidence", async () => {
    const denied = Object.assign(new Error("denied"), { code: "EACCES" });
    mocks.access.mockRejectedValue(denied);
    mocks.ortBackend.mockResolvedValue({ name: "primary" });
    await expect(loadNodeModel("/models/caller")).rejects.toBe(denied);
  });
});

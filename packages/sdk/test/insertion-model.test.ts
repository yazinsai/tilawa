import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ access: vi.fn(), readFile: vi.fn(), ortBackend: vi.fn() }));
vi.mock("node:fs/promises", async importOriginal => ({
  ...await importOriginal<typeof import("node:fs/promises")>(),
  access: mocks.access,
  readFile: mocks.readFile,
}));
vi.mock("../src/ort.js", () => ({ ortBackend: mocks.ortBackend }));

import { loadNodeModel } from "../src/node.js";

const manifest = { file: "primary.onnx", tokens: "بت", inventory: "muaalem-v3_2-phonemes", supported: ["added_word"] };

beforeEach(() => {
  vi.resetAllMocks();
  mocks.readFile.mockImplementation(async (path: string) => {
    if (path.endsWith("predictive-phone-asset.json")) throw Object.assign(new Error("absent"), {code:"ENOENT"});
    return JSON.stringify(manifest);
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


describe("Node Bayesian side-scorer assets", () => {
  const fixture = () => {
    const matrix = Array.from({length:8},(_,i)=>Array.from({length:8},(_,j)=>i===j?3:0));
    const emission = {mean:Array(8).fill(0),kappa:2,nu:12,psi:matrix,count:1};
    const value = {schema:1,inventory:manifest.inventory,tokens:manifest.tokens,frameMs:40,observationMs:10,dimensions:8,
      center:Array(8).fill(0),scale:Array(8).fill(1),phones:[emission,emission,emission],stay:.8,draws:128,maxFrames:2048};
    const bytes = Buffer.from(JSON.stringify(value));
    return {bytes,descriptor:JSON.stringify({schema:1,file:"predictive-phone-emissions.json",bytes:bytes.length,
      sha256:createHash("sha256").update(bytes).digest("hex")})};
  };
  const primary = () => ({name:"primary",run:vi.fn(),release:vi.fn()});
  it("verifies the declared asset and retains the caller's model", async () => {
    const p=primary(), f=fixture();
    mocks.access.mockRejectedValue(Object.assign(new Error("no teacher"),{code:"ENOENT"}));
    mocks.ortBackend.mockResolvedValue(p);
    mocks.readFile.mockImplementation(async (path:string)=>path.endsWith("manifest.json")?JSON.stringify({...manifest,sampleRate:16000,frameMs:40}):
      path.endsWith("asset.json")?f.descriptor:f.bytes);
    const m=await loadNodeModel("/models/caller");
    expect(m.backend).toBe(p);
    expect(m.predictivePhone?.phones).toHaveLength(3);
    expect(p.run).not.toHaveBeenCalled();
  });
  it("rejects a missing declared file and releases the backend", async () => {
    const p=primary(), f=fixture();
    mocks.access.mockRejectedValue(Object.assign(new Error("no teacher"),{code:"ENOENT"}));
    mocks.ortBackend.mockResolvedValue(p);
    mocks.readFile.mockImplementation(async (path:string)=> {
      if (path.endsWith("manifest.json")) return JSON.stringify(manifest);
      if (path.endsWith("asset.json")) return f.descriptor;
      throw Object.assign(new Error("missing emissions"),{code:"ENOENT"});
    });
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("missing emissions");
    expect(p.release).toHaveBeenCalledOnce();
  });
  it("rejects altered model bytes instead of silently falling back", async () => {
    const p=primary(), f=fixture();
    mocks.access.mockRejectedValue(Object.assign(new Error("no teacher"),{code:"ENOENT"}));
    mocks.ortBackend.mockResolvedValue(p);
    const corrupt=Buffer.from(f.bytes); corrupt[0]=32;
    mocks.readFile.mockImplementation(async (path:string)=>path.endsWith("manifest.json")?JSON.stringify(manifest):
      path.endsWith("asset.json")?f.descriptor:corrupt);
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("checksum");
    expect(p.release).toHaveBeenCalledOnce();
  });
});

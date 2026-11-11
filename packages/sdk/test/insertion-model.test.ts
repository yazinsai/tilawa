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
  mocks.readFile.mockImplementation(async (path:string) => {
    if (path.endsWith("training-support-asset.json")) throw Object.assign(new Error("absent"), {code:"ENOENT"});
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


describe("recording-support bundle loading", () => {
  const fixture = () => {
    const phone = {mean:Array(8).fill(0),variance:Array(8).fill(1),count:1};
    const fit = {center:Array(8).fill(0),scale:Array(8).fill(1),phones:[phone,phone,phone]};
    const value = {schema:1,inventory:manifest.inventory,tokens:manifest.tokens,frameMs:40,observationMs:10,dimensions:8,
      stay:.8,maxFrames:2048,cutoff:0,full:fit,deletions:[fit,fit]};
    const bytes = Buffer.from(JSON.stringify(value));
    return {bytes,descriptor:JSON.stringify({schema:1,file:"training-support-emissions.json",bytes:bytes.length,
      sha256:createHash("sha256").update(bytes).digest("hex")})};
  };
  for (const state of ["valid", "missing", "corrupt"] as const) it(`handles a ${state} declared bundle`, async () => {
    const p={name:"primary",run:vi.fn(),release:vi.fn()}, f=fixture();
    mocks.access.mockRejectedValue(Object.assign(new Error("no teacher"),{code:"ENOENT"}));
    mocks.ortBackend.mockResolvedValue(p);
    mocks.readFile.mockImplementation(async (path:string)=> {
      if (path.endsWith("manifest.json")) return JSON.stringify(manifest);
      if (path.endsWith("asset.json")) return f.descriptor;
      if (state==="missing") throw Object.assign(new Error("missing emissions"),{code:"ENOENT"});
      return state==="corrupt" ? Buffer.from("altered") : f.bytes;
    });
    if (state==="valid") {
      const m=await loadNodeModel("/models/caller");
      expect(m.trainingSupport?.deletions).toHaveLength(2);
      expect(m.backend).toBe(p);
      expect(p.run).not.toHaveBeenCalled();
    } else {
      await expect(loadNodeModel("/models/caller")).rejects.toThrow(state==="missing"?"missing emissions":"checksum");
      expect(p.release).toHaveBeenCalledOnce();
    }
  });
});

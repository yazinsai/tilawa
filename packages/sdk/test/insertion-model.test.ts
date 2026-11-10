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

const manifest = { file: "primary.onnx", tokens: "بت", inventory: "muaalem-v3_2-phonemes", supported: ["added_word"] };

beforeEach(() => {
  vi.resetAllMocks();
  mocks.readFile.mockImplementation(async (path: string) => {
    if (path.endsWith("depth-trajectory-asset.json")) throw Object.assign(new Error("absent"), { code: "ENOENT" });
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

describe("declared depth scorer integrity", () => {
  it("loads a complete matched bundle and releases both graph instances", async () => {
    const primaryRelease=vi.fn();
    const primary={name:"primary",release:primaryRelease}, extractor={name:"extractor",release:vi.fn()};
    const hash=(bytes:Buffer)=>createHash("sha256").update(bytes).digest("hex");
    const graph=Buffer.from("primary graph"), sideGraph=Buffer.from("frozen graph");
    const declared={...manifest,sampleRate:16000,frameMs:40,sha256:hash(graph)};
    const eye=Array.from({length:16},(_,i)=>Array.from({length:16},(_,j)=>i===j?1:0));
    const arm={center:Array(16).fill(0),scale:Array(16).fill(1),phones:Array.from({length:3},()=>({
      mean:Array(16).fill(0),precision:eye,logDet:0,count:1}))};
    const fitted={schema:1,dimensions:16,tokens:manifest.tokens,inventory:manifest.inventory,primarySha256:declared.sha256,
      outputs:["a","b","c","d"],projections:Array.from({length:4},()=>eye.slice(0,4)),controlProjection:eye,
      trajectory:arm,control:arm};
    const emissions=Buffer.from(JSON.stringify(fitted));
    const descriptor={schema:1,files:[{file:"depth-trajectory-emissions.json",bytes:emissions.length,sha256:hash(emissions)},
      {file:"depth-trajectory.onnx",bytes:sideGraph.length,sha256:hash(sideGraph)}]};
    mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
    mocks.ortBackend.mockResolvedValueOnce(primary).mockResolvedValueOnce(extractor);
    mocks.readFile.mockImplementation(async (path:string)=> {
      if (path.endsWith("manifest.json")) return JSON.stringify(declared);
      if (path.endsWith("depth-trajectory-asset.json")) return JSON.stringify(descriptor);
      if (path.endsWith("depth-trajectory-emissions.json")) return emissions;
      return path.endsWith("depth-trajectory.onnx")?sideGraph:graph;
    });
    const model=await loadNodeModel("/models/caller");
    expect(model.depthTrajectory).toEqual({model:fitted,backend:extractor});
    await model.backend.release();
    expect(primaryRelease).toHaveBeenCalledOnce();
    expect(extractor.release).toHaveBeenCalledOnce();
  });
  it("fails and releases the primary backend when a declared side file is missing", async () => {
    const primary={name:"primary",release:vi.fn()};
    const entry={bytes:1,sha256:"0".repeat(64)};
    mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
    mocks.ortBackend.mockResolvedValue(primary);
    mocks.readFile.mockImplementation(async (path:string)=> {
      if (path.endsWith("manifest.json")) return JSON.stringify(manifest);
      if (path.endsWith("depth-trajectory-asset.json")) return JSON.stringify({schema:1,files:[
        {...entry,file:"depth-trajectory-emissions.json"},{...entry,file:"depth-trajectory.onnx"}]});
      throw Object.assign(new Error("missing side asset"),{code:"ENOENT"});
    });
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("missing side asset");
    expect(primary.release).toHaveBeenCalledOnce();
  });
  it("rejects corrupt side emissions before starting the extractor", async () => {
    const primary={name:"primary",release:vi.fn()};
    mocks.access.mockRejectedValue(Object.assign(new Error("absent"),{code:"ENOENT"}));
    mocks.ortBackend.mockResolvedValue(primary);
    mocks.readFile.mockImplementation(async (path:string)=> {
      if (path.endsWith("manifest.json")) return JSON.stringify(manifest);
      if (path.endsWith("depth-trajectory-asset.json")) return JSON.stringify({schema:1,files:[
        {file:"depth-trajectory-emissions.json",bytes:1,sha256:"0".repeat(64)},
        {file:"depth-trajectory.onnx",bytes:1,sha256:"0".repeat(64)}]});
      return Buffer.from("x");
    });
    await expect(loadNodeModel("/models/caller")).rejects.toThrow("checksum mismatch");
    expect(mocks.ortBackend).toHaveBeenCalledTimes(1);
    expect(primary.release).toHaveBeenCalledOnce();
  });
});

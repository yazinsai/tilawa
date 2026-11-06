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
  mocks.readFile.mockResolvedValue(JSON.stringify(manifest));
});

describe("Node insertion teacher loading", () => {
  it("retains the caller's primary model and loads the adjacent teacher once for repeated crops", async () => {
    const primary = { name: "primary", run: vi.fn(), release: vi.fn() };
    const rows = Float32Array.from([.9, .05, .05], Math.log);
    const teacher = { name: "teacher", run: vi.fn().mockResolvedValue({
      log_probs: { type: "float32", data: rows, dims: [1, 1, 3] },
    }), release: vi.fn() };
    mocks.access.mockImplementation(async (path: string) => {
      if (path.endsWith("word-encoder.onnx")) throw Object.assign(new Error("absent"), {code:"ENOENT"});
    });
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

  it("loads the declared word ear lazily alongside the teacher and releases both word backends", async () => {
    const primary = { name:"primary", run:vi.fn(), release:vi.fn() };
    const encoder = { name:"encoder", run:vi.fn().mockResolvedValue({encoder_hidden_states:{type:"float32",data:new Float32Array(1),dims:[1,1,1]}}), release:vi.fn() };
    const decoder = { name:"decoder", run:vi.fn().mockResolvedValue({logits:{type:"float32",data:Float32Array.from([10,0]),dims:[1,1,2]}}), release:vi.fn() };
    mocks.access.mockResolvedValue(undefined);
    mocks.readFile.mockImplementation(async (path: string) => JSON.stringify(path.endsWith("word-tokenizer.json")
      ? {schema:1,samples:16,prefix:[1],eos:0,maxTokens:4,suppressed:[],vocabulary:["", "a"]} : manifest));
    mocks.ortBackend.mockResolvedValueOnce(primary).mockResolvedValueOnce(encoder).mockResolvedValueOnce(decoder);
    const model = await loadNodeModel("/models/caller");
    expect(model.wordEar).toBeDefined();
    expect(mocks.ortBackend).toHaveBeenCalledOnce();
    await model.wordEar!.decode(new Float32Array(8));
    await model.wordEar!.decode(new Float32Array(8));
    expect(mocks.ortBackend).toHaveBeenCalledTimes(3);
    expect(primary.run).not.toHaveBeenCalled();
    await model.backend.release();
    expect(encoder.release).toHaveBeenCalledOnce();
    expect(decoder.release).toHaveBeenCalledOnce();
    expect(primary.release).toHaveBeenCalledOnce();
  });
});

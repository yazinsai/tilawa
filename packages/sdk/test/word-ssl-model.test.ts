import { beforeEach, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({ access: vi.fn(), readFile: vi.fn(), ortBackend: vi.fn() }));
vi.mock("node:fs/promises", async original => ({ ...await original<typeof import("node:fs/promises")>(),
  access: mocks.access, readFile: mocks.readFile }));
vi.mock("../src/ort.js", () => ({ ortBackend: mocks.ortBackend }));
import { loadNodeModel } from "../src/node.js";
const manifest = { file: "primary.onnx", tokens: "بت", inventory: "muaalem-v3_2-phonemes", supported: ["added_word"] };
const tokenizer = { schema: 1, samples: 4000, prefix: [1], eos: 0, maxTokens: 10, suppressed: [], vocabulary: ["", ""] };
const spec = { tokens: manifest.tokens, inventory: manifest.inventory, geometry: { left: 2, chunk: 4, right: 1 } };
beforeEach(() => {
  vi.resetAllMocks();
  mocks.access.mockImplementation(async (path: string) => {
    if (path.includes("teacher.onnx")) throw Object.assign(new Error("absent"), { code: "ENOENT" });
  });
  mocks.readFile.mockImplementation(async (path: string) => JSON.stringify(
    path.endsWith("manifest.json") ? manifest : path.endsWith("word-tokenizer.json") ? tokenizer : spec));
});
it("loads each independent ear once, preserves primary inference, and releases every backend", async () => {
  const primary = { name: "primary", run: vi.fn(), release: vi.fn() };
  const primaryRelease = primary.release;
  const encoder = { name: "word-encoder", run: vi.fn().mockResolvedValue({ encoder_hidden_states: {} }), release: vi.fn() };
  const decoder = { name: "word-decoder", run: vi.fn().mockResolvedValue({ logits: { data: new Float32Array([0, -10]) } }), release: vi.fn() };
  const ssl = { name: "ssl", run: vi.fn().mockResolvedValue({ log_probs: {
    type: "float32", data: Float32Array.from([0,0,0,0].flatMap(() => [-20,0,-20])), dims: [1,4,3],
  } }), release: vi.fn() };
  mocks.ortBackend.mockResolvedValueOnce(primary).mockResolvedValueOnce(encoder).mockResolvedValueOnce(decoder).mockResolvedValueOnce(ssl);
  const model = await loadNodeModel("/models/caller", { threads: 3 });
  expect(model.backend).toBe(primary);
  expect(model.insertionTeacher).toBeUndefined();
  expect(mocks.ortBackend).toHaveBeenCalledTimes(1);
  const audio = new Float32Array(3201), spans = [{ fromFrame: 0, toFrame: 4 }];
  await model.wordEar!.decode(audio); await model.wordEar!.decode(audio);
  await model.sslEar!.decodeSeams(audio, spans); await model.sslEar!.decodeSeams(audio, spans);
  expect(mocks.ortBackend).toHaveBeenCalledTimes(4);
  expect(encoder.run).toHaveBeenCalledTimes(2);
  expect(ssl.run).toHaveBeenCalledTimes(2);
  expect(primary.run).not.toHaveBeenCalled();
  await model.backend.release();
  expect(primaryRelease).toHaveBeenCalledOnce();
  for (const b of [encoder, decoder, ssl]) expect(b.release).toHaveBeenCalledOnce();
});
it("requires both ears and their complete files", async () => {
  mocks.ortBackend.mockResolvedValue({ name: "primary", release: vi.fn() });
  mocks.access.mockImplementation(async (path: string) => {
    if (path.endsWith("ssl-ear.onnx") || path.includes("teacher.onnx"))
      throw Object.assign(new Error("absent"), { code: "ENOENT" });
  });
  const model = await loadNodeModel("/models/caller");
  expect(model.wordEar).toBeUndefined(); expect(model.sslEar).toBeUndefined();
  mocks.access.mockImplementation(async (path: string) => {
    if (path.endsWith("word-decoder.onnx") || path.includes("teacher.onnx"))
      throw Object.assign(new Error("absent"), { code: "ENOENT" });
  });
  await expect(loadNodeModel("/models/caller")).rejects.toThrow("absent");
});
it("rejects mismatched phoneme inventories before installing the ears", async () => {
  mocks.ortBackend.mockResolvedValue({ name: "primary", release: vi.fn() });
  mocks.readFile.mockImplementation(async (path: string) => JSON.stringify(
    path.endsWith("manifest.json") ? manifest : path.endsWith("word-tokenizer.json") ? tokenizer : { ...spec, tokens: "بج" }));
  await expect(loadNodeModel("/models/caller")).rejects.toThrow("inventory");
});

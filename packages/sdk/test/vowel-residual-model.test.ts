import { createHash } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ files: new Map<string, Buffer>(), ort: vi.fn(), access: vi.fn() }));
vi.mock("node:fs/promises", async original => ({ ...await original<typeof import("node:fs/promises")>(),
  readFile: async (path: string, encoding?: string) => {
    const file = mocks.files.get(path.split("/").at(-1)!);
    if (!file) throw Object.assign(new Error("absent"), { code: "ENOENT" });
    return encoding ? file.toString("utf8") : file;
  }, access: mocks.access,
}));
vi.mock("../src/ort.js", () => ({ ortBackend: mocks.ort }));
import { loadNodeModel } from "../src/node.js";

const zeros = (n: number, d: number) => Array.from({ length: n }, () => Array(d).fill(0));
const manifest = { file: "model.onnx", inventory: "test", tokens: "بتَُِ", sha256: "a".repeat(64), sampleRate: 16000, frameMs: 40, fbank: { mels: 80 } };
const head = () => ({ weight: zeros(32, 32), bias: Array(32).fill(0), output: zeros(6, 32), intercept: Array(6).fill(0) });
const parameters = () => ({ schema: 1, inventory: manifest.inventory, tokens: manifest.tokens, primarySha256: manifest.sha256,
  featureOutput: "hidden", projection: zeros(32, 32), center: Array(32).fill(0), scale: Array(32).fill(1), vowels: [[3], [4], [5]],
  durations: [.5, .5], predictor: zeros(32, 10), shuffledPredictor: zeros(32, 10), residual: head(), original: head(), shuffled: head() });
function install() {
  mocks.files.set("vowel-residual.json", Buffer.from(JSON.stringify(parameters())));
  mocks.files.set("vowel-residual-features.onnx", Buffer.from("separate-feature-graph"));
  mocks.files.set("vowel-residual-assets.json", Buffer.from(JSON.stringify(Object.fromEntries(
    ["vowel-residual.json", "vowel-residual-features.onnx"].map(file => {
      const data = mocks.files.get(file)!;
      return [file, { bytes: data.length, sha256: createHash("sha256").update(data).digest("hex") }];
    })))));
}
beforeEach(() => {
  vi.resetAllMocks(); mocks.files.clear();
  mocks.files.set("manifest.json", Buffer.from(JSON.stringify(manifest)));
  mocks.access.mockRejectedValue(Object.assign(new Error("absent"), { code: "ENOENT" }));
});

describe("declared vowel residual assets", () => {
  it("retains the ordinary listener when no scorer is declared", async () => {
    const primary = { name: "primary", release: vi.fn() }; mocks.ort.mockResolvedValue(primary);
    const loaded = await loadNodeModel("/model");
    expect(loaded.backend).toBe(primary); expect(loaded.vowelResidual).toBeUndefined();
    expect(primary.release).not.toHaveBeenCalled();
  });
  it("loads every matched arm and releases the separate graph with the primary", async () => {
    install();
    const release = vi.fn(), primary = { name: "primary", release }, auxiliary = { name: "features", release: vi.fn() };
    mocks.ort.mockResolvedValueOnce(primary).mockResolvedValueOnce(auxiliary);
    const loaded = await loadNodeModel("/model", { threads: 2 });
    expect(loaded.backend).toBe(primary); expect(loaded.vowelResidual?.backend).toBe(auxiliary);
    expect(loaded.vowelResidual?.model.shuffled).toBeDefined();
    expect(mocks.ort.mock.calls[1][3]).toMatchObject({ intraOpNumThreads: 2, graphOptimizationLevel: "disabled" });
    await loaded.backend.release();
    expect(auxiliary.release).toHaveBeenCalledOnce();
    expect(release).toHaveBeenCalledOnce();
    expect(mocks.ort).toHaveBeenCalledTimes(2);
  });
  it.each(["missing", "corrupt", "wrong-primary"])("rejects a %s declared bundle before primary decoding", async failure => {
    install();
    if (failure === "missing") mocks.files.delete("vowel-residual-features.onnx");
    if (failure === "corrupt") mocks.files.set("vowel-residual-features.onnx", Buffer.from("tampered"));
    if (failure === "wrong-primary") {
      const data = Buffer.from(JSON.stringify({ ...parameters(), primarySha256: "b".repeat(64) }));
      mocks.files.set("vowel-residual.json", data);
      const descriptor = JSON.parse(mocks.files.get("vowel-residual-assets.json")!.toString());
      descriptor["vowel-residual.json"] = { bytes: data.length, sha256: createHash("sha256").update(data).digest("hex") };
      mocks.files.set("vowel-residual-assets.json", Buffer.from(JSON.stringify(descriptor)));
    }
    const release = vi.fn(), primary = { name: "primary", release }; mocks.ort.mockResolvedValue(primary);
    await expect(loadNodeModel("/model")).rejects.toThrow();
    expect(release).toHaveBeenCalledOnce(); expect(mocks.ort).toHaveBeenCalledOnce();
  });
});

import { mkdtemp,rm,writeFile,mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach,describe,expect,it,vi } from "vitest";
import type { ModelManifest } from "../src/model.js";

const backends=vi.hoisted(() => [] as { release: ReturnType<typeof vi.fn> }[]);
vi.mock("../src/ort.js",() => ({ ortBackend: async () => {
  const backend={ name: "stub",run: vi.fn(),release: vi.fn(async () => {}) };
  backends.push({ release: backend.release }); return backend;
} }));
import { loadNodeModel } from "../src/node.js";

let directory: string;
afterEach(async () => { if (directory) await rm(directory,{ recursive: true,force: true }); backends.length=0; });
const manifest: ModelManifest={ schema: 1,id: "test",format: "test",arch: "zipformer2-streaming-ctc",inventory: "test",
  tokens: "بت",blank: 0,sampleRate: 16000,frameMs: 40,fbank: { mels: 80,padValue: -23.0259 },
  window: 45,shift: 32,states: [],file: "model.onnx",bytes: 0,sha256: "" };
async function bundle(frameMs=40) {
  directory=await mkdtemp(join(tmpdir(),"tilawa-pause-model-"));
  await mkdir(join(directory,"pause"));
  await writeFile(join(directory,"manifest.json"),JSON.stringify({ ...manifest,pauseMaskModel: { manifest: "pause/manifest.json" } }));
  await writeFile(join(directory,"pause/manifest.json"),JSON.stringify({ ...manifest,frameMs,pauseMask: { input: "pause_floor",clockMs: 20 } }));
}
describe("declared pause-trained model loading",() => {
  it("resolves the auxiliary from its bundle and releases both sessions",async () => {
    await bundle();
    const model=await loadNodeModel(directory);
    expect(model.pauseModel?.manifest.pauseMask).toEqual({ input: "pause_floor",clockMs: 20 });
    expect(model.manifest.tokens).toBe(manifest.tokens);
    await model.backend.release();
    expect(backends.map(b => b.release.mock.calls.length)).toEqual([1,1]);
  });
  it("rejects a changed output clock and releases the loaded sessions",async () => {
    await bundle(80);
    await expect(loadNodeModel(directory)).rejects.toThrow("differs from the deployed primary regime");
    expect(backends.map(b => b.release.mock.calls.length)).toEqual([1,1]);
  });
});

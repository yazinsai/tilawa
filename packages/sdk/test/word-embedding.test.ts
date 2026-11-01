import { describe, expect, it, vi } from "vitest";
import { createWordEmbedding, trimWordGap } from "../src/word-embedding.js";
import type { WordEmbeddingAsset } from "../src/word-embedding.js";
import type { InferenceBackend } from "../src/model.js";

const file = { file: "unused", sha256: "unused", bytes: 0 };
const asset: WordEmbeddingAsset = { schema: 1, sampleRate: 16000, dimensions: 2, keys: ["neighbour", "candidate"],
  radius: .3, encoder: file, words: file, prefixes: file, prefixKeys: file };
const audio = Float32Array.from({ length: 8000 }, (_, i) => .2 * Math.sin(i * .17));
function model(z = [0, 1], prefix = [1, 0], radius = .3) {
  const run = vi.fn(async () => ({ embedding: { type: "float32" as const, data: new Float32Array(z), dims: [1, 2] } }));
  const backend: InferenceBackend = { name: "test", run, release: vi.fn(async () => {}) };
  return { run, word: createWordEmbedding({ ...asset, radius }, backend,
    new Float32Array([1, 0, 0, 1]), new Float32Array(prefix), new Uint32Array([0])) };
}

describe("whole-word acoustic comparison", () => {
  it("accepts a non-neighbour inside the fixed radius and feeds the actual crop duration", async () => {
    const { word, run } = model();
    expect(await word.compare(audio, 16000, ["neighbour"])).toEqual({ word: "candidate", distance: 0, neighbourDistance: 1, radius: .3 });
    const feeds = (run.mock.calls as unknown as [Parameters<InferenceBackend["run"]>[0]][])[0][0];
    expect(feeds.x.dims).toEqual([1, 48, 80]);
    expect(Array.from(feeds.duration.data)).toEqual([.5]);
  });
  it("rejects whole-word repeats, tied prefixes, and words outside calibration", async () => {
    expect(await model([1, 0]).word.compare(audio, 16000, ["neighbour"])).toBeNull();
    expect(await model([0, 1], [0, 1]).word.compare(audio, 16000, ["neighbour"])).toBeNull();
    expect(await model([.6, .8], [1, 0], .1).word.compare(audio, 16000, ["neighbour"])).toBeNull();
  });
  it("abstains before inference on missing neighbour coverage, silence, or unsupported audio", async () => {
    const { word, run } = model();
    expect(await word.compare(audio, 16000, ["missing"])).toBeNull();
    expect(await word.compare(new Float32Array(16000), 16000, ["neighbour"])).toBeNull();
    expect(await word.compare(audio, 8000, ["neighbour"])).toBeNull();
    expect(run).not.toHaveBeenCalled();
  });
  it("keeps endpoint padding around speech and removes empty input", () => {
    const padded = new Float32Array(16000);
    padded.set(audio, 4000);
    const trimmed = trimWordGap(padded);
    expect(trimmed.length).toBeGreaterThan(audio.length);
    expect(trimmed.length).toBeLessThan(padded.length);
    expect(trimWordGap(new Float32Array(16000))).toHaveLength(0);
  });
});

import { describe, expect, it, vi } from "vitest";
import { createListener } from "../src/listener.js";
import type { LoadedModel, ModelManifest, Tensor } from "../src/model.js";
import { posteriorProductModel } from "../src/posterior-product.js";

const manifest: ModelManifest = {
  schema: 1, id: "stub", format: "onnx", arch: "zipformer2-streaming-ctc", inventory: "stub",
  tokens: "ab", blank: 0, sampleRate: 16000, frameMs: 40, fbank: { mels: 80, padValue: 0 },
  window: 45, shift: 32, states: [{ name: "cache", shape: [1], dtype: "float32" }],
  file: "stub.onnx", bytes: 0, sha256: "",
};
function model(rows: number[][], width = 1): LoadedModel {
  return {
    manifest: { ...manifest, states: [{ name: "cache", shape: [width], dtype: "float32" }] },
    supported: ["added_word", "wrong_word"],
    backend: {
      name: "stub", release: vi.fn(async () => {}),
      run: vi.fn(async (feeds: Record<string, Tensor>) => {
        const step = Number(feeds.cache.data[0]);
        const row = rows[step % rows.length];
        return {
          new_cache: { type: "float32", data: new Float32Array(width).fill(step + 1), dims: [width] },
          log_probs: { type: "float32", data: Float32Array.from(Array.from({ length: 8 }, () => row).flat(), Math.log), dims: [1, 8, 3] },
        };
      }),
    },
  };
}
const tensor = (n = 1): Tensor => ({ type: "float32", data: new Float32Array(n), dims: [n] });
const feeds = () => ({ x: tensor(3600), ear0_cache: tensor(), ear1_cache: tensor(2) });

describe("equal-weight posterior product", () => {
  it("pools full distributions before selection, and keeps the posterior normalized", async () => {
    const first = model([[0.6, 0.1, 0.3]]), second = model([[0.1, 0.6, 0.3]], 2);
    const pair = posteriorProductModel(first, second);
    const input = feeds();
    const out = await pair.backend.run(input);
    const p = Array.from(out.log_probs.data.slice(0, 3), v => Math.exp(Number(v)));
    // The shared runner-up beats both individual argmaxes; a top-1 vote would lose it.
    const total = 2 * Math.sqrt(0.06) + 0.3;
    expect(p[0]).toBeCloseTo(Math.sqrt(0.06) / total, 6);
    expect(p[1]).toBeCloseTo(p[0], 6);
    expect(p[2]).toBeCloseTo(0.3 / total, 6);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(vi.mocked(first.backend.run).mock.calls[0][0].x).toBe(input.x);
    expect(vi.mocked(second.backend.run).mock.calls[0][0].x).toBe(input.x);
    expect(out.new_ear0_cache.dims).toEqual([1]);
    expect(out.new_ear1_cache.dims).toEqual([2]);
    expect(input.ear0_cache.data[0]).toBe(0);
    await pair.backend.release();
    expect(first.backend.release).toHaveBeenCalledOnce();
    expect(second.backend.release).toHaveBeenCalledOnce();
  });

  it("preserves identical ears and stays finite for extremely small probabilities", async () => {
    const p = [1e-300, 0.8, 0.2];
    const pair = posteriorProductModel(model([p]), model([p], 2));
    const out = await pair.backend.run(feeds());
    expect(Number(out.log_probs.data[0])).toBeCloseTo(Math.log(p[0]), 4);
    expect(Math.exp(Number(out.log_probs.data[1]))).toBeCloseTo(0.8, 6);
    expect(Math.exp(Number(out.log_probs.data[2]))).toBeCloseTo(0.2, 6);
  });

  it("rejects misaligned clocks, vocabularies and output frames", async () => {
    for (const change of [{ frameMs: 20 }, { tokens: "ba" }, { shift: 16 }, { fbank: { mels: 40, padValue: 0 } }]) {
      const second = model([[0.2, 0.3, 0.5]]);
      second.manifest = { ...second.manifest, ...change };
      expect(() => posteriorProductModel(model([[0.2, 0.3, 0.5]]), second)).toThrow(/matching/);
    }
    const bad = model([[0.2, 0.3, 0.5]]);
    bad.backend.run = async () => ({ log_probs: { type: "float32", data: new Float32Array(3), dims: [1, 1, 3] } });
    await expect(posteriorProductModel(model([[0.2, 0.3, 0.5]]), bad).backend.run(feeds())).rejects.toThrow(/frames/);
  });

  it("carries both caches through CTC chunks, independently for every listener and cold crop", async () => {
    const rows = [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8]];
    const pair = posteriorProductModel(model(rows), model(rows, 2));
    const audio = new Float32Array(16000 * 2);
    async function decode(step: number) {
      const listener = createListener(pair, { posteriors: true, topK: 3 });
      const chunks = [];
      for (let i = 0; i < audio.length; i += step) {
        const e = await listener.push(audio.subarray(i, i + step), 16000);
        if (e) chunks.push(e);
      }
      const e = await listener.end();
      if (e) chunks.push(e);
      return { phones: chunks.flatMap(c => c.phonemes), probs: chunks.flatMap(c => Array.from(c.posteriors!.probs)) };
    }
    const [small, large] = await Promise.all([decode(320), decode(audio.length)]);
    expect(small).toEqual(large);
    expect(small.phones.map(p => p.p)).toEqual(["a", "b", "a", "b"]);
    expect(await decode(1600)).toEqual(small);
  });
});

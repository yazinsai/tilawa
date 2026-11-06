import { describe, expect, it } from "vitest";
import type { InferenceBackend } from "../src/model.js";
import { createSslEar, sslSeamSpans, sslSeamSpellings, sslWindow } from "../src/ssl-ear.js";
import type { PhonemeEvidence, ReferenceIndex } from "@tilawa/core";

describe("undistilled SSL streaming ear", () => {
  it("spells a whole isolated surplus but never an interior lexical fragment or neighbour", () => {
    const texts = ["بَت", "ثِج", "دَذ", "رَز", "سَش", "صَض", "طَظ", "فَق"];
    let offset = 0;
    const index = { inventoryId: "test", phonemes: texts.join(""),
      words: texts.map((text, i) => {
        const start = offset; offset += text.length;
        return { surah: 1, ayah: i < 3 ? 1 : 2, word: i < 3 ? i + 1 : i - 2, start, end: offset };
      }) } as unknown as ReferenceIndex;
    const native = (text: string): PhonemeEvidence => ({ inventory: "test", frameMs: 20,
      phonemes: [...text].map((p, i) => ({ p, start: i * 2, end: i * 2, conf: 1 })),
      posteriors: { startFrame: 0, nFrames: text.length * 2, k: 1,
        ids: [...text].flatMap(() => [1, 0]), probs: new Array(text.length * 2).fill(1) } });
    const spans = [{ word: 3, fromFrame: 0, toFrame: 100, pauses: [] }];
    expect(sslSeamSpellings(native("فَق"), spans, index)).toMatchObject([{ said: "فَق" }]);
    expect(sslSeamSpellings(native("هفَقن"), spans, index)).toEqual([]);
    expect(sslSeamSpellings(native("رَز"), spans, index)).toEqual([]);
    expect(sslSeamSpellings(native("رَ"), spans, index)).toEqual([]);
  });

  it("bounds a seam by its original primary entry and word-2 step across local backsteps", () => {
    const primary: PhonemeEvidence = { inventory: "test", frameMs: 40,
      phonemes: [{ p: "ب", start: 2, end: 10, conf: 1 }, { p: "ت", start: 21, end: 30, conf: 1 }],
      posteriors: { startFrame: 0, nFrames: 40, k: 1, ids: new Array(40).fill(0), probs: new Array(40).fill(1) } };
    const locks = [[0, 2], [1, 18], [0, 25], [1, 30], [2, 40]].map(([wi, frame]) => ({ wi, timeMs: frame * 40 }));
    const expected = [{ word: 1, fromFrame: 22, toFrame: 80, pauses: [{ fromFrame: 22, toFrame: 42 }] }];
    expect(sslSeamSpans(primary, locks, word => word === 1)).toEqual(expected);
    expect(sslSeamSpans(primary, locks, () => false)).toEqual([]);
    // Blank argmax alone cannot label a pause under the folded detector.
    primary.posteriors!.probs[15] = .49;
    expect(sslSeamSpans(primary, locks, () => true)).toEqual([]);
  });

  it("requires stored rows covering the whole pause and honors their frame offset", () => {
    const primary: PhonemeEvidence = { inventory: "test", frameMs: 40,
      phonemes: [{ p: "ب", start: 2, end: 10, conf: 1 }, { p: "ت", start: 21, end: 30, conf: 1 }],
      posteriors: { startFrame: 11, nFrames: 10, k: 1, ids: new Array(10).fill(0), probs: new Array(10).fill(1) } };
    const locks = [{ wi: 0, timeMs: 80 }, { wi: 2, timeMs: 1600 }];
    expect(sslSeamSpans(primary, locks, () => true)).toHaveLength(1);
    primary.posteriors!.nFrames = 9;
    expect(sslSeamSpans(primary, locks, () => true)).toEqual([]);
  });

  it("normalizes available samples independently of padding and retains stacked phases", () => {
    const frames = [0, 2, 4, 6].map(x => new Float32Array(80).fill(x));
    const geometry = { left: 2, chunk: 2, right: 1 };
    const window = sslWindow(frames, 0, geometry);
    expect(Array.from(window.mask)).toEqual([0n, 0n, 1n, 1n, 0n]);
    expect(window.data.slice(0, 320).every(x => x === 0)).toBe(true);
    expect(window.data[320]).toBeCloseTo(-3 / Math.sqrt(20 / 3), 5);
    expect(window.data[400]).toBeCloseTo(-1 / Math.sqrt(20 / 3), 5);
  });

  it("decodes all full and partial chunks on one clock, with CTC collapse across calls", async () => {
    let calls = 0;
    const backend: InferenceBackend = { name: "mock", release: async () => {}, async run(feeds) {
      expect(feeds.input_features.dims).toEqual([1, 7, 160]);
      const labels = calls++ % 2 ? [1, 0, 1, 2] : [1, 2, 1, 1];
      const data = Float32Array.from(labels.flatMap(id => [0, 1, 2].map(i => i === id ? 0 : -20)));
      return { log_probs: { type: "float32", data, dims: [1, 4, 3] } };
    } };
    const e = await createSslEar(backend, "بت", "test", { left: 2, chunk: 4, right: 1 }).decode(new Float32Array(3201), 3);
    expect(e.frameMs).toBe(20);
    expect(e.posteriors!.nFrames).toBe(9);
    expect(calls).toBe(3);
    expect(e.posteriors!.ids.length).toBe(27);
    expect(e.phonemes.map(p => [p.p, p.start, p.end])).toEqual([["ب", 0, 0], ["ت", 1, 1], ["ب", 2, 4], ["ب", 6, 6], ["ت", 7, 7], ["ب", 8, 8]]);
  });

  it("rejects invalid geometry and a mismatched exported output", async () => {
    const backend: InferenceBackend = { name: "mock", release: async () => {}, run: async () => ({
      log_probs: { type: "float32", data: new Float32Array(1), dims: [1] },
    }) };
    expect(() => createSslEar(backend, "بت", "test", { left: 2, chunk: 0, right: 1 })).toThrow();
    await expect(createSslEar(backend, "بت", "test", { left: 2, chunk: 4, right: 1 }).decode(new Float32Array(4000), 4)).rejects.toThrow("width");
    await expect(createSslEar(backend, "بت", "test", { left: 2, chunk: 4, right: 1 }).decode(new Float32Array(4000), 3)).rejects.toThrow("output");
  });

  it("evaluates seam chunks once without moving their phase or joining tokens across gaps", async () => {
    let calls = 0;
    const masks: bigint[][] = [];
    const backend: InferenceBackend = { name: "mock", release: async () => {}, async run(feeds) {
      calls++;
      masks.push(Array.from(feeds.attention_mask.data as BigInt64Array));
      return { log_probs: { type: "float32", data: Float32Array.from([0, 0, 0, 0].flatMap(() => [-20, 0, -20])), dims: [1, 4, 3] } };
    } };
    const ear = createSslEar(backend, "بت", "test", { left: 2, chunk: 4, right: 1 });
    const samples = new Float32Array(3201);
    const full = await ear.decode(samples, 1);
    const fullMasks = masks.splice(0);
    expect(full.phonemes.map(p => [p.start, p.end])).toEqual([[0, 8]]);
    calls = 0;
    const selected = await ear.decodeSeams(samples, [
      { fromFrame: 0, toFrame: 1 }, { fromFrame: 8, toFrame: 9 }, { fromFrame: 8, toFrame: 10 },
    ]);
    expect(calls).toBe(2);
    expect(masks).toEqual([fullMasks[0], fullMasks[2]]);
    expect(selected.map(p => [p.p, p.start, p.end])).toEqual([["ب", 0, 0], ["ب", 8, 8]]);
    calls = 0;
    expect(await ear.decodeSeams(samples, [])).toEqual([]);
    expect(calls).toBe(0);
    await expect(ear.decodeSeams(samples, [{ fromFrame: 4, toFrame: 4 }])).rejects.toThrow("span");
  });
});

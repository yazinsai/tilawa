import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "../src/ctc.js";
import { createLocalSearch } from "../src/local.js";
import { boundaryFeatures, boundaryLikelihood } from "../src/word-boundary.js";
import type { BoundaryModel } from "../src/word-boundary.js";
import type { ReferenceIndex } from "../src/reference.js";

const model: BoundaryModel = { mean: [0], scale: [1], weights: [1], intercept: 0, prior: 0.25, temperature: 1 };
const index: ReferenceIndex = {
  version: "test", inventoryId: "test", tokens: new Set("abcd"), phonemes: "abac",
  words: [{ surah: 1, ayah: 1, word: 1, start: 0, end: 2 }, { surah: 1, ayah: 1, word: 2, start: 2, end: 4 }],
  gramSize: 2, grams: new Map(), wordAt: (offset) => offset < 2 ? 0 : 1,
};

function frames() {
  const f = new PosteriorFrames(2);
  f.append({ k: 3, startFrame: 0, nFrames: 3, ids: [1, 0, 2, 0, 1, 2, 2, 0, 1],
    probs: [0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1] });
  return f;
}

describe("posterior word boundary channel", () => {
  it("normalizes boundary and within likelihoods against their common acoustic pool", () => {
    for (const x of [-100, -1, 0, 1, 100]) {
      const result = boundaryLikelihood([x], model);
      expect(model.prior * Math.exp(result.boundary) + (1 - model.prior) * Math.exp(result.within)).toBeCloseTo(1, 12);
      expect(Number.isFinite(result.boundary) && Number.isFinite(result.within)).toBe(true);
    }
    const pooled = boundaryLikelihood([Math.log(model.prior / (1-model.prior))], model);
    expect(pooled.boundary).toBeCloseTo(0);
    expect(pooled.within).toBeCloseTo(0);
  });

  it("uses posterior change across the seam, not only a gap duration", () => {
    const f = frames();
    const a = { p: "a", start: 0, end: 0, conf: 0.8 }, b = { p: "b", start: 2, end: 2, conf: 0.8 };
    const result = boundaryFeatures(f, a, b, 20)!;
    expect(result).toHaveLength(13);
    expect(result[0]).toBeCloseTo(Math.log1p(0.02));
    expect(result[2]).toBeCloseTo(0.8);
    expect(result[7]).toBeCloseTo(0.8);
    expect(result[10]).toBeCloseTo(0.2);
    f.dropBefore(1);
    expect(boundaryFeatures(f, a, b, 20)).toBeNull();
  });

  it("does not invent blank evidence for adjacent decoded spikes", () => {
    const f = frames();
    const result = boundaryFeatures(f, { p: "a", start: 0, end: 0, conf: 1 }, { p: "b", start: 1, end: 1, conf: 1 }, 20)!;
    expect([result[0], result[2], result[5], result[8], result[12]]).toEqual([0, 0, 0, 0, 0]);
  });

  it("applies the same seam channel to a restart and a within-word continuation", () => {
    const start = { offset: 1, origin: 0, edits: 0 };
    const ordinary = createLocalSearch(index, start);
    ordinary.step("b");
    expect(ordinary.position().offset).toBe(2);
    const supported = createLocalSearch(index, start);
    supported.step("a", { boundary: 10, within: -10 });
    expect(supported.position().origin).toBe(2);
    const unsupported = createLocalSearch(index, start);
    unsupported.step("a", { boundary: -10, within: 10 });
    expect(unsupported.position().origin).toBe(0);
  });

  it("restores the acoustic word ownership when a phrase rewinds the local search", () => {
    const t = createLocalSearch(index, { offset: 1, origin: 0, edits: 0 });
    const snapshot = t.save();
    t.step("a", { boundary: 10, within: -10 });
    t.restore(snapshot);
    t.step("b", { boundary: -10, within: 10 });
    expect(t.position().offset).toBe(2);
    expect(t.position().origin).toBe(0);
  });
});

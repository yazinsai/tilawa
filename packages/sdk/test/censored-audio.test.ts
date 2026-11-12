import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "@tilawa/core";
import { CensoredPosteriorFrames, constantAudioIntervals, omissionDependsOnMissingAudio } from "../src/censored-audio.js";

const frames = (rows: number[][]) => {
  const f = new PosteriorFrames(rows[0].length - 1);
  f.append({ startFrame: 0, nFrames: rows.length, k: rows[0].length,
    ids: rows.flatMap(r => r.map((_, i) => i)), probs: rows.flat() });
  return f;
};

describe("constant sample candidates", () => {
  it("records exact maximal plateaus and both boundary jumps, including held nonzero samples", () => {
    expect(constantAudioIntervals(Float32Array.of(1, 0, 0, 0, -1, 2, 2, 1))).toEqual([
      { from: 1, to: 4, value: 0, entryJump: 1, exitJump: 1 },
      { from: 5, to: 7, value: 2, entryJump: 3, exitJump: 1 },
    ]);
  });
  it("does not turn low energy into unavailable samples or invent stream-edge boundaries", () => {
    expect(constantAudioIntervals(Float32Array.of(1e-8, 2e-8, 3e-8))).toEqual([]);
    expect(constantAudioIntervals(Float32Array.of(0, 0))).toEqual([
      { from: 0, to: 2, value: 0, entryJump: null, exitJump: null },
    ]);
  });
});

describe("missing observation marginalization", () => {
  it("sums over the unavailable class and retains elapsed time instead of cutting the frame", () => {
    const f = frames([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]);
    const missing = new CensoredPosteriorFrames(f, 1, [[1, 2]]);
    // For label 1, valid paths are 001, 010, 011, 100, 110, 111.
    const sum = 0.9 * 0.1 + 0.9 * 0.9 + 0.9 * 0.1 + 0.1 * 0.9 + 0.1 * 0.9 + 0.1 * 0.1;
    expect(Math.exp(-missing.nll([1], 0, 3))).toBeCloseTo(sum, 12);
    expect(missing.nll([1], 0, 3)).not.toBeCloseTo(f.nll([1], 0, 3, undefined, [1, 2]));
    expect(f.row(1)[0]).toBeCloseTo(Math.log(0.9), 12);
    expect(missing.nll([], 0, 3)).toBeCloseTo(-Math.log(0.9 * 0.9), 12);
  });
  it("abstains only on a strict reversal, and intact silence receives no exception", () => {
    const f = frames([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01]]);
    expect(omissionDependsOnMissingAudio(f, 1, [1], [], 0, 3, [[1, 2]])).toBe(true);
    expect(omissionDependsOnMissingAudio(f, 1, [1], [], 0, 3, [])).toBe(false);
    expect(omissionDependsOnMissingAudio(f, 1, [1, 1, 1], [], 0, 3, [[1, 2]])).toBe(false);
  });
  it("is identical to ordinary scoring outside the mask and rejects invalid mask bounds", () => {
    const f = frames([[0.6, 0.4], [0.7, 0.3]]);
    const missing = new CensoredPosteriorFrames(f, 1, []);
    expect(missing.nll([1], 0, 2)).toBe(f.nll([1], 0, 2));
    expect(() => new CensoredPosteriorFrames(f, 1, [[0, 3]])).toThrow(RangeError);
  });
});

import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "../src/ctc.js";

const framesOf = (rows: number[][]) => {
  const frames = new PosteriorFrames(rows[0].length - 1);
  frames.append({ startFrame: 0, nFrames: rows.length, k: rows[0].length,
    ids: rows.flatMap(row => row.map((_, id) => id)), probs: rows.flat() });
  return frames;
};

describe("erasure-tolerant CTC labels", () => {
  it("sums the phone, blank and vowel arcs once at an erased label", () => {
    // Blank, written consonant, vowel, contradictory consonant.
    const frames = framesOf([[0.4, 0.05, 0.5, 0.05]]);
    expect(frames.nll([1], 0, 1)).toBeCloseTo(-Math.log(0.05));
    expect(frames.nll([1], 0, 1, undefined, new Map([[0, [0, 2, 2, 1]]]))).toBeCloseTo(-Math.log(0.95));
  });

  it("preserves the plain and repeat forward sums when no label has alternatives", () => {
    const frames = framesOf([[0.05, 0.9, 0.05], [0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]);
    for (const repeats of [undefined, [[0, 1] as const]]) {
      expect(frames.nll([1, 2], 0, 3, repeats, new Map())).toBe(frames.nll([1, 2], 0, 3, repeats));
    }
  });

  it("cannot turn an unrelated consonant into erasure evidence", () => {
    const frames = framesOf([[0.02, 0.01, 0.02, 0.95]]);
    const relaxed = frames.nll([1], 0, 1, undefined, new Map([[0, [0, 2]]]));
    expect(relaxed - frames.nll([3], 0, 1)).toBeCloseTo(Math.log(0.95 / 0.05));
  });

  it("still requires the other written labels and a frame for each erased position", () => {
    const frames = framesOf([[0.9, 0.05, 0.05]]);
    const erased = new Map([[0, [0]]]);
    expect(frames.nll([1], 0, 1, undefined, erased)).toBeCloseTo(-Math.log(0.95));
    expect(frames.nll([1, 2], 0, 1, undefined, erased)).toBe(Infinity);
    expect(frames.nll([1], 0, 0, undefined, erased)).toBe(Infinity);
  });
});

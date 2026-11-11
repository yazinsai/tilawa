import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "../src/ctc.js";

function framesOf(rows: number[][]) {
  const frames = new PosteriorFrames(rows[0].length - 1);
  frames.append({ startFrame: 0, nFrames: rows.length, k: rows[0].length,
    ids: rows.flatMap(row => row.map((_, id) => id)), probs: rows.flat() });
  return frames;
}

/** Sum acoustic paths whose collapsed transcription equals the entire requested sequence. */
function exhaustive(rows: number[][], labels: number[]) {
  let mass = 0;
  const visit = (path: number[], probability: number) => {
    if (path.length === rows.length) {
      const text = path.filter((id, i) => id !== 0 && (i === 0 || id !== path[i - 1]));
      if (text.length === labels.length && text.every((id, i) => id === labels[i])) mass += probability;
      return;
    }
    rows[path.length].forEach((p, id) => visit([...path, id], probability * p));
  };
  visit([], 1);
  return mass;
}

describe("complete CTC label acceptance", () => {
  it("matches exhaustive complete transcriptions, including repeated labels and empty text", () => {
    const rows = [[.2, .5, .3], [.4, .4, .2], [.3, .1, .6], [.6, .2, .2]];
    const frames = framesOf(rows);
    for (const labels of [[], [1], [2], [1, 2], [1, 1], [1, 2, 1], [1, 1, 2], [1, 2, 1, 2]]) {
      expect(Math.exp(-frames.nll(labels, 0, rows.length)))
        .toBeCloseTo(exhaustive(rows, labels), 12);
    }
  });

  it("excludes a strong prefix when the full pronunciation cannot finish", () => {
    const frames = framesOf([[.01, .98, .01]]);
    expect(frames.nll([1], 0, 1)).toBeCloseTo(-Math.log(.98), 12);
    expect(frames.nll([1, 2], 0, 1)).toBe(Infinity);
    expect(frames.nll([1, 1], 0, 1)).toBe(Infinity);
  });

  it("permits final-label continuation and trailing blanks after consuming every label", () => {
    const rows = [[.1, .8, .1], [.1, .1, .8], [.1, .1, .8], [.8, .1, .1]];
    expect(Math.exp(-framesOf(rows).nll([1, 2], 0, rows.length)))
      .toBeCloseTo(exhaustive(rows, [1, 2]), 12);
  });

  it("does not finish repeated articulations without a separating blank", () => {
    expect(framesOf([[.1, .8, .1], [.1, .8, .1]]).nll([1, 1], 0, 2)).toBe(Infinity);
    const rows = [[.1, .8, .1], [.8, .1, .1], [.1, .8, .1]];
    expect(Math.exp(-framesOf(rows).nll([1, 1], 0, 3))).toBeCloseTo(.8 ** 3, 12);
  });
});

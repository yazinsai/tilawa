import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "../src/ctc.js";
import { excisedNeighbourMargins, gridSpan, surplusNasal } from "../src/seam-excision.js";
import { PHRASE_SWITCH } from "../src/parallel-phrases.js";

const tokens = "abcdefgh";
const id = (p: string) => tokens.indexOf(p) + 1;
function framesOf(text: string, blur = false) {
  const frames = new PosteriorFrames(tokens.length);
  const ids: number[] = [], probs: number[] = [];
  for (const p of text) {
    ids.push(id(p), 0, id(p) === 8 ? 7 : 8);
    probs.push(blur ? 0.42 : 0.9, 0.05, blur ? 0.5 : 0.02);
    ids.push(0, id(p), 8);
    probs.push(0.95, 0.02, 0.01);
  }
  frames.append({ startFrame: 0, nFrames: text.length * 2, k: 3, ids, probs });
  return frames;
}

describe("seam span excision", () => {
  it("preserves a surplus-adjacent nasal as nun, but leaves a bare tail to the strip", () => {
    expect(surplusNasal("وَءِںںں")).toBe("وَءِن");
    expect(surplusNasal("فَءِںںں")).toBe("فَءِن");
    expect(surplusNasal("ںںں")).toBe("ںںں");
    expect(surplusNasal("ۦںں")).toBe("ۦںں");
  });

  it("is identical to concatenating the surviving posterior rows, including repeat arcs", () => {
    const all = framesOf("abghabcdef"), remaining = framesOf("ababcdef");
    const labels = [..."abcdef"].map(id);
    const repeats: [number, number][] = [[0, 1]];
    expect(all.nll(labels, 0, all.endFrame, repeats, [4, 8]))
      .toBe(remaining.nll(labels, 0, remaining.endFrame, repeats));
    expect(all.nll(labels, 0, all.endFrame, repeats, [0, all.endFrame])).toBe(Infinity);
    expect(all.nll([], 0, all.endFrame, [], [0, all.endFrame])).toBe(0);
    expect(all.nll([..."abcdef"].map(id), 0, all.endFrame, [], [0, 8]))
      .toBe(framesOf("abcdef").nll(labels, 0, 12));
  });

  it("certifies blurred neighbours when the cut removes only the extra word", () => {
    const frames = framesOf("abghcdef", true);
    const margins = excisedNeighbourMargins(frames, id, [["ab"], ["cd"], ["ef"]], 0, frames.endFrame, [4, 8]);
    expect(margins).not.toBeNull();
    expect(margins!.every(m => m >= PHRASE_SWITCH)).toBe(true);
  });

  it("rejects a lexical cut that consumes a neighbour's only reading", () => {
    const frames = framesOf("abcdef");
    const margins = excisedNeighbourMargins(frames, id, [["ab"], ["cd"], ["ef"]], 0, frames.endFrame, [4, 8]);
    expect(margins === null || margins.some(m => m < PHRASE_SWITCH)).toBe(true);
  });

  it("does not let a favourable pausal choice weaken the omitted competitor", () => {
    const frames = framesOf("abef");
    const margins = excisedNeighbourMargins(frames, id, [["ab", "a"], ["cd"], ["ef"]], 0, frames.endFrame, [0, 2]);
    expect(margins === null || margins.some(m => m < PHRASE_SWITCH)).toBe(true);
  });
});

describe("same-frame shifted greedy spelling", () => {
  const rows = new Map([0, 1, 1, 0, 1, 2, 2, 0].map((id, f) => [f, id]));
  it("collapses repeats, retains blank-separated phones and omits earlier onsets", () => {
    expect(gridSpan(rows, ["ن", "م"], 1, 8)).toBe("ننم");
    expect(gridSpan(rows, ["ن", "م"], 2, 7)).toBe("نم");
  });
  it("fails closed on unavailable, uncovered or invalid frames", () => {
    expect(gridSpan(rows, ["ن", "م"], 0, 9)).toBeNull();
    expect(gridSpan(new Map([[0, -1]]), ["ن"], 0, 1)).toBeNull();
    expect(gridSpan(new Map([[0, 2]]), ["ن"], 0, 1)).toBeNull();
  });
});

import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "@tilawa/core";
import { localizationInformation, wordBoundaryDistribution } from "../src/localization-information.js";

function rows(probabilities: number[][], startFrame = 0) {
  const frames = new PosteriorFrames(probabilities[0].length - 1);
  frames.append({ startFrame, nFrames: probabilities.length, k: probabilities[0].length,
    ids: probabilities.flatMap(r => r.map((_, i) => i)), probs: probabilities.flat() });
  return frames;
}

// Independent enumeration of legal frame-state paths for a tiny three-phone read. The middle
// word may be repeated and the complete ayah may be repeated, including via facing-edge blanks.
function enumerate(frames: PosteriorFrames, T: number) {
  const token = [0, 1, 0, 2, 0, 3, 0];
  const total = new Map<string, number>();
  const visit = (path: number[], probability: number) => {
    if (path.length === T) {
      if (path.at(-1)! < 5) return;
      const end = path.lastIndexOf(3), start = (() => {
        let s = end;
        while (s && path[s - 1] === 3) s--;
        return s;
      })();
      const key = `${start}:${end}`;
      total.set(key, (total.get(key) ?? 0) + probability);
      return;
    }
    const previous = path.at(-1);
    for (let state = 0; state < 7; state++) {
      const ordinary = previous === undefined ? state <= 1 : state === previous || state === previous + 1 ||
        (state % 2 === 1 && state === previous + 2 && token[state] !== token[previous]);
      const repeat = previous !== undefined && ((state === 3 && previous === 4) || (state === 1 && previous >= 5));
      if (ordinary || repeat) visit([...path, state], probability * Math.exp(frames.row(path.length)[token[state]]));
    }
  };
  visit([], 1);
  const z = [...total.values()].reduce((a, b) => a + b, 0);
  return { total, z };
}

describe("word boundary localization information", () => {
  it("matches exhaustive joint path mass and the ordinary forced-read partition with repeat arcs", () => {
    const frames = rows(Array.from({ length: 7 }, (_, t) => t === 3 ? [.2, .1, .6, .1] : [.4, .2, .15, .25]));
    const repeats: [number, number][] = [[1, 1], [0, 2], [0, 2]];
    const result = wordBoundaryDistribution(frames, [1, 2, 3], [1, 1], 0, 7, repeats)!;
    const exact = enumerate(frames, 7);
    let sum = 0;
    result.joint.forEach((r, s) => r.forEach((p, d) => {
      const expected = (exact.total.get(`${s}:${s + d}`) ?? 0) / exact.z;
      expect(Math.exp(p)).toBeCloseTo(expected, 12);
      sum += Math.exp(p);
    }));
    expect(sum).toBeCloseTo(1, 12);
    expect(result.logPartition).toBeCloseTo(Math.log(exact.z), 12);
    expect(result.logPartition).toBeCloseTo(-frames.nll([1, 2, 3], 0, 7, repeats), 12);
  });

  it("has zero information when word identities are uniform despite blank and neighbour localization", () => {
    const frames = rows([[.1,.7,.1,.1], [.8,.05,.05,.1], [.1,.3,.3,.3], [.8,.05,.05,.1], [.1,.1,.1,.7]]);
    // Sharp blank/nonblank timing remains in the control even without any phone identity.
    const uniform = rows([[.1,.3,.3,.3], [.8,1/15,1/15,1/15], [.4,.2,.2,.2], [.9,1/30,1/30,1/30], [.1,.3,.3,.3]]);
    expect(localizationInformation(uniform, [1,2,3], [1,1], 0, 5)).toBeCloseTo(0, 12);
    expect(localizationInformation(frames, [1,2,3], [1,1], 0, 5)!).toBeGreaterThan(0);
  });

  it("retains the joint dependence of two-phone boundaries, including mandatory repeated-label blanks", () => {
    const frames = rows([[.1,.7,.2], [.7,.2,.1], [.1,.7,.2], [.5,.2,.3], [.1,.2,.7], [.2,.2,.6]]);
    const d = wordBoundaryDistribution(frames, [1,1,2], [0,1], 0, 6, [[0,1], [2,2], [0,2]])!;
    expect(d.logPartition).toBeCloseTo(-frames.nll([1,1,2], 0, 6, [[0,1], [2,2], [0,2]]), 12);
    expect(d.joint.flatMap(r => [...r]).reduce((s,p) => s + Math.exp(p), 0)).toBeCloseTo(1, 12);
    // Two identical labels cannot fit one or two frames: CTC requires an intervening blank.
    for (const r of d.joint) {
      expect(r[0]).toBe(-Infinity);
      if (r.length > 1) expect(r[1]).toBe(-Infinity);
    }
  });

  it("keeps absolute frame coordinates and normalizes first and final word boundaries", () => {
    const frames = rows(Array.from({ length: 5 }, () => [.4,.3,.3]), 20);
    for (const word of [[0,0], [1,1]] as const) {
      const d = wordBoundaryDistribution(frames, [1,2], word, 20, 25, [[0,1]], true)!;
      expect(d.fromFrame).toBe(20);
      expect(d.joint.flatMap(r => [...r]).reduce((s,p) => s + Math.exp(p), 0)).toBeCloseTo(1, 12);
    }
    expect(wordBoundaryDistribution(frames, [1,2], [0,0], 19, 25)).toBeNull();
    expect(wordBoundaryDistribution(frames, [1,2], [0,0], 20, 26)).toBeNull();
    expect(wordBoundaryDistribution(frames, [1,1], [0,1], 20, 22)).toBeNull();
  });
});

import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "@tilawa/core";
import type { SequenceEvidenceCandidate } from "@tilawa/core";
import { conditionedComparison } from "../src/conditioned-ctc.js";

const tokens = new Map([["a", 1], ["b", 2]]);
function setup(rows: number[][], written: string, accused: string, a: number, b: number) {
  const frames = new PosteriorFrames(2);
  frames.append({ startFrame: 0, nFrames: rows.length, k: 3,
    ids: rows.flatMap(() => [0, 1, 2]), probs: rows.flat() });
  const c: SequenceEvidenceCandidate = { word: { surah: 1, ayah: 1, word: 1 }, written, accused,
    fromFrame: a, toFrame: b, prefix: a ? [{ p: "a", start: 0, end: a - 1, conf: 1 }] : [],
    suffix: b < rows.length ? [{ p: "b", start: b, end: rows.length - 1, conf: 1 }] : [],
    margin: frames.nll([...written].map(p => tokens.get(p)!), a, b) - frames.nll([...accused].map(p => tokens.get(p)!), a, b), need: 10 };
  return { frames, c };
}
/** Enumerate raw paths independently of the DP, fixing only where phones first appear. */
function mass(rows: number[][], text: string, a: number, b: number) {
  let total = 0;
  const visit = (path: number[], prob: number) => {
    if (path.length < rows.length) { rows[path.length].forEach((p, id) => visit([...path, id], prob * p)); return; }
    const births = path.flatMap((p, f) => p && (f === 0 || path[f - 1] !== p) ? [{ p, f }] : []);
    if (births.map(x => x.p === 1 ? "a" : "b").join("") !== "a" + text + "b") return;
    if (births[0].f >= a || births.at(-1)!.f < b || births.slice(1, -1).some(x => x.f < a || x.f >= b)) return;
    total += prob;
  };
  visit([], 1); return total;
}

describe("conditioned CTC boundaries", () => {
  it("matches exhaustive paths with active-label tails and repeated-label separation", () => {
    const rows = [[.1,.8,.1],[.4,.3,.3],[.1,.4,.5],[.2,.2,.6],[.3,.1,.6]];
    const { frames, c } = setup(rows, "a", "b", 1, 3);
    const got = conditionedComparison(frames, c, tokens)!;
    expect(got.margin).toBeCloseTo(Math.log(mass(rows, "b", 1, 3) / mass(rows, "a", 1, 3)), 12);
    expect(got.incomingBlank).toBeCloseTo(0, 12);
  });
  it("preserves a fresh comparison when exterior blank rows reset both boundaries", () => {
    const rows = [[.01,.98,.01],[1,0,0],[.1,.55,.35],[.3,.2,.5],[1,0,0],[.01,.01,.98]];
    const { frames, c } = setup(rows, "a", "b", 2, 4);
    expect(conditionedComparison(frames, c, tokens)!.margin).toBeCloseTo(c.margin, 4);
  });
  it("reverses a fresh-entry advantage when the accused label is already active", () => {
    const { frames, c } = setup([[.01,.98,.01],[.05,.6,.35],[.05,.6,.35],[1,0,0],[.01,.01,.98]], "b", "a", 1, 3);
    expect(c.margin).toBeGreaterThan(0);
    const got = conditionedComparison(frames, c, tokens)!;
    expect(got.margin).toBeLessThan(0);
    expect(got.reverses).toBe(true);
  });
  it("cannot drop an interior target phone to make a repeated label cheap", () => {
    const { frames, c } = setup([[.1,.8,.1],[.1,.8,.1],[.1,.1,.8]], "aa", "b", 1, 2);
    expect(conditionedComparison(frames, c, tokens)).toBeNull();
  });
  it("abstains without exterior state or with missing rows", () => {
    const { frames, c } = setup([[.2,.4,.4]], "a", "b", 0, 1);
    expect(conditionedComparison(frames, c, tokens)).toBeNull();
    expect(conditionedComparison(frames, { ...c, toFrame: 2 }, tokens)).toBeNull();
  });
});

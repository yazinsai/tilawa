import { describe, expect, it } from "vitest";
import { LocationBelief, BELIEF_FEATURES } from "../src/location-belief.js";
import type { LocationBeliefModel } from "../src/location-belief.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { boundaryDistribution, smoothLocations } from "../train/belief-smoother.js";
import { trainingNoneLabels } from "../train/belief-eligibility.js";

const model = (control?: "nonrecurrent"): LocationBeliefModel => ({
  version: 1, input: [[1, ...new Array(BELIEF_FEATURES - 1).fill(0)]], recurrent: [[.8]],
  bias: [0], outputBias: [0], output: [1], scale: .5, control,
});
const x = (v: number) => [v, ...new Array(BELIEF_FEATURES - 1).fill(0)];
const index = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "abcdef" },
  ayahs: [{ s: 1, a: 1, w: ["ab", "cd", "ef"], p: "abcdef", o: [0, 2, 4], iso: ["ab", "cd", "ef"] }] });

describe("causal candidate memory", () => {
  it("carries memory through a jump and overlapping window recentering", () => {
    const b = new LocationBelief(model());
    b.advance(10, [x(1), x(0)], Int32Array.of(-1, -1));
    // A new run at another reference cell still descends from earlier consumed speech.
    const carried = b.advance(11, [x(0), x(0)], Int32Array.of(10, 11));
    const reset = new LocationBelief(model()).advance(11, [x(0), x(0)], Int32Array.of(10, 11));
    expect(carried[0]).toBeGreaterThan(reset[0]);
    expect(carried[1]).toBe(reset[1]);
  });

  it("rewinds memory without mutating retained snapshots", () => {
    const b = new LocationBelief(model());
    b.advance(0, [x(1)], Int32Array.of(-1));
    const saved = b.save(), original = [...saved.hidden];
    const first = b.advance(0, [x(.2)], Int32Array.of(0));
    b.advance(0, [x(.7)], Int32Array.of(0));
    expect([...saved.hidden]).toEqual(original);
    b.restore(saved);
    expect(b.advance(0, [x(.2)], Int32Array.of(0))).toEqual(first);
  });

  it("makes the same-size instantaneous control independent of previous speech", () => {
    const a = new LocationBelief(model("nonrecurrent")), b = new LocationBelief(model("nonrecurrent"));
    a.advance(0, [x(1)], Int32Array.of(-1));
    b.advance(0, [x(0)], Int32Array.of(-1));
    expect(a.advance(0, [x(.2)], Int32Array.of(0))).toEqual(b.advance(0, [x(.2)], Int32Array.of(0)));
  });

  it("rejects invalid dimensions and nonfinite calibration", () => {
    expect(() => new LocationBelief({ ...model(), scale: NaN })).toThrow();
    expect(() => new LocationBelief({ ...model(), recurrent: [[]] })).toThrow();
  });

  it("preserves baseline positions in audit mode and at zero calibration", () => {
    const start = { offset: 0, origin: 0, edits: 0 };
    const baseline = createLocalSearch(index, start);
    const audited = createLocalSearch(index, start, { observe() {} });
    const zero = createLocalSearch(index, start, { belief: { ...model(), scale: 0 } });
    for (const p of "abefcdabcdef") {
      for (const t of [baseline, audited, zero]) t.step(p);
      expect(audited.position()).toEqual(baseline.position());
      expect(zero.position()).toEqual(baseline.position());
    }
  });

  it("restores the edit search and candidate state together", () => {
    const t = createLocalSearch(index, { offset: 0, origin: 0, edits: 0 }, { belief: model() });
    t.step("a"); t.step("b");
    const saved = t.save();
    t.step("e"); t.step("f");
    const expected = t.position(), expectedState = t.save();
    t.restore(saved); t.step("e"); t.step("f");
    expect(t.position()).toEqual(expected);
    expect(t.save()).toEqual(expectedState);
  });

  it("keeps candidate memory when the actual local run origin restarts", () => {
    const ref = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "abcdefghijkl" },
      ayahs: [{ s: 1, a: 1, w: ["abcd", "efgh", "ijkl"], p: "abcdefghijkl", o: [0, 4, 8], iso: ["abcd", "efgh", "ijkl"] }] });
    const m = { ...model(), scale: 0, input: [[0, 0, 0, 0, 1, ...new Array(7).fill(0)]] };
    const start = { offset: 0, origin: 0, edits: 0 };
    const carried = createLocalSearch(ref, start, { belief: m });
    const reset = createLocalSearch(ref, start, { belief: m, resetBelief: true });
    for (const p of "abcdijk") { carried.step(p); reset.step(p); }
    expect(carried.position().origin).toBe(8);
    expect(carried.position()).toEqual(reset.position());
    const a = carried.save() as { belief: { hidden: Float64Array } };
    const b = reset.save() as { belief: { hidden: Float64Array } };
    expect(a.belief.hidden).not.toEqual(b.belief.hidden);
  });
});

describe("training supervision", () => {
  it("normalizes full-sequence location beliefs and boundary distributions", () => {
    const q = smoothLocations("abcdef", "abdef");
    for (const row of q) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(q.at(-1)!.at(-1)).toBeCloseTo(1);
    const boundary = boundaryDistribution(q, 2);
    expect(boundary.every(v => v >= 0)).toBe(true);
    expect(boundary.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(boundary.filter(p => p > .01).length).toBeGreaterThan(1);
  });

  it("uses future speech in teacher supervision while the student is prefix-only", () => {
    const first = smoothLocations("abac", "aba")[1];
    const future = smoothLocations("abac", "abac")[1];
    expect(first).not.toEqual(future);
    const a = new LocationBelief(model()), b = new LocationBelief(model());
    expect(a.advance(0, [x(.4)], Int32Array.of(-1))).toEqual(b.advance(0, [x(.4)], Int32Array.of(-1)));
    b.advance(0, [x(.7)], Int32Array.of(0));
    // Later observations replace states; an already returned prefix decision remains unchanged.
    expect(a.save().hidden).not.toEqual(b.save().hidden);
  });

  it("excludes release speakers, mixed-split speakers, scripted mistakes and repeats", () => {
    const r = { id: "clean", speaker: "train-a", split: "train", kind: "none", ayahs: ["1:1"], wav: "clean.wav" };
    const rows = [r, { ...r, id: "heldout", speaker: "heldout", split: "test" },
      { ...r, id: "mixed", speaker: "mixed" }, { ...r, id: "mixed-test", speaker: "mixed", split: "test" },
      { ...r, id: "mistake", events: [{}] }, { ...r, id: "repeat", non_errors: [{}] }];
    expect(trainingNoneLabels(rows).map(r => r.id)).toEqual(["clean"]);
  });
});

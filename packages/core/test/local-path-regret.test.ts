import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import type { LocalRegretOptions } from "../src/local.js";
import { BASE_EDIT_COSTS } from "../src/path-regret.js";
import { loadReferenceIndex } from "../src/reference.js";

const words = ["بتجده", "صضطعء", "حخذرس", "قفزثس", "نمكلو"];
const index = loadReferenceIndex({ version: "toy", text: "toy", profile: {}, inventory: { id: "toy", tokens: words.join("") },
  ayahs: [{ s: 1, a: 1, w: words, p: words.join(""), o: words.map((_, i) => i * 5), iso: words }] });
const start = { offset: 5, origin: 0, edits: 0 };
const uncertainty = { models: [{ costs: [...BASE_EDIT_COSTS] }] };
type Comparison = Parameters<NonNullable<LocalRegretOptions["onComparison"]>>[0];

describe("local complete-path regret", () => {
  it("keeps audit positions identical and reproduces every raw path cost over one interval", () => {
    const original = createLocalSearch(index, start), comparisons: Comparison[] = [];
    const audit = createLocalSearch(index, start, { uncertainty, mode: "audit", onComparison: c => comparisons.push(c) });
    let n = 0;
    for (const p of "قفزثسنمكلو") {
      n++; original.step(p); audit.step(p);
      expect(audit.position()).toEqual(original.position());
      const c = comparisons.at(-1);
      if (c) for (const t of c.trajectories) {
        expect(t.observations).toBeLessThan(n + 1);
        expect(t.fixedCost + t.counts.reduce((s, count, j) => s + count * BASE_EDIT_COSTS[j], 0)).toBeCloseTo(t.acousticCost, 8);
      }
    }
    expect(comparisons.length).toBeGreaterThan(0);
    expect(comparisons.every(c => c.trajectories.every(t => t.observations === c.trajectories[0].observations))).toBe(true);
    expect(comparisons.some(c => c.trajectories.some(t => t.fixedCost >= 20))).toBe(true);
  });

  it("selects an existing trajectory's entire metadata and preserves rewind/replay", () => {
    const comparisons: Comparison[] = [];
    const local = createLocalSearch(index, start, { uncertainty: { models: [
      { costs: [1, ...BASE_EDIT_COSTS.slice(1)] }, { costs: [30, ...BASE_EDIT_COSTS.slice(1)] },
    ] }, onComparison: c => comparisons.push(c) });
    const step = (p: string) => {
      const before = comparisons.length;
      local.step(p);
      if (comparisons.length > before) {
        const c = comparisons.at(-1)!, selected = c.trajectories[c.decision.selected];
        const got = local.position();
        expect({ offset: got.offset, origin: got.origin, edits: got.edits }).toEqual({
          offset: selected.offset, origin: selected.origin, edits: selected.edits,
        });
      }
      return local.position();
    };
    for (const p of "قفز") step(p);
    const saved = local.save(), before = comparisons.length;
    const positions = [..."ثسنمكلو"].map(step);
    const decisions = comparisons.slice(before);
    local.restore(saved); comparisons.length = before;
    const replay = [..."ثسنمكلو"].map(step);
    expect(replay).toEqual(positions);
    expect(comparisons.slice(before)).toEqual(decisions);
    for (const c of comparisons) {
      expect(c.decision.selected).toBeLessThan(c.trajectories.length);
      expect(c.decision.regrets[c.decision.selected]).toBe(Math.min(...c.decision.regrets));
    }
  });

  it("owns input models and callback witnesses", () => {
    const costs: number[] = [...BASE_EDIT_COSTS], first: Comparison[] = [], second: Comparison[] = [];
    const make = (onComparison: NonNullable<LocalRegretOptions["onComparison"]>) => createLocalSearch(index, start,
      { uncertainty: { models: [{ costs }] }, mode: "audit", onComparison });
    const a = make(c => { first.push(structuredClone(c)); c.trajectories[0].counts = Array(9).fill(999); });
    const b = make(c => second.push(c));
    costs[0] = 100;
    for (const p of "قفزثسنمكلو") { a.step(p); b.step(p); }
    expect(first.length).toBeGreaterThan(0);
    expect(first).toEqual(second);
    expect(a.position()).toEqual(b.position());
  });

  it("leaves exact in-order progress unchanged under a singleton original model", () => {
    const a = createLocalSearch(index, start), b = createLocalSearch(index, start, { uncertainty });
    for (const p of words.slice(1).join("")) { a.step(p); b.step(p); expect(b.position()).toEqual(a.position()); }
  });
});

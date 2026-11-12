import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import type { EditOrderComparison, EditOrderModel, LocalEditOrderOptions } from "../src/edit-order.js";

const ayahs = ["بتث جحخ", "دذر زسش", "صضط ظعغ", "فقك لمن", "هوي بتث"];
const index = loadReferenceIndex({ version: "toy", text: "toy", profile: {}, inventory: { id: "toy", tokens: ayahs.join("").replaceAll(" ", "") },
  ayahs: ayahs.map((text, i) => ({ s: 1, a: i + 1, w: text.split(" "), p: text.replaceAll(" ", ""), o: [0, 3], iso: text.split(" ") })) });
const start = { offset: 3, origin: 0, edits: 0 };
const model: EditOrderModel = { version: 1, ordered: Array.from({ length: 25 }, (_, i) => Math.sin(i)), shuffled: Array(25).fill(0), scale: 1 };
const heard = "جحخدذرزسشجحخدذرصضطظعغفقك";
const run = (options?: LocalEditOrderOptions) => {
  const search = createLocalSearch(index, start, options);
  return [...heard].map(p => { search.step(p); return search.position(); });
};

describe("local edit-order recurrence", () => {
  it("preserves original decisions in audit and zero-calibration modes over progress, repeats and skips", () => {
    const comparisons: EditOrderComparison[] = [];
    expect(run({ model, mode: "audit", onComparison: c => comparisons.push(c) })).toEqual(run());
    expect(comparisons.length).toBeGreaterThan(0);
    expect(run({ model: { ...model, scale: 0 }, mode: "ordered" })).toEqual(run());
    expect(run({ model, mode: "shuffled" })).toEqual(run());
    for (const c of comparisons) {
      expect(c.candidates.some(p => p.offset === c.weightedSelected)).toBe(true);
      expect(c.candidates.every(p => p.counts[0] + p.counts[1] + p.counts[2] === c.observations)).toBe(true);
    }
  });

  it("restores complete recurrent evidence and takes ownership of model weights and diagnostic counts", () => {
    const comparisons: EditOrderComparison[] = [];
    const mutable = { ...model, ordered: [...model.ordered] };
    const a = createLocalSearch(index, start, { model: mutable, mode: "ordered", onComparison: c => comparisons.push(c) });
    const b = createLocalSearch(index, start, { model, mode: "ordered" });
    mutable.ordered.fill(1000);
    for (const p of heard.slice(0, 8)) { a.step(p); b.step(p); }
    const saved = a.save();
    const replay = () => [...heard.slice(8)].map(p => { a.step(p); return a.position(); });
    const first = replay();
    a.restore(saved);
    expect(replay()).toEqual(first);
    for (const p of heard.slice(8)) b.step(p);
    expect(a.position()).toEqual(b.position());
    if (comparisons.length) {
      const before = [...comparisons[0].candidates[0].counts];
      a.step("ب");
      expect(comparisons[0].candidates[0].counts).toEqual(before);
    }
  });
});

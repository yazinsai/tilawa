import { expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import type { LocalParameterComparison } from "../src/local.js";
import { fitEditParameters } from "../src/edit-parameter.js";
import type { ReferenceIndex } from "../src/reference.js";

function fixture() {
  let phonemes = "";
  const words = ["بتد", "درز", "سشص", "بتد", "فنق"].map((p, i) => {
    const start = phonemes.length; phonemes += p;
    return { surah: 1, ayah: 1, word: i + 1, start, end: phonemes.length };
  });
  const phones = [...new Set(phonemes)];
  const index: ReferenceIndex = { phonemes, words, tokens: new Set(phones), version: "unit", inventoryId: "unit",
    gramSize: 4, grams: new Map(), wordAt: offset => {
      for (let i = words.length - 1; i >= 0; i--) if (words[i].start <= offset) return i;
      return 0;
    } };
  const posterior = fitEditParameters([{ written: phonemes, heard: phonemes }], phones, 4);
  return { index, posterior };
}

const start = { offset: 0, origin: 0, edits: 0 };

it("accounts for the entire heard prefix without changing the original audit-mode recurrence", () => {
  const { index, posterior } = fixture();
  let n = 0;
  const comparisons: LocalParameterComparison[] = [];
  const original = createLocalSearch(index, start);
  const audit = createLocalSearch(index, start, { posterior, mode: "audit", onComparison(c) {
    expect(c.trajectories.every(p => p.statistics.observations === n)).toBe(true);
    comparisons.push(c);
  } });
  for (const p of "بتدبتدفنق") {
    n++; original.step(p); audit.step(p);
    expect(audit.position()).toEqual(original.position());
  }
  expect(comparisons).toHaveLength(1);
  const c = comparisons[0], restart = c.trajectories[c.weightedSelected];
  expect(restart.origin).toBe(9);
  expect(restart.statistics.observations).toBe(7);
  expect(restart.statistics.fixedCost).toBeCloseTo(20.006, 12);
  expect(restart.acousticCost).toBeCloseTo(20.006, 12);
});

it("rewinds parameter counts and leaves owned audit snapshots independent of retained history", () => {
  const { index, posterior } = fixture();
  const comparisons: LocalParameterComparison[] = [];
  const audit = createLocalSearch(index, start, { posterior, mode: "audit", onComparison(c) { comparisons.push(c); } });
  for (const p of "بتدبتد") audit.step(p);
  const saved = audit.save();
  audit.step("ف");
  const expected = audit.position(), comparison = structuredClone(comparisons[0]);
  comparisons[0].trajectories.forEach(p => p.statistics.counts.fill(999));
  audit.step("ن");
  audit.restore(saved); audit.step("ف");
  expect(audit.position()).toEqual(expected);
  expect(comparisons[1]).toEqual(comparison);
});

it("uses an existing complete trajectory in either likelihood mode and freezes the supplied model", () => {
  const { index, posterior } = fixture();
  for (const mode of ["mean", "marginal"] as const) {
    const comparisons: LocalParameterComparison[] = [];
    const input = structuredClone(posterior);
    const local = createLocalSearch(index, start, { posterior: input, mode, onComparison(c) { comparisons.push(c); } });
    input.operations.forEach(row => row.fill(100000));
    const unmodified = createLocalSearch(index, start, { posterior, mode });
    for (const p of "بتدبتدفنق") {
      local.step(p); unmodified.step(p);
      expect(local.position()).toEqual(unmodified.position());
    }
    expect(comparisons.length).toBeGreaterThan(0);
    // Recreate the first comparison alone so its selected existing endpoint can be checked before later progress.
    const first = comparisons[0], selected = first.trajectories[first.decision[mode].selected];
    const atFirst = createLocalSearch(index, start, { posterior, mode });
    for (const p of "بتدبتدف") atFirst.step(p);
    expect(atFirst.position().offset).toBe(selected.offset);
    expect(atFirst.position().origin).toBe(selected.origin);
  }
});

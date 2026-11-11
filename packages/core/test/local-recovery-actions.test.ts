import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import type { ReferenceIndex } from "../src/reference.js";

const index: ReferenceIndex = {
  version: "test", inventoryId: "test", tokens: new Set("بتمس"), phonemes: "بتمسبتمس",
  words: [
    { surah: 1, ayah: 1, word: 1, start: 0, end: 2 },
    { surah: 1, ayah: 1, word: 2, start: 2, end: 4 },
    { surah: 1, ayah: 2, word: 1, start: 4, end: 6 },
    { surah: 1, ayah: 2, word: 2, start: 6, end: 8 },
  ], gramSize: 4, grams: new Map(),
  wordAt: offset => Math.min(3, Math.max(0, Math.floor(offset / 2))),
};
const start = { offset: 2, origin: 0, edits: 0 };

describe("retained local recovery actions", () => {
  it("preserves search costs and positions when the observer keeps the selected action", () => {
    const plain = createLocalSearch(index, start);
    const observed = createLocalSearch(index, start, (_previous, selected, actions) => {
      expect(actions.some(a => a.offset === selected)).toBe(true);
      expect(actions.every(a => Number.isFinite(a.cost))).toBe(true);
      return selected;
    });
    for (const p of "مسبتمس") {
      plain.step(p); observed.step(p);
      expect(observed.position()).toEqual(plain.position());
    }
  });

  it("selects an existing alternative and can restore the pre-action state", () => {
    const search = createLocalSearch(index, start, (_previous, selected, actions) => actions.find(a => a.offset !== selected)?.offset);
    const saved = search.save();
    const before = search.position();
    search.step("م");
    const intervened = search.position();
    search.restore(saved);
    expect(search.position()).toEqual(before);
    search.step("م");
    expect(search.position()).toEqual(intervened);
  });

  it("rejects actions outside the existing search state", () => {
    const search = createLocalSearch(index, start, () => 100);
    expect(() => search.step("م")).toThrow("not a retained local endpoint");
  });
});

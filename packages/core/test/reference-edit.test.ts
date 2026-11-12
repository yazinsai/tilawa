import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import { editedReference, jumpFeature } from "../src/reference-edit.js";
import type { ReferenceIndex } from "../src/reference.js";

function reference(parts: string[]): ReferenceIndex {
  let phonemes = "";
  const words = parts.map((p, i) => { const start = phonemes.length; phonemes += p;
    return { surah: 1, ayah: 1, word: i + 1, start, end: phonemes.length }; });
  return { version: "test", inventoryId: "test", tokens: new Set(phonemes), phonemes, words,
    gramSize: 4, grams: new Map(), wordAt: at => { let i = words.length - 1; while (i > 0 && words[i].start > at) i--; return i; } };
}

describe("paired reference copies", () => {
  it("keeps exact ordinal correspondence for unchanged words under all three edits", () => {
    const original = reference(["abc", "def", "ghi", "jkl"]);
    for (const kind of ["delete", "insert", "substitute"] as const) {
      const copy = editedReference(original, 1, kind, "xyz");
      copy.correspondence.forEach((ordinal, i) => {
        if (ordinal < 0) return;
        const a = original.words[ordinal], b = copy.index.words[i];
        expect(copy.index.phonemes.slice(b.start, b.end)).toBe(original.phonemes.slice(a.start, a.end));
        expect(copy.index.wordAt(b.start)).toBe(i);
      });
      expect(copy.correspondence).toEqual(kind === "delete" ? [0, 2, 3] : kind === "insert" ? [0, -1, 1, 2, 3] : [0, -1, 2, 3]);
    }
    expect(original.phonemes).toBe("abcdefghijkl");
  });

  it("distinguishes a one-word lexical jump from forward movement and repeat", () => {
    const index = reference(["abc", "def", "ghi", "jkl"]);
    expect(jumpFeature(index, 4, 6)).toBe(3);
    expect(jumpFeature(index, 4, 9)).toBe(4);
    expect(jumpFeature(index, 4, 0)).toBe(5);
  });

  it("identity weights reproduce every position and preserve rewind state", () => {
    const index = reference(["abc", "def", "ghi", "jkl"]);
    const start = { offset: 0, origin: 0, edits: 0 };
    const baseline = createLocalSearch(index, start);
    const corrected = createLocalSearch(index, start, { logScale: [0, 0, 0, 0, 0, 0] });
    for (const p of "abcghixjkl") { baseline.step(p); corrected.step(p); expect(corrected.position()).toEqual(baseline.position()); }
    const saved = corrected.save(), before = corrected.position();
    corrected.step("z"); corrected.restore(saved);
    expect(corrected.position()).toEqual(before);
  });

  it("can price a lexical discrepancy separately while reporting its unchanged raw fit cost", () => {
    const index = reference(["abc", "def", "ghi", "jkl"]);
    const start = { offset: 3, origin: 0, edits: 0 };
    const t = createLocalSearch(index, start, { logScale: [0, 0, 0, -2, 0, 0] });
    t.step("g");
    expect(t.position().origin).toBe(6);
    expect(t.position().stepCost).toBeCloseTo(2.0003);
    for (const p of "hijkl") t.step(p);
    expect(t.position().offset).toBe(index.phonemes.length);
  });
});

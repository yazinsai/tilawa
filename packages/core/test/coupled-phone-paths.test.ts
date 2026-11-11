import { describe, expect, it } from "vitest";
import { coupledPhoneCosts } from "../src/coupled.js";
import type { CoupledPath } from "../src/coupled.js";
import { createLocalSearch } from "../src/local.js";
import { coupledLocalCosts } from "../src/tracker.js";
import type { ReferenceIndex } from "../src/reference.js";

const path = (start: number, assignments: readonly (number | null)[], cost: number): CoupledPath => ({
  start, cost, phones: assignments.map((observed, i) => ({ at: start + i, entry: 0, consumes: false,
    observed, frontier: observed ?? assignments.slice(0, i).filter(p => p !== null).length })),
});

describe("paired pronunciation binding", () => {
  it("preserves already identical shared assignments exactly", () => {
    const left = path(0, [0, 1], 0), right = path(2, [0, 1], 20);
    right.phones[0].entry = 20;
    right.phones[0].consumes = true;
    expect(coupledPhoneCosts("ABAB", "AB", left, right)).toEqual({
      left: 0, right: 20, correspondences: 1, disputed: 0,
    });
  });

  it("retains separate edit prices on differing spans and is symmetric", () => {
    const left = path(0, [0, 1], 0), right = path(2, [null, 0, 1], 10);
    const forward = coupledPhoneCosts("ABXAB", "AB", left, right);
    const reverse = coupledPhoneCosts("ABXAB", "AB", right, left);
    expect(forward.correspondences).toBe(1);
    expect(forward.disputed).toBe(0);
    expect(forward.left).toBeCloseTo(0);
    expect(forward.right).toBeCloseTo(10);
    expect(reverse.left).toBe(forward.right);
    expect(reverse.right).toBe(forward.left);
  });

  it("normalizes all four expanded-noon correspondences without extra match credit", () => {
    const left = path(0, [0, null, null, null, 1, 2], 6);
    const right = path(6, [0, 1, 2], 0);
    const result = coupledPhoneCosts("ننننذرنذر", "نذر", left, right);
    expect(result.correspondences).toBe(4);
    expect(result.disputed).toBe(3);
    expect(result.left).toBeCloseTo(6);
    expect(result.right).toBeCloseTo(0);
  });

  it("leaves pronunciations with no common phones unchanged", () => {
    expect(coupledPhoneCosts("ABXY", "AB", path(0, [0, 1], 0), path(2, [0, 1], 20)))
      .toEqual({ left: 0, right: 20, correspondences: 0, disputed: 0 });
  });

  it("charges both candidates when their shared phone requires incompatible token assignments", () => {
    // AY puts Y on observation 1, whereas YB puts it on 0. A common deletion or substitution must resolve this.
    const left = path(0, [0, 1], 10), right = path(2, [0, 1], 10);
    const result = coupledPhoneCosts("AYYB", "AB", left, right);
    expect(result.disputed).toBe(1);
    expect(result.left).toBeGreaterThan(left.cost);
    expect(result.right).toBeGreaterThan(right.cost);
    const reverse = coupledPhoneCosts("AYYB", "AB", right, left);
    expect(reverse.left).toBeCloseTo(result.right);
    expect(reverse.right).toBeCloseTo(result.left);
  });

  it("can change the preferred path using binding alone, while retaining a private entry price", () => {
    // A short-vowel deletion and a consonant deletion have different existing prices. Independent optima put
    // their common Y on opposite observations; symmetric coupling need not raise the two costs equally.
    const left = path(0, [0, 1], 10), right = path(2, [0, 1], 11);
    right.phones[1].entry = 1;
    const result = coupledPhoneCosts("َYYب", "َب", left, right);
    expect(left.cost).toBeLessThan(right.cost);
    expect(result.right).toBeLessThan(result.left);
    const plain = coupledPhoneCosts("َYYب", "َب", left, path(2, [0, 1], 10));
    expect(result.right - plain.right).toBeCloseTo(1);
    expect(result.left).toBeCloseTo(plain.left);
  });
});

const index = (): ReferenceIndex => {
  const phonemes = "ABCDEABXDE";
  const words = [{ surah: 1, ayah: 1, word: 1, start: 0, end: 5 },
    { surah: 1, ayah: 1, word: 2, start: 5, end: 10 }];
  return { version: "synthetic", inventoryId: "synthetic", tokens: new Set(phonemes), phonemes, words,
    gramSize: 4, grams: new Map(), wordAt: at => at < 5 ? 0 : 1 };
};

describe("local paired comparison", () => {
  it("provenance and a neutral comparison preserve the original recurrence, including a repeated word", () => {
    const ref = index(), start = { offset: 0, origin: 0, edits: 0 };
    const original = createLocalSearch(ref, start);
    const instrumented = createLocalSearch(ref, start, (a, b) => ({ left: a.cost, right: b.cost }));
    for (const p of "ABCDEABCDEABXDE") {
      original.step(p); instrumented.step(p);
      expect(instrumented.position()).toEqual(original.position());
    }
  });

  it("rewinds immutable paired histories and reproduces subsequent choices and costs", () => {
    const ref = index(), search = createLocalSearch(ref, { offset: 0, origin: 0, edits: 0 },
      (a, b) => coupledLocalCosts(ref.phonemes, a, b));
    for (const p of "ABCDE") search.step(p);
    const saved = search.save();
    const tail = () => [..."ABCDEABXDE"].map(p => { search.step(p); return search.position(); });
    const expected = tail();
    search.restore(saved);
    expect(tail()).toEqual(expected);
  });
});

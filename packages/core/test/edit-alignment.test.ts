import { describe, expect, it } from "vitest";
import { editAlignmentLogP, fitEditAlignment, logAdd } from "../src/edit-alignment.js";
import type { EditAlignmentModel } from "../src/edit-alignment.js";
import { editTrainingReference } from "../src/edit-alignment-training.js";
import type { EditTrainingLabel } from "../src/edit-alignment-training.js";
import type { ReferenceIndex } from "../src/reference.js";
import { createLocalSearch } from "../src/local.js";

const model: EditAlignmentModel = { phones: ["a", "b"], insertion: 0.1, inserted: [0.6, 0.4],
  deleted: [0.1, 0.2], emitted: [[0.9, 0.1], [0.2, 0.8]] };
const probability = (w: string, h: string) => Math.exp(editAlignmentLogP(w, h, model));

describe("normalized edit alignment marginal", () => {
  it("sums all three one-token routes with their divided probability mass", () => {
    const match = 0.9 * 0.9 * 0.9 * 0.9;
    const insertionDeletion = 0.1 * 0.6 * 0.9 * 0.1 * 0.9;
    expect(probability("a", "a")).toBeCloseTo(match + 2 * insertionDeletion, 12);
    expect(Math.exp(editAlignmentLogP("a", "a", model, "max"))).toBeCloseTo(match, 12);
  });

  it("has total output probability one, including empty speech and arbitrary insertions", () => {
    let total = 0, strings = [""];
    for (let n = 0; n <= 12; n++) {
      total += strings.reduce((sum, s) => sum + probability("ab", s), 0);
      strings = strings.flatMap(s => [s + "a", s + "b"]);
    }
    expect(total).toBeLessThanOrEqual(1 + 1e-12);
    expect(total).toBeCloseTo(1, 9);
    expect(probability("", "aa")).toBeCloseTo(0.06 ** 2 * 0.9, 12);
  });

  it("can reverse a minimum-path preference through normalized alignment ambiguity", () => {
    const ambiguous: EditAlignmentModel = { phones: ["a"], insertion: 0.01, inserted: [1],
      deleted: [0.6], emitted: [[1]] };
    expect(editAlignmentLogP("a", "a", ambiguous, "max")).toBeGreaterThan(editAlignmentLogP("aa", "a", ambiguous, "max"));
    expect(editAlignmentLogP("a", "a", ambiguous)).toBeLessThan(editAlignmentLogP("aa", "a", ambiguous));
  });

  it("fits deterministically with normalized, nonzero rows and keeps exact speech ahead", () => {
    const pairs = Array.from({ length: 50 }, () => ({ written: "aabbaabb", heard: "aabbaabb" }));
    const fitted = fitEditAlignment(["a", "b"], pairs);
    expect(fitted).toEqual(fitEditAlignment(["a", "b"], pairs));
    for (const row of [fitted.inserted, ...fitted.emitted]) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      expect(row.every(p => p > 0 && p < 1)).toBe(true);
    }
    for (let w = 0; w < 2; w++) {
      const mass = fitted.insertion + (1 - fitted.insertion) * (fitted.deleted[w]
        + (1 - fitted.deleted[w]) * fitted.emitted[w].reduce((a, b) => a + b, 0));
      expect(mass).toBeCloseTo(1, 12);
    }
    expect(editAlignmentLogP("aabbaabb", "aabbaabb", fitted)).toBeGreaterThan(editAlignmentLogP("bbaabbaa", "aabbaabb", fitted));
    expect(() => editAlignmentLogP("c", "a", fitted)).toThrow("inventory");
  });

  it("stays finite on long evidence and does not erase a difficult prefix", () => {
    expect(logAdd(-Infinity, -1000)).toBe(-1000);
    expect(logAdd(-1000, -1001)).toBeCloseTo(-1000 + Math.log1p(Math.exp(-1)), 12);
    expect(Number.isFinite(editAlignmentLogP("ab".repeat(500), "ba".repeat(500), model))).toBe(true);
    expect(editAlignmentLogP("ab", "bbbbab", model)).toBeLessThan(editAlignmentLogP("ab", "ab", model));
  });

  it("removes scripted omissions and expands repeats before fitting acoustic errors", () => {
    const index = { phonemes: "aabbaa", words: [
      { surah: 1, ayah: 1, word: 1, start: 0, end: 2 },
      { surah: 1, ayah: 1, word: 2, start: 2, end: 4 },
      { surah: 1, ayah: 1, word: 3, start: 4, end: 6 },
    ] } as ReferenceIndex;
    const label: EditTrainingLabel = { split: "train", scored: true, kind: "none", ayahs: ["1:1"], events: [], non_errors: [], unsupported: [] };
    expect(editTrainingReference(index, { ...label, kind: "skip_word", events: [
      { errorClass: "omitted_word", location: index.words[1] },
    ] })).toBe("aaaa");
    expect(editTrainingReference(index, { ...label, kind: "repeat", non_errors: [
      { kind: "repeat", location: index.words[1] },
    ] })).toBe("aabbbbaa");
    expect(editTrainingReference(index, { ...label, kind: "substitution" })).toBeNull();
    expect(() => editTrainingReference(index, { ...label, split: "eval" })).toThrow("held-out");
    expect(editTrainingReference(index, { ...label, excluded: "not audible" })).toBeNull();
  });

  it("retains ordinary progress and rewinds marginalized local state", () => {
    const text = "abcdefghijklmnopqrstuvwxy";
    const words = Array.from({ length: 5 }, (_, i) => ({ surah: 1, ayah: 1, word: i + 1, start: i * 5, end: i * 5 + 5 }));
    const index = { phonemes: text, words, wordAt: (at: number) => Math.min(4, Math.floor(at / 5)) } as ReferenceIndex;
    const trained = fitEditAlignment([...text], Array.from({ length: 100 }, () => ({ written: text, heard: text })));
    const local = createLocalSearch(index, { offset: 0, origin: 0, edits: 0 }, trained);
    for (const [i, p] of [...text.slice(0, 10)].entries()) {
      local.step(p);
      expect(local.position().offset).toBe(i + 1);
      expect(local.position().origin).toBe(0);
    }
    expect(local.position().offset).toBe(10);
    expect(local.position().origin).toBe(0);
    expect(local.position().recentCost).toBe(0);
    const saved = local.save();
    for (const p of text.slice(15)) local.step(p);
    const result = local.position();
    expect(result.offset).toBe(25);
    expect(result.origin).toBe(15);
    local.restore(saved);
    for (const p of text.slice(15)) local.step(p);
    expect(local.position()).toEqual(result);
  });
});

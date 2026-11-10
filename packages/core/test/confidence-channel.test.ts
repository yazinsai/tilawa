import { describe, expect, it } from "vitest";
import { confidenceCost, confidenceLogLikelihood, confidenceObservation, fitConfidenceChannel } from "../src/confidence-channel.js";
import type { ConfidenceRecording } from "../src/confidence-channel.js";
import { createLocalSearch } from "../src/local.js";
import type { ReferenceIndex } from "../src/reference.js";

const clip = (written: string, heard: string, conf: number): ConfidenceRecording => ({ written, heard: [...heard].map(p => ({ p, conf })) });

describe("confidence-bearing edit channel", () => {
  // Correct tokens are reliable, while acoustic substitutions have low confidence. No word/location features.
  const records = Array.from({ length: 12 }, () => [clip("بت", "بت", 0.95), clip("بت", "تت", 0.15)]).flat();
  const fitted = fitConfidenceChannel(records, ["ب", "ت", "د"], 2, 2);

  it("learns a confidence distinction while keeping every joint emission distribution normalized", () => {
    for (const row of fitted.identity) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    for (const row of [...Object.values(fitted.confidence), ...fitted.background]) {
      expect(row.every(p => p > 0 && p <= 1)).toBe(true);
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    }
    const difference = (c: number) => confidenceCost(fitted, "ب", "ت", c) - confidenceCost(fitted, "ت", "ت", c);
    expect(difference(0.15)).toBeLessThan(0);
    expect(difference(0.95)).toBeGreaterThan(0);
    expect(Number.isFinite(confidenceLogLikelihood(fitted, clip("بت", "تت", 0.15)))).toBe(true);
  });

  it("shares evidence with unseen pairs in the same phonetic classes", () => {
    expect(confidenceCost(fitted, "د", "ت", 0.15)).toBe(confidenceCost(fitted, "ب", "ت", 0.15));
  });

  it("contributes no preference when confidence is missing or marginalized", () => {
    for (const c of [undefined, NaN, Infinity, -0.1, 1.1]) expect(confidenceCost(fitted, "ب", "ت", c)).toBe(0);
    expect(confidenceCost(fitted, "unknown", "ت", 0.1)).toBe(0);
    expect(confidenceCost(fitted, "ب", "unknown", 0.1)).toBe(0);
    const row = fitted.background[1];
    const ratioMass = row.reduce((sum, p, b) => sum + p * Math.exp(-3 * confidenceCost(fitted, "ب", "ت", (b + 0.5) / fitted.bins) / 10), 0);
    expect(ratioMass).toBeCloseTo(1, 12);
    const pooled = fitConfidenceChannel(records, ["ب", "ت", "د"], 1, 2);
    expect(confidenceCost(pooled, "ب", "ت", 0.15)).toEqual(0);
  });

  it("includes insertion confidence and accepts both endpoints without a cutoff", () => {
    const observation = confidenceObservation(fitted, "ت", 1);
    expect(observation.substitution("ب")).toBe(confidenceCost(fitted, "ب", "ت", 1));
    expect(observation.insertion).toBe(confidenceCost(fitted, null, "ت", 1));
    expect(Number.isFinite(confidenceCost(fitted, null, "ت", 0))).toBe(true);
  });

  it("fits ambiguous edits and unequal lengths without assigning one fixed token alignment", () => {
    const ambiguous = fitConfidenceChannel([clip("ببت", "بت", 0.9), clip("بت", "ببت", 0.2)], ["ب", "ت"], 2, 4);
    expect(Number.isFinite(confidenceLogLikelihood(ambiguous, clip("ببت", "بت", 0.9)))).toBe(true);
    expect(Number.isFinite(confidenceLogLikelihood(ambiguous, clip("بت", "ببت", 0.2)))).toBe(true);
    expect(() => fitConfidenceChannel([clip("x", "ب", 0.5)], ["ب", "ت"], 2, 4)).toThrow("outside inventory");
  });

  it("compares confidence densities on a common scale across candidate bin counts", () => {
    const uniform = (bins: number) => ({ ...fitted, bins,
      confidence: Object.fromEntries(Object.keys(fitted.confidence).map(k => [k, Array(bins).fill(1 / bins) as number[]])),
      background: fitted.background.map(() => Array(bins).fill(1 / bins) as number[]),
    });
    const recording = clip("ببت", "بت", 0.9);
    expect(confidenceLogLikelihood(uniform(1), recording)).toBeCloseTo(confidenceLogLikelihood(uniform(8), recording), 12);
  });

  const index: ReferenceIndex = {
    version: "test", inventoryId: "test", tokens: new Set(["ب", "ت", "د"]), phonemes: "بتد",
    words: [{ surah: 1, ayah: 1, word: 1, start: 0, end: 1 }, { surah: 1, ayah: 1, word: 2, start: 1, end: 2 }, { surah: 1, ayah: 1, word: 3, start: 2, end: 3 }],
    gramSize: 4, grams: new Map(), wordAt: offset => Math.min(2, Math.max(0, offset)),
  };
  const start = { offset: 0, origin: 0, edits: 0 };

  it("passes confidence into local substitution costs and restores every score when rewound", () => {
    const low = createLocalSearch(index, start, fitted), high = createLocalSearch(index, start, fitted);
    const saved = low.save();
    low.step("ت", 0.15);
    high.step("ت", 0.95);
    // The written onset is ب; only reliability differs between these two observations.
    expect(low.position().stepCost).toBeLessThan(high.position().stepCost);
    const first = low.position();
    low.restore(saved);
    low.step("ت", 0.15);
    expect(low.position()).toEqual(first);
  });

  it("preserves local identity, duration, deletion and movement costs with uninformative confidence", () => {
    const plain = createLocalSearch(index, start), absent = createLocalSearch(index, start, fitted);
    const pooled = createLocalSearch(index, start, fitConfidenceChannel(records, ["ب", "ت", "د"], 1, 2));
    for (const p of "بتتدبت") {
      plain.step(p);
      absent.step(p);
      pooled.step(p, 0.9);
      expect(absent.position()).toEqual(plain.position());
      expect(pooled.position()).toEqual(plain.position());
    }
  });
});

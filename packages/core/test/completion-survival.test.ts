import { describe, expect, it } from "vitest";
import { completionDepartureCost, completionFeatures, completionLogLikelihood, completionProbability, fitCompletionModel } from "../src/completion-survival.js";
import type { CompletionExample, CompletionModel, WordPrefix } from "../src/completion-survival.js";
import { createLocalSearch } from "../src/local.js";
import type { LocalSnapshot } from "../src/local.js";
import type { ReferenceIndex } from "../src/reference.js";

const prefix: WordPrefix = { word: 0, entry: 1, progress: 2, length: 8, observations: 2, matches: 2, editCost: 0, lastCost: 0, stalled: 0 };
const half: CompletionModel = { weights: Array(8).fill(0), endpointOnly: false, completeRate: 0.2, abandonRate: 0.5 };
const index: ReferenceIndex = { version: "synthetic", inventoryId: "synthetic", tokens: new Set("abcdefgh"), phonemes: "abcdefgh",
  words: [{ surah: 1, ayah: 1, word: 1, start: 0, end: 4 }, { surah: 1, ayah: 1, word: 2, start: 4, end: 8 }],
  gramSize: 2, grams: new Map(), wordAt: at => at < 4 ? 0 : 1 };

describe("completion survival", () => {
  it("uses history while the matched control sees identical endpoints", () => {
    const stalled = { ...prefix, observations: 7, matches: 1, lastCost: 5, stalled: 3 };
    expect(completionFeatures(prefix, true)).toEqual(completionFeatures(stalled, true));
    expect(completionFeatures(prefix, false)).not.toEqual(completionFeatures(stalled, false));
    expect(completionFeatures(prefix, true)).toHaveLength(completionFeatures(prefix, false).length);
  });
  it("right censoring contributes the mixture survival, never a failed completion", () => {
    const e: CompletionExample = { prefix, delay: 3, outcome: "censored" };
    expect(completionLogLikelihood(half, [e])).toBeCloseTo(Math.log(0.5 * 0.8 ** 3 + 0.5 * 0.5 ** 3));
    expect(completionLogLikelihood(half, [e])).not.toBeCloseTo(completionLogLikelihood(half, [{ ...e, outcome: "abandon" }]));
  });
  it("fits recoverable versus abandoned trajectories and retains normalized finite probabilities", () => {
    const lost = { ...prefix, observations: 8, matches: 1, lastCost: 10, stalled: 6 };
    const examples: CompletionExample[] = Array.from({ length: 12 }, () => [
      { prefix, delay: 5, outcome: "complete" as const }, { prefix: lost, delay: 2, outcome: "abandon" as const },
      { prefix, delay: 1, outcome: "censored" as const },
    ]).flat();
    const model = fitCompletionModel(examples, false, 0.01);
    expect(completionProbability(model, prefix)).toBeGreaterThan(completionProbability(model, lost));
    expect(Number.isFinite(completionLogLikelihood(model, examples))).toBe(true);
    expect(model.completeRate).toBeGreaterThan(0); expect(model.completeRate).toBeLessThan(1);
    expect(model.abandonRate).toBeGreaterThan(0); expect(model.abandonRate).toBeLessThan(1);
  });
  it("marginalizes a departure and leaves a completed word free", () => {
    const cost = completionDepartureCost(half, prefix, 20);
    expect(cost).toBeGreaterThan(0); expect(cost).toBeLessThan(20);
    expect(cost).toBeCloseTo(-Math.log(0.5 + 0.5 * Math.exp(-6)) * 10 / 3);
    expect(completionDepartureCost(half, { ...prefix, progress: 8 }, 20)).toBe(0);
  });
  it("captures each candidate's trajectory without changing an uncalibrated run, and rewinds it", () => {
    const snapshots: LocalSnapshot[] = [], start = { offset: 0, origin: 0, edits: 0 };
    const base = createLocalSearch(index, start), observed = createLocalSearch(index, start, { observe: s => snapshots.push(s) });
    for (const p of "ab") { base.step(p); observed.step(p); expect(observed.position()).toEqual(base.position()); }
    const saved = observed.save(), old = JSON.stringify(snapshots), at = observed.position();
    observed.step("x"); const branch = observed.position(); observed.restore(saved);
    expect(observed.position()).toEqual(at); observed.step("x"); expect(observed.position()).toEqual(branch);
    expect(JSON.stringify(snapshots.slice(0, 2))).toBe(old);
    expect(snapshots[1].selected?.progress).toBe(2);
    expect(snapshots[1].selected?.matches).toBe(2);
    expect(snapshots[1].candidates.length).toBeGreaterThan(1);
  });
  it("uses pending status at the existing departure edge without closing the open frontier", () => {
    const text = "abqqqqwxyz";
    const ref: ReferenceIndex = { ...index, phonemes: text,
      words: [{ surah: 1, ayah: 1, word: 1, start: 0, end: 6 }, { surah: 1, ayah: 1, word: 2, start: 6, end: 10 }],
      wordAt: at => at < 6 ? 0 : 1 };
    const run = (intercept: number) => {
      const t = createLocalSearch(ref, { offset: 0, origin: 0, edits: 0 },
        { completion: { ...half, weights: [intercept, 0, 0, 0, 0, 0, 0, 0] } });
      for (const p of "abwxy") t.step(p);
      return t.position();
    };
    expect(run(-10).origin).toBe(6);
    expect(run(10).origin).toBe(0);
    expect(run(10).offset).toBe(5);
  });
});

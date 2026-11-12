import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { PosteriorFrames } from "../src/ctc.js";
import { interruptionFeatures, interruptionProbability, restartAfterAttempt, wordAttemptState } from "../src/acoustic-interruption.js";
import type { InterruptionModel } from "../src/acoustic-interruption.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { createTracker, trackerInput } from "../src/tracker.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { PhonemeEvidence } from "../src/evidence.js";

const model: InterruptionModel = { mean: Array(13).fill(0), scale: Array(13).fill(1), weights: Array(13).fill(0),
  intercept: 0, temperature: 1, threshold: 0.8 };
const phone = (p: string, start: number, end = start) => ({ p, start, end, conf: 0.9 });
const rows = () => {
  const f = new PosteriorFrames(2);
  f.append({ startFrame: 0, nFrames: 8, k: 3, ids: Array(8).fill([0, 1, 2]).flat(),
    probs: [[0.1,0.8,0.1], [0.1,0.8,0.1], [0.9,0.05,0.05], [0.9,0.05,0.05],
      [0.1,0.1,0.8], [0.1,0.1,0.8], [0.1,0.1,0.8], [0.1,0.1,0.8]].flat() });
  return f;
};

describe("acoustic interruption is independent of lexical matching", () => {
  it("an incomplete match cannot provide the cutoff exception", () => {
    const state = wordAttemptState(2, 11, 0.1, model);
    expect(state).toEqual({ lexical: "incomplete", acoustic: "continuous" });
    expect(restartAfterAttempt(state, false)).toBe(false);
  });
  it("positive acoustic interruption admits a genuine interrupted attempt", () => {
    const state = wordAttemptState(2, 11, 0.9, model);
    expect(state.acoustic).toBe("interrupted");
    expect(restartAfterAttempt(state, false)).toBe(true);
    expect(wordAttemptState(11, 11, 0.9, model).lexical).toBe("complete");
  });
  it("unavailable evidence supplies no cutoff, and a labelled pause remains sufficient", () => {
    const state = wordAttemptState(2, 11, null, model);
    expect(state.acoustic).toBe("unavailable");
    expect(restartAfterAttempt(state, false)).toBe(false);
    expect(restartAfterAttempt(state, true)).toBe(true);
  });
});

describe("posterior trajectory features", () => {
  it("retains blank, entropy and distribution change at the source/resume seam", () => {
    const x = interruptionFeatures(rows(), phone("a",0,1), phone("b",4,5), 40)!;
    expect(x).toHaveLength(13);
    expect(x.every(Number.isFinite)).toBe(true);
    expect(x[2]).toBeCloseTo(0.9);
    expect(x[12]).toBeCloseTo(0.9);
    expect(x[0]).toBeCloseTo(Math.log1p(0.08));
    expect(interruptionProbability(x, model)).toBe(0.5);
  });
  it("adjacent spikes have no fabricated blank gap", () => {
    const x = interruptionFeatures(rows(), phone("a",0,1), phone("b",2,3), 40)!;
    expect(x[0]).toBe(0);
    expect(x[2]).toBe(0);
    expect(x[12]).toBe(0);
  });
  it("missing rows or overlapping runs are unavailable", () => {
    const f = rows(); f.dropBefore(2);
    expect(interruptionFeatures(f, phone("a",0,1), phone("b",4,5), 40)).toBeNull();
    expect(interruptionFeatures(rows(), phone("a",0,4), phone("b",4,5), 40)).toBeNull();
    expect(interruptionFeatures(rows(), phone("a",0,1), phone("b",9), 40)).toBeNull();
  });
  it("rejects malformed models rather than inventing evidence", () => {
    expect(() => interruptionProbability(Array(13).fill(0), { ...model, temperature: 0 })).toThrow();
    expect(() => interruptionProbability([0], model)).toThrow();
  });
  it("preserves probabilities separately from acquisition without mutating evidence", () => {
    const e: PhonemeEvidence = { inventory: "test", frameMs: 40, phonemes: [phone("a",0)],
      posteriors: { startFrame: 0, nFrames: 1, k: 1, ids: [1], probs: [1] } };
    const result = trackerInput(e);
    expect(result.posteriors).toBeUndefined();
    expect(result.interruptionPosteriors).toBe(e.posteriors);
    expect(result.phonemes).toBe(e.phonemes);
    expect(e.posteriors).toBeDefined();
  });
  it("leaves the default tracker decoded when posterior evidence is present", () => {
    const real = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
    const f = syntheticFixture(real, { id: "interruption-control", from: { surah: 1, ayah: 1, word: 1 },
      to: { surah: 1, ayah: 5, word: 4 }, edits: [], chunkSize: 1 });
    const baseline = createTracker(real), candidate = createTracker(real);
    const a = f.evidence.flatMap(e => baseline.push(e));
    const b = f.evidence.flatMap(e => candidate.push(trackerInput(withPosteriors(real, e))));
    expect([...b, ...candidate.end()]).toEqual([...a, ...baseline.end()]);
  });
});

describe("local restart eligibility", () => {
  const words = ["aaaaa", "bbbbb", "ccccc", "zzzzz"];
  const index = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "abcz" },
    ayahs: [{ s: 1, a: 1, w: words, p: words.join(""), o: [0,5,10,15], iso: words }] });
  const search = () => createLocalSearch(index, { offset: 2, origin: 0, edits: 0 });
  it("blocks an unsupported mid-ayah restart without changing ordinary edit transitions", () => {
    const allowed = search(), denied = search();
    for (const p of "zzzz") { allowed.step(p); denied.step(p, dest => dest !== 15); }
    expect(allowed.position().origin).toBe(15);
    expect(denied.position().origin).not.toBe(15);
  });
  it("restoring a gated search reproduces its complete state", () => {
    const s = search(), before = s.save();
    for (const p of "zzzz") s.step(p, () => false);
    s.restore(before);
    for (const p of "zzzz") s.step(p);
    const original = search();
    for (const p of "zzzz") original.step(p);
    expect(s.position()).toEqual(original.position());
  });
  it("the next-word and single-deletion continuation remain usable when restarts are gated", () => {
    const s = search();
    for (const p of "bbbbb") s.step(p, dest => dest === 5 || dest === 10);
    expect(s.position().offset).toBe(10);
  });
});

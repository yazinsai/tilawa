import { describe, expect, it } from "vitest";
import { WordOccurrenceMemory, subsequenceDtw, validateOccurrenceCalibration } from "../src/recurring-word.js";
import type { OccurrenceCalibration, OccurrenceStep } from "../src/recurring-word.js";
import { fitOccurrenceCalibration } from "../src/recurring-word-fit.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { trackerInput } from "../src/tracker.js";
import { createTracker } from "../src/tracker.js";
import { readFileSync } from "node:fs";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";

const a = [0.05, 0.85, 0.1], b = [0.05, 0.1, 0.85];
const calibration: OccurrenceCalibration = { version: 1, feature: "posterior-subsequence-dtw-v1", trainingOnly: true,
  speakers: 2, folds: 2, pairs: 8, scaleNats: 3 };
const step = (word: number, pronunciation: string, fromFrame: number, extra: Partial<OccurrenceStep> = {}): OccurrenceStep =>
  ({ word, pronunciation, fromFrame, run: "in-order", unambiguous: true, ...extra });
function append(memory: WordOccurrenceMemory, rows: number[][], startFrame = 0) {
  memory.append({ k: 3, startFrame, nFrames: rows.length, ids: rows.flatMap(() => [0, 1, 2]), probs: rows.flat() });
}
function remembered() {
  const memory = new WordOccurrenceMemory(2);
  append(memory, [a, a, a, b, b, a, a, a].map(x => [...x]));
  for (const s of [step(0, "prefix", 0), step(1, "near", 1), step(2, "far", 3), step(3, "suffix", 5)]) memory.commit(s);
  append(memory, [[...a], [...a]], 8);
  return memory;
}

describe("posterior subsequence DTW", () => {
  it("preserves temporal order, while tolerating stretching and an exemplar prefix/suffix", () => {
    expect(subsequenceDtw([a, a, b], [b, a, b, a])).toBeCloseTo(0, 12);
    expect(subsequenceDtw([a, b, a], [a, b, a])).toBeCloseTo(0, 12);
    expect(subsequenceDtw([a, b, a], [b, a, b])).toBeGreaterThan(0);
  });
  it("consumes every query frame and keeps sub-blank class distinctions", () => {
    expect(subsequenceDtw([a, b], [a])).toBeGreaterThan(0);
    expect(subsequenceDtw([[0.8, 0.15, 0.05]], [[0.8, 0.05, 0.15]])).toBeGreaterThan(0);
    expect(subsequenceDtw([], [a])).toBeNull();
  });
  it("normalizes posterior mass and rejects an incompatible inventory", () => {
    expect(subsequenceDtw([a.map(x => x * 2)], [a])).toBeCloseTo(0, 12);
    expect(() => subsequenceDtw([[1, 0]], [a])).toThrow();
    expect(() => subsequenceDtw([[0, 0, 0]], [a])).toThrow();
  });
});

describe("earlier acoustic occurrence memory", () => {
  it("requires both earlier pronunciations, and returns a calibrated relative likelihood", () => {
    const memory = remembered();
    const match = memory.compare("near", "far", 8, 10, calibration);
    expect(match.continuationExemplars).toBe(1);
    expect(match.restartExemplars).toBe(1);
    expect(match.continuationDistance).toBeCloseTo(0, 12);
    expect(match.continuationLogLikelihood).toBeGreaterThan(0);
    expect(memory.compare("near", "absent", 8, 10, calibration).continuationLogLikelihood).toBeNull();
    expect(memory.compare("near", "far", 8, 10).continuationLogLikelihood).toBeNull();
  });
  it("excludes an exemplar overlapping or following the comparison interval", () => {
    const result = remembered().compare("near", "far", 4, 5, calibration);
    expect(result.continuationExemplars).toBe(1);
    expect(result.restartExemplars).toBe(0);
    expect(result.continuationLogLikelihood).toBeNull();
  });
  it("keeps the entire disputed trajectory across chunks without admitting unclear words", () => {
    const memory = remembered();
    memory.retainFrom(8);
    memory.interrupt();
    append(memory, [[...a], [...a]], 10);
    const result = memory.compare("near", "far", 8, 12, calibration);
    expect(result.continuationDistance).toBeCloseTo(0, 12);
    expect(result.continuationLogLikelihood).toBeGreaterThan(0);
  });
  it("does not admit a word across a jump, a hold or an unclear neighbour", () => {
    for (const middle of [step(1, "near", 1, { unambiguous: false }), step(1, "near", 1, { run: "restart" }), step(4, "near", 1)]) {
      const memory = new WordOccurrenceMemory(2);
      append(memory, [a, a, a, a, a, a]);
      memory.commit(step(0, "prefix", 0)); memory.commit(middle); memory.commit(step(2, "suffix", 3));
      expect(memory.compare("near", "far", 4, 6).continuationExemplars).toBe(0);
    }
  });
  it("waits for separation inside all three bracket words", () => {
    const saved: unknown[] = [], memory = new WordOccurrenceMemory(2, e => saved.push(e));
    append(memory, [a, a, a, b, b, a, a, a]);
    memory.commit(step(0, "prefix", 0, { supported: false })); memory.support(0, true);
    memory.commit(step(1, "near", 1, { supported: false })); memory.support(1, true);
    memory.commit(step(2, "far", 3, { supported: false }));
    expect(saved.length).toBe(0);
    memory.support(2, true);
    expect(saved.length).toBe(1);
    memory.support(2, true);
    expect(saved.length).toBe(1);
  });
  it("owns snapshots, indexes by pronunciation, and waits for the right bracket", () => {
    const saved: unknown[] = [];
    const memory = new WordOccurrenceMemory(2, e => { saved.push(e); (e.trajectory[0] as number[])[1] = 0; });
    const rows = [a, a, a, b, b, a, a, a].map(x => [...x]);
    append(memory, rows);
    memory.commit(step(10, "prefix", 0)); memory.commit(step(11, "near", 1));
    expect(saved.length).toBe(0);
    memory.commit(step(12, "far", 3)); memory.commit(step(13, "suffix", 5));
    expect(saved.length).toBe(2);
    rows[1][1] = 0;
    append(memory, [[...a], [...a]], 8);
    expect(memory.compare("near", "far", 8, 10, calibration).continuationDistance).toBeCloseTo(0, 12);
  });
  it("rejects uncalibrated or non-training models", () => {
    expect(() => validateOccurrenceCalibration({ ...calibration, trainingOnly: false } as unknown as OccurrenceCalibration)).toThrow();
    expect(() => validateOccurrenceCalibration({ ...calibration, folds: 1 })).toThrow();
    expect(() => validateOccurrenceCalibration({ ...calibration, scaleNats: NaN })).toThrow();
  });
});

describe("speaker-separated scale estimation", () => {
  const training = ["one", "two", "three"].flatMap(speaker => ["take1", "take2"].flatMap(recording => [
    { speaker, recording, pronunciation: "ab", trajectory: [a, a] },
    { speaker, recording, pronunciation: "ac", trajectory: [b, b] },
  ]));
  it("fits a positive likelihood scale with whole speakers held out", () => {
    const fit = fitOccurrenceCalibration(training, { trainingOnly: true });
    expect(fit.model.speakers).toBe(3);
    expect(fit.model.folds).toBe(3);
    expect(fit.model.scaleNats).toBeGreaterThan(0);
    expect(fit.validation.every(v => v.foldNll.length === 3 && v.nll < Math.log(2))).toBe(true);
  });
  it("cannot calibrate using unspecified same-recording intervals or only one speaker", () => {
    expect(() => fitOccurrenceCalibration(training.filter(x => x.recording === "take1"), { trainingOnly: true })).toThrow();
    expect(() => fitOccurrenceCalibration(training.filter(x => x.speaker === "one"), { trainingOnly: true })).toThrow();
  });
  it("allows a same-recording exemplar only when its complete interval predates the query", () => {
    const repeated = training.map(x => {
      const fromFrame = (x.recording === "take1" ? 0 : 10) + (x.pronunciation === "ab" ? 0 : 3);
      return { ...x, recording: "repeat", fromFrame, toFrame: fromFrame + 2 };
    });
    expect(fitOccurrenceCalibration(repeated, { trainingOnly: true }).model.scaleNats).toBeGreaterThan(0);
    expect(() => fitOccurrenceCalibration(repeated.map(x => ({ ...x, fromFrame: 0, toFrame: 5 })), { trainingOnly: true })).toThrow();
  });
});

describe("local scoring fallback", () => {
  const index = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "abcd" }, ayahs: [
    { s: 1, a: 1, w: ["one", "two", "three"], p: "abacbd", o: [0, 2, 4], iso: ["ab", "ac", "bd"] },
  ] });
  it("retains every local decision when the comparison supplies no likelihood", () => {
    const start = { offset: 2, origin: 0, edits: 0 };
    const raw = createLocalSearch(index, start), memory = createLocalSearch(index, start, { compare: () => null });
    for (const p of "acbdbdac") {
      raw.step(p); memory.step(p);
      const { wordLead: _, wordEvidencePhonemes: __, ...actual } = memory.position();
      expect(actual).toEqual(raw.position());
    }
  });
  it("does not call the scorer on ordinary unambiguous forward progress", () => {
    let calls = 0;
    const local = createLocalSearch(index, { offset: 2, origin: 0, edits: 0 }, { compare: () => { calls++; return 100; } });
    for (const p of "acbd") local.step(p);
    expect(calls).toBe(0);
    expect(local.position().offset).toBe(6);
  });
  it("can reverse an original restart through relative likelihood", () => {
    const long = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "abc" }, ayahs: [
      { s: 1, a: 1, w: ["first", "second", "third"], p: "aaaabbbbcccc", o: [0, 4, 8], iso: ["aaaa", "bbbb", "cccc"] },
    ] });
    const start = { offset: 8, origin: 4, edits: 0 };
    const raw = createLocalSearch(long, start);
    let calls = 0;
    const learned = createLocalSearch(long, start, { compare: c => {
      calls++; expect(c.continuation.word === c.restart.word).toBe(false); return 100;
    } });
    for (const p of "aaa") { raw.step(p); learned.step(p); }
    expect(calls).toBeGreaterThan(0);
    expect(raw.position().origin).toBe(0);
    expect(learned.position().origin).toBe(4);
  });
  it("retains rows separately from acquisition scoring without changing decoded tokens", () => {
    const rows = { k: 1, startFrame: 0, nFrames: 1, ids: [0], probs: [1] };
    const evidence = { inventory: "test", frameMs: 20, phonemes: [{ p: "a", start: 0, end: 0, conf: 1 }], posteriors: rows };
    const input = trackerInput(evidence);
    expect(input.posteriors).toBeUndefined();
    expect(input.occurrencePosteriors).toBe(rows);
    expect(input.phonemes).toBe(evidence.phonemes);
  });
  it("retains identical tracker events in read-only audit mode while collecting complete real-word trajectories", () => {
    const reference = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
    const fixture = syntheticFixture(reference, { id: "memory-unit", from: { surah: 2, ayah: 255, word: 1 },
      to: { surah: 2, ayah: 255, word: 35 }, edits: [], chunkSize: 1 });
    const raw = createTracker(reference), occurrences: unknown[] = [];
    const audit = createTracker(reference, { acousticMemory: { audit: true, occurrence: x => occurrences.push(x) } });
    const evidence = fixture.evidence.map(e => withPosteriors(reference, e));
    const first = evidence.flatMap(e => raw.push(trackerInput(e)));
    const second = evidence.flatMap(e => audit.push(trackerInput(e)));
    expect([...second, ...audit.end()]).toEqual([...first, ...raw.end()]);
    expect(occurrences.length).toBeGreaterThan(0);
  });
});

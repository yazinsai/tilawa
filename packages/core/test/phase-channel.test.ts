import { describe, expect, it } from "vitest";
import { channelPhase, fitPhaseChannel, inferencePhase, phaseLogLikelihood, phaseObservation } from "../src/phase-channel.js";
import type { PhaseChannel, PhaseRecording } from "../src/phase-channel.js";
import { trackerInput } from "../src/tracker.js";
import { createLocalSearch } from "../src/local.js";
import type { ReferenceIndex } from "../src/reference.js";
import type { DecodedPhoneme, PhonemeEvidence } from "../src/evidence.js";

const token = (frame: number): DecodedPhoneme => ({ p: "ب", start: frame, end: frame, conf: .8,
  inference: { frame, chunkStart: Math.floor(frame / 8) * 8, chunkFrames: 8, model: "model" } });
const records: PhaseRecording[] = Array.from({ length: 8 }, () => ({ written: "بتب", views: [
  [{ p: "ب", phase: 0 }, { p: "ت", phase: .5 }, { p: "ب", phase: .25 }],
  [{ p: "ت", phase: .5 }, { p: "ت", phase: 0 }, { p: "ت", phase: .75 }],
] }));

describe("native inference phase", () => {
  it("uses original inference coordinates through merged pushes and shifted crops", () => {
    const phonemes = [token(7), token(8), token(11)];
    const evidence: PhonemeEvidence = { inventory: "fixture", frameMs: 40, phonemes };
    const all = trackerInput(evidence).phonemes.map(p => p.chunkPhase);
    const split = phonemes.flatMap(p => trackerInput({ ...evidence, phonemes: [p] }).phonemes).map(p => p.chunkPhase);
    const cropped = trackerInput({ ...evidence, phonemes: phonemes.map(p => ({ ...p, start: p.start - 5, end: p.end - 5 })) });
    expect(all).toEqual([7 / 8, 0, 3 / 8]);
    expect(split).toEqual(all);
    expect(cropped.phonemes.map(p => p.chunkPhase)).toEqual(all);
    expect(evidence.phonemes[0].chunkPhase).toBeUndefined();
  });

  it("does not trust a push boundary or a stale derived phase when native geometry is missing or invalid", () => {
    const ph = { ...token(9), inference: undefined, chunkPhase: .5 };
    expect(trackerInput({ inventory: "fixture", frameMs: 40, phonemes: [ph] }).phonemes[0].chunkPhase).toBeUndefined();
    expect(inferencePhase({ ...token(9), inference: { ...token(9).inference!, chunkFrames: 0 } })).toBeUndefined();
    expect(inferencePhase({ ...token(9), inference: { ...token(9).inference!, chunkStart: 10 } })).toBeUndefined();
    const fitted = { ...fitPhaseChannel(records, [..."بت"], 2, 16), geometry: { model: "model", chunkFrames: 8 } };
    expect(channelPhase(fitted, token(9))).toBe(1 / 8);
    expect(channelPhase(fitted, { ...token(9), inference: { ...token(9).inference!, model: "other" } })).toBeUndefined();
    expect(channelPhase(fitted, { ...token(9), inference: { ...token(9).inference!, chunkFrames: 16 } })).toBeUndefined();
  });
});

describe("paired phase observation channel", () => {
  it("normalizes every emitted-identity distribution and learns opposite preferences by phase", () => {
    const model = fitPhaseChannel(records, [..."بت"], 2, 1);
    for (const row of [...model.pooled, ...model.conditioned.flat()]) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
      expect(row.every(p => p > 0 && p < 1)).toBe(true);
    }
    expect(phaseObservation(model, "ب", 0).substitution("ب")).toBeLessThan(0);
    expect(phaseObservation(model, "ب", .5).substitution("ب")).toBeGreaterThan(0);
    expect(Number.isFinite(phaseLogLikelihood(model, records[0]))).toBe(true);
  });

  it("backs off to a pooled control and ignores unavailable observations", () => {
    const model = fitPhaseChannel(records, [..."بت"], 1, 16);
    expect(phaseObservation(model, "ب", .75).substitution("ت")).toBe(0);
    expect(phaseObservation(model, "ب", .75).insertion).toBe(0);
    const phase = fitPhaseChannel(records, [..."بت"], 2, 16);
    expect(phaseObservation(phase, "ب").substitution("ت")).toBe(0);
    expect(phaseObservation(phase, "ب", 1).insertion).toBe(0);
    expect(phaseObservation(phase, "unknown", 0).substitution("ب")).toBe(0);
    expect(phaseObservation(phase, "ب", 0).substitution("unknown")).toBe(0);
  });

  it("gives a recording total weight one, regardless of repeated paired views", () => {
    const a = fitPhaseChannel(records, [..."بت"], 2, 16);
    const b = fitPhaseChannel(records.map(r => ({ ...r, views: [...r.views, ...r.views] })), [..."بت"], 2, 16);
    a.conditioned.forEach((rows, r) => rows.forEach((row, phase) => row.forEach((p, h) => expect(p).toBeCloseTo(b.conditioned[r][phase][h], 10))));
    expect(phaseLogLikelihood(a, records[0])).toBeCloseTo(phaseLogLikelihood(a, { ...records[0], views: [...records[0].views, ...records[0].views] }), 10);
  });

  it("marginalizes indels rather than requiring equal decoded lengths", () => {
    const model = fitPhaseChannel([{ written: "بت", views: [[], [{ p: "ب", phase: 0 }],
      [{ p: "ب", phase: 0 }, { p: "ب", phase: .5 }, { p: "ت", phase: .5 }]] }], [..."بت"], 2, 16);
    expect(model.conditioned.flat(2).every(p => Number.isFinite(p) && p > 0)).toBe(true);
    expect(() => fitPhaseChannel([{ written: "x", views: [[{ p: "ب", phase: 0 }]] }], [..."بت"], 2, 16)).toThrow();
  });

  it("adds the same channel to local continuation and restart edges and preserves rewind", () => {
    const reference: ReferenceIndex = { version: "fixture", inventoryId: "fixture", tokens: new Set("بت"), phonemes: "بت",
      words: [{ surah: 1, ayah: 1, word: 1, start: 0, end: 1 }, { surah: 1, ayah: 1, word: 2, start: 1, end: 2 }],
      gramSize: 1, grams: new Map(), wordAt: offset => Math.min(1, offset) };
    const channel: PhaseChannel = { phones: [..."بت"], bins: 2, mass: 1, pooled: [[.5, .5], [.5, .5], [.5, .5]],
      conditioned: [[[.9, .1], [.1, .9]], [[.001, .999], [.999, .001]], [[.001, .999], [.999, .001]]] };
    const start = { offset: 0, origin: 0, edits: 0 };
    const local = createLocalSearch(reference, start, channel), saved = local.save();
    local.step("ب", 0);
    const first = local.position();
    expect(first.offset).toBe(1);
    expect(first.stepCost).toBeCloseTo(-Math.log(.9 / .5) / 3);
    local.step("ت", .5);
    local.restore(saved); local.step("ب", 0);
    expect(local.position()).toEqual(first);
    const plain = createLocalSearch(reference, start), unavailable = createLocalSearch(reference, start, channel);
    for (const p of "بتت") { plain.step(p); unavailable.step(p); expect(unavailable.position()).toEqual(plain.position()); }
  });
});

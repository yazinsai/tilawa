import { describe, expect, it } from "vitest";
import { fitPredictiveChannel, predictiveCalibration, predictiveContribution, predictiveLogP, snapshotPredictivePair } from "../src/predictive-channel.js";
import type { PredictiveChannel } from "../src/predictive-channel.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";

const tiny = (): PredictiveChannel => ({ phones: ["ب", "ت", "َ"], context: 1, pooling: 8, insert: 0.2, delete: 0.1,
  inserted: [1 / 3, 1 / 3, 1 / 3], emitted: [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], conditional: {} });

describe("predictive observation transducers", () => {
  it("sums ambiguous edit placements, rather than selecting one edit script", () => {
    const model = tiny();
    // One reference, one observation: match then stop; or insert+delete in either order then stop.
    const match = 0.8 * 0.9 * 0.8 * 0.8;
    const indel = (0.2 / 3) * (0.8 * 0.1) * 0.8;
    expect(Math.exp(predictiveLogP(model, { written: "ب", heard: "ب" }))).toBeCloseTo(match + 2 * indel, 12);
  });
  it("has a normalized complete-string probability, including geometric insertions", () => {
    const model = tiny();
    let mass = 0;
    for (let n = 0; n <= 12; n++) {
      // Summing identities of each inserted token gives one, so all strings of length n have this total mass.
      mass += Math.exp(predictiveLogP(model, { written: "", heard: "ب".repeat(n) })) * 3 ** n;
    }
    expect(mass).toBeCloseTo(1, 8);
    expect(Math.exp(predictiveLogP(model, { written: "ب", heard: "" }))).toBeCloseTo(0.8 * 0.1 * 0.8, 12);
  });
  it("uses subsequent observations to interpret an earlier mismatch without any additional tokens", () => {
    const forward = tiny(), reverse = tiny();
    reverse.conditional["ت/vowel/vowel"] = [0.8, 0.1, 0.1];
    const pair = { forward, reverse, natsPerEdit: 3 };
    const both = predictiveContribution(pair, { written: "تَ", heard: "بَ" });
    expect(both.reverse).toBeGreaterThan(both.forward);
    expect(predictiveContribution(pair, { written: "ت", heard: "ب" }).edits).toBe(0);
  });
  it("direction is the only capacity difference and context-free predictions agree on reversal", () => {
    const model = tiny(), record = { written: "بتَ", heard: "تَ" };
    expect(predictiveLogP(model, record, true)).toBeCloseTo(predictiveLogP(model, record), 10);
  });
  it("EM preserves normalized rows and fits each direction from complete strings", () => {
    const records = Array.from({ length: 8 }, () => ({ written: "تَبتَ", heard: "بَبتَ" }));
    for (const reverse of [false, true]) {
      const model = fitPredictiveChannel(records, tiny().phones, 1, 8, reverse, 4);
      for (const row of [model.inserted, ...model.emitted, ...Object.values(model.conditional)]) {
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
        expect(row.every(x => x > 0)).toBe(true);
      }
      expect(Number.isFinite(predictiveLogP(model, records[0], reverse))).toBe(true);
      expect(predictiveCalibration([model])).toBeGreaterThan(0);
    }
  });
  it("rejects observations outside the trained inventory", () => {
    expect(() => predictiveLogP(tiny(), { written: "ب", heard: "ك" })).toThrow(/Unknown/);
  });
  it("keeps movement and rewind identical when direction supplies no relative evidence", () => {
    const index = loadReferenceIndex({ version: "tiny", text: "synthetic", profile: {}, inventory: { id: "tiny", tokens: "بتَ" },
      ayahs: [{ s: 1, a: 1, w: ["a", "b", "c"], p: "ب".repeat(10) + "تَ".repeat(5), o: [0, 1, 10], iso: ["ب", "ب".repeat(9), "تَ".repeat(5)] }] });
    const start = { offset: 1, origin: 0, edits: 0 };
    const plain = createLocalSearch(index, start);
    const comparisons: string[] = [];
    const model = { forward: tiny(), reverse: tiny(), natsPerEdit: 3 };
    const predictive = createLocalSearch(index, start, { model, mode: "forward", onComparison: c => {
      comparisons.push(c.heard);
      expect(c.candidates.every(x => Math.abs(x.edits) < 1e-10)).toBe(true);
    } });
    for (const p of "تَتَتَ") { plain.step(p); predictive.step(p); expect(predictive.position()).toEqual(plain.position()); }
    expect(comparisons.length).toBeGreaterThan(0);
    const saved = predictive.save(), before = predictive.position();
    predictive.step("ب"); predictive.restore(saved);
    expect(predictive.position()).toEqual(before);
    predictive.step("ت"); const first = predictive.position();
    predictive.restore(saved); predictive.step("ت"); expect(predictive.position()).toEqual(first);
  });
  it("snapshots calibration and rejects a control with different capacity", () => {
    const pair = { forward: tiny(), reverse: tiny(), natsPerEdit: 3 };
    const copy = snapshotPredictivePair(pair);
    pair.forward.emitted[0][0] = 0;
    expect(copy.forward.emitted[0][0]).toBe(0.8);
    expect(() => snapshotPredictivePair({ ...copy, reverse: { ...copy.reverse, context: 2 } })).toThrow(/Unmatched/);
  });
});

import { describe, expect, it } from "vitest";
import { fitTokenEditModel, logAdd, tokenPartition, tokenRecordingLogP, validateTokenRestart } from "../src/heldout-token.js";
import type { TokenEditModel, TokenGeometry, TokenInterval } from "../src/heldout-token.js";

const model: TokenEditModel = { phones: ["ب", "ت"], rows: [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
  insertion: 0.1, inserted: [0.5, 0.5], block: 1 };

describe("held-out token identity", () => {
  it("preserves every token position and exchanges contiguous block partitions", () => {
    const a = tokenPartition(13, 3, 0), b = tokenPartition(13, 3, 1);
    expect(a).toEqual([true, true, true, false, false, false, true, true, true, false, false, false, true]);
    expect(b).toEqual(a.map(x => !x));
    expect(() => tokenPartition(3, 0, 0)).toThrow();
  });

  it("marginalizes a masked identity exactly rather than dropping the observation", () => {
    const masked = tokenRecordingLogP(model, { written: "بت", heard: "بت" }, [true, false]);
    const summed = logAdd(tokenRecordingLogP(model, { written: "بت", heard: "بب" }),
      tokenRecordingLogP(model, { written: "بت", heard: "بت" }));
    expect(masked).toBeCloseTo(summed, 12);
    const shorter = tokenRecordingLogP(model, { written: "بت", heard: "ب" });
    expect(Math.abs(masked - shorter)).toBeGreaterThan(0.1);
  });

  it("predictive partition ratios normalize over all withheld identities", () => {
    const denominator = tokenRecordingLogP(model, { written: "بت", heard: "بت" }, [true, false]);
    const probabilities = model.phones.map(p => Math.exp(tokenRecordingLogP(model, { written: "بت", heard: "ب" + p }) - denominator));
    expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it("also sums hidden identity evidence in duration-discounted insertion transitions", () => {
    const duration: TokenEditModel = { phones: ["ا", "ت"], rows: model.rows.map(r => [...r]), insertion: 0.1, inserted: [0.5, 0.5], block: 1 };
    const masked = tokenRecordingLogP(duration, { written: "اات", heard: "ات" }, [true, false]);
    const summed = logAdd(tokenRecordingLogP(duration, { written: "اات", heard: "اا" }),
      tokenRecordingLogP(duration, { written: "اات", heard: "ات" }));
    expect(masked).toBeCloseTo(summed, 12);
  });

  it("selects both locations without reading the held-out identities", () => {
    const geometry: TokenGeometry = { reference: "ببتت", starts: new Set([0, 2]), movement: () => 20, entry: () => 0 };
    const interval = (heard: string): TokenInterval => ({ lo: 0, cost: [0, Infinity, Infinity, Infinity, Infinity],
      origin: [0, 0, 0, 0, 0], nearOrigin: 0, proposedOrigin: 2,
      layers: [...heard].map(p => ({ lo: 0, size: 5, heard: p })) });
    const a = validateTokenRestart(model, geometry, interval("بببت"));
    const b = validateTokenRestart(model, geometry, interval("بتبب"));
    expect(a.directions[0].near).toEqual(b.directions[0].near);
    expect(a.directions[0].restart).toEqual(b.directions[0].restart);
    expect(a.directions[0].nearPredictive).not.toEqual(b.directions[0].nearPredictive);
    for (const d of a.directions) {
      expect(d.near).not.toBeNull();
      expect(d.restart).not.toBeNull();
      expect(d.nearPredictive!).toBeLessThanOrEqual(0);
      expect(d.restartPredictive!).toBeLessThanOrEqual(0);
    }
  });

  it("does not certify a restart without two nonempty evidence partitions", () => {
    const result = validateTokenRestart({ ...model, block: 8 }, {
      reference: "بت", starts: new Set([0, 1]), movement: () => 20, entry: () => 0,
    }, { lo: 0, cost: [0, Infinity, Infinity], origin: [0, 0, 0], nearOrigin: 0, proposedOrigin: 1,
      layers: [{ lo: 0, size: 3, heard: "ت" }] });
    expect(result.retain).toBe(false);
    expect(result.directions.every(d => d.gain === null)).toBe(true);
  });

  it("retains predictive agreement in both directions and rejects a one-partition fit", () => {
    const reference = "ب".repeat(8) + "ت".repeat(8);
    const geometry: TokenGeometry = { reference, starts: new Set(), movement: () => Infinity, entry: () => 0 };
    const interval = (heard: string): TokenInterval => {
      const cost = Array(reference.length + 1).fill(Infinity), origin = Array(reference.length + 1).fill(0);
      cost[0] = 0; cost[8] = 0; origin[8] = 8;
      return { lo: 0, cost, origin, nearOrigin: 0, proposedOrigin: 8,
        layers: [...heard].map(p => ({ lo: 0, size: reference.length + 1, heard: p })) };
    };
    const jump = validateTokenRestart(model, geometry, interval("تتتت"));
    expect(jump.retain).toBe(true);
    expect(jump.directions.every(d => d.gain! > 0)).toBe(true);
    const fluctuation = validateTokenRestart(model, geometry, interval("تبتب"));
    expect(fluctuation.directions[0].gain!).toBeLessThanOrEqual(0);
    expect(fluctuation.directions[1].gain!).toBeGreaterThan(0);
    expect(fluctuation.retain).toBe(false);
  });

  it("fits directed edits and retains normalized probability for unseen phones", () => {
    const fitted = fitTokenEditModel([{ written: "بت", heard: "ب" }, { written: "ب", heard: "بت" }], model.phones, 1, 2);
    expect(fitted.rows[1][2]).toBeGreaterThan(fitted.rows[0][2]);
    expect(fitted.inserted[1]).toBeGreaterThan(fitted.inserted[0]);
    for (const row of [...fitted.rows, fitted.inserted]) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      expect(row.every(x => x > 0)).toBe(true);
    }
  });
});

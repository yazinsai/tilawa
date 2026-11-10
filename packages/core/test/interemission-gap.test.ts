import { describe, expect, it } from "vitest";
import { GAP_BINS, fitGapCounts, gapAlignmentRatio, gapBin, gapContext, gapKeys, gapLogRatio, gapTrainingObservations } from "../src/interemission-gap.js";
import type { GapContext, GapObservation } from "../src/interemission-gap.js";
const context: GapContext = { left: "ب", right: "ت", between: "َ", boundary: false };
const observations: GapObservation[] = Array.from({ length: 200 }, (_, i) => ({
  ...context, between: i < 100 ? "َ" : "", frames: i < 100 ? 4 : 0, weight: 1,
}));
const model = fitGapCounts(observations, 8);
const heard = [{ p: "ب", start: 0, end: 1, conf: 1 }, { p: "ت", start: 6, end: 6, conf: 1 }];

describe("inter-emission gap channel", () => {
  it("counts blank frames between inclusive runs, including a zero-frame gap", () => {
    expect(gapBin(0)).toBe(0);
    expect(gapBin(1)).toBe(1);
    expect(gapBin(3)).toBe(2);
    expect(gapBin(1e6)).toBe(GAP_BINS - 1);
    expect(gapContext("بَت", new Set([1]), 1, 2, false)).toEqual({ ...context, boundary: true });
  });
  it("backs an unseen context all the way off to a normalized pooled distribution", () => {
    const m = fitGapCounts(observations, 8);
    m.contexts = {};
    for (let b = 0; b < GAP_BINS; b++) expect(gapLogRatio(m, context, 2 ** b - 1)).toBe(0);
    expect(gapLogRatio(m, context, NaN)).toBe(0);
    expect(gapAlignmentRatio("بَت", new Set(), heard, m)).toBe(0);
  });
  it("keeps each hierarchical distribution on the probability simplex", () => {
    const n = model.pooled.reduce((a, b) => a + b, 0);
    let sum = 0;
    for (let b = 0; b < GAP_BINS; b++) {
      const pooled = (model.pooled[b] + 0.5) / (n + 0.5 * GAP_BINS);
      sum += pooled * Math.exp(gapLogRatio(model, context, 2 ** b - 1));
    }
    expect(sum).toBeCloseTo(1, 12);
    expect(gapKeys(context)).toHaveLength(4);
  });
  it("uses erased reference phones to distinguish a missing nucleus from adjacency", () => {
    expect(gapLogRatio(model, context, 4)).toBeGreaterThan(0);
    expect(gapLogRatio(model, { ...context, between: "" }, 4)).toBeLessThan(0);
    expect(gapLogRatio(model, context, 0)).toBeLessThan(0);
    const erased = gapAlignmentRatio("بَت", new Set(), heard, model);
    const adjacent = gapAlignmentRatio("بت", new Set(), heard, model);
    expect(erased).toBeGreaterThan(adjacent);
  });
  it("marginalizes edit placement without counting one observed gap repeatedly", () => {
    const obs = gapTrainingObservations("بَت", new Set(), heard);
    expect(obs.reduce((n, o) => n + o.weight, 0)).toBeCloseTo(heard.length - 1, 10);
    expect(obs.some(o => o.between === "َ" && o.weight > 0)).toBe(true);
    expect(obs.some(o => o.between === "" && o.weight > 0)).toBe(true);
    expect(obs.every(o => o.frames === 4)).toBe(true);
  });
});

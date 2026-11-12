import { describe, expect, it } from "vitest";
import { BASE_EDIT_COSTS, EDIT_KINDS, editKind, fitAcousticCosts, minimaxPathRegret,
  trainingCostScale, trainingEditCounts } from "../src/path-regret.js";
import type { PathCostWitness, PathUncertaintySet } from "../src/path-regret.js";

const counts = (hard = 0) => [hard, ...Array(EDIT_KINDS.length - 1).fill(0)];
const path = (hard: number, fixedCost = 0): PathCostWitness => ({ counts: counts(hard), fixedCost, observations: 12 });
const model = (hard: number) => ({ costs: [hard, ...BASE_EDIT_COSTS.slice(1)] });

describe("worst-case trajectory regret", () => {
  it("chooses a robust available path even when averaging prefers the fragile one", () => {
    const paths = [path(0, 6), path(1)];
    const set = { models: [model(1), model(1), model(12)] };
    const got = minimaxPathRegret(set, paths, 1);
    expect(got.regrets).toEqual([5, 6]);
    expect(got.selected).toBe(0);
    expect(got.costs[1].reduce((a, b) => a + b) / 3).toBeLessThan(6);
  });

  it("leaves a uniformly preferred destination selected", () => {
    const got = minimaxPathRegret({ models: [model(2), model(20)] }, [path(2), path(1)], 1);
    expect(got.selected).toBe(1);
    expect(got.regrets[1]).toBe(0);
  });

  it("compares excess cost against a different best path in each model", () => {
    const got = minimaxPathRegret({ models: [model(1), model(12)] }, [path(0, 6), path(1)]);
    expect(got.costs).toEqual([[6, 6], [1, 12]]);
    expect(got.regrets).toEqual([5, 6]);
  });

  it("preserves correlated vertices instead of fabricating independent extrema", () => {
    const set: PathUncertaintySet = { models: [
      { costs: [1, 9, ...BASE_EDIT_COSTS.slice(2)] }, { costs: [9, 1, ...BASE_EDIT_COSTS.slice(2)] },
    ] };
    const both: PathCostWitness = { counts: [1, 1, ...counts().slice(2)], observations: 12, fixedCost: 0 };
    const got = minimaxPathRegret(set, [path(0, 11), both]);
    expect(got.selected).toBe(1);
    expect(got.regrets[1]).toBe(0);
  });

  it("is invariant to a common prefix and includes movement costs", () => {
    const set = { models: [model(1), model(12)] };
    const paths = [path(0, 6), path(1)];
    const first = minimaxPathRegret(set, paths);
    const prefix = paths.map(p => ({ ...p, counts: p.counts.map((n, j) => n + (j === 0 ? 10 : 0)),
      fixedCost: p.fixedCost + 20, observations: p.observations + 10 }));
    expect(minimaxPathRegret(set, prefix).regrets).toEqual(first.regrets);
    expect(minimaxPathRegret(set, [path(0, 20), path(1)]).selected).toBe(1);
  });

  it("keeps the acoustic winner on a regret tie", () => {
    expect(minimaxPathRegret({ models: [model(1)] }, [path(1), path(1)], 1).selected).toBe(1);
  });

  it("rejects unequal heard horizons and malformed models", () => {
    expect(() => minimaxPathRegret({ models: [model(1)] }, [path(1), { ...path(2), observations: 11 }])).toThrow();
    expect(() => minimaxPathRegret({ models: [] }, [path(1)])).toThrow();
    expect(() => minimaxPathRegret({ models: [model(0)] }, [path(1)])).toThrow();
  });
});

describe("training edit prices", () => {
  it("preserves substitution direction, soft classes, vowel and duration indels", () => {
    expect(editKind("diagonal", "ص", 0, "س")).toBe(1);
    expect(editKind("diagonal", "َ", 0, "ِ")).toBe(2);
    expect(editKind("diagonal", "ب", 0, "ب")).toBe(null);
    expect(editKind("insert", "ب", 0, "س")).toBe(3);
    expect(editKind("insert", "ب", 0, "َ")).toBe(4);
    expect(editKind("delete", "نننن", 1)).toBe(8);
    expect(editKind("insert", "نننن", 1, "ن")).toBe(5);
  });

  it("counts the whole training transcript, including a missing terminal phone", () => {
    const clean = trainingEditCounts({ written: "بَت", heard: "بَت" });
    expect(clean).toEqual({ counts: counts(), matches: 3 });
    const terminal = trainingEditCounts({ written: "بَت", heard: "بَ" });
    expect(terminal.matches).toBe(2);
    expect(terminal.counts[6]).toBe(1);
  });

  it("fits finite positive prices and one shared training unit conversion", () => {
    const records = [trainingEditCounts({ written: "بَت", heard: "بَك" }),
      trainingEditCounts({ written: "جُد", heard: "جِد" })];
    const nats = fitAcousticCosts(records), scale = trainingCostScale(nats, records);
    expect(nats.every(c => Number.isFinite(c) && c > 0)).toBe(true);
    expect(Number.isFinite(scale) && scale > 0).toBe(true);
    expect(nats[0]).toBeLessThan(nats[3]);
    expect(() => trainingCostScale(nats, [trainingEditCounts({ written: "بت", heard: "بت" })])).toThrow();
  });
});

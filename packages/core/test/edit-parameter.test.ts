import { describe, expect, it } from "vitest";
import { dirichletLogPredictive, editParameterCounter, editParameterLogP, fitEditParameters, parameterEdits, parameterTrajectoryDecision } from "../src/edit-parameter.js";

describe("shared training parameter uncertainty", () => {
  it("integrates one draw over the whole trajectory rather than one mean per observation", () => {
    const alpha = [2, 2];
    expect(Math.exp(dirichletLogPredictive(alpha, [2, 0]))).toBeCloseTo(0.3, 12);
    expect(Math.exp(dirichletLogPredictive(alpha, [2, 0], true))).toBeCloseTo(0.25, 12);
    expect(Math.exp(dirichletLogPredictive(alpha, [1, 1]))).toBeCloseTo(0.2, 12);
    // The ordered paths AA, AB, BA, BB exhaust the probability mass.
    expect(2 * Math.exp(dirichletLogPredictive(alpha, [2, 0])) + 2 * Math.exp(dirichletLogPredictive(alpha, [1, 1]))).toBeCloseTo(1, 12);
  });

  it("converges to point estimates when training rates are well supported", () => {
    const mean = dirichletLogPredictive([900000, 100000], [5, 1], true);
    expect(dirichletLogPredictive([900000, 100000], [5, 1])).toBeCloseTo(mean, 4);
  });

  it("can reverse an identical mean-channel preference on complete trajectories", () => {
    const model = fitEditParameters([{ written: "بت", heard: "بت" }], ["ب", "ت"], 4);
    model.operations[0] = [2, 2, 1];
    model.substituted[0][1] = 1;
    const counter = editParameterCounter(model);
    const start = counter.empty();
    const a = counter.extend(counter.extend(start, { kind: "diagonal", reference: "ب", heard: "ب" }),
      { kind: "diagonal", reference: "ب", heard: "ب" }, 0.1);
    const b = counter.extend(counter.extend(start, { kind: "diagonal", reference: "ب", heard: "ب" }),
      { kind: "diagonal", reference: "ب", heard: "ت" });
    const result = parameterTrajectoryDecision(model, [a, b])!;
    expect(result.mean.selected).toBe(1);
    expect(result.marginal.selected).toBe(0);
    expect(() => parameterTrajectoryDecision(model, [a, start])).toThrow();
  });

  it("retains fixed duration and movement charges in both comparison arms", () => {
    const model = fitEditParameters([{ written: "بنننن", heard: "بن" }], ["ب", "ن"], 4);
    const counter = editParameterCounter(model);
    const empty = counter.empty();
    const duration = counter.extend(empty, { kind: "delete", reference: "ن", duration: true }, 20);
    expect(duration.counts).toEqual(empty.counts);
    expect(duration.fixedCost).toBe(22);
    expect(editParameterLogP(model, duration)).toBeCloseTo(-6.6, 12);
    expect(editParameterLogP(model, duration, true)).toBeCloseTo(-6.6, 12);
    expect(empty.fixedCost).toBe(0);
  });

  it("conditions the normalized operation rows on the written phone and keeps unknown phones explicit", () => {
    const model = fitEditParameters([{ written: "بتبت", heard: "بببب" }], ["ب", "ت", "د"], 1);
    expect(model.operations[0][0]).toBeGreaterThan(model.operations[1][0]);
    expect(model.substituted.every(row => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-12)).toBe(true);
    const counter = editParameterCounter(model);
    const start = counter.empty();
    const next = counter.extend(start, { kind: "diagonal", reference: "ت", heard: "ب" });
    expect(next.observations).toBe(1);
    expect(next.counts[4]).toBe(1);
    expect(start.counts.every(x => x === 0)).toBe(true);
    expect(() => counter.extend(start, { kind: "insert", heard: "?", duration: false })).toThrow();
  });

  it("distinguishes duration deletions from ordinary directed phone deletions", () => {
    const edits = parameterEdits({ written: "بنننن", heard: "بن" });
    expect(edits.filter(e => e.kind === "delete" && e.duration)).toHaveLength(3);
    expect(parameterEdits({ written: "بت", heard: "ب" })).toContainEqual({ kind: "delete", reference: "ت", duration: false });
    expect(parameterEdits({ written: "ب", heard: "بت" })).toContainEqual({ kind: "insert", heard: "ت", duration: false });
    expect(() => dirichletLogPredictive([1, 1], [0.5, 0])).toThrow();
  });
});

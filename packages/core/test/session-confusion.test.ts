import { describe, expect, it } from "vitest";
import { committedConfusionHistory, confusionPosterior, confusionRecording, confusionSpanCost, fitConfusionModel, freezeConfusion, phoneFeatures } from "../src/session-confusion.js";
import type { ConfusionModel, ConfusionRecording } from "../src/session-confusion.js";

const phones = ["د", "ت", "ض", "ط"];
const recording = (pairs: [number, number, number][]): ConfusionRecording => {
  const counts = phones.map(() => phones.map(() => 0));
  for (const [w, h, n] of pairs) counts[w][h] = n;
  return { pairs: counts };
};
const clean = recording([[0, 0, 100], [1, 1, 100]]);
const distorted = recording([[0, 1, 100], [1, 1, 100]]);

describe("recording-level confusion posterior", () => {
  it("shares devoicing evidence with an unobserved phone pair and normalizes every emission row", () => {
    expect(phoneFeatures("د")).toEqual(phoneFeatures("ض"));
    expect(phoneFeatures("ت")).toEqual(phoneFeatures("ط"));
    const model = fitConfusionModel([clean, distorted, clean, distorted], phones, 2, 16);
    for (const mapping of model.emissions) for (const row of mapping) {
      expect(row.every(x => x > 0 && Number.isFinite(x))).toBe(true);
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    }
    const a = freezeConfusion(model, confusionPosterior(model, [clean]));
    const b = freezeConfusion(model, confusionPosterior(model, [distorted]));
    const score = (ch: typeof a) => ch.marginal(model.prior.map((_, k) => ch.substitution("ض", "ط", k)));
    expect(score(b)).toBeLessThan(score(a));
  });

  it("keeps one latent mapping shared across tokens rather than redrawing it for each sound", () => {
    const model: ConfusionModel = { phones: ["د", "ت"], prior: [0.5, 0.5], pooled: [[0.5, 0.5], [0.5, 0.5]],
      emissions: [[[0.99, 0.01], [0.01, 0.99]], [[0.01, 0.99], [0.99, 0.01]]], mass: 16 };
    const ch = freezeConfusion(model, model.prior);
    const first = [0, 1].map(k => ch.substitution("د", "د", k));
    const second = [0, 1].map(k => ch.substitution("د", "ت", k));
    expect(ch.marginal(first.map((c, k) => c + second[k]))).toBeGreaterThan(ch.marginal(first) + ch.marginal(second));
  });

  it("freezes the posterior and model costs so later training cannot contaminate a comparison", () => {
    const model = fitConfusionModel([clean, distorted], phones, 2, 16);
    const posterior = confusionPosterior(model, [clean]);
    const frozen = freezeConfusion(model, posterior);
    const before = frozen.marginal([1, 7]);
    const sub = frozen.substitution("د", "ت", 0);
    posterior.reverse();
    model.emissions[0][0][1] = 0.9;
    expect(frozen.marginal([1, 7])).toBe(before);
    expect(frozen.substitution("د", "ت", 0)).toBe(sub);
  });

  it("uses the training prior with no earlier evidence and leaves unknown inventory phones unchanged", () => {
    const model = fitConfusionModel([clean, distorted], phones, 2, 16);
    expect(confusionPosterior(model, [])).toEqual(model.prior);
    const frozen = freezeConfusion(model, model.prior);
    expect(frozen.substitution("س", "ز", 0)).toBe(10);
    expect(() => freezeConfusion(model, [1, 1])).toThrow("Invalid confusion posterior");
  });

  it("marginalizes uncertain edit placement without teaching deletions as identity substitutions", () => {
    const exact = confusionRecording("د", "د", phones).pairs;
    expect(exact[0][0]).toBeGreaterThan(0.99);
    expect(exact[0][0]).toBeLessThan(1);
    const empty = confusionRecording("ضضض", "", phones).pairs;
    expect(empty.flat().reduce((a, b) => a + b, 0)).toBe(0);
    const ambiguous = confusionRecording("دد", "د", phones).pairs;
    expect(ambiguous[0][0]).toBeGreaterThan(0.9);
    expect(ambiguous[0][0]).toBeLessThanOrEqual(1);
  });

  it("requires prior committed, checker-acknowledged, unaccused speech outside the comparison", () => {
    const model = fitConfusionModel([clean, distorted], phones, 2, 16);
    const history = committedConfusionHistory(model);
    history.commit({ from: 0, to: 10, words: [0], recording: clean });
    history.commit({ from: 10, to: 20, words: [1], recording: distorted });
    expect(history.freezeBefore(30).posterior).toEqual(model.prior);
    history.feedback(20, []);
    const frozen = history.freezeBefore(10);
    expect(frozen.posterior).toEqual(confusionPosterior(model, [clean]));
    history.feedback(30, [0]);
    expect(history.freezeBefore(10).posterior).toEqual(model.prior);
    expect(frozen.posterior).toEqual(confusionPosterior(model, [clean]));
    expect(history.freezeBefore(20).posterior).toEqual(confusionPosterior(model, [distorted]));
    expect(() => history.commit({ from: 19, to: 21, words: [2], recording: clean })).toThrow("must not overlap");
  });

  it("keeps duration charges, paid initial prefixes and an open terminal frontier", () => {
    // Unlisted phones take exactly the existing directed edit costs in this fixture.
    const model = fitConfusionModel([clean, distorted], phones, 2, 16);
    const ch = freezeConfusion(model, model.prior);
    expect(confusionSpanCost("نننن", "ن", ch)).toBeCloseTo(0, 12);
    expect(confusionSpanCost("ننننس", "نس", ch)).toBeCloseTo(6, 12);
    expect(confusionSpanCost("بنس", "نس", ch)).toBeCloseTo(10, 12);
    expect(confusionSpanCost("نس", "ن", ch)).toBeCloseTo(0, 12);
  });
});

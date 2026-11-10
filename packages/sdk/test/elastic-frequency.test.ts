import { describe, expect, it } from "vitest";
import { deformationPrior, elasticComparison, elasticFeatures, elasticReversal, scoreElasticSequence, validateElasticModel,
  warpSpectrum, ElasticObservations, type ElasticModel } from "../src/elastic-frequency.js";

const zeros = () => Array<number>(24).fill(0);
const model = (): ElasticModel => ({ schema: 1, inventory: "toy", tokens: "ab", frameMs: 40, observationMs: 10,
  dimensions: 24, scale: 1, means: Array.from({ length: 3 }, () => [zeros(), zeros(), zeros()]),
  variances: Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => Array<number>(24).fill(.02))),
  counts: [0, 10, 10], stay: [.6, .7, .6], displacements: [-2, -1, 0, 1, 2], regularization: .5, maxFrames: 2048 });

describe("joint elastic time-frequency acoustic scoring", () => {
  it("matches enumeration of complete timing and continuous warp paths across a phone boundary", () => {
    const m = model(), raw = Array.from({ length: 7 }, () => zeros());
    m.means[1][0][7] = .1; m.means[2][2][9] = .1;
    const prior = deformationPrior(m, true);
    let acoustic = 0, completion = 0;
    const emission = (t: number, s: number, w: number) => {
      const p = s < 3 ? 1 : 2, q = s % 3;
      const mu = warpSpectrum(m.means[p][q], prior.displacements[w]);
      const v = warpSpectrum(m.variances[p][q], prior.displacements[w]);
      return mu.reduce((sum, mean, b) => sum - .5 * (Math.log(2 * Math.PI * v[b]) + (raw[t][b] - mean) ** 2 / v[b]), 0);
    };
    const visit = (t: number, s: number, w: number, mass: number, likelihood: number) => {
      likelihood += emission(t, s, w);
      if (t === raw.length - 1) {
        if (s === 5) { completion += mass; acoustic += mass * Math.exp(likelihood); }
        return;
      }
      for (const next of [s, s + 1]) if (next < 6) {
        const timing = next === s ? m.stay[s % 3] : 1 - m.stay[s % 3];
        for (let v = 0; v < 5; v++) if (Number.isFinite(prior.transitions[w][v]))
          visit(t + 1, next, v, mass * timing * Math.exp(prior.transitions[w][v]), likelihood);
      }
    };
    for (let w = 0; w < 5; w++) visit(0, 0, w, Math.exp(prior.initial[w]), 0);
    expect(scoreElasticSequence(m, raw, "ab", true)).toBeCloseTo(Math.log(acoustic / completion), 10);
  });
  it("normalizes warp mass and leaves frequency-invariant spectra equal to the rigid control", () => {
    const m = model(), prior = deformationPrior(m, true);
    expect(prior.initial.reduce((s, x) => s + Math.exp(x), 0)).toBeCloseTo(1, 12);
    for (let w = 0; w < 5; w++) {
      expect(prior.transitions[w].reduce((s, x) => s + Math.exp(x), 0)).toBeCloseTo(1, 12);
      for (let v = 0; v < 5; v++) if (Math.abs(w - v) > 1) expect(prior.transitions[w][v]).toBe(-Infinity);
    }
    const raw = Array.from({ length: 12 }, () => zeros());
    expect(scoreElasticSequence(m, raw, "ab", true)).toBeCloseTo(scoreElasticSequence(m, raw, "ab", false)!, 10);
  });
  it("recovers a shifted trajectory while retaining a real rival realization", () => {
    const m = model(), mean = zeros().map((_, b) => 3 * Math.exp(-(((b - 10) / 1.8) ** 2)));
    const shifted = warpSpectrum(mean, 2);
    m.means[1] = [mean, mean, mean];
    m.means[2] = Array.from({ length: 3 }, () => shifted.map((v, b) => v + (b % 2 ? .25 : -.25)));
    const c = { word: { surah: 1, ayah: 1, word: 1 }, written: "a", rival: "b", fromFrame: 0, toFrame: 3 };
    const result = elasticComparison(m, Array.from({ length: 12 }, () => shifted), c)!;
    expect(elasticReversal(result)).toBe(true);
    const rival = elasticComparison(m, Array.from({ length: 12 }, () => m.means[2][0]), c)!;
    expect(elasticReversal(rival)).toBe(false);
    expect(rival.elasticRival).toBeGreaterThan(rival.elasticWritten);
  });
  it("requires complete repeated articulations and abstains on unsupported or invalid inputs", () => {
    const m = model(), raw = Array.from({ length: 5 }, () => zeros());
    expect(scoreElasticSequence(m, raw, "aa", true)).toBeNull();
    expect(scoreElasticSequence(m, raw, "c", true)).toBeNull();
    expect(scoreElasticSequence(m, [Array<number>(24).fill(NaN)], "a", true)).toBeNull();
    expect(() => validateElasticModel({ ...m, displacements: [-3, 0, 3] }, m.inventory, m.tokens)).toThrow();
    expect(validateElasticModel(m, m.inventory, m.tokens)).toEqual(m);
  });
  it("pins endpoints and keeps every permitted spectral warp monotonic", () => {
    const x = Array.from({ length: 24 }, (_, i) => i);
    for (const d of [-2, -1, 0, 1, 2]) {
      const y = warpSpectrum(x, d);
      expect(y[0]).toBe(0); expect(y[23]).toBe(23);
      expect(y.slice(1).every((v, i) => v > y[i])).toBe(true);
    }
  });
  it("uses the same centered spectral observations at every audio push size", () => {
    const pcm = Float32Array.from({ length: 2000 }, (_, i) => .2 * Math.sin(i * .15));
    const all = elasticFeatures(pcm);
    expect(all.every(row => Math.abs(row.reduce((a, b) => a + b, 0)) < 1e-8)).toBe(true);
    const incremental = new ElasticObservations();
    for (let i = 0; i < pcm.length; i += 137) incremental.push(pcm.subarray(i, i + 137));
    expect(incremental.frames).toEqual(all);
  });
});

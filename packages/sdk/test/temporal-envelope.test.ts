import { describe, expect, it } from "vitest";
import { EnvelopeObservations, envelopeFeatures, envelopeFrame, frequencyPrediction, scoreEnvelopeSequence, validateEnvelopeModel, envelopeReversal } from "../src/temporal-envelope.js";
import type { EnvelopeFrame, EnvelopeModel } from "../src/temporal-envelope.js";

const dimensions = 19;
const frame = (x: number): EnvelopeFrame => ({ envelope: Array(dimensions).fill(x), control: Array(dimensions).fill(x) });
const model = (): EnvelopeModel => ({ schema: 1, inventory: "toy", tokens: "بن", frameMs: 40, order: 2, ridge: .001,
  maxFrames: 4096, phoneCounts: [10, 10], envelopeScale: Array(dimensions).fill(1), controlScale: Array(dimensions).fill(1),
  envelope: [0, 1, -1].map(v => ({ mean: Array(dimensions).fill(v), variance: Array(dimensions).fill(1) })),
  control: [0, 1, -1].map(v => ({ mean: Array(dimensions).fill(v), variance: Array(dimensions).fill(1) })) });

describe("frequency-domain temporal-envelope observations", () => {
  it("fits along frequency with regularized, stable reflection coefficients", () => {
    const r = -20 / 30.03;
    const next = -(11 + r * 20) / (30.03 * (1 - r * r));
    const actual = frequencyPrediction([1, 2, 3, 4], 2, .001);
    expect(actual[0]).toBeCloseTo(Math.atanh(r), 12);
    expect(actual[1]).toBeCloseTo(Math.atanh(next), 12);
    expect(frequencyPrediction(Array(50).fill(0), 4, .001).every(x => x === 0)).toBe(true);
  });
  it("retains temporal interruption structure and gives both arms equal capacity", () => {
    const sustained = Float64Array.from({ length: 2048 }, (_, i) => Math.sin(i * .19) + .3 * Math.sin(i * .47));
    const interrupted = sustained.map((v, i) => i > 900 && i < 1040 ? v * .01 : v);
    for (const order of [2, 4] as const) {
      const a = envelopeFrame(sustained, order, .001), b = envelopeFrame(interrupted, order, .001);
      expect(a.envelope.length).toBe(1 + 6 * (order + 1));
      expect(a.control.length).toBe(a.envelope.length);
      expect([...a.envelope, ...a.control, ...b.envelope, ...b.control].every(Number.isFinite)).toBe(true);
      expect(a.envelope.slice(1).some((v, i) => Math.abs(v - b.envelope[i + 1]) > .01)).toBe(true);
    }
  });
  it("keeps frame coordinates and values independent of audio push size", () => {
    const audio = Float32Array.from({ length: 6400 }, (_, i) => Math.sin(i * .27) * (.2 + i / 64000));
    const stream = new EnvelopeObservations(2, .001);
    for (let i = 0; i < audio.length; i += 137) stream.push(audio.subarray(i, i + 137));
    stream.push(new Float32Array(0), true);
    expect(stream.frames).toEqual(envelopeFeatures(audio, 2, .001));
    expect(stream.frames.length).toBe(10);
  });
});

describe("complete-pronunciation latent-boundary scoring", () => {
  it("agrees with exhaustive summation of a small phone/silence graph", () => {
    const m = model(), raw = [frame(1), frame(0), frame(-1)], states = [0, 1, 0, 2, 0];
    const emissions = raw.map(r => m.envelope.map(e => r.envelope.reduce((s, x, d) => s - .5 * (Math.log(2 * Math.PI) + (x - e.mean[d]) ** 2), 0)));
    let mass = 0, prior = 0;
    function visit(t: number, s: number, p: number, likelihood: number) {
      const nextLikelihood = likelihood * Math.exp(emissions[t][states[s]]);
      if (t === raw.length - 1) { if (s >= states.length - 2) { mass += p * nextLikelihood; prior += p; } return; }
      const dest = s === 4 ? [[4, 1]] : s % 2 ? [[s, .6], [s + 1, .2], [Math.min(4, s + 2), .2]] : [[s, .6], [s + 1, .4]];
      for (const [to, weight] of dest) visit(t + 1, to, p * weight, nextLikelihood);
    }
    visit(0, 0, .5, 1); visit(0, 1, .5, 1);
    expect(scoreEnvelopeSequence(m, raw, "بن", true)).toBeCloseTo(Math.log(mass / prior), 10);
  });
  it("requires every phone, preserves repetition and abstains outside learned coverage", () => {
    const m = model();
    expect(scoreEnvelopeSequence(m, [frame(1)], "ب", true)).not.toBeNull();
    expect(scoreEnvelopeSequence(m, [frame(1)], "بب", true)).toBeNull();
    expect(scoreEnvelopeSequence(m, [frame(1), frame(1)], "بب", true)).not.toBeNull();
    m.phoneCounts[1] = 0;
    expect(scoreEnvelopeSequence(m, [frame(-1)], "ن", true)).toBeNull();
    expect(scoreEnvelopeSequence(m, [frame(1)], "ت", true)).toBeNull();
    expect(scoreEnvelopeSequence(m, [frame(NaN)], "ب", true)).toBeNull();
  });
  it("gives equal comparisons to identical hypotheses and a strict two-arm reversal", () => {
    const m = model(), raw = [frame(1), frame(1)];
    expect(scoreEnvelopeSequence(m, raw, "ب", true)).toBe(scoreEnvelopeSequence(m, raw, "ب", false));
    const c = { word: { surah: 1, ayah: 1, word: 1 }, written: "ب", rival: "ن", fromFrame: 1, toFrame: 3,
      controlWritten: -2, controlRival: -1, envelopeWritten: -1, envelopeRival: -2 };
    expect(envelopeReversal(c)).toBe(true);
    expect(envelopeReversal({ ...c, envelopeWritten: -2 })).toBe(false);
    expect(envelopeReversal({ ...c, controlWritten: -1 })).toBe(false);
    expect(envelopeReversal({ ...c, envelopeWritten: Infinity })).toBe(false);
  });
  it("rejects corrupt or mismatched assets before inference", () => {
    expect(validateEnvelopeModel(model(), "toy", "بن")).toEqual(model());
    expect(() => validateEnvelopeModel(model(), "other", "بن")).toThrow();
    const corrupt = model(); corrupt.envelope[1].variance[0] = 0;
    expect(() => validateEnvelopeModel(corrupt, "toy", "بن")).toThrow();
  });
});

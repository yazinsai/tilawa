import { describe, expect, it } from "vitest";
import type { HeldBandEvidence, PhonemeEvidence, WordEvidenceRequest } from "@tilawa/core";
import { createHeldBandScorer, timingPartition, carryHeldBandEvidence, type HeldBandModel } from "../src/held-band.js";
import { SpectralBands } from "../src/fbank.js";
import { insertionCells } from "../src/seam-insertion.js";

const request: WordEvidenceRequest = { word: { surah: 1, ayah: 1, word: 1 }, written: "ب", observed: "ت", fromFrame: 0, toFrame: 2 };
const bundle = (): HeldBandModel => {
  const predictor = (mean: number) => ({ weights: [[mean], [0]], variance: [1] });
  const partition = (visible: number[], held: number[]) => ({ visible, held,
    background: predictor(0), phones: [predictor(0), predictor(2), predictor(-2)] });
  return { schema: 1, inventory: "test", tokens: "بت", edges: [1, 3, 5], mean: [0, 0], scale: [1, 1],
    phoneMean: [[0, 0], [0, 0], [0, 0]], phoneVariance: [[1, 1], [1, 1], [1, 1]], stay: .8,
    phoneCount: [1, 1, 1],
    partitions: [partition([0], [1]), partition([1], [0])] };
};

describe("held-band prediction", () => {
  it("prefers complete written speech in both partitions, and actual rival speech reverses both preferences", () => {
    const scorer = createHeldBandScorer(bundle());
    const written = scorer.compare(Array.from({ length: 8 }, () => Float64Array.of(2, 2)), request, 40)!;
    const rival = scorer.compare(Array.from({ length: 8 }, () => Float64Array.of(-2, -2)), request, 40)!;
    expect(written.every(m => m > 0)).toBe(true);
    expect(rival.every(m => m < 0)).toBe(true);
    expect(rival[0]).toBeCloseTo(-written[0], 10);
    expect(written[1]).toBeCloseTo(written[0], 10);
  });

  it("requires complementary agreement and abstains on absent observations or unknown pronunciations", () => {
    const scorer = createHeldBandScorer(bundle());
    const frames = Array.from({ length: 8 }, () => Float64Array.of(2, -2));
    const mixed = scorer.compare(frames, request, 40)!;
    expect(mixed[0]).toBeLessThan(0);
    expect(mixed[1]).toBeGreaterThan(0);
    expect(scorer.compare(frames.slice(0, 7), request, 40)).toBeNull();
    expect(scorer.compare(frames, { ...request, written: "ن" }, 40)).toBeNull();
    expect(scorer.compare(frames, { ...request, written: "ب".repeat(20) }, 40)).toBeNull();
    const uncovered = bundle(); uncovered.phoneCount[1] = 0;
    expect(createHeldBandScorer(uncovered).compare(frames, request, 40)).toBeNull();
  });

  it("does not count predictable candidate-independent background as lexical advantage", () => {
    const model = bundle(), observations = Array.from({ length: 8 }, () => Float64Array.of(2, 2));
    const before = createHeldBandScorer(model).compare(observations, request, 40);
    model.partitions.forEach(p => { p.background.weights[0][0] = 100; });
    const after = createHeldBandScorer(model).compare(observations, request, 40)!;
    expect(after[0]).toBeCloseTo(before![0], 7);
    expect(after[1]).toBeCloseTo(before![1], 7);
  });

  it("rejects overlapping partitions and nonfinite or degenerate models", () => {
    const model = bundle();
    model.partitions[1].visible = [1, 0];
    expect(() => createHeldBandScorer(model)).toThrow();
    const degenerate = bundle(); degenerate.partitions[0].phones[1].variance[0] = 0;
    expect(() => createHeldBandScorer(degenerate)).toThrow();
  });
});

describe("normalized timing and complete acceptance", () => {
  it("matches exhaustive weighted paths with repeated labels and optional blanks", () => {
    const emissions = [Float64Array.of(-.2, -.8, -1), Float64Array.of(-.9, -.3, -.4), Float64Array.of(-.1, -.7, -.9)];
    for (const labels of [[1], [1, 2], [1, 1]]) {
      const ids = Array.from({ length: labels.length * 2 + 1 }, (_, s) => s % 2 ? labels[(s - 1) / 2] : 0);
      const paths: number[] = [];
      const visit = (t: number, s: number, weight: number) => {
        weight += emissions[t][ids[s]];
        if (t === emissions.length - 1) { if (s >= ids.length - 2) paths.push(weight); return; }
        if (s === ids.length - 1) { visit(t + 1, s, weight); return; }
        if (s % 2 === 0) {
          visit(t + 1, s, weight + Math.log(.5)); visit(t + 1, s + 1, weight + Math.log(.5));
        } else {
          const direct = s + 2 < ids.length && ids[s] !== ids[s + 2];
          visit(t + 1, s, weight + Math.log(.8));
          visit(t + 1, s + 1, weight + Math.log(.2 / (direct ? 2 : 1)));
          if (direct) visit(t + 1, s + 2, weight + Math.log(.1));
        }
      };
      visit(0, 0, Math.log(.5)); visit(0, 1, Math.log(.5));
      expect(timingPartition(emissions, labels, .8)).toBeCloseTo(Math.log(paths.reduce((s, p) => s + Math.exp(p), 0)), 12);
    }
    expect(timingPartition(emissions.slice(0, 2), [1, 1], .8)).toBe(-Infinity);
  });
});

describe("spectral observations and certificates", () => {
  it("extracts identical disjoint observations across arbitrary audio pushes", () => {
    const pcm = Float32Array.from({ length: 16000 }, (_, i) => Math.sin(i * .07) * .1);
    const whole = new SpectralBands([1, 10, 50, 256]).push(pcm);
    const bands = new SpectralBands([1, 10, 50, 256]);
    const split: Float64Array[] = [];
    for (let i = 0; i < pcm.length; i += 137) split.push(...bands.push(pcm.subarray(i, i + 137)));
    expect(split).toEqual(whole);
    expect(() => new SpectralBands([1, 50, 10])).toThrow();
  });

  it("preserves tokens and posteriors through JSON and ordinary engine rechunking", () => {
    const chunks: PhonemeEvidence[] = [{ inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 0, end: 1, conf: .9 }],
      posteriors: { startFrame: 0, nFrames: 2, k: 1, ids: [1, 0], probs: [.9, .9] } }];
    const metadata: HeldBandEvidence[] = [{ ...request, margins: [2, 3] }];
    const before = structuredClone(chunks), enriched = carryHeldBandEvidence(chunks, metadata);
    const cells = insertionCells(JSON.parse(JSON.stringify(enriched)));
    expect(cells.flatMap(c => c.phonemes)[0].heldBands).toEqual(metadata);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(chunks).toEqual(before);
    expect(cells.flatMap(c => c.phonemes).map(({ heldBands: _, ...p }) => p)).toEqual(chunks[0].phonemes);
  });
});

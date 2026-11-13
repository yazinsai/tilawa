import { describe, expect, it } from "vitest";
import { vowelResidualVeto } from "@tilawa/core";
import { carryVowelResidualEvidence, sequenceLogProbability, vowelContext, residualComparisons, validateVowelResidualModel } from "../src/vowel-residual.js";
import type { PhoneReadout, VowelResidualModel } from "../src/vowel-residual.js";
import type { ModelManifest } from "../src/model.js";

const zeros = (n: number, d: number) => Array.from({ length: n }, () => Array(d).fill(0));
const head = (v: number): PhoneReadout => ({ weight: zeros(32, 32), bias: Array(32).fill(0), output: zeros(v, 32), intercept: Array(v).fill(0) });
const tokens = "بتَُِ", manifest = { tokens, inventory: "test", sha256: "a".repeat(64) } as ModelManifest;
const parameters = (): VowelResidualModel => ({ schema: 1, inventory: manifest.inventory, tokens, primarySha256: manifest.sha256,
  featureOutput: "hidden", projection: zeros(32, 32), center: Array(32).fill(0), scale: Array(32).fill(1),
  vowels: [[3], [4], [5]], durations: [.25, .5, .25], predictor: zeros(32, 10), shuffledPredictor: zeros(32, 10),
  residual: head(6), original: head(6), shuffled: head(6) });
const candidate = { word: { surah: 1, ayah: 1, word: 1 }, written: "بَ", rival: "تَ", fromFrame: 0, toFrame: 6 };
const certificate = { ...candidate, residualMargin: 1, originalMargin: -1, shuffledMargin: -1, consonantMargin: 1 };

describe("vowel-conditioned acoustic residual", () => {
  it("marginalizes vowel intervals and their normalized phases independently of pronunciations", () => {
    const rows = [[.05, .05, .05, .8, .025, .025], [.2, .05, .05, .65, .025, .025], [.05, .8, .05, .05, .025, .025]].map(r => r.map(Math.log));
    const phi = vowelContext(rows, [[3], [4], [5]], [.2, .5, .3]);
    for (const r of phi) {
      expect(r[0] + r[1] + r[4] + r[7]).toBeCloseTo(1, 8);
      for (const q of [1, 4, 7]) {
        expect(r[q + 1]).toBeGreaterThanOrEqual(-1e-10);
        expect(r[q + 1]).toBeLessThanOrEqual(r[q] + 1e-10);
        expect(r[q + 2]).toBeLessThanOrEqual(r[q + 1] + 1e-10);
      }
    }
    expect(phi[0][1]).toBeGreaterThan(phi[0][4]);
    expect(phi[0][2] / phi[0][1]).toBeLessThan(phi[1][2] / phi[1][1]);
    expect(vowelContext(rows, [[3], [4], [5]], [.2, .5, .3])).toEqual(phi);
  });
  it("sums all CTC paths but requires every phone, including repeated consonants", () => {
    const lp = [[.2, .7, .1], [.4, .4, .2], [.2, .1, .7]].map(r => r.map(Math.log));
    const paths = Array.from({ length: 27 }, (_, i) => [Math.floor(i / 9), Math.floor(i / 3) % 3, i % 3]);
    let mass = 0;
    for (const p of paths) {
      const collapsed = p.filter((v, i) => i === 0 || v !== p[i - 1]).filter(Boolean);
      if (collapsed.join() === "1,2") mass += Math.exp(p.reduce((s, v, t) => s + lp[t][v], 0));
    }
    expect(sequenceLogProbability(lp, [1, 2])).toBeCloseTo(Math.log(mass), 10);
    expect(sequenceLogProbability(lp.slice(0, 2), [1, 1])).toBe(-Infinity);
    expect(sequenceLogProbability(lp, [1, 2, 1, 2])).toBe(-Infinity);
  });
  it("requires both rival-preferring controls and positive written-consonant evidence", () => {
    expect(vowelResidualVeto(certificate)).toBe(true);
    for (const field of ["residualMargin", "originalMargin", "shuffledMargin", "consonantMargin"] as const) {
      expect(vowelResidualVeto({ ...certificate, [field]: 0 })).toBe(false);
      expect(vowelResidualVeto({ ...certificate, [field]: NaN })).toBe(false);
    }
    expect(vowelResidualVeto({ ...certificate, rival: "بِ" })).toBe(false);
    const m = parameters();
    // Matching vowels and opposing consonants: controls favor ت, residual favors ب.
    m.residual.intercept[1] = 3; m.original.intercept[2] = 3; m.shuffled.intercept[2] = 3;
    const x = zeros(6, 32), context = zeros(6, 10); context.forEach(r => { r[1] = 1; r[2] = .5; r[3] = .25; });
    const result = residualComparisons(m, x, context, candidate)!;
    expect(vowelResidualVeto(result)).toBe(true);
    // Vowels favor the written word, but its consonant remains unsupported.
    m.residual.intercept[1] = 0; m.residual.intercept[2] = 1; m.residual.intercept[3] = 4;
    const vowelOnly = residualComparisons(m, x, context, { ...candidate, rival: "تِ" })!;
    expect(vowelOnly.residualMargin).toBeGreaterThan(0);
    expect(vowelOnly.consonantMargin).toBeLessThan(0);
    expect(vowelResidualVeto(vowelOnly)).toBe(false);
    expect(residualComparisons(m, x, context, { ...candidate, written: "غ" })).toBeNull();
  });
  it("validates all arms and preserves primary evidence through certificate serialization", () => {
    const m = parameters(); expect(validateVowelResidualModel(m, manifest)).toBe(m);
    expect(() => validateVowelResidualModel({ ...m, shuffledPredictor: [] }, manifest)).toThrow();
    expect(() => validateVowelResidualModel({ ...m, vowels: [[3], [3], [5]] }, manifest)).toThrow();
    expect(() => validateVowelResidualModel({ ...m, primarySha256: "bad" }, manifest)).toThrow();
    const chunks = [{ inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 0, end: 1, conf: .9 }],
      posteriors: { k: 1, nFrames: 2, startFrame: 0, ids: [1, 1], probs: [.9, .9] } }];
    const enriched = carryVowelResidualEvidence(chunks, [certificate]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toMatchObject(chunks[0].phonemes[0]);
    expect(JSON.parse(JSON.stringify(enriched))[0].phonemes[0].vowelResidualVetoes).toEqual([certificate]);
    expect(chunks[0].phonemes[0]).not.toHaveProperty("vowelResidualVetoes");
  });
});

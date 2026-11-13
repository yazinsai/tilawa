import { describe, expect, it } from "vitest";
import { wordScrubVeto } from "@tilawa/core";
import { carryWordScrubEvidence, sequenceLogProbability, scrubComparisons, validateWordScrubModel } from "../src/word-scrub.js";
import type { PhoneReadout, WordScrubModel } from "../src/word-scrub.js";
import type { ModelManifest } from "../src/model.js";

const zeros = (n: number, d: number) => Array.from({ length: n }, () => Array(d).fill(0));
const head = (v: number): PhoneReadout => ({ weight: zeros(32, 32), bias: Array(32).fill(0), residual: zeros(32, 32), output: zeros(v, 32), intercept: Array(v).fill(0) });
const tokens = "بتَُِ", manifest = { tokens, inventory: "test", sha256: "a".repeat(64) } as ModelManifest;
const parameters = (): WordScrubModel => ({ schema: 1, inventory: manifest.inventory, tokens, primarySha256: manifest.sha256,
  featureOutput: "hidden", projection: zeros(32, 32), center: Array(32).fill(0), scale: Array(32).fill(1),
  residual: head(6), original: head(6), shuffled: head(6) });
const candidate = { word: { surah: 1, ayah: 1, word: 1 }, written: "بَ", rival: "تَ", fromFrame: 0, toFrame: 6 };
const certificate = { ...candidate, residualMargin: 1, originalMargin: -1, shuffledMargin: -1 };

describe("conditional lexical acoustic scrub", () => {
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
  it("requires both rival-preferring controls and complete written evidence", () => {
    expect(wordScrubVeto(certificate)).toBe(true);
    for (const field of ["residualMargin", "originalMargin", "shuffledMargin"] as const) {
      expect(wordScrubVeto({ ...certificate, [field]: 0 })).toBe(false);
      expect(wordScrubVeto({ ...certificate, [field]: NaN })).toBe(false);
    }
    expect(wordScrubVeto({ ...certificate, rival: candidate.written })).toBe(false);
    const m = parameters();
    // Matching vowels and opposing consonants: controls favor ت, residual favors ب.
    m.residual.intercept[1] = 3; m.original.intercept[2] = 3; m.shuffled.intercept[2] = 3;
    const x = zeros(6, 32);
    expect(wordScrubVeto(scrubComparisons(m, x, candidate)!)).toBe(true);
    m.original.intercept[2] = 0;
    expect(wordScrubVeto(scrubComparisons(m, x, candidate)!)).toBe(false);
    expect(scrubComparisons(m, x, { ...candidate, written: "غ" })).toBeNull();
    expect(scrubComparisons(m, x.slice(0, 1), candidate)).toBeNull();
  });
  it("validates all arms and preserves primary evidence through certificate serialization", () => {
    const m = parameters(); expect(validateWordScrubModel(m, manifest)).toBe(m);
    expect(() => validateWordScrubModel({ ...m, shuffled: { ...m.shuffled, residual: [] } }, manifest)).toThrow();
    expect(() => validateWordScrubModel({ ...m, primarySha256: "bad" }, manifest)).toThrow();
    const chunks = [{ inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 0, end: 1, conf: .9 }],
      posteriors: { k: 1, nFrames: 2, startFrame: 0, ids: [1, 1], probs: [.9, .9] } }];
    const enriched = carryWordScrubEvidence(chunks, [certificate]);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    expect(enriched[0].phonemes[0]).toMatchObject(chunks[0].phonemes[0]);
    expect(JSON.parse(JSON.stringify(enriched))[0].phonemes[0].wordScrubVetoes).toEqual([certificate]);
    expect(chunks[0].phonemes[0]).not.toHaveProperty("wordScrubVetoes");
  });
});

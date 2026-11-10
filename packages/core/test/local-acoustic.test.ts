import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/reference.js";
import { createLocalSearch } from "../src/local.js";
import { localAcousticCandidates } from "../src/local-acoustic.js";
import { UNIT } from "../src/fuzzy.js";
import { trackerInput } from "../src/tracker.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const first = index.words.find((w) => w.surah === 112 && w.ayah === 1 && w.word === 1)!;
const search = () => createLocalSearch(index, { offset: first.start, origin: first.start, edits: 0 });

describe("local lattice composition", () => {
  it("retains every existing edit/movement cost for a single acoustic string", () => {
    const s = search();
    const checkpoint = s.save();
    const phones = index.phonemes.slice(first.start, first.end);
    for (const p of phones) s.step(p);
    const original = s.candidates().filter((c) => Number.isFinite(c.cost));
    const composed = localAcousticCandidates(s, checkpoint, [{ phones, runs: [], logProbability: -2 }], 3);
    expect(composed).toHaveLength(original.length);
    for (const c of composed) {
      const old = original.find((o) => o.offset === c.offset)!;
      expect(c.origin).toBe(old.origin);
      expect(c.cost - old.cost).toBeCloseTo(2 * UNIT / 3, 10);
    }
  });

  it("marginalizes different strings once each and leaves the live search unchanged", () => {
    const s = search();
    const checkpoint = s.save();
    s.step("ق");
    const before = s.position(), candidates = s.candidates();
    const prefixes = [{ phones: "ق", runs: [], logProbability: Math.log(0.7) }, { phones: "ك", runs: [], logProbability: Math.log(0.3) }];
    const one = prefixes.map((p) => localAcousticCandidates(s, checkpoint, [p], 3));
    const combined = localAcousticCandidates(s, checkpoint, prefixes, 3);
    for (const c of combined) {
      const a = one[0].find((o) => o.offset === c.offset)!, b = one[1].find((o) => o.offset === c.offset)!;
      const expected = -Math.log(Math.exp(-a.cost * 3 / UNIT) + Math.exp(-b.cost * 3 / UNIT)) * UNIT / 3;
      expect(c.cost).toBeCloseTo(expected, 10);
    }
    expect(s.position()).toEqual(before);
    expect(s.candidates()).toEqual(candidates);
  });

  it("restores complete-interval costs when rewinding a normalized DP row", () => {
    const s = search();
    s.step("ز");
    const checkpoint = s.save();
    const original = s.candidates();
    s.step("ر");
    s.restore(checkpoint);
    expect(s.candidates()).toEqual(original);
  });

  it("applies marginal frontier costs without changing movement costs on later tokens", () => {
    const s = search();
    const checkpoint = s.save();
    const phones = index.phonemes.slice(first.start, first.end);
    for (const p of phones) s.step(p);
    const before = s.position();
    const cs = localAcousticCandidates(s, checkpoint, [{ phones, runs: [], logProbability: -2 }], 3);
    const original = s.save();
    s.step("ه");
    const next = s.position();
    s.restore(original);
    s.rescore(cs, before.offset);
    expect(s.position()).toEqual(before);
    s.step("ه");
    expect(s.position()).toEqual(next);
  });

  it("routes local rows without changing acquisition or listener evidence", () => {
    const posteriors = { k: 2, startFrame: 0, nFrames: 1, ids: [1, 0], probs: [0.7, 0.3] };
    const evidence = { inventory: index.inventoryId, frameMs: 40, phonemes: [{ p: "ق", start: 0, end: 0, conf: 0.7 }], posteriors };
    const input = trackerInput(evidence);
    expect(input.posteriors).toBe(undefined);
    expect(input.trackerPosteriors).toBe(posteriors);
    expect(input.phonemes).toBe(evidence.phonemes);
    expect(evidence.posteriors).toBe(posteriors);
  });

  it("does not count another boundary witness for an already-summed string as new probability", () => {
    const s = search(), checkpoint = s.save();
    const p = { phones: "ق", runs: [{ p: "ق", start: 0, end: 1 }], logProbability: Math.log(0.7) };
    const alternativeBoundary = { ...p, runs: [{ p: "ق", start: 1, end: 1 }] };
    expect(localAcousticCandidates(s, checkpoint, [p, alternativeBoundary], 3)).toEqual(localAcousticCandidates(s, checkpoint, [p], 3));
  });
});

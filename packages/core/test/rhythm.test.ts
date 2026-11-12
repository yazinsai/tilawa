import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import { rhythmClasses, rhythmCopula, rhythmRatios } from "../src/rhythm.js";
import type { RhythmModel, RhythmRow } from "../src/rhythm.js";
import type { ReferenceIndex } from "../src/reference.js";

const model = (rho = 0): RhythmModel => ({
  mean: Array(8).fill(0), sd: Array(8).fill(1),
  rho: Array.from({ length: 8 }, () => Array(8).fill(rho)),
});
const ref = "بَننناَدَ";
const starts = new Set([0, 5]);
const rows: RhythmRow[] = [
  { lo: 0, cost: Float64Array.from([0, ...Array(ref.length).fill(Infinity)]), best: 0, p: "ب", duration: 2 },
  { lo: 0, cost: Float64Array.from([10, 0, 5, 7, 9, 11, 20, 30, 40, 50]), best: 1, p: "َ", duration: 5 },
  { lo: 0, cost: Float64Array.from([20, 10, 0, 2, 4, 6, 10, 15, 20, 25]), best: 2, p: "ن", duration: 1 },
];
const score = (m: RhythmModel, r = rows) => rhythmRatios(ref, rhythmClasses(ref), r, m, starts, () => 0, () => 0, 20, .001);

describe("relative rhythm", () => {
  it("ties the copula's two marginals to the independent control", () => {
    // Integrate the joint/independent ratio against the second marginal. It must preserve the first marginal
    // even when the first sound is unusually short or long, rather than relaxing a duration threshold.
    const h = .002;
    for (const rho of [-.6, .6]) for (const z1 of [-2, 0, 2]) {
      let mass = 0;
      for (let z2 = -10; z2 < 10; z2 += h) {
        mass += Math.exp(rhythmCopula(z1, z2, rho) - z2*z2/2) / Math.sqrt(2*Math.PI)*h;
      }
      expect(mass).toBeCloseTo(1, 5);
    }
  });

  it("adds exactly zero for the matched independent model, including jumps and deletions", () => {
    expect([...score(model())]).toEqual(Array(rows[0].cost.length).fill(0));
  });

  it("uses adjacent coordination and stays invariant to a common speaking-rate change", () => {
    const m = model(.35);
    m.mean[1] = .4;
    m.mean[4] = -.3;
    m.sd[4] = .6;
    const original = [...score(m)];
    const faster = [...score(m, rows.map(r => ({ ...r, duration: r.duration*8 })))];
    for (let i = 0; i < original.length; i++) expect(faster[i]).toBeCloseTo(original[i], 10);
    const rearranged = [...score(m, rows.map((r, i) => ({ ...r, duration: [2, 1, 5][i] })))];
    expect(rearranged.some((v, i) => Math.abs(v - original[i]) > .01)).toBe(true);
  });

  it("does not fabricate timing when a token has no valid duration", () => {
    expect([...score(model(.5), rows.map((r, i) => ({ ...r, duration: i === 1 ? NaN : r.duration })))])
      .toEqual(Array(rows[0].cost.length).fill(0));
  });

  it("conditions lengthened nasals on pronunciation run structure", () => {
    expect([...rhythmClasses("نَناانننں")]).toEqual([3, 1, 3, 2, 2, 4, 4, 4, 4]);
  });

  it("rewinds timed observations with the local search, and rho zero keeps acoustic behavior", () => {
    const index: ReferenceIndex = {
      version: "test", inventoryId: "test", phonemes: ref, tokens: new Set(ref), gramSize: 2, grams: new Map(),
      words: [{ surah: 1, ayah: 1, word: 1, start: 0, end: 5 }, { surah: 1, ayah: 1, word: 2, start: 5, end: ref.length }],
      wordAt: offset => offset < 5 ? 0 : 1,
    };
    const initial = { offset: 1, origin: 0, edits: 0 };
    const timed = createLocalSearch(index, initial, model());
    const plain = createLocalSearch(index, initial);
    for (const [p, d] of [["َ", 5], ["ن", 1], ["ب", 2], ["َ", 4]] as const) {
      timed.step(p, d); plain.step(p);
      expect(timed.position()).toEqual(plain.position());
    }
    const m = model(.4);
    m.mean[4] = -.3;
    m.sd[4] = .6;
    const learned = createLocalSearch(index, initial, m);
    for (let i = 0; i < 16; i++) learned.step(ref[i % ref.length], [2, 5, 1][i % 3]);
    const saved = learned.save();
    for (const p of "بَدَ") learned.step(p, 6);
    const first = learned.position();
    learned.restore(saved);
    for (const p of "بَدَ") learned.step(p, 6);
    expect(learned.position()).toEqual(first);
  });
});

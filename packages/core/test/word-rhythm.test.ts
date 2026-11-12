import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import { nucleusCorrelation, wordRhythmCopula, wordRhythmRatios, wordRhythmStructure } from "../src/word-rhythm.js";
import type { WordRhythmModel, WordRhythmRow } from "../src/word-rhythm.js";
import type { ReferenceIndex } from "../src/reference.js";

const model = (rho = 0): WordRhythmModel => ({ mean: Array(8).fill(0), sd: Array(8).fill(1), rho: Array(8).fill(rho) });
const ref = "بَتِدُ";
const words = [{ start: 0, end: ref.length }];
const rows: WordRhythmRow[] = [...ref].map((p, t) => ({
  lo: 0, cost: Float64Array.from(Array.from({ length: ref.length+1 }, (_, i) => i === t ? 0 : Infinity)),
  best: t, p, duration: [2, 8, 2, 1, 2, 6][t],
}));
const score = (m: WordRhythmModel, r = rows, ws = words) => wordRhythmRatios(ref, wordRhythmStructure(ref, ws), r, m,
  new Set(ws.map(w => w.start)), () => 0, () => 0, Infinity, 0);

describe("word-internal nucleus rhythm", () => {
  it("preserves every independent duration marginal", () => {
    const h = .002;
    for (const rho of [-.7, .7]) for (const left of [-2, 0, 2]) {
      let mass = 0;
      for (let right = -10; right < 10; right += h) mass +=
        Math.exp(wordRhythmCopula(left, right, rho)-right*right/2)/Math.sqrt(2*Math.PI)*h;
      expect(mass).toBeCloseTo(1, 5);
    }
  });

  it("compiles one lengthened nucleus with distinct intervening consonant groups", () => {
    const s = wordRhythmStructure("بَااتِكدُ", [{ start: 0, end: 9 }]);
    expect(s.starts).toEqual([1, 5, 8]);
    expect([...s.nucleus]).toEqual([-1, 0, 0, 0, -1, 1, -1, -1, 2]);
    expect(s.links).toEqual([-1, 4, 1]);
    expect([...s.classes].slice(1, 4)).toEqual([2, 2, 2]);
  });

  it("marginalizes missing emitted nuclei through correlation products, and never links different words", () => {
    const s = wordRhythmStructure(ref, words), m = model(.6);
    expect(nucleusCorrelation(s, m, 0, 1)).toBeCloseTo(.6);
    expect(nucleusCorrelation(s, m, 0, 2)).toBeCloseTo(.36);
    const split = wordRhythmStructure(ref, [{ start: 0, end: 2 }, { start: 2, end: ref.length }]);
    expect(nucleusCorrelation(split, m, 0, 2)).toBe(0);
    expect(nucleusCorrelation(s, m, 0, 0)).toBe(0);
  });

  it("uses word grouping while sharing the identical observations and duration marginals", () => {
    const m = model(.65);
    const joined = [...score(m)];
    const split = [...score(m, rows, [{ start: 0, end: 2 }, { start: 2, end: 4 }, { start: 4, end: 6 }])];
    expect(joined.some(v => Math.abs(v) > .01)).toBe(true);
    for (const v of split) expect(v).toBeCloseTo(0, 10);
    expect([...score(model())]).toEqual(Array(ref.length+1).fill(0));
    const scaled = [...score(m, rows.map(r => ({ ...r, duration: 10*r.duration })))];
    joined.forEach((v, i) => expect(scaled[i]).toBeCloseTo(v, 10));
  });

  it("retains finite common-interval evidence with a missing nucleus and an inserted consonant", () => {
    const r = rows.filter((_, i) => i !== 3).map((r, t) => ({ ...r, best: t,
      p: t === 2 ? "ك" : r.p }));
    const got = score(model(.5), r);
    expect([...got].every(Number.isFinite)).toBe(true);
    expect(got[ref.length]).not.toBe(0);
    expect([...score(model(.5), r.map((r, i) => ({ ...r, duration: i === 1 ? NaN : r.duration })))])
      .toEqual(Array(ref.length+1).fill(0));
  });

  it("rewinds retained timing with search state and keeps rho-zero behavior identical", () => {
    const index: ReferenceIndex = {
      version: "test", inventoryId: "test", phonemes: ref, tokens: new Set(ref), gramSize: 2, grams: new Map(),
      words: [{ surah: 1, ayah: 1, word: 1, start: 0, end: 2 }, { surah: 1, ayah: 1, word: 2, start: 2, end: ref.length }],
      wordAt: offset => offset < 2 ? 0 : 1,
    };
    const start = { offset: 1, origin: 0, edits: 0 };
    const plain = createLocalSearch(index, start), control = createLocalSearch(index, start, model());
    for (let i = 0; i < 24; i++) {
      const r = rows[i % rows.length];
      plain.step(r.p); control.step(r.p, r.duration);
      expect(control.position()).toEqual(plain.position());
    }
    const grouped = createLocalSearch(index, start, model(.6));
    for (let i = 0; i < 16; i++) grouped.step(rows[i % rows.length].p, 1+i%3);
    const saved = grouped.save();
    for (const p of "كَتِدُ") grouped.step(p, 5);
    const first = grouped.position();
    grouped.restore(saved);
    for (const p of "كَتِدُ") grouped.step(p, 5);
    expect(grouped.position()).toEqual(first);
  });
});

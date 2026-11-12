import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import { fitRestartChannel } from "../src/sequential-restart.js";
import type { ReferenceIndex } from "../src/reference.js";
const ref = "بببببتتتتتثثثثثددددد";
const index: ReferenceIndex = { version: "toy", inventoryId: "toy", tokens: new Set(ref), phonemes: ref,
  words: [0, 5, 10, 15].map((start, i) => ({ surah: 1, ayah: 1, word: i + 1, start, end: start + 5 })),
  gramSize: 4, grams: new Map(), wordAt: offset => Math.min(3, Math.max(0, Math.floor(offset / 5))) };
const channel = fitRestartChannel([{ written: ref, heard: ref }], [...index.tokens], 1);
const seed = { offset: 0, origin: 0, edits: 0 };
describe("local persistent evidence integration", () => {
  it("preserves all original positions and costs in read-only audit mode", () => {
    const original = createLocalSearch(index, seed);
    const audit = createLocalSearch(index, seed, { calibration: { channel, threshold: 0 }, audit: true });
    for (const p of "بببتتثددددد") { original.step(p); audit.step(p); expect(audit.position()).toEqual(original.position()); }
  });
  it("continues an available path when the proposed restart lacks calibrated evidence", () => {
    const original = createLocalSearch(index, seed), comparisons: unknown[] = [];
    const guarded = createLocalSearch(index, seed, { calibration: { channel, threshold: Number.MAX_VALUE }, onComparison: c => comparisons.push(c) });
    for (const p of "بببببددددد") { original.step(p); guarded.step(p); }
    expect(comparisons.length).toBeGreaterThan(0);
    expect(original.position().origin).toBe(15);
    expect(guarded.position().origin).toBe(0);
  });
  it("restores evidence and local search together across provisional comparisons", () => {
    const s = createLocalSearch(index, seed, { calibration: { channel, threshold: Number.MAX_VALUE } });
    for (const p of "ببب") s.step(p);
    const saved = s.save();
    const tail = () => [..."بددددد"].map(p => { s.step(p); return s.position(); });
    const first = tail(); s.restore(saved); expect(tail()).toEqual(first);
  });
});

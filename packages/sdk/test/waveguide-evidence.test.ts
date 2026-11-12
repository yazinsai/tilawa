import { describe, expect, it } from "vitest";
import { waveguideSupportsRival, type WaveguideComparison } from "../src/waveguide-evidence.js";

const candidate = { word: { surah: 1, ayah: 1, word: 1 }, written: "بِسم", rival: "قِسم", fromFrame: 30, toFrame: 50, frameMs: 40 };
const comparison: WaveguideComparison = { candidate, scores: { physical: { written: -1, rival: -2 }, control: { written: -2, rival: -1 } } };
describe("waveguide comparison transport", () => {
  it("requires finite greater reconstruction likelihood for the complete written pronunciation", () => {
    expect(waveguideSupportsRival(candidate, 40, [comparison])).toBe(false);
    for (const physical of [{ written: -2, rival: -1 }, { written: -1, rival: -1 }, { written: NaN, rival: -2 }, { written: null, rival: -2 }]) {
      expect(waveguideSupportsRival(candidate, 40, [{ ...comparison, scores: { physical } }])).toBe(true);
    }
  });
  it("never lets the training-only control decide a veto", () => {
    expect(waveguideSupportsRival(candidate, 40, [{ ...comparison, scores: { physical: comparison.scores.physical } }])).toBe(false);
  });
  it("requires the exact clock, window, word and both complete pronunciations", () => {
    for (const c of [{ ...candidate, toFrame: 51 }, { ...candidate, written: "بِسمِ" }, { ...candidate, rival: "قَسم" }, { ...candidate, word: { ...candidate.word, word: 2 } }]) {
      expect(waveguideSupportsRival(c, 40, [comparison])).toBe(true);
    }
    expect(waveguideSupportsRival(candidate, 20, [comparison])).toBe(true);
  });
});

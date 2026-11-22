import { describe, expect, it } from "vitest";
import { gapCarriesAddition, interWordGaps, placeGap, recoverGapPhonemes } from "../../src/engine/gap-decode.js";
import type { GapPhoneme } from "../../src/engine/gap-decode.js";

const ph = (p: string, start: number, conf = 0.9): GapPhoneme => ({ p, start, end: start, conf });

describe("context-free gap phonemes", () => {
  it("drops the anchored neighbours and keeps the insertion between them", () => {
    const decoded = [ph("ب", 0), ph("س", 10), ph("َ", 11), ph("ل", 12), ph("ِ", 13), ph("ج", 22)];
    expect(recoverGapPhonemes(decoded, 4, 20, "اَب", "جَد").map((p) => p.p).join("")).toBe("سَلِ");
  });

  it("drops a held copy of the left word's last phone that leaks into the pause", () => {
    const decoded = [ph("ب", 10), ph("ب", 11), ph("س", 12), ph("َ", 13), ph("ل", 14)];
    expect(recoverGapPhonemes(decoded, 8, 20, "اَب", "جَد").map((p) => p.p).join("")).toBe("سَل");
  });

  it("keeps nothing when the cropped pause is only the neighbours", () => {
    const decoded = [ph("ا", 8), ph("َ", 9), ph("ب", 10), ph("ج", 14), ph("َ", 15)];
    expect(recoverGapPhonemes(decoded, 8, 16, "اَب", "جَد")).toEqual([]);
  });

  it("replaces phones inside the gap and leaves the anchored words", () => {
    const gap = { fromFrame: 4, toFrame: 20, left: "اَب", right: "جَد" };
    const obs = [ph("ا", 0), ph("ب", 2), ph("ن", 10), ph("ج", 22)];
    const kept = [ph("س", 9), ph("َ", 11), ph("ل", 14)];
    expect(placeGap(obs, gap, kept).map((p) => p.p).join("")).toBe("ابسَلج");
  });

  it("treats two consonants over three phones as an addition and a held vowel as not one", () => {
    expect(gapCarriesAddition([{ p: "س" }, { p: "َ" }, { p: "ل" }, { p: "ِ" }])).toBe(true);
    expect(gapCarriesAddition([{ p: "ب" }, { p: "ب" }])).toBe(false);
    expect(gapCarriesAddition([{ p: "َ" }, { p: "ر" }, { p: "ِ" }])).toBe(false);
  });
});

describe("inter-word gaps", () => {
  const words = ["اَب", "جَد", "هُو"];
  const phonemes = words.join("");
  const wordAt = (offset: number) => words.findIndex((_, i) => offset < words.slice(0, i + 1).join("").length);
  const wordText = (wi: number) => words[wi] ?? "";

  it("opens a gap on the inserted phones between two confident words and not on a linked boundary", () => {
    const obs = [
      ph("ا", 0), ph("َ", 1), ph("ب", 2),
      ph("س", 12), ph("َ", 13), ph("ل", 14), ph("ِ", 15),
      ph("ج", 24), ph("َ", 25), ph("د", 26),
      ph("ه", 27), ph("ُ", 28), ph("و", 29),
    ];
    const gaps = interWordGaps({ phonemes, from: 0, to: phonemes.length, wordAt, wordText, obs, minFrames: 8 });
    expect(gaps).toEqual([{ fromFrame: 3, toFrame: 24, left: "اَب", right: "جَد" }]);
  });

  it("does not anchor a gap on a word heard only below the confidence gate", () => {
    const obs = [ph("ا", 0, 0.4), ph("َ", 1, 0.4), ph("ب", 2, 0.4), ph("ج", 20), ph("َ", 21), ph("د", 22)];
    expect(interWordGaps({ phonemes, from: 0, to: phonemes.length, wordAt, wordText, obs, minFrames: 8 })).toEqual([]);
  });
});

import { describe, expect, it } from "vitest";
import { openEdgeUtteranceSplits } from "../src/utterance-parse.js";

describe("open utterance edges", () => {
  it("consumes optional neighbour suffixes and prefixes without losing valid splits", () => {
    expect(openEdgeUtteranceSplits("بتجحخ", ["دب", "دبت"], ["خز", "حخز"]))
      .toEqual([{ from: 0, to: 4 }, { from: 1, to: 5 }, { from: 0, to: 3 },
        { from: 1, to: 4 }, { from: 2, to: 5 }, { from: 1, to: 3 }, { from: 2, to: 4 }, { from: 2, to: 3 }]);
  });
  it("rejects arbitrary trimming and fragments from the wrong neighbour edge", () => {
    expect(openEdgeUtteranceSplits("بتجح", ["بد"], ["زح"])).toEqual([]);
    expect(openEdgeUtteranceSplits("بتجح", ["دك"], ["خز"])).toEqual([]);
  });
  it("uses existing soft costs but rejects hard boundary substitutions", () => {
    expect(openEdgeUtteranceSplits("كَبت", ["دقِ"], []))
      .toContainEqual({ from: 2, to: 4 });
    expect(openEdgeUtteranceSplits("فَبت", ["دقِ"], []))
      .not.toContainEqual({ from: 2, to: 4 });
  });
  it("keeps a nonempty middle and cannot spend the same token on both edges", () => {
    expect(openEdgeUtteranceSplits("ب", ["دب"], ["بد"])).toEqual([]);
    expect(openEdgeUtteranceSplits("بت", ["دب"], ["تد"]))
      .toEqual([{ from: 0, to: 1 }, { from: 1, to: 2 }]);
  });
  it("matches held realizations without splitting a held run", () => {
    const splits = openEdgeUtteranceSplits("ۦۦبتاا", ["دۦ"], ["ااز"]);
    expect(splits).toContainEqual({ from: 2, to: 4 });
    expect(splits.every(s => s.from !== 1 && s.to !== 5)).toBe(true);
    const nasal = openEdgeUtteranceSplits("نننبتممم", ["دنن"], ["ممز"]);
    expect(nasal).toContainEqual({ from: 3, to: 5 });
    expect(nasal.every(s => ![1, 2].includes(s.from) && ![6, 7].includes(s.to))).toBe(true);
  });
  it("leaves whole utterances and complete neighbour words to the existing path", () => {
    expect(openEdgeUtteranceSplits("بت", [], [])).toEqual([]);
    expect(openEdgeUtteranceSplits("بتج", ["بت"], [])).toEqual([]);
  });
});

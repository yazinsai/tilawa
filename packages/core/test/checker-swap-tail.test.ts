import { describe, expect, it } from "vitest";
import { lexicalWords } from "../src/checker-lexical.js";
import { swapTailUnion } from "../src/checker-swap-tail.js";
import { minGapFrames } from "../src/gap-decode.js";
import { loadReferenceIndex } from "../src/reference.js";

// A small vocabulary keeps the test about a split word, independent of any lab passage.
const forms = ["بَتَثَ", "دَشَسَ", "ظَبَطَ", "زَجَعَ", "حَخَفَ", "نَمَلَ", "دَرَجَكُم"];
const index = loadReferenceIndex({
  version: "swap-tail-test", text: "", profile: {}, inventory: { id: "test", tokens: [...new Set(forms.join(""))].join("") },
  ayahs: [
    { s: 1, a: 1, w: [forms[0]], p: forms[0], o: [0], iso: [forms[0]] },
    { s: 1, a: 2, w: forms, p: forms.join(""), o: forms.map((_, i) => forms.slice(0, i).join("").length), iso: forms },
  ],
});
const lex = lexicalWords(index);
const wrong = { cls: "wrong_word" as const, word: 2, last: 2, obsFrom: 0, obsTo: 6 };
const added = { cls: "added_word" as const, word: 3, last: 3, obsFrom: 6, obsTo: 9 };
const phones = [..."دَرَجَكُم"];
const context = {
  nextWord: (wi: number) => wi + 1,
  visits: [] as { obsFrom: number; obsTo: number }[],
  labelledPause: (_prev: number) => false,
  heard: (span: { obsFrom: number; obsTo: number }) => phones.slice(span.obsFrom, span.obsTo).join(""),
  spellsWord: lex.surplusWord,
};

describe("a wrong word's added tail", () => {
  it("keeps a short inter-phone blank inside one attested word", () => {
    const leftEnd = 40, rightStart = 45, frameMs = 30;
    expect(rightStart - leftEnd - 1).toBeLessThan(minGapFrames(frameMs));
    const union = swapTailUnion(wrong, added, {
      ...context,
      labelledPause: prev => prev === 5 && rightStart - leftEnd - 1 >= minGapFrames(frameMs),
    });
    expect(union).toEqual({ obsFrom: 0, obsTo: 9, observed: phones.join("") });
  });

  it("keeps a separate addition across a labelled pause", () => {
    expect(swapTailUnion(wrong, added, { ...context, labelledPause: prev => prev === 5 })).toBeNull();
  });

  it("checks every boundary when uncounted phones separate the spans", () => {
    expect(swapTailUnion({ ...wrong, obsTo: 4 }, added, {
      ...context, labelledPause: prev => prev === 4,
    })).toBeNull();
    expect(swapTailUnion({ ...wrong, obsTo: 4 }, added, context)?.observed).toBe(phones.join(""));
  });

  it("keeps an addition when a counting visit separates the spans, including a repeated word", () => {
    expect(swapTailUnion({ ...wrong, obsTo: 4 }, added, {
      ...context, visits: [{ obsFrom: 4, obsTo: 6 }],
    })).toBeNull();
    expect(swapTailUnion({ ...wrong, obsTo: 4 }, added, {
      ...context, visits: [{ obsFrom: 3, obsTo: 5 }],
    })).toBeNull();
  });

  it("ignores the visits ending at the head or beginning at the tail", () => {
    expect(swapTailUnion(wrong, added, {
      ...context, visits: [{ obsFrom: 0, obsTo: 6 }, { obsFrom: 6, obsTo: 12 }],
    })).not.toBeNull();
  });

  it("requires one whole attested non-neighbour word under the folded soft spell test", () => {
    expect(swapTailUnion(wrong, added, { ...context, heard: () => "دُرِجَكُم" })).not.toBeNull();
    for (const heard of ["دَرَجَكُ", "دَرَجَكُمن", "دَرَجَكُلب", forms[3], forms[3].slice(0, 4)]) {
      expect(swapTailUnion(wrong, added, { ...context, heard: () => heard })).toBeNull();
    }
  });

  it("requires a preceding wrong-word accusation and leaves spell-rule emissions alone", () => {
    expect(swapTailUnion(undefined, added, context)).toBeNull();
    expect(swapTailUnion({ ...wrong, cls: "omitted_word" }, added, context)).toBeNull();
    expect(swapTailUnion({ ...wrong, last: 1 }, added, context)).toBeNull();
    expect(swapTailUnion(wrong, { ...added, lexicon: true }, context)).toBeNull();
    expect(swapTailUnion(wrong, { ...added, obsFrom: 5 }, context)).toBeNull();
  });
});

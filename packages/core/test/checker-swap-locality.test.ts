import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { swapLocality } from "../src/checker-swap-locality.js";
import { lexicalWords } from "../src/checker-lexical.js";
import { channelFromCounts, emptyCounts } from "../src/noisy-channel.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";

const phrases = [
  ["كَتَبَ", "شَفَعَ", "رَفَعَ", "دَخَلَ", "بَعُدَ", "نَزَلَ", "سَمِعَ", "بَرَقَ"],
  ["كَتَبَ", "بَقِيَ", "رَفَعَ", "دَخَلَ", "مَسَحَ", "غَسَلَ", "ذَهَبَ"],
];
const tokens = [...new Set(phrases.flat().join(""))].join("");
const index = loadReferenceIndex({
  version: "test", text: "test", profile: {}, inventory: { id: "test", tokens },
  ayahs: phrases.flatMap((words, s) => [words.slice(0, 1), words.slice(1)].map((part, a) => ({
    s: s + 1, a: a + 1, w: part, iso: part, p: part.join(""),
    o: part.map((_, i) => part.slice(0, i).join("").length),
  }))),
});
const counts = emptyCounts([...tokens]);
counts.phones.forEach((_, i) => { counts.sub[i][i] = 1000; counts.nWritten += 1000; });
const channel = channelFromCounts(counts);
const locality = swapLocality(index);

describe("attested swap locality", () => {
  it("keeps a whole word elsewhere in the current surah", () => {
    expect(locality.assess(1, "سَمِعَ", channel)).toMatchObject({ sameSurah: true, keep: true });
  });

  it("keeps a cross-surah parallel completion for this exact slot", () => {
    expect(locality.assess(1, "بَقِيَ", channel)).toMatchObject({ sameSurah: false, parallel: true, keep: true });
  });

  it("vetoes a word attested only outside both locality links", () => {
    expect(locality.assess(1, "غَسَلَ", channel)).toMatchObject({ rival: "غَسَلَ", sameSurah: false, parallel: false, keep: false });
  });

  it("does not borrow another slot's parallel link", () => {
    expect(locality.assess(5, "بَقِيَ", channel)).toMatchObject({ rival: "بَقِيَ", parallel: false, keep: false });
  });

  it("ranks a blurred non-word before checking its locality", () => {
    expect(lexicalWords(index).surplusWord(1, "غَسَل")).toBeNull();
    expect(locality.assess(1, "غَسَل", channel)).toMatchObject({ rival: "غَسَلَ", keep: false });
  });

  it("uses the lexical candidate's named rival rather than ranking a different word", () => {
    expect(locality.assess(1, "سَمِعَ", channel, "غَسَلَ")).toMatchObject({ rival: "غَسَلَ", keep: false });
  });

  it("prefers an exact local pronunciation over a soft spelling with a different case ending", () => {
    const cases = loadReferenceIndex({
      version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: tokens + "ن" },
      ayahs: [
        { s: 1, a: 1, w: ["سَمِعًا"], p: "سَمِعَن", iso: ["سَمِعَن"], o: [0] },
        { s: 2, a: 1, w: ["كَتَبَ", "رَفَعَ", "دَخَلَ", "بَعُدَ", "نَزَلَ", "رَفَعَ", "دَخَلَ", "بَعُدَ", "سَمِعٌ"],
          p: "كَتَبَرَفَعَدَخَلَبَعُدَنَزَلَرَفَعَدَخَلَبَعُدَسَمِعُن",
          iso: ["كَتَبَ", "رَفَعَ", "دَخَلَ", "بَعُدَ", "نَزَلَ", "رَفَعَ", "دَخَلَ", "بَعُدَ", "سَمِعُن"],
          o: [0, 6, 12, 18, 24, 30, 36, 42, 48] },
      ],
    });
    expect(lexicalWords(cases).surplusWord(5, "سَمِعُن")).toBe("سَمِعَن");
    expect(swapLocality(cases).assess(5, "سَمِعُن", channel)).toMatchObject({
      rival: "سَمِعُن", sameSurah: true, keep: true,
    });
  });

  it("abstains without mushaf text or a channel for a non-word", () => {
    expect(swapLocality({ ...index, wordText: undefined }).assess(1, "غَسَلَ", channel)).toBeNull();
    expect(locality.assess(1, "غَسَل", channelFromCounts(emptyCounts([])))).toBeNull();
  });
});

function check(said?: string, wi = 1) {
  const from = index.words[0], to = index.words[7], target = index.words[wi];
  const fixture = syntheticFixture(index, {
    id: "swap-locality", from, to, chunkSize: 4,
    edits: said ? [
      { kind: "del", at: target.start, count: target.end - target.start },
      { kind: "ins", at: target.start, text: said },
    ] : [],
  });
  const checker = createChecker(index, { passage: { from, to }, policy: { confirm: { after: "next-word" } } });
  for (const chunk of fixture.evidence) checker.push(withPosteriors(index, chunk), []);
  checker.end();
  return checker.mistakes().filter(e => e.errorClass === "wrong_word");
}

describe("wrong-word confirmation with locality", () => {
  it("vetoes a remote string-path rival at an ayah onset", () => { expect(check("غَسَلَ")).toEqual([]); });
  it("keeps a remote substitution inside the ayah", () => { expect(check("غَسَلَ", 3)).toHaveLength(1); });
  it("preserves a local swap", () => { expect(check("سَمِعَ")).toHaveLength(1); });
  it("preserves a parallel swap", () => { expect(check("بَقِيَ")).toHaveLength(1); });
  it("stays quiet on clean text", () => { expect(check()).toEqual([]); });
});

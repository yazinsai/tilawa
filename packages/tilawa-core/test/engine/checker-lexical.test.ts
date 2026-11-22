import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../../src/engine/checker.js";
import { hardConsonantEdits, lexicalWords } from "../../src/engine/checker-lexical.js";
import { loadReferenceIndex } from "../../src/engine/reference.js";
import { syntheticFixture, withPosteriors } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";
import type { CheckerEvent, WordRef } from "../../src/engine/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find(x => key(x) === key(r))!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

function check(from: WordRef, to: WordRef, edits: (at: (r: WordRef) => number) => Edit[], acoustics: "decoded" | "missing" | "neutral" = "decoded") {
  const at = (r: WordRef) => find(r).start - find(from).start;
  const fixture = syntheticFixture(index, { id: "lexical-word", from, to, edits: edits(at), chunkSize: 4 });
  const checker = createChecker(index, { passage: { from, to }, policy: { confirm: { after: "next-word" } } });
  const events: CheckerEvent[] = [];
  const neutral: [string, number][] = ["", ...index.tokens].map(p => [p, 1 / (index.tokens.size + 1)]);
  for (const evidence of fixture.evidence) {
    const chunk = acoustics === "missing" ? evidence : acoustics === "neutral"
      ? withPosteriors(index, evidence, () => neutral, neutral.length)
      : withPosteriors(index, evidence);
    events.push(...checker.push(chunk, []));
  }
  events.push(...checker.end());
  return { events, mistakes: checker.mistakes().map(e => `${e.errorClass} ${key(e.location)}`) };
}

const replace = (at: (r: WordRef) => number, target: WordRef, said: string): Edit[] => [
  { kind: "del", at: at(target), count: text(target).length },
  { kind: "ins", at: at(target), text: said },
];

describe("consonants independent of vowels", () => {
  it("counts the two real edits in either direction for نوح and صالح", () => {
    const noah = text(w(26, 106, 5)), salih = text(w(26, 142, 5));
    expect(hardConsonantEdits(noah, salih)).toBe(2);
    expect(hardConsonantEdits(salih, noah)).toBe(2);
    expect(hardConsonantEdits("نُح", "نَااح")).toBe(0);
  });

  it.each([
    { target: w(20, 53, 9), said: "مُبُلَا" },
    { target: w(28, 58, 14), said: "خَلِۦۦلَا" },
  ])("finds only one hard edit in a supported form for $target", ({ target, said }) => {
    const wi = index.words.indexOf(find(target));
    expect(Math.min(...index.wordForms![wi].map(form => hardConsonantEdits(form, said)))).toBe(1);
    expect(lexicalWords(index).alternative(wi, said)).toBe(false);
  });

  it("discounts two separate confusable substitutions", () => {
    expect(hardConsonantEdits("حَق", "هَك")).toBe(0);
  });

  it.each(["ن", "م"])("ignores held %s length but retains its genuine doubling", sound => {
    expect(hardConsonantEdits(sound.repeat(3), sound.repeat(5))).toBe(0);
    expect(hardConsonantEdits(sound.repeat(3), sound)).toBe(1);
    expect(hardConsonantEdits(sound, sound.repeat(3))).toBe(1);
  });

  it("retains ordinary gemination", () => {
    expect(hardConsonantEdits("عِزز", "عِز")).toBe(1);
    expect(hardConsonantEdits("عِز", "عِزز")).toBe(1);
  });

  it("does not turn vowels between repeated consonants into a held run", () => {
    expect(hardConsonantEdits("نَنَن", "نَن")).toBe(1);
    expect(hardConsonantEdits("بَب", "ب")).toBe(1);
  });

  it("requires one complete supported word, not a fragment or joined words", () => {
    const target = index.words.indexOf(find(w(26, 106, 5)));
    const salih = text(w(26, 142, 5));
    expect(lexicalWords(index).alternative(target, salih, { before: text(w(26, 106, 4)), after: text(w(26, 106, 6)) })).toBe(true);
    expect(lexicalWords(index).alternative(target, salih.slice(0, 4))).toBe(false);
    expect(lexicalWords(index).alternative(target, salih + salih)).toBe(false);
  });
});

describe("complete lexical substitutions", () => {
  it("catches صالح for نوح when vowel alignment hides a second consonant edit", () => {
    const target = w(26, 106, 5);
    expect(check(w(26, 106, 1), w(26, 107, 3), at => replace(at, target, text(w(26, 142, 5)))).mistakes)
      .toEqual([`wrong_word ${key(target)}`]);
  });

  it("catches the reverse lexical substitution", () => {
    const target = w(26, 142, 5);
    expect(check(w(26, 142, 1), w(26, 143, 3), at => replace(at, target, text(w(26, 106, 5)))).mistakes)
      .toEqual([`wrong_word ${key(target)}`]);
  });

  it("keeps the لوط to صالح recovery in its matching neighbor context", () => {
    const target = w(26, 161, 5);
    expect(check(w(26, 161, 1), w(26, 162, 3), at => replace(at, target, text(w(26, 142, 5)))).mistakes)
      .toEqual([`wrong_word ${key(target)}`]);
  });

  it.each(["missing", "neutral"] as const)("abstains on the hidden substitution with %s acoustic support", acoustics => {
    const target = w(26, 106, 5);
    const result = check(w(26, 106, 1), w(26, 107, 3), at => replace(at, target, text(w(26, 142, 5))), acoustics);
    expect(result.events.filter(e => key(e.location) === key(target))).toEqual([]);
    expect(result.mistakes).toEqual([]);
  });

  it("preserves an unrelated wrong-word detection", () => {
    const target = w(2, 255, 23);
    expect(check(w(2, 255, 18), w(2, 255, 30), at => replace(at, target, text(w(108, 1, 3)))).mistakes)
      .toEqual([`wrong_word ${key(target)}`]);
  });
});

describe("lexical false-warning protections", () => {
  it.each([
    { from: w(20, 53, 5), to: w(20, 54, 4), target: w(20, 53, 9), said: "مُبُلَا" },
    { from: w(28, 58, 10), to: w(28, 59, 3), target: w(28, 58, 14), said: "خَلِۦۦلَا" },
  ])("does not revive the historical warning at $target", ({ from, to, target, said }) => {
    // Each changes only one consonant, despite the different vowel/ending alignment.
    expect(check(from, to, at => replace(at, target, said)).mistakes).toEqual([]);
  });

  it("does not count discounted confusable consonants as two hard edits", () => {
    const target = w(2, 255, 16);
    const said = text(target).replaceAll("س", "ص").replaceAll("ت", "ط");
    expect(check(w(2, 255, 12), w(2, 255, 20), at => replace(at, target, said)).mistakes).toEqual([]);
  });

  it("leaves a shortened genuine doubled consonant alone", () => {
    const target = w(19, 81, 8);
    expect(check(w(19, 80, 1), w(19, 84, 3), at => [
      { kind: "del", at: at(target) + text(target).indexOf("زز"), count: 1 },
    ]).mistakes).toEqual([]);
  });

  it("does not turn held ghunna at a boundary into another word", () => {
    const target = w(2, 7, 12);
    expect(check(w(2, 6, 1), w(2, 11, 8), at => [
      { kind: "ins", at: at(target) + text(target).length - 1, text: "مم" },
    ]).mistakes).toEqual([]);
  });
});

describe("a spelled word outside the neighbourhood", () => {
  const lex = lexicalWords(index);
  const wi = index.words.indexOf(find(w(2, 255, 23)));

  it("accepts a distant mushaf word and rejects the written word, a ±3 neighbour and a non-word", () => {
    expect(lex.surplusWord(wi, text(w(2, 171, 9)))).toBeTruthy();
    expect(lex.surplusWord(wi, text(w(2, 255, 23)))).toBeNull();
    expect(lex.surplusWord(wi, text(w(2, 255, 25)))).toBeNull();
    expect(lex.surplusWord(wi, "يَغقَعُ")).toBeNull();
  });
});

describe("lexical candidate boundaries", () => {
  it("abstains when the substituted word has no following neighbor", () => {
    const target = w(26, 106, 5);
    const result = check(w(26, 106, 1), target, at => replace(at, target, text(w(26, 142, 5))));
    expect(result.events.filter(e => key(e.location) === key(target))).toEqual([]);
  });

  it("abstains when the substituted word has no preceding neighbor", () => {
    const target = w(26, 106, 5);
    const result = check(target, w(26, 107, 3), at => replace(at, target, text(w(26, 142, 5))));
    expect(result.events.filter(e => key(e.location) === key(target))).toEqual([]);
  });

  it("leaves clean words alone", () => {
    expect(check(w(26, 106, 1), w(26, 107, 3), () => []).mistakes).toEqual([]);
  });

  it("preserves a correct repeat around the target", () => {
    expect(check(w(26, 106, 1), w(26, 107, 3), at => [
      { kind: "repeat", from: at(w(26, 106, 4)), to: at(w(26, 107, 1)) },
    ]).mistakes).toEqual([]);
  });

  it("preserves a wrong attempt followed by the correct word", () => {
    const target = w(2, 255, 23);
    const attempt = text(target).replace("ش", "س").replace("ف", "م");
    const result = check(w(2, 255, 18), w(2, 255, 30), at => [
      { kind: "ins", at: at(target), text: attempt },
    ]);
    expect(result.mistakes).toEqual([]);
    expect(result.events.some(e => e.state === "resolved" && key(e.location) === key(target))).toBe(true);
  });

  it("keeps a neighboring addition located on that addition", () => {
    const before = w(2, 255, 23);
    expect(check(w(2, 255, 18), w(2, 255, 30), at => [
      { kind: "ins", at: at(before), text: text(w(108, 1, 3)) },
    ]).mistakes).toEqual([`added_word ${key(before)}`]);
  });
});

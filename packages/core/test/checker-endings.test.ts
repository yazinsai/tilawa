import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { lexicalWords } from "../src/checker-lexical.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { CheckerEvent, WordRef } from "../src/events.js";
import type { ReferenceAsset } from "../src/reference.js";

const asset: ReferenceAsset = {
  version: "ending-test",
  text: "synthetic",
  profile: {},
  inventory: { id: "ending-test", tokens: "بَسَلَهِفتمَاُۥكذخ" },
  ayahs: [
    { s: 1, a: 1, w: ["before", "short", "after"], p: "بَسَلَهِفِ", o: [0, 2, 8],
      iso: ["ب", "سَلَه", "ف"] },
    { s: 1, a: 2, w: ["before", "long", "after"], p: "تَسَلَهِمَاكَ", o: [0, 2, 11],
      iso: ["ت", "سَلَهِمَا", "ك"] },
    { s: 1, a: 3, w: ["bare", "dual"], p: "ذَهَبَذَهَبَا", o: [0, 5],
      iso: ["ذَهَب", "ذَهَبَا"] },
  ],
};
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find(x => key(x) === key(r))!;
const atWord = (r: WordRef) => index.words.indexOf(find(r));
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const iso = (r: WordRef) => index.isolatedWordForms![atWord(r)];

function check(from: WordRef, to: WordRef, edits: (at: (r: WordRef) => number) => Edit[], acoustics: "decoded" | "missing" | "neutral" = "decoded") {
  const at = (r: WordRef) => find(r).start - find(from).start;
  const fixture = syntheticFixture(index, { id: "word-ending", from, to, edits: edits(at), chunkSize: 4 });
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

describe("attested endings", () => {
  const words = lexicalWords(index);
  const short = w(1, 1, 2), long = w(1, 2, 2);

  it("reports which side supplies the consonant-bearing attested ending", () => {
    expect(words.ending(atWord(short), text(long))).toEqual({ observedConsonant: true });
    expect(words.ending(atWord(long), text(short))).toEqual({ observedConsonant: false });
  });

  it("rejects its own forms, held vowels, and consonant-free endings", () => {
    expect(words.ending(atWord(short), text(short))).toBeNull();
    expect(words.ending(atWord(short), iso(short))).toBeNull();
    expect(words.ending(atWord(short), text(short) + "اا")).toBeNull();
    expect(words.ending(atWord(w(1, 3, 1)), text(w(1, 3, 2)))).toBeNull();
    expect(words.ending(atWord(w(1, 3, 2)), text(w(1, 3, 1)))).toBeNull();
  });
});

describe("suffix swaps", () => {
  it("catches an attested consonant-bearing suffix", () => {
    const target = w(1, 1, 2);
    expect(check(w(1, 1, 1), w(1, 1, 3), at => replace(at, target, text(w(1, 2, 2)))).mistakes)
      .toEqual([`wrong_word ${key(target)}`]);
  });

  it("abstains when the changed consonant is missing and has no decoded acoustic anchor", () => {
    const target = w(1, 2, 2);
    expect(check(w(1, 2, 1), w(1, 2, 3), at => replace(at, target, text(w(1, 1, 2)))).mistakes)
      .toEqual([]);
  });

  it("does not turn an unattested extra consonant into a suffix", () => {
    const target = w(1, 1, 2);
    expect(check(w(1, 1, 1), w(1, 1, 3), at => [
      { kind: "ins", at: at(target) + text(target).length, text: "م" },
    ]).mistakes).toEqual([]);
  });

  it.each(["missing", "neutral"] as const)("abstains on the ending swap with %s acoustic support", acoustics => {
    const target = w(1, 1, 2);
    const result = check(w(1, 1, 1), w(1, 1, 3), at => replace(at, target, text(w(1, 2, 2))), acoustics);
    expect(result.events.filter(e => key(e.location) === key(target))).toEqual([]);
    expect(result.mistakes).toEqual([]);
  });
});

describe("endings that are not a different word", () => {
  it("stays quiet when the word is paused", () => {
    const target = w(1, 1, 2);
    expect(check(w(1, 1, 1), w(1, 1, 3), at => replace(at, target, iso(target))).mistakes).toEqual([]);
  });

  it("stays quiet when a vowel is held on past the word", () => {
    const target = w(1, 1, 2);
    expect(check(w(1, 1, 1), w(1, 1, 3), at => [
      { kind: "ins", at: at(target) + text(target).length, text: "اا" },
    ]).mistakes).toEqual([]);
  });

  it("does not treat a dual alif as a consonant ending", () => {
    const target = w(1, 3, 1);
    expect(check(target, w(1, 3, 2), at => replace(at, target, text(w(1, 3, 2)))).mistakes).toEqual([]);
  });

  it("leaves the passage alone", () => {
    expect(check(w(1, 1, 1), w(1, 1, 3), () => []).mistakes).toEqual([]);
  });
});

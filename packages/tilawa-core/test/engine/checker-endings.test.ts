import { describe, expect, it } from "vitest";
import { createChecker } from "../../src/engine/checker.js";
import { lexicalWords } from "../../src/engine/checker-lexical.js";
import { loadReferenceIndex } from "../../src/engine/reference.js";
import { syntheticFixture, withPosteriors } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";
import type { CheckerEvent, WordRef } from "../../src/engine/events.js";
import type { ReferenceAsset } from "../../src/engine/reference.js";

const asset: ReferenceAsset = {
  version: "ending-test",
  text: "synthetic",
  profile: {},
  inventory: { id: "ending-test", tokens: "بَسَلَهِفتمَاُۥكذخ" },
  ayahs: [
    { s: 1, a: 1, w: ["before", "short", "after"], p: "بَسَلَهِفِ", o: [0, 2, 8],
      iso: ["ب", "سَلَه", "ف"] },
    // The last word is only there so the alignment cannot stop inside the ending: كَ's vowel matches the suffix.
    { s: 1, a: 2, w: ["before", "long", "after", "tail"], p: "تَسَلَهِمَاكَذَ", o: [0, 2, 11, 13],
      iso: ["ت", "سَلَهِمَا", "ك", "ذ"] },
    { s: 1, a: 3, w: ["bare", "dual"], p: "ذَهَبَذَهَبَا", o: [0, 5],
      iso: ["ذَهَب", "ذَهَبَا"] },
    // The word after the long form starts with the ending's consonant, so a single heard م can be lent to either.
    { s: 1, a: 4, w: ["long", "follows", "tail"], p: "سَلَهِمَامَكذَ", o: [0, 9, 12],
      iso: ["سَلَهِمَا", "مَك", "ذ"] },
  ],
};
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find(x => key(x) === key(r))!;
const atWord = (r: WordRef) => index.words.indexOf(find(r));
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const iso = (r: WordRef) => index.isolatedWordForms![atWord(r)];

function check(from: WordRef, to: WordRef, edits: (at: (r: WordRef) => number) => Edit[], acoustics: "decoded" | "missing" | "neutral" = "decoded", chunkSize = 4) {
  const at = (r: WordRef) => find(r).start - find(from).start;
  const fixture = syntheticFixture(index, { id: "word-ending", from, to, edits: edits(at), chunkSize });
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

  it("lists only shorter forms whose missing tail holds a consonant", () => {
    expect(words.droppedSuffixes(atWord(long)).sort()).toEqual([iso(short), text(short)].sort());
    expect(words.droppedSuffixes(atWord(short))).toEqual([]);
    expect(words.droppedSuffixes(atWord(w(1, 3, 1)))).toEqual([]);
    expect(words.droppedSuffixes(atWord(w(1, 3, 2)))).toEqual([]);
  });
});

describe("suffix swaps", () => {
  it("catches an attested consonant-bearing suffix", () => {
    const target = w(1, 1, 2);
    expect(check(w(1, 1, 1), w(1, 1, 3), at => replace(at, target, text(w(1, 2, 2)))).mistakes)
      .toEqual([`wrong_word ${key(target)}`]);
  });

  it("catches a dropped consonant ending when nothing was spoken in its place", () => {
    const target = w(1, 2, 2);
    expect(check(w(1, 2, 1), w(1, 2, 4), at => replace(at, target, text(w(1, 1, 2)))).mistakes)
      .toEqual([`wrong_word ${key(target)}`]);
  });

  it("catches the drop when the next word's onset is the ending consonant, so one م cannot pay twice", () => {
    const target = w(1, 4, 1), stem = text(w(1, 1, 2));
    expect(check(target, w(1, 4, 3), at => [
      { kind: "del", at: at(target) + stem.length, count: text(target).length - stem.length },
    ]).mistakes).toEqual([`wrong_word ${key(target)}`]);
  });

  it("abstains when the heard consonant still accounts for the ending", () => {
    const target = w(1, 2, 2), stem = text(w(1, 1, 2));
    expect(check(w(1, 2, 1), w(1, 2, 4), at => [
      { kind: "del", at: at(target) + stem.length + 1, count: text(target).length - stem.length - 1 },
    ]).mistakes).toEqual([]);
  });

  it("abstains when the hole could still have held the consonant and its long vowel", () => {
    const target = w(1, 2, 2), stem = text(w(1, 1, 2));
    // 12 silent frames plus the boundary frame is 260 ms at this fixture's 20 ms stride, past the 240 ms hold.
    expect(check(w(1, 2, 1), w(1, 2, 4), at => [
      ...replace(at, target, text(w(1, 1, 2))),
      { kind: "gap", at: at(target) + stem.length, frames: 12 },
    ], "decoded", 10_000).mistakes).toEqual([]);
  });

  it.each(["missing", "neutral"] as const)("abstains on a dropped ending with %s acoustic support", acoustics => {
    const target = w(1, 2, 2);
    const result = check(w(1, 2, 1), w(1, 2, 4), at => replace(at, target, text(w(1, 1, 2))), acoustics);
    expect(result.events.filter(e => key(e.location) === key(target))).toEqual([]);
    expect(result.mistakes).toEqual([]);
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
    expect(check(w(1, 2, 1), w(1, 2, 4), () => []).mistakes).toEqual([]);
    expect(check(w(1, 4, 1), w(1, 4, 3), () => []).mistakes).toEqual([]);
  });
});

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { mannerChanges, mannerSkeleton, mannerWords } from "../src/manner.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { DecodedPhoneme } from "../src/evidence.js";
import type { WordRef } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find((x) => key(x) === key(r))!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const at = (from: WordRef, target: WordRef) => find(target).start - find(from).start;

const from = w(9, 47, 14);
const to = w(9, 48, 3);
const target = w(9, 47, 17);
const written = text(target);
const muttaqin = text(w(9, 44, 14));
const neutral: [string, number][] = ["", ...index.tokens].map((p) => [p, 1 / (index.tokens.size + 1)]);

function soften(phonemes: DecodedPhoneme[], needle: string): DecodedPhoneme[] {
  const s = phonemes.map((p) => p.p).join("");
  const i = s.indexOf(needle);
  if (i < 0) return phonemes;
  return phonemes.map((p, j) => (j >= i && j < i + [...needle].length ? { ...p, conf: 0.5 } : p));
}

function run(said: string, o: { conf?: boolean; neutral?: boolean } = {}) {
  const edits: Edit[] = [
    { kind: "del", at: at(from, target), count: written.length },
    { kind: "ins", at: at(from, target), text: said },
  ];
  const fx = syntheticFixture(index, { id: "manner-skeleton", from, to, edits });
  const checker = createChecker(index, { passage: { from, to }, policy: { confirm: { after: "next-word" } } });
  for (const evidence of fx.evidence) {
    const phonemes = o.conf ? soften(evidence.phonemes, said) : evidence.phonemes;
    const chunk = o.neutral
      ? withPosteriors(index, { ...evidence, phonemes }, () => neutral, neutral.length)
      : withPosteriors(index, { ...evidence, phonemes });
    checker.push({ ...chunk, phonemes }, []);
  }
  checker.end();
  return checker.mistakes().map((e) => `${e.errorClass} ${key(e.location)}`);
}

describe("manner skeleton of a word", () => {
  const words = mannerWords(index);
  const wi = index.words.indexOf(find(target));

  it("counts manner changes and not same-manner pairs", () => {
    expect(mannerSkeleton(muttaqin)).toBe(mannerSkeleton("بِلمُتتَقِۦۦن"));
    expect(mannerChanges(written, muttaqin)).toBeGreaterThanOrEqual(2);
    expect(mannerChanges(written, written.replaceAll("ظ", "ط"))).toBe(0);
    expect(mannerChanges("نَمَفَك", "مَنَكَف")).toBe(0);
    expect(words.outside(wi, muttaqin)).toBe(true);
    expect(words.outside(wi, written)).toBe(false);
    expect(words.outside(wi, text(w(9, 47, 16)))).toBe(false);
  });

  it("confirms a low-confidence swap whose manner skeleton is another mushaf word", () => {
    expect(run(muttaqin, { conf: true })).toEqual([`wrong_word ${key(target)}`]);
  });

  it("keeps the 0.6 cut when the consonants stay in the same manner", () => {
    expect(run(written.replaceAll("ظ", "ط"), { conf: true })).toEqual([]);
  });

  it("does not call a neighbour's skeleton a different word", () => {
    // عليم is shorter than بالظالمين, so the ordinary path can call the word omitted. It must not become a wrong word.
    expect(run(text(w(9, 47, 16)), { conf: true })).not.toContain(`wrong_word ${key(target)}`);
  });

  it("keeps the 0.6 cut for a single manner change", () => {
    expect(run(written.slice(0, -1) + "ل", { conf: true })).toEqual([]);
  });

  it("does not confirm a manner swap the posteriors do not support", () => {
    expect(run(muttaqin, { conf: true, neutral: true })).toEqual([]);
  });

  it("stays quiet on the written word", () => {
    const fx = syntheticFixture(index, { id: "manner-clean", from, to, edits: [] });
    const checker = createChecker(index, { passage: { from, to }, policy: { confirm: { after: "next-word" } } });
    for (const evidence of fx.evidence) checker.push(withPosteriors(index, evidence), []);
    checker.end();
    expect(checker.mistakes()).toEqual([]);
  });
});

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { maddSites } from "../src/madd-drops.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { CheckerEvent, WordRef } from "../src/events.js";
import type { ReferenceAsset } from "../src/reference.js";

const quran = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const qw = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const qkey = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const qat = (r: WordRef) => quran.words.findIndex((x) => qkey(x) === qkey(r));
const qphone = (r: WordRef) => quran.phonemes.slice(quran.words[qat(r)]!.start, quran.words[qat(r)]!.end);

describe("madd munfasil vowel drops", () => {
  const sites = maddSites(quran);

  it("keeps the twin that drops only the madd, and not a second spelling of the same vowel", () => {
    const go = qat(qw(20, 43, 1));
    expect(sites.munfasil.get(go)).toBe(go + 1);
    expect(sites.drops.get(go)).toEqual({ next: go + 1, alt: qphone(qw(20, 24, 1)) });
    expect(sites.drops.has(qat(qw(24, 9, 5)))).toBe(true);
    // بها and بهۦ share a letter skeleton; the parallel still has the madd.
    expect(sites.drops.has(qat(qw(7, 106, 7)))).toBe(false);
    expect(sites.munfasil.has(qat(qw(20, 24, 1)))).toBe(false);
    const li = qat(qw(20, 26, 2));
    expect(sites.munfasil.has(li)).toBe(true);
    expect(sites.drops.has(li)).toBe(false);
  });
});

const asset: ReferenceAsset = {
  version: "madd-test",
  text: "synthetic",
  profile: {},
  inventory: { id: "madd-test", tokens: "ءبتكلمنهَاِغظ" },
  ayahs: [
    { s: 1, a: 1, w: ["غظ"], p: "غَظَ", o: [0], iso: ["غَظَ"] },
    // با is madd munfasil before أت. Dropping its alef is not a mushaf word.
    { s: 1, a: 2, w: ["با", "أت", "كل"], p: "بَااااءَتَكَلَ", o: [0, 6, 10], iso: ["بَاااا", "ءَتَ", "كَلَ"] },
    // كتا إلى من نه. Dropping the alef is كت, and that phrase is ayah 4.
    { s: 1, a: 3, w: ["كتا", "إل", "من", "نه"], p: "كَتَااااءِلَمَنَنَهَ", o: [0, 8, 12, 16],
      iso: ["كَتَاااا", "ءِلَ", "مَنَ", "نَهَ"] },
    { s: 1, a: 4, w: ["كت", "إل", "من", "نه"], p: "كَتَءِلَمَنَنَهَ", o: [0, 4, 8, 12],
      iso: ["كَتَ", "ءِلَ", "مَنَ", "نَهَ"] },
    // لا before أم is munfasil, and dropping the alef is not a word. The short span must stay quiet.
    { s: 1, a: 5, w: ["لا", "أم", "نك", "هب"], p: "لَااااءَمَنَكَهَبَ", o: [0, 6, 10, 14],
      iso: ["لَاااا", "ءَمَ", "نَكَ", "هَبَ"] },
  ],
};
const index = loadReferenceIndex(asset);
const w = (ayah: number, word: number): WordRef => ({ surah: 1, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find((x) => key(x) === key(r))!;

function check(from: WordRef, to: WordRef, edits: Edit[]) {
  const fixture = syntheticFixture(index, { id: "madd", from, to, edits });
  const checker = createChecker(index, { passage: { from, to }, policy: { confirm: { after: "next-word" } } });
  const events: CheckerEvent[] = [];
  for (const evidence of fixture.evidence) events.push(...checker.push(withPosteriors(index, evidence), []));
  events.push(...checker.end());
  return checker.mistakes().map((e) => `${e.errorClass} ${key(e.location)}`);
}

const at = (from: WordRef, target: WordRef) => find(target).start - find(from).start;

describe("connected madd length", () => {
  const from = w(2, 1);
  const to = w(3, 4);
  const target = w(3, 1);
  const dropAlef = (start: WordRef): Edit => ({ kind: "del", at: at(start, target) + 4, count: 4 });

  it("flags a vowel drop shorter than this reciter's own madd munfasil", () => {
    expect(check(from, to, [dropAlef(from)])).toEqual([`wrong_word ${key(target)}`]);
  });

  it("stays quiet when the madd is held as long as the earlier one", () => {
    expect(check(from, to, [])).toEqual([]);
  });

  it("stays quiet with no earlier madd to compare against", () => {
    expect(check(target, to, [{ kind: "del", at: 4, count: 4 }])).toEqual([]);
  });

  it("stays quiet when a labelled pause sits between the word and the hamza", () => {
    // Eight extra madd letters make the baseline longer than the dropped vowel plus a 16-frame pause
    // (past the 300 ms silence). The pause is what keeps the short span from winning.
    expect(check(from, to, [
      { kind: "ins", at: 2, text: "اااااااا" },
      dropAlef(from),
      { kind: "gap", at: at(from, target) + 8, frames: 16 },
    ])).toEqual([]);
  });

  it("does not flag a short vowel that is not another mushaf phrase", () => {
    const far = w(5, 4);
    const la = w(5, 1);
    expect(check(from, far, [{ kind: "del", at: at(from, la) + 2, count: 4 }])).toEqual([]);
  });
});

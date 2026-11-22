import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../../src/engine/checker.js";
import { loadReferenceIndex } from "../../src/engine/reference.js";
import { syntheticFixture } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";
import type { WordRef } from "../../src/engine/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find(x => key(x) === key(r))!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

function check(from: WordRef, to: WordRef, edits: (at: (r: WordRef) => number) => Edit[]) {
  const at = (r: WordRef) => find(r).start - find(from).start;
  const f = syntheticFixture(index, { id: "joint-addition", from, to, edits: edits(at), chunkSize: 4 });
  const checker = createChecker(index, { passage: { from, to } });
  for (const e of f.evidence) checker.push(e, []);
  checker.end();
  return checker.mistakes().map(e => `${e.errorClass} ${key(e.location)}`);
}

describe("added speech between neighboring words", () => {
  it("reads a paused قبل outside the neighbouring words as an added word", () => {
    const from = w(51, 16, 1), to = w(51, 21, 4), added = w(51, 18, 1);
    expect(check(from, to, at => [{ kind: "ins", at: at(added), text: "قَبڇل" }])).toEqual([`added_word ${key(added)}`]);
  });

  it("does not mistake a one-consonant detour for a correctly repeated word", () => {
    const from = w(35, 18, 1), to = w(35, 23, 1), added = w(35, 22, 1);
    expect(check(from, to, at => [{ kind: "ins", at: at(added), text: "تَزُۥۥلَاا" }])).toContain(`added_word ${key(added)}`);
  });

  it("recovers a forward word detour in another chapter", () => {
    const from = w(9, 60, 1), to = w(9, 62, 4), added = w(9, 61, 1);
    expect(check(from, to, at => [{ kind: "ins", at: at(added), text: "فِۦۦهَاا" }])).toContain(`added_word ${key(added)}`);
  });

  it("retains the earlier addition repair when a neighboring vowel is noisy", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8), added = w(2, 8, 1), before = w(2, 7, 12);
    expect(check(from, to, at => [
      { kind: "sub", at: at(before) + 1, to: "ُ" },
      { kind: "ins", at: at(added), text: "قَلِۦۦلَاا" },
    ])).toContain(`added_word ${key(added)}`);
  });

  it("preserves a repeated preceding word with one tolerated consonant change", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8), before = w(2, 7, 12), next = w(2, 8, 1);
    const repeated = text(before).replace("ع", "غ");
    expect(check(from, to, at => [{ kind: "ins", at: at(next), text: repeated }])).toEqual([]);
  });

  it("retains an added word when its sounds can take a reference-word detour", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8), added = w(2, 8, 1);
    expect(check(from, to, at => [{ kind: "ins", at: at(added), text: "قَلِۦۦلَاا" }])).toContain(`added_word ${key(added)}`);
  });

  it("retains added speech in a different passage", () => {
    const from = w(37, 104, 1), to = w(37, 111, 4), added = w(37, 108, 1);
    expect(check(from, to, at => [{ kind: "ins", at: at(added), text: "قَلِۦۦلَاا" }])).toContain(`added_word ${key(added)}`);
  });

  it("does not turn an ambiguous short addition into a warning on another word", () => {
    const from = w(38, 2, 1), to = w(38, 7, 1), added = w(38, 4, 6);
    const events = check(from, to, at => [
      { kind: "ins", at: at(added), text: "ءِننن" },
      { kind: "del", at: at(added), count: 1 },
    ]);
    expect(events.every(e => e.endsWith(key(added)))).toBe(true);
  });

  it("preserves another addition after the right-hand word", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8), first = w(2, 8, 1), second = w(2, 8, 2);
    const events = check(from, to, at => [
      { kind: "ins", at: at(first), text: "قَلِۦۦلَاا" },
      { kind: "ins", at: at(second), text: "كَوثَر" },
    ]);
    expect(events).toContain(`added_word ${key(second)}`);
  });

  it("preserves a genuine single-word repeat", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8), repeated = w(2, 7, 12);
    expect(check(from, to, at => [{ kind: "repeat", from: at(repeated), to: at(w(2, 8, 1)) }])).toEqual([]);
  });

  it("leaves lengthened sounds at the boundary alone", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8), before = w(2, 7, 12);
    expect(check(from, to, at => [{ kind: "ins", at: at(before) + text(before).length - 1, text: "مم" }])).toEqual([]);
  });

  it("leaves the same unedited neighboring words alone", () => {
    expect(check(w(2, 6, 1), w(2, 11, 8), () => [])).toEqual([]);
  });

  it("preserves a genuine multiword repeat between those neighbors", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8);
    expect(check(from, to, at => [{ kind: "repeat", from: at(w(2, 7, 1)), to: at(w(2, 8, 1)) }])).toEqual([]);
  });
});

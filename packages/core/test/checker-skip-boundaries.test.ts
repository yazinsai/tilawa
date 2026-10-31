import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { ReferenceAsset, WordRef } from "../src/index.js";
import { replayChecked, syntheticFixture } from "../src/fixtures/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")) as ReferenceAsset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find(x => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const skip = (from: WordRef, to: WordRef, first: WordRef, last: WordRef, chunkSize = 8) => {
  const start = find(from).start;
  const fixture = syntheticFixture(index, { id: "skip-boundary", from, to, chunkSize,
    edits: [{ kind: "del", at: find(first).start - start, count: find(last).end - find(first).start }] });
  return replayChecked(index, fixture, {}, { passage: { from, to } }).filter(e => e.state === "confirmed");
};

describe("skip placement at similar boundary words", () => {
  it("keeps an article-bearing preceding word outside the skipped ayah", () => {
    const events = skip(w(26, 100, 1), w(26, 105, 3), w(26, 103, 1), w(26, 103, 8));
    expect(events.map(e => `${e.errorClass} ${key(e.location)}`)).toEqual(["skipped_passage 26:103:1"]);
  });

  it("uses the heard boundary in another surah with an article difference", () => {
    const events = skip(w(36, 12, 1), w(36, 16, 5), w(36, 14, 1), w(36, 14, 11));
    expect(events.map(e => `${e.errorClass} ${key(e.location)}`)).toEqual(["skipped_passage 36:14:1"]);
  });

  it("leaves a repeated phrase resolved rather than turning it into a skip", () => {
    const from = w(2, 255, 1), to = w(2, 256, 10);
    const start = find(from).start;
    const fixture = syntheticFixture(index, { id: "boundary-repeat", from, to, chunkSize: 8,
      edits: [{ kind: "repeat", from: find(w(2, 255, 22)).start - start, to: find(w(2, 255, 24)).start - start }] });
    expect(replayChecked(index, fixture, {}, { passage: { from, to } }).filter(e => e.state === "confirmed")).toEqual([]);
  });

  it("keeps repeated short words on the right side of an ayah skip", () => {
    const events = skip(w(26, 128, 1), w(26, 133, 3), w(26, 130, 1), w(26, 130, 3));
    expect(events.map(e => `${e.errorClass} ${key(e.location)}`)).toEqual(["skipped_passage 26:130:1"]);
  });

  it.each([2, 8, 13])("preserves the full skipped span after exact-repeat relocation with chunks of %i", chunkSize => {
    // The exact-repeat rule can move the gap past the first heard ولا. The acoustic pass must use the next
    // visit after that relocated gap, not the stale visit now contained in the gap.
    const first = w(109, 3, 1), last = w(109, 3, 5);
    const events = skip(w(109, 2, 1), w(109, 5, 5), first, last, chunkSize);
    expect(events.map(e => ({ cls: e.errorClass, at: key(e.location), expected: e.expected }))).toEqual([
      { cls: "skipped_passage", at: key(first), expected: index.phonemes.slice(find(first).start, find(last).end) },
    ]);
  });

  it("retains an unambiguous within-ayah skip", () => {
    const events = skip(w(2, 255, 1), w(2, 257, 5), w(2, 255, 30), w(2, 255, 39));
    expect(events.map(e => `${e.errorClass} ${key(e.location)}`)).toEqual(["skipped_passage 2:255:30"]);
  });

  it("keeps a clean passage quiet", () => {
    const from = w(26, 100, 1), to = w(26, 105, 3);
    const fixture = syntheticFixture(index, { id: "skip-clean", from, to, chunkSize: 8 });
    expect(replayChecked(index, fixture, {}, { passage: { from, to } }).filter(e => e.state === "confirmed")).toEqual([]);
  });
});

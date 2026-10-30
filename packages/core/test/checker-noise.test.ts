import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { WordRef } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find(x => key(x) === key(r))!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

function check(from: WordRef, to: WordRef, edits: (at: (r: WordRef) => number) => Edit[]) {
  const at = (r: WordRef) => find(r).start - find(from).start;
  const f = syntheticFixture(index, { id: "noisy-gap", from, to, edits: edits(at), chunkSize: 4 });
  const checker = createChecker(index, { passage: { from, to } });
  for (const e of f.evidence) checker.push(e, []);
  checker.end();
  return checker.mistakes().map(e => `${e.errorClass} ${key(e.location)}`);
}

describe("word gaps surrounded by listener noise", () => {
  it("keeps an inserted doubled ending out of a following word whose onset was lost", () => {
    const added = w(38, 4, 6);
    expect(check(w(38, 4, 1), w(38, 5, 4), at => [
      { kind: "ins", at: at(added), text: "ءِننن" },
      { kind: "del", at: at(added), count: 1 },
    ])).toEqual([`added_word ${key(added)}`]);
  });

  it("hears a bounded excursion as an addition when it returns straight to the next word", () => {
    const added = w(77, 13, 1);
    expect(check(w(77, 10, 1), w(77, 18, 3), at => [
      { kind: "ins", at: at(added), text: "فَءِذَاا" },
    ])).toEqual([`added_word ${key(added)}`]);
  });

  it("keeps a shortened doubled consonant with the word before an omission", () => {
    const from = w(19, 80, 1), to = w(19, 84, 3), before = w(19, 81, 8), omitted = w(19, 82, 1), after = w(19, 82, 2);
    expect(check(from, to, at => [
      { kind: "del", at: at(before) + text(before).indexOf("زز"), count: 1 },
      { kind: "del", at: at(omitted), count: text(omitted).length },
      { kind: "sub", at: at(after), to: "ف" },
    ])).toEqual([`omitted_word ${key(omitted)}`]);
  });

  it.each([w(77, 13, 1), w(77, 16, 1)])("keeps a doubled ending inside an added word before %j", added => {
    expect(check(w(77, 10, 1),w(77, 18, 3),at => [
      { kind: "ins", at: at(added), text: "فَءِننن" },
    ])).toEqual([`added_word ${key(added)}`]);
  });

  it("does not count one borrowed consonant as a recited short word", () => {
    const omitted = w(77, 41, 3), after = w(77, 41, 4);
    expect(check(w(77, 39, 1),w(77, 43, 6),at => [
      { kind: "del", at: at(omitted), count: text(omitted).length },
      { kind: "sub", at: at(after), to: "ف" },
      { kind: "sub", at: at(after) + 1, to: "َ" },
      { kind: "sub", at: at(after) + 7, to: "َ" },
    ])).toContain(`omitted_word ${key(omitted)}`);
  });

  it("does not let a stray vowel tail erase an omission", () => {
    const omitted = w(51, 25, 6), before = w(51, 25, 5), after = w(51, 25, 7);
    expect(check(w(51, 24, 1),w(51, 28, 6),at => [
      { kind: "del", at: at(before), count: text(before).length },
      { kind: "ins", at: at(before), text: "سَلَاامَاا" },
      { kind: "del", at: at(omitted), count: text(omitted).length },
      { kind: "del", at: at(after), count: text(after).length },
      { kind: "ins", at: at(after), text: "سَلَاام" },
    ])).toEqual([`omitted_word ${key(omitted)}`]);
  });

  it("leaves clean words alone when only their doubled sounds change length", () => {
    expect(check(w(19, 80, 1),w(19, 84, 3),at => [
      { kind: "del", at: at(w(19, 81, 8)) + text(w(19, 81, 8)).indexOf("زز"), count: 1 },
    ])).toEqual([]);
  });
});

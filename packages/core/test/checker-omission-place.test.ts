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
  const f = syntheticFixture(index, { id: "omission-place", from, to, edits: edits(at) });
  const checker = createChecker(index, { passage: { from, to } });
  for (const e of f.evidence) checker.push(e, []);
  checker.end();
  return checker.mistakes().map(e => `${e.errorClass} ${key(e.location)}`);
}

describe("two-candidate omission placement", () => {
  const from = w(20, 24, 1), to = w(20, 26, 3);
  const li = w(20, 25, 4), sadr = w(20, 25, 5);

  it("puts the omission on the candidate with fewer hard edits", () => {
    // لي is لِۦۦ and صدري is صَدڇرِۦۦ. The segment لِۦ matches لي at soft cost and
    // صدري with one fewer hard edit once each word's deletion is counted: omitting
    // صدري is hard 3 / 45, omitting لي is hard 2 / 46. Total charge picks صدري.
    expect(check(from, to, at => [
      { kind: "del", at: at(li), count: text(li).length },
      { kind: "del", at: at(sadr), count: text(sadr).length },
      { kind: "ins", at: at(sadr), text: "لِۦ" },
    ])).toEqual([`omitted_word ${key(li)}`]);
  });

  it("leaves an omission where the neighbour was said and hard edits already agree", () => {
    expect(check(from, to, at => [
      { kind: "del", at: at(li), count: text(li).length },
    ])).toEqual([`omitted_word ${key(li)}`]);
  });

  it("does not move a fully heard word's omission onto that word", () => {
    // الذي heard in full, يشفع deleted. The segment charges الذي nothing, so the omission stays.
    const start = w(2, 255, 1), omitted = w(2, 255, 23);
    expect(check(start, w(2, 257, 5), at => [
      { kind: "del", at: at(omitted), count: text(omitted).length },
    ])).toContain(`omitted_word ${key(omitted)}`);
  });
});

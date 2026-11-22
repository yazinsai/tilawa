import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { WordRef } from "../src/events.js";
import type { ChannelCounts } from "../src/noisy-channel.js";
import { emptyCounts } from "../src/noisy-channel.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find(x => key(x) === key(r))!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const off = emptyCounts([]);

function check(from: WordRef, to: WordRef, edits: (at: (r: WordRef) => number) => Edit[], channel?: ChannelCounts) {
  const at = (r: WordRef) => find(r).start - find(from).start;
  const f = syntheticFixture(index, { id: "channel-place", from, to, edits: edits(at) });
  const checker = createChecker(index, { passage: { from, to }, channel: channel ?? off });
  for (const e of f.evidence) checker.push(e, []);
  checker.end();
  return checker.mistakes().map(e => `${e.errorClass} ${key(e.location)}`);
}

/** Identity channel, then `edit` overwrites the cells that should win the placement. */
function channel(edit: (c: ChannelCounts, at: (p: string) => number) => void): ChannelCounts {
  const phones = [...index.tokens];
  const c = emptyCounts(phones);
  const at = (p: string) => phones.indexOf(p);
  for (let i = 0; i < phones.length; i++) c.sub[i][i] = 80;
  c.nWritten = 80 * phones.length;
  edit(c, at);
  return c;
}

describe("two-candidate omission placement by channel likelihood", () => {
  const from = w(20, 24, 1), to = w(20, 26, 3);
  const li = w(20, 25, 4), sadr = w(20, 25, 5);
  const split = (at: (r: WordRef) => number): Edit[] => [
    { kind: "del", at: at(li), count: text(li).length },
    { kind: "del", at: at(sadr), count: text(sadr).length },
    { kind: "ins", at: at(sadr), text: "لِۦ" },
  ];

  it("leaves the alignment's placement when the channel is off", () => {
    expect(check(from, to, split)).toEqual([`omitted_word ${key(sadr)}`]);
  });

  it("moves the omission onto the word the channel says was not the survivor", () => {
    // لِۦ is لي's own tail. A channel that deletes صدري's onset and hears ل for what is left
    // still loses to an exact لي unless deletions of صدري are the common event and ل is not
    // a match for لي. Make every phone of صدري cheap to delete and ل a common insertion,
    // and make ل|ل unseen, so the stretch is صدري surviving.
    const preferSadr = channel((c, at) => {
      for (const p of text(sadr)) c.del[at(p)] = 500;
      c.ins[at("ل")] = 500;
      c.sub[at("ل")][at("ل")] = 0;
      c.nWritten += 500 * text(sadr).length;
    });
    expect(check(from, to, split, preferSadr)).toEqual([`omitted_word ${key(li)}`]);
  });

  it("does not move an omission off a word whose neighbour was heard in full", () => {
    const gone = (at: (r: WordRef) => number): Edit[] => [
      { kind: "del", at: at(sadr), count: text(sadr).length },
    ];
    const diagonal = channel(() => {});
    expect(check(from, to, gone, diagonal)).toEqual([`omitted_word ${key(sadr)}`]);
  });
});

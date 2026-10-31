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
  const f = syntheticFixture(index, { id: "single-accusation", from, to, edits: edits(at), chunkSize: 4 });
  const checker = createChecker(index, { passage: { from, to } });
  for (const e of f.evidence) checker.push(e, []);
  checker.end();
  return checker.mistakes();
}

describe("one spoken edit", () => {
  it("does not also accuse the word preceding an insertion", () => {
    const from = w(26, 3, 1), to = w(26, 8, 5), inserted = w(26, 5, 1);
    const events = check(from, to, at => [{ kind: "ins", at: at(inserted), text: "قَلِۦۦلُمن" }]);
    expect(events).toHaveLength(1);
    expect(events[0].errorClass).toBe("added_word");
    expect(key(events[0].location)).toBe(key(inserted));
  });
  it("keeps an inserted word in another passage a single accusation", () => {
    const from = w(26, 171, 1), to = w(26, 177, 5), inserted = w(26, 175, 1);
    const events = check(from, to, at => [{ kind: "ins", at: at(inserted), text: "قَلِۦلُمن" }]);
    expect(events).toHaveLength(1);
    expect(events[0].errorClass).toBe("added_word");
    expect(key(events[0].location)).toBe(key(inserted));
  });

  it("preserves a separate wrong word and addition in another surah", () => {
    const from = w(2, 255, 1), to = w(2, 256, 10), changed = w(2, 255, 23), inserted = w(2, 255, 24);
    const events = check(from, to, at => [
      { kind: "del", at: at(changed), count: text(changed).length },
      { kind: "ins", at: at(changed), text: "كَوثَر" },
      { kind: "ins", at: at(inserted), text: "مُستَقِۦۦم" },
    ]).map(e => `${e.errorClass} ${key(e.location)}`);
    expect(events).toContain(`wrong_word ${key(changed)}`);
    expect(events).toContain(`added_word ${key(inserted)}`);
  });

  it("preserves an insertion with a different consonant pattern", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8), inserted = w(2, 8, 1);
    const events = check(from, to, at => [{ kind: "ins", at: at(inserted), text: "كَوثَر" }]);
    expect(events).toHaveLength(1);
    expect(events[0].errorClass).toBe("added_word");
    expect(key(events[0].location)).toBe(key(inserted));
  });

  it("leaves clean speech and a repeated boundary word alone", () => {
    const from = w(2, 6, 1), to = w(2, 11, 8);
    expect(check(from, to, () => [])).toEqual([]);
    expect(check(from, to, at => [{ kind: "repeat", from: at(w(2, 7, 12)), to: at(w(2, 8, 1)) }])).toEqual([]);
  });
});

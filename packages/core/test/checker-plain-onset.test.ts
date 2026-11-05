import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { CheckerEvent, WordRef } from "../src/events.js";
import type { ReferenceAsset } from "../src/reference.js";

// The first word's tanween is assimilated into the second, doubling its onset (وجوهٌ يومئذ: يييَومَذِن).
const asset: ReferenceAsset = {
  version: "plain-onset-test",
  text: "synthetic",
  profile: {},
  inventory: { id: "plain-onset-test", tokens: "بُيَومذِنكخ" },
  ayahs: [
    { s: 1, a: 1, w: ["before", "day", "after", "tail"], p: "بُيييَومَذِنكَخُ", o: [0, 2, 12, 14],
      iso: ["بُن", "يَومَذ", "ك", "خ"] },
  ],
};
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const find = (r: WordRef) => index.words.find(x => key(x) === key(r))!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

function mistakes(said: string) {
  const from = w(1, 1, 1), to = w(1, 1, 4), day = w(1, 1, 2);
  const at = (r: WordRef) => find(r).start - find(from).start;
  const edits: Edit[] = [{ kind: "del", at: at(day), count: text(day).length }, { kind: "ins", at: at(day), text: said }];
  const fixture = syntheticFixture(index, { id: "plain-onset", from, to, edits, chunkSize: 4 });
  const checker = createChecker(index, { passage: { from, to }, policy: { confirm: { after: "next-word" } } });
  const events: CheckerEvent[] = [];
  for (const evidence of fixture.evidence) events.push(...checker.push(withPosteriors(index, evidence), []));
  events.push(...checker.end());
  return checker.mistakes().map(e => `${e.errorClass} ${key(e.location)}`);
}

describe("a word opened plainly after a pause", () => {
  it("is one of the word's own forms: connected tail, isolated onset", () => {
    expect(index.wordForms![1]).toContain("يَومَذِن");
    expect(index.wordForms![0]).not.toContain("ُن"); // only a doubled onset is trimmed
  });

  it("still catches a different word in its place", () => {
    expect(mistakes("كَخِن")).toContain("wrong_word 1:1:2");
  });
});

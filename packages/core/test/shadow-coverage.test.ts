import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { replay, replayChecked, syntheticFixture } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { PhraseAsset, PositionEvent, ReferenceAsset, WordRef } from "../src/index.js";
import { loadReferenceIndex } from "../src/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const phones = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const from = w(54, 33, 1);
const at = (r: WordRef) => find(r).start - find(from).start;

function run(edits: Edit[]) {
  const f = syntheticFixture(index, { id: "shadow", from, to: w(54, 42, 1), edits, chunkSize: 8 });
  const pos = replay(index, f, { phrases }).filter((e): e is PositionEvent => e.type === "position");
  const checked = replayChecked(index, f, { phrases }, { phrases });
  const skips = checked.filter((e) => e.state === "confirmed" && e.errorClass === "skipped_passage").map((e) => key(e.location));
  const covered = pos.flatMap((e) => e.coverage ?? []);
  const words = pos.filter((e) => e.state === "locked" && e.observed[0]).map((e) => key(e.observed[0].word));
  return { pos, skips, covered, words };
}

const omit = (r: WordRef): Edit => ({ kind: "del", at: at(r), count: find(r).end - find(r).start });
/** Hear `alt` where `src` is written. */
const asWord = (src: WordRef, alt: WordRef): Edit[] => [omit(src), { kind: "ins", at: at(src), text: phones(alt) }];

describe("shadow coverage without rebinding", () => {
  it("certifies the near ayah when a one-deletion continuation beats the far step, and leaves that step locked", () => {
    // فتماروا omitted, and بالنذر heard as the rhyme look-alike ونذر, then 54:37 in order.
    const { skips, covered, words } = run([...asWord(w(54, 36, 5), w(54, 37, 9)), omit(w(54, 36, 4))]);
    expect(covered.some((a) => a.surah === 54 && a.ayah === 36)).toBe(true);
    expect(words).toContain("54:37:9");
    expect(skips).not.toContain("54:36:1");
  }, 20000);

  it("emits nothing on an in-order reading", () => {
    const { skips, covered, pos } = run([]);
    expect(covered).toEqual([]);
    expect(pos.some((e) => e.coveragePending)).toBe(false);
    expect(skips).toEqual([]);
  }, 20000);

  it("keeps a skip whose landing no near continuation fits", () => {
    // After بطشتنا the next heard word is صبحهم, unique to 54:38 and past the one-deletion shadow.
    const fromWord = w(54, 36, 4);
    const toWord = w(54, 38, 2);
    const { skips, covered, words } = run([{ kind: "del", at: at(fromWord), count: find(toWord).start - find(fromWord).start }]);
    expect(covered.some((a) => a.surah === 54 && a.ayah === 36)).toBe(false);
    expect(words).toContain("54:38:2");
    expect(skips).toContain("54:36:4");
  }, 20000);
});

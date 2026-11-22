import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { replay, replayChecked, syntheticFixture } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";
import type { PhraseAsset, PositionEvent, ReferenceAsset, WordRef } from "../../src/engine/index.js";
import { loadReferenceIndex } from "../../src/engine/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const phones = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const from = w(26, 120, 1);
const at = (r: WordRef) => find(r).start - find(from).start;

function run(edits: Edit[]) {
  const f = syntheticFixture(index, { id: "ex", from, to: w(26, 130, 1), edits, chunkSize: 8 });
  const pos = replay(index, f, { phrases }).filter((e): e is PositionEvent => e.type === "position");
  const checked = replayChecked(index, f, { phrases }, { phrases });
  const moved = pos.filter((e) => e.relocated && key(e.relocated.from) !== key(e.relocated.to));
  const skips = checked.filter((e) => e.state === "confirmed" && e.errorClass === "skipped_passage").map((e) => key(e.location));
  return { moved, skips };
}

const noise: Edit = { kind: "noise", at: at(w(26, 123, 1)), length: 40, seed: 3 };
const swap = (src: WordRef, alt: WordRef): Edit[] => [
  { kind: "del", at: at(src), count: find(src).end - find(src).start },
  { kind: "ins", at: at(src), text: phones(alt) },
];
const skip126: Edit = { kind: "del", at: at(w(26, 126, 1)), count: find(w(26, 127, 1)).start - find(w(26, 126, 1)).start };

describe("an excursion onto a parallel span", () => {
  it("rebinds to the in-order member when the continuation returns and that text fits", () => {
    const { moved, skips } = run([noise, ...swap(w(26, 123, 2), w(26, 141, 2)), skip126]);
    expect(moved.some((e) => key(e.relocated!.from) === "26:140:1" && key(e.relocated!.to) === "26:122:1")).toBe(true);
    expect(skips).not.toContain("26:144:1");
  });

  it("keeps the other copy when its words are what was recited", () => {
    const { moved } = run([
      noise,
      ...swap(w(26, 123, 2), w(26, 141, 2)),
      ...swap(w(26, 124, 5), w(26, 142, 5)),
      skip126,
    ]);
    expect(moved.filter((e) => key(e.relocated!.to) === "26:122:1")).toEqual([]);
  });

  it("stays put on an in-order reading of the same block", () => {
    const { moved, skips } = run([]);
    expect(moved).toEqual([]);
    expect(skips).toEqual([]);
  });
});

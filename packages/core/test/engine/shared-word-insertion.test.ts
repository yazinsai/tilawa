import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, createTracker, loadReferenceIndex } from "../../src/engine/index.js";
import type { PhraseAsset, PositionEvent, ReferenceAsset, WordRef } from "../../src/engine/index.js";
import { syntheticFixture } from "../fixtures/synthetic.js";
import type { Edit } from "../fixtures/synthetic.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

describe("an inserted word that completes a distant pair", () => {
  // 2:205:8-9 is ويهلك الحرث. الحرث is also 2:71:12, and the word after it there is not والنسل.
  const from = w(2, 71, 1);
  const at = find(w(2, 71, 12)).start - find(from).start;
  const edits: Edit[] = [{ kind: "ins", at, text: text(w(2, 205, 8)) }];

  it("stays on the anchor and confirms the addition", () => {
    const f = syntheticFixture(index, { id: "ins", from, to: w(2, 71, 18), edits, chunkSize: 8 });
    const tracker = createTracker(index, { phrases });
    const checker = createChecker(index, { phrases });
    const pos: PositionEvent[] = [];
    const out = [];
    for (const e of f.evidence) {
      const p = tracker.push(e);
      pos.push(...p.filter((x): x is PositionEvent => x.type === "position"));
      out.push(...checker.push(e, p));
    }
    const end = tracker.end();
    pos.push(...end.filter((x): x is PositionEvent => x.type === "position"));
    out.push(...checker.end(end));
    for (const e of pos) if (e.state === "locked" && e.observed[0]) {
      expect(e.observed[0].word.surah).toBe(2);
      expect(e.observed[0].word.ayah).toBe(71);
    }
    expect(pos.at(-1)!.observed[0].word).toEqual(w(2, 71, 18));
    const added = out.filter((e) => e.errorClass === "added_word" && key(e.location) === "2:71:12");
    expect(added.some((e) => e.state === "confirmed")).toBe(true);
  });
});

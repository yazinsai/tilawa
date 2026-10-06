import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/index.js";
import type { PhraseAsset, ReferenceAsset, WordRef } from "../src/index.js";
import { attribute, scoreClip } from "../src/bench/acceptance.js";
import type { AcceptanceClip } from "../src/bench/acceptance.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const text = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);

const clip = (from: WordRef, to: WordRef, ayahs: string[], edits: Edit[], events: AcceptanceClip["events"] = []): AcceptanceClip => {
  const e = withPosteriors(index, syntheticFixture(index, { id: "c", from, to, edits }).evidence[0]);
  const p = e.posteriors!;
  return {
    id: "c",
    ayahs,
    events,
    nonErrors: [],
    evidence: { inventory: e.inventory, frameMs: e.frameMs, nFrames: p.nFrames, phonemes: e.phonemes, posteriors: { k: p.k, ids: [...p.ids], probs: [...p.probs] } },
  };
};

// 26:106 with نوح said as هود is exactly 26:124, and 26:107 is the same text as 26:125.
const nuh = w(26, 106, 5);
const swapped = () => {
  const at = find(nuh).start - find(w(26, 106, 1)).start;
  return clip(w(26, 106, 1), w(26, 107, 4), ["26:106", "26:107"], [
    { kind: "del", at, count: text(nuh).length },
    { kind: "ins", at, text: text(w(26, 124, 5)) },
  ], [{ errorClass: "wrong_word", state: "confirmed", location: nuh }]);
};

describe("declared-passage arm", () => {
  it("catches a substitution that forms another valid ayah, which the free arm follows", () => {
    const c = swapped();
    const free = scoreClip(index, c, { arm: "free", phrases });
    const declared = scoreClip(index, c, { arm: "declared", phrases });
    expect(free.misses).toHaveLength(1);
    expect(declared.misses).toEqual([]);
    expect(declared.hits).toBe(1);
  });
});

describe("E0 attribution", () => {
  const both = (c: AcceptanceClip) => attribute(index, c, scoreClip(index, c, { arm: "free", phrases }), scoreClip(index, c, { arm: "declared", phrases }));

  it("blames the tracker for a miss the declared passage catches", () => {
    expect(both(swapped()).map((a) => a.cause)).toEqual(["tracker"]);
  });

  it("blames the evidence when the decode shows the word as written", () => {
    const c = clip(w(26, 106, 1), w(26, 107, 4), ["26:106", "26:107"], [], [{ errorClass: "wrong_word", state: "confirmed", location: nuh }]);
    expect(both(c).map((a) => a.cause)).toEqual(["evidence"]);
  });
});

describe("start gate", () => {
  it("counts a clean clip located within 5 s, with no false jump", () => {
    const g = scoreClip(index, clip(w(2, 255, 1), w(2, 255, 30), ["2:255"], []), { arm: "free", phrases }).gate;
    expect(g).toMatchObject({ correctWithin5s: true, wrongLockFirst: false, falseJumps: 0 });
  });
});

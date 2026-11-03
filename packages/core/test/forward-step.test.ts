import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { replay, replayChecked, syntheticFixture } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";
import type { PhraseAsset, PositionEvent, ReferenceAsset, WordRef } from "../src/index.js";
import { commitMargin, forwardStepChoice, loadReferenceIndex } from "../src/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const phrases: PhraseAsset = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const w = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const find = (r: WordRef) => index.words.find((x) => x.surah === r.surah && x.ayah === r.ayah && x.word === r.word)!;
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const phones = (r: WordRef) => index.phonemes.slice(find(r).start, find(r).end);
const margin = (n: number) => commitMargin(n);

const positions = (from: WordRef, to: WordRef, edits: Edit[] = []) => {
  const f = syntheticFixture(index, { id: "fwd", from, to, edits, chunkSize: 1 });
  return replay(index, f, { phrases }).filter((e): e is PositionEvent => e.type === "position" && e.state === "locked" && !!e.observed[0]);
};

describe("forward step choice", () => {
  const heard = "بببببببببب";
  const far = { index: 2, phonemes: heard };

  it("keeps the nearer word when a farther one is cheaper by less than the commit margin", () => {
    // Two substitutions: 2 edits, under the margin of 3 for a 10-phone span, and per 0.2 still fits.
    const near = { index: 1, phonemes: "ببببببببذت" };
    expect(forwardStepChoice(heard, [near, far], margin, 0.35)).toBe(1);
  });

  it("takes the farther word when it beats the nearer one by the commit margin", () => {
    // Three substitutions: 3 edits, which clears the margin, and per 0.3 still fits.
    const near = { index: 1, phonemes: "بببببببذتس" };
    expect(forwardStepChoice(heard, [near, far], margin, 0.35)).toBe(2);
  });

  it("returns null while no forward word fits", () => {
    expect(forwardStepChoice("قققق", [{ index: 1, phonemes: heard }], margin, 0.35)).toBeNull();
  });
});

describe("a rhyme look-alike ahead of an omitted word", () => {
  const from = w(54, 35, 1);
  const at = (r: WordRef) => find(r).start - find(from).start;
  // The listener's phones for the word after فتماروا, which also soft-matches the rhyme ten words ahead.
  const heard = "فِلنُذُر";

  it("steps to the nearer word instead of restarting on the farther rhyme", () => {
    const drop = at(w(54, 36, 5)) - at(w(54, 36, 4));
    const events = positions(from, w(54, 38, 2), [
      { kind: "del", at: at(w(54, 36, 4)), count: drop },
      { kind: "ins", at: at(w(54, 36, 4)), text: heard },
    ]);
    const words = events.map((e) => key(e.observed[0].word));
    const unique: string[] = [];
    for (const word of words) if (word !== unique.at(-1)) unique.push(word);
    const at371 = unique.indexOf("54:37:1");
    expect(at371).toBeGreaterThan(unique.indexOf("54:36:5"));
    let seen371 = false;
    for (const e of events) {
      if (key(e.observed[0].word) === "54:37:1") seen371 = true;
      if (!seen371) expect(key(e.observed[0].span.start)).not.toBe("54:37:9");
    }
    const checked = replayChecked(index, syntheticFixture(index, {
      id: "fwd-check", from, to: w(54, 38, 2), chunkSize: 1,
      edits: [
        { kind: "del", at: at(w(54, 36, 4)), count: drop },
        { kind: "ins", at: at(w(54, 36, 4)), text: heard },
      ],
    }), { phrases }, { phrases });
    const skips = checked.filter((e) => e.state === "confirmed" && e.errorClass === "skipped_passage").map((e) => key(e.location));
    expect(skips).not.toContain("54:36:1");
  });

  it("does not restart a clean reading on the rhyme", () => {
    const events = positions(from, w(54, 38, 2));
    for (const e of events) {
      if (key(e.observed[0].word) === "54:37:9") expect(key(e.observed[0].span.start)).not.toBe("54:37:9");
    }
    const words = events.map((e) => key(e.observed[0].word));
    expect(words.indexOf("54:37:9")).toBeGreaterThan(words.indexOf("54:37:1"));
  });
});

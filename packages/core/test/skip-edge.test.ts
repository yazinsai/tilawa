import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "../src/ctc.js";
import { PHRASE_SWITCH } from "../src/parallel-phrases.js";
import type { ReferenceAsset } from "../src/reference.js";
import { loadReferenceIndex } from "../src/reference.js";
import { sequenceNll } from "../src/skip-edge.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const tokens = [...index.tokens];
const idOf = new Map(tokens.map((t, i) => [t, i + 1]));

/** Three frames per phone, each peaked on that phone. Word ends are every phone, one word per phone, unless grouped. */
function peaked(phones: string): PosteriorFrames {
  const k = 4;
  const ids: number[] = [];
  const probs: number[] = [];
  for (const p of phones) {
    const id = idOf.get(p);
    if (!id) throw new Error(`missing token ${p}`);
    for (let f = 0; f < 3; f++) {
      for (const [i, pr] of [[id, 0.92], [0, 0.04], [1, 0.02], [2, 0.01]] as [number, number][]) {
        ids.push(i);
        probs.push(pr);
      }
    }
  }
  const frames = new PosteriorFrames(tokens.length);
  frames.append({ k, startFrame: 0, nFrames: phones.length * 3, ids, probs });
  return frames;
}

const phones = (surah: number, ayah: number, word: number) => {
  const w = index.words.find((x) => x.surah === surah && x.ayah === ayah && x.word === word)!;
  return index.phonemes.slice(w.start, w.end);
};

describe("skip-edge forward sum", () => {
  it("prefers the path that omits an unsaid ayah by the switch penalty", () => {
    // 26:131:1 فاتقوا then 26:132:2 الذي, the landing after 26:130. The frames are only that landing.
    const heard = [phones(26, 131, 1), phones(26, 132, 2)];
    const ayah = [phones(26, 130, 1), phones(26, 130, 2), phones(26, 130, 3), phones(26, 130, 4)];
    const said = heard.join("");
    const frames = peaked(said);
    const ids = (parts: string[]) => parts.join("").split("").map((p) => idOf.get(p)!);
    const wordLast = (parts: string[]) => {
      const out: number[] = [];
      let n = 0;
      for (const part of parts) {
        n += part.length;
        out.push(n - 1);
      }
      return out;
    };
    const skip = sequenceNll(frames, ids(heard), wordLast(heard), 0, said.length * 3);
    const read = sequenceNll(frames, ids([...ayah, ...heard]), wordLast([...ayah, ...heard]), 0, said.length * 3);
    expect(ayah.join("").length).toBeGreaterThan(8);
    expect(read - skip).toBeGreaterThanOrEqual(PHRASE_SWITCH);
  });

  it("keeps a finite score when the audio repeats a word the path says once", () => {
    const word = phones(26, 132, 2);
    const frames = peaked(word + word);
    const ids = [...word].map((p) => idOf.get(p)!);
    const once = sequenceNll(frames, ids, [ids.length - 1], 0, word.length * 6);
    const other = sequenceNll(frames, [...phones(26, 130, 4)].map((p) => idOf.get(p)!), [phones(26, 130, 4).length - 1], 0, word.length * 6);
    expect(Number.isFinite(once)).toBe(true);
    expect(once).toBeLessThan(other);
  });
});

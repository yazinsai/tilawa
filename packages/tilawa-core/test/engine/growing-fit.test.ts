import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { alignSubstring, growingFit } from "../../src/engine/fuzzy.js";
import { loadReferenceIndex } from "../../src/engine/index.js";
import type { ReferenceAsset } from "../../src/engine/index.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const ref = index.phonemes;

/** A deterministic pseudo-random stream, so a failure replays. */
function random(seed: number) {
  let x = seed;
  return () => ((x = (x * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);
}

describe("growingFit", () => {
  it("keeps the cost a strict-start alignSubstring finds, phoneme by phoneme", () => {
    const next = random(7);
    const tokens = [...index.tokens];
    for (let t = 0; t < 60; t++) {
      const word = index.words[Math.floor(next() * index.words.length)];
      if (word.end <= word.start) continue;
      const lo = word.start;
      const hi = Math.min(ref.length, lo + 96);
      // Speech from an opening, with substitutions, insertions, deletions and held sounds.
      let speech = "";
      for (let k = lo; k < Math.min(hi, lo + 40) && speech.length < 48; k++) {
        const r = next();
        if (r < 0.08) continue;
        if (r < 0.16) speech += tokens[Math.floor(next() * tokens.length)];
        else if (r < 0.22) speech += ref[k] + ref[k];
        else speech += ref[k];
        if (next() < 0.05) speech += tokens[Math.floor(next() * tokens.length)];
      }
      const fit = growingFit(ref, lo, hi);
      for (let j = 1; j <= speech.length; j++) {
        fit.push(speech[j - 1]);
        const whole = alignSubstring(ref, lo, hi, speech.slice(0, j), (o) => o === lo, { strictStart: true });
        expect(fit.cost(), `${t}: ${j} of ${speech.length}`).toBe(whole?.cost ?? null);
      }
      expect(fit.heard).toBe(speech.length);
    }
  });

  it("fits a phrase the same way, from its first phoneme", () => {
    const phrase = ref.slice(index.words[0].start, index.words[3].end);
    const fit = growingFit(phrase, 0, phrase.length);
    const speech = `${phrase.slice(0, 8)}زظ${phrase.slice(8)}ثخ`;
    for (let j = 1; j <= speech.length; j++) {
      fit.push(speech[j - 1]);
      expect(fit.cost()).toBe(alignSubstring(phrase, 0, phrase.length, speech.slice(0, j), (o) => o === 0, { strictStart: true })?.cost ?? null);
    }
  });
});

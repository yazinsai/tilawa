import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../../src/engine/index.js";
import type { ReferenceAsset, WordRef } from "../../src/engine/index.js";
import { disambiguationWords, lcpArray, suffixArray } from "../../src/engine/disambiguation.js";

const asset: ReferenceAsset = JSON.parse(readFileSync(new URL("../../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const at = (r: WordRef) => index.words.findIndex((w) => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word);

describe("suffix array", () => {
  it("matches a naive sort and naive common prefixes", () => {
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let t = 0; t < 50; t++) {
      const s = Array.from({ length: 1 + Math.floor(rnd() * 60) }, () => "ab"[Math.floor(rnd() * 2)]).join("");
      const sa = suffixArray(s);
      expect([...sa]).toEqual([...s].map((_, i) => i).sort((a, b) => (s.slice(a) < s.slice(b) ? -1 : 1)));
      const { lcp } = lcpArray(s, sa);
      for (let r = 1; r < s.length; r++) {
        const a = s.slice(sa[r - 1]), b = s.slice(sa[r]);
        let h = 0;
        while (a[h] && a[h] === b[h]) h++;
        expect(lcp[r]).toBe(h);
      }
    }
  });
});

describe("disambiguationWords", () => {
  const k = disambiguationWords(index);
  it("needs the next ayah for the Al-Rahman refrain", () => {
    const i = at({ surah: 55, ayah: 13, word: 1 });
    expect(index.words[i + k[i]].ayah).toBe(14);
  });
  // Mid-ayah 2:255 keeps القيوم's case ending; 3:2 stops on it, so the phonemes part at word 7, not word 8.
  it("runs to the end of the words 2:255 shares with 3:2", () => {
    expect(k[at({ surah: 2, ayah: 255, word: 1 })]).toBe(6);
  });
  it("is 0 for a word that occurs once", () => {
    expect(k[at({ surah: 108, ayah: 1, word: 2 })]).toBe(0);
  });
  it("never points before the start word or past the end", () => {
    k.forEach((x, i) => { if (x >= 0) expect(i + x).toBeLessThan(index.words.length); });
  });
});

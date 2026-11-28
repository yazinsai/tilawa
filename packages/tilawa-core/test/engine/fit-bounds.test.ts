import { describe, expect, it } from "vitest";
import { UNIT, fitBounds, growingFit } from "../../src/engine/fuzzy.js";
import { ayah, index, spoken } from "./takes.js";

/**
 * `fitBounds`: the openings whose fit of the speech may cost less than a bar, found at once by walking the openings as
 * a trie, as refinements.declaredStart looks for rivals of the declared opening. Every opening whose fit
 * (`growingFit`) costs less than the bar must be found, each with a bound below the bar, cheapest first.
 */

const P = index.phonemes;
/** The reference a fit of an opening reads: the tracker's, twice its 48-phoneme window. */
const REACH = 96;
const end = (start: number) => Math.min(P.length, start + REACH);
/** Every ayah's first spoken phoneme, sorted by the text a fit of it reads. */
const starts = (() => {
  const out: number[] = [];
  let last = "";
  index.words.forEach((w, i) => {
    if (!spoken(i) || `${w.surah}:${w.ayah}` === last) return;
    last = `${w.surah}:${w.ayah}`;
    out.push(w.start);
  });
  const text = (s: number) => P.slice(s, end(s));
  return Int32Array.from(out.sort((a, b) => (text(a) < text(b) ? -1 : text(a) > text(b) ? 1 : a - b)));
})();
const fitCost = (start: number, obs: string) => {
  const fit = growingFit(P, start, end(start));
  for (const p of obs) fit.push(p);
  return fit.cost()!;
};
const opening = (s: number, a: number, n: number) => P.slice(index.words[ayah(s, a)[0]].start, index.words[ayah(s, a)[0]].start + n);

describe("fitBounds", () => {
  const speech: [string, string][] = [
    ["36:1's first 8 phonemes", opening(36, 1, 8)],
    ["26:1's first 8 phonemes", opening(26, 1, 8)],
    ["2:96's first 10 phonemes, the third dropped", opening(2, 96, 10).slice(0, 2) + opening(2, 96, 10).slice(3)],
    ["stray sounds, then 67:1's first 8 phonemes", "زظغذ" + opening(67, 1, 8)],
  ];
  for (const [name, obs] of speech) {
    it(`${name}: every opening that fits under the bar is found, cheapest first`, () => {
      const costs = new Map([...starts].map((s) => [s, fitCost(s, obs)]));
      const best = Math.min(...costs.values());
      for (const bar of [best + UNIT, best + 3 * UNIT]) {
        const found = fitBounds(P, starts, end, obs, bar);
        const byStart = new Map(found.map((f) => [f.start, f.bound]));
        for (const [s, cost] of costs) {
          if (cost < bar) expect(byStart.has(s), `${name}, bar ${bar}: opening at ${s} (fit ${cost})`).toBe(true);
          if (byStart.has(s)) expect(byStart.get(s)!, `${name}, bar ${bar}: bound at ${s}`).toBeLessThan(bar);
        }
        expect(found.length, `${name}, bar ${bar}: some opening fits`).toBeGreaterThan(0);
        for (let k = 1; k < found.length; k++) expect(found[k].bound).toBeGreaterThanOrEqual(found[k - 1].bound);
      }
    }, 60_000);
  }
});

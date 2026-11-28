import { describe, expect, it } from "vitest";
import { DEFAULT, ayah, ayahs, drawn, guessed, index, passage, pause, recite, sameWord, say, shownAt, speech } from "./takes.js";
import type { Piece, Speech, Take } from "./takes.js";

/**
 * refinements.declaredMove with the default policy, where a lead-in from before the passage and an ayah after the
 * declared one open alike: while near rivals fit the speech alike the take's opening contest stays open, the move goes
 * to the one that stands apart, it withdraws only words the page showed, and the first onset after a pause is judged
 * again, so the declared ayah lights from its first word when the reciter says it. Each take runs at 1 and 3 phonemes
 * a push, beside the same passage started correctly. The takes are synthetic phoneme streams built from the reference
 * (`./takes.ts`).
 */

const STEPS = [1, 3];

/** The words of ayah `from` on in surah `s` said and dark on the page at the take's end. */
const darkFrom = (take: Speech, t: Take, s: number, from: number) => {
  const page = drawn(t.shown);
  return [...take.saidAt.keys()].filter((k) => {
    const [x, y] = k.split(":").map(Number);
    return x === s && y >= from && !page.has(k);
  });
};
/** Every word said and dark on the page at the take's end. */
const dark = (take: Speech, t: Take) => [...take.saidAt.keys()].filter((k) => !drawn(t.shown).has(k));
/** How long after its end (each phoneme 120 ms) the first word of ayah `a` of surah `s` shows, in ms. */
const firstLag = (take: Speech, t: Take, s: number, a: number) => {
  const wi = ayah(s, a)[0];
  const end = take.saidAt.get(`${s}:${a}:1`)! + (index.words[wi].end - index.words[wi].start) * 120;
  return (shownAt(t.shown).get(`${s}:${a}:1`) ?? Infinity) - end;
};

describe("refinements.declaredMove: a lead-in from before the passage that opens like an ayah after it", () => {
  it("53:45-46 declared, 53:44's first three words, a stop, then 53:45-46: every word said lit, nothing unsaid lit", () => {
    // 53:43, 53:44, 53:48 and 53:49 open وَأَنَّهُۥ هُوَ, 53:45 وَأَنَّهُۥ. Before, the page jumped to 53:48 while 53:44
    // was said, lit 53:48:3 and 53:50:2, never said, and left 53:45:1 dark.
    const take = speech([say(53, 44, 1, 3), pause(20), ...ayahs(53, 45, 46)]);
    const control = speech(ayahs(53, 45, 46));
    for (const step of STEPS) {
      const t = recite(take, { passage: passage(53, 45, 46), step, refinements: DEFAULT });
      expect(dark(take, t), `step ${step}: said, dark`).toEqual([]);
      expect(guessed(take, drawn(t.shown)).filter((w) => w.includes("never")), `step ${step}: lit, never said`).toEqual([]);
      expect(guessed(take, shownAt(t.shown)), `step ${step}: lit before said`).toEqual([]);
      const c = recite(control, { passage: passage(53, 45, 46), step, refinements: DEFAULT });
      expect(dark(control, c), `step ${step}, started correctly: said, dark`).toEqual([]);
      expect(firstLag(control, c, 53, 45), `step ${step}, started correctly: 53:45:1 shown after its end (ms)`).toBeLessThanOrEqual(600);
    }
  });

  it("2:69-70 declared, the whole of 2:68 led in: no word lit before it is said, none said left dark", () => {
    // 2:68 opens with 2:69's first seven words and 2:70's first eight. Before, the move went to 2:70 on order alone,
    // 2:69:9-13 lit about 19 s before they were said, and 2:68's first eight words stayed dark.
    const take = speech(ayahs(2, 68, 70));
    const control = speech(ayahs(2, 69, 70));
    for (const step of STEPS) {
      const t = recite(take, { passage: passage(2, 69, 70), step, refinements: DEFAULT, checker: false });
      expect(guessed(take, shownAt(t.shown)), `step ${step}: lit, not said there`).toEqual([]);
      expect(dark(take, t), `step ${step}: said, dark`).toEqual([]);
      const c = recite(control, { passage: passage(2, 69, 70), step, refinements: DEFAULT, checker: false });
      expect(guessed(control, shownAt(c.shown)), `step ${step}, started correctly: lit, not said there`).toEqual([]);
      expect(dark(control, c), `step ${step}, started correctly: said, dark`).toEqual([]);
      expect(firstLag(control, c, 2, 69), `step ${step}, started correctly: 2:69:1 shown after its end (ms)`).toBeLessThanOrEqual(600);
    }
  });

  it("2:69-70 declared, a slip into 2:68 (its first eight words), a stop, then 2:69-70: the passage lit as said", () => {
    const take = speech([say(2, 68, 1, 8), pause(20), ...ayahs(2, 69, 70)]);
    for (const step of STEPS) {
      const t = recite(take, { passage: passage(2, 69, 70), step, refinements: DEFAULT, checker: false });
      expect(guessed(take, shownAt(t.shown)), `step ${step}: lit, not said there`).toEqual([]);
      expect(darkFrom(take, t, 2, 69), `step ${step}: passage words said, dark`).toEqual([]);
    }
  });

  it("lights no word before it is said in any lead-in where an ayah after the declared one opens more like it", async () => {
    // From the text: every lead-in from an ayah one or two before a declared ayah d, where an ayah up to three after d
    // shares more opening words with the lead-in's ayah than d does. Each take runs from the lead-in through d+1. Before,
    // the default policy lit words before they were said in 7 of the 23.
    const shared = (x: number[], y: number[]) => {
      let k = 0;
      while (k < x.length && k < y.length && sameWord(x[k], y[k])) k++;
      return k;
    };
    const places: { name: string; take: Speech; s: number; d: number }[] = [];
    for (const w of index.words) {
      if (w.word !== 1) continue;
      const [s, d] = [w.surah, w.ayah];
      for (const back of [1, 2]) {
        const b = d - back;
        if (b < 1 || !ayah(s, d + 1).length) continue;
        const sd = shared(ayah(s, b), ayah(s, d));
        if (sd < 1) continue;
        if (![d + 1, d + 2, d + 3].some((f) => ayah(s, f).length && shared(ayah(s, b), ayah(s, f)) > sd)) continue;
        const pieces: Piece[] = [];
        for (let x = b; x <= d + 1; x++) pieces.push(say(s, x), pause(15));
        places.push({ name: `${s}:${d} declared, from ${s}:${b}`, take: speech(pieces), s, d });
      }
    }
    expect(places.length).toBe(23);
    for (const step of STEPS) {
      const early: string[] = [];
      const darkIn: string[] = [];
      for (const { name, take, s, d } of places) {
        const t = recite(take, { passage: passage(s, d, d + 1), step, refinements: DEFAULT, checker: false });
        const g = guessed(take, shownAt(t.shown));
        if (g.length) early.push(`${name}: ${g.join(" ")}`);
        const k = darkFrom(take, t, s, d);
        if (k.length) darkIn.push(`${name}: ${k.join(" ")}`);
        // A long scan: let the test runner's worker answer between takes.
        await new Promise((done) => setImmediate(done));
      }
      expect(early, `step ${step}: lit before said`).toEqual([]);
      expect(darkIn, `step ${step}: passage words said, dark`).toEqual([]);
    }
  }, 300_000);
});

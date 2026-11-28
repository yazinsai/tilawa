import { describe, expect, it } from "vitest";
import { UNIT, growingFit } from "../../src/engine/fuzzy.js";
import { ALL, FRAME_MS, ayah, guessed, index, passage, pause, recite, say, shownAt, speech, spoken, without } from "./takes.js";

/**
 * refinements.declaredStart: every ayah opening in the mushaf is a rival of the declared one, judged by how well the
 * speech heard so far fits it. While rivals are in play, the declared opening's words show only as far as every rival
 * in play shares them, so of ayahs that open with the same words the declared one, the one next in order, lights as
 * they are said. A student who starts another surah or ayah whose opening only sounds like the declared one (a
 * different first sound, a short vowel, a confusable pair) is shown no word of the declared passage, and is followed
 * where they are once the speech tells the openings apart. The takes are synthetic phoneme streams built from the
 * reference (`./takes.ts`).
 */

/** Every ayah's first spoken word, in mushaf order. */
const openings = (() => {
  const out: { s: number; a: number; at: number }[] = [];
  let last = "";
  index.words.forEach((w, i) => {
    if (!spoken(i) || `${w.surah}:${w.ayah}` === last) return;
    last = `${w.surah}:${w.ayah}`;
    out.push({ s: w.surah, a: w.ayah, at: w.start });
  });
  return out;
})();
const P = index.phonemes;

/**
 * The `n` openings elsewhere in the mushaf (not the declared ayah's surah within three ayahs) whose first five to eight
 * phonemes fit the declared opening most cheaply, within the tracker's landing cost (0.34 of an edit a phoneme): the
 * openings a reciter is most likely to be shown the declared passage on.
 */
const lookAlikes = (d: { s: number; a: number; at: number }, n: number) => {
  const found: { s: number; a: number; cost: number }[] = [];
  for (const o of openings) {
    if (o.s === d.s && Math.abs(o.a - d.a) <= 3) continue;
    const fit = growingFit(P, d.at, Math.min(P.length, d.at + 96));
    for (let k = 1; k <= 8; k++) {
      fit.push(P[o.at + k - 1]);
      const cost = fit.cost();
      if (k >= 5 && cost !== null && cost / UNIT / k <= 0.34) {
        found.push({ s: o.s, a: o.a, cost });
        break;
      }
    }
  }
  return found.sort((x, y) => x.cost - y.cost).slice(0, n);
};
const hasAyah = (s: number, a: number) => ayah(s, a).length > 0;

describe("refinements.declaredStart: rivals by fit", () => {
  it("a surah whose opening differs from the declared one in a sound shows no word of the declared passage", () => {
    // 36:1 (يَااسِۦۦۦۦۦۦن) and 26:1, 27:1, 28:1 (طَااسِ…) differ in their first sound; 19:1 and 50:1 open on كاف and
    // قاف; 20:1 and 36:1 differ in their first sound (طه, يس).
    for (const [declared, said] of [[36, 26], [36, 27], [36, 28], [19, 50], [20, 36], [36, 20]]) {
      const s = speech([say(said, 1), pause(15), say(said, 2)]);
      for (const step of [1, 3]) {
        const t = recite(s, { passage: passage(declared, 1, 3), checker: false, step, refinements: ALL });
        const unsaid = [...shownAt(t.positions).keys()].filter((k) => !s.saidAt.has(k));
        expect(unsaid, `declared ${declared}:1–3, said ${said}:1–2, step ${step}`).toEqual([]);
      }
    }
  }, 60_000);

  it("the openings elsewhere that fit a declared opening best show no word that was not said, but on the same words", () => {
    // A spread of declared openings through the mushaf, each against its three closest look-alikes.
    for (let d = 7; d < openings.length; d += 701) {
      const D = openings[d];
      if (!hasAyah(D.s, D.a + 1)) continue;
      for (const S of lookAlikes(D, 3)) {
        const s = speech([say(S.s, S.a), pause(15), ...(hasAyah(S.s, S.a + 1) ? [say(S.s, S.a + 1)] : [])]);
        for (const step of [1, 3]) {
          const t = recite(s, { passage: passage(D.s, D.a, D.a + 1), checker: false, step, refinements: ALL });
          expect(guessed(s, shownAt(t.positions)), `declared ${D.s}:${D.a}, said ${S.s}:${S.a}, step ${step}`).toEqual([]);
        }
      }
    }
  }, 300_000);

  it("a declared opening that another ayah opens like shows its first word by its second word's second sound", () => {
    // 113:1 and 114:1 share قُلْ أَعُوذُ بِرَبِّ; 82:1 and 84:1 إِذَا ٱلسَّمَآءُ; 18:1 and 6:1
    // ٱلْحَمْدُ لِلَّهِ ٱلَّذِى; 67:1 and 25:1 تَبَارَكَ ٱلَّذِى; 2:1 and 3:1 are both الٓمٓ, 26:1 and 28:1 طسٓمٓ.
    // Those shared words light as the in-order ayah's. A first word that another opening's extends (قُلْ and
    // قُلْنَا, الٓمٓ and الٓمٓر) waits for the first sound after it, or the second when only a vowel tells them apart.
    // Without the field the first word waits until global search tells them apart (two words in, or a whole ayah).
    for (const [s, a] of [[113, 1], [114, 1], [82, 1], [84, 1], [18, 1], [6, 1], [67, 1], [2, 1], [26, 1]] as const) {
      const take = speech([say(s, a), pause(15), say(s, a + 1)]);
      const first = `${s}:${a}:1`;
      const second = take.saidAt.get(ayah(s, a).length > 1 ? `${s}:${a}:2` : `${s}:${a + 1}:1`)!;
      for (const step of [1, 3]) {
        const now = shownAt(recite(take, { passage: passage(s, a, a + 1), checker: false, step, refinements: ALL }).positions).get(first);
        const rc1 = shownAt(recite(take, { passage: passage(s, a, a + 1), checker: false, step, refinements: without("declaredStart") }).positions).get(first);
        expect(now, `${first}, step ${step}`).toBeDefined();
        expect(now!, `${first}, step ${step}`).toBeLessThanOrEqual(second + 5 * FRAME_MS);
        expect(now!, `${first}, step ${step}: sooner than without it`).toBeLessThan(rc1 ?? Infinity);
      }
    }
  }, 60_000);

  it("Ayat al-Kursi and 3:2, which open alike for seven words, show their first word on the second word's first sound", () => {
    // ٱللَّهُ also opens the takbir (ٱللَّهُ أَكْبَرُ), a phrase set aside before the passage: until the speech
    // leaves it, the opening could be the takbir, so the lock waits for the second word's first sound, not for the
    // seventh.
    for (const [s, a] of [[2, 255], [3, 2]] as const) {
      const take = speech([say(s, a), pause(15), say(s, a + 1)]);
      const second = take.saidAt.get(`${s}:${a}:2`)!;
      for (const step of [1, 3]) {
        const now = shownAt(recite(take, { passage: passage(s, a, a + 1), checker: false, step, refinements: ALL }).positions).get(`${s}:${a}:1`);
        expect(now, `${s}:${a}, step ${step}`).toBeDefined();
        expect(now!, `${s}:${a}, step ${step}`).toBeLessThanOrEqual(second + 3 * FRAME_MS);
      }
    }
  });

  it("a declared opening that sounds like no other shows on its first sounds, before its second word is said", () => {
    // Openings of a long first word that no other ayah's opening fits within an edit.
    for (const [s, a] of [[36, 6], [7, 22], [40, 44], [55, 5]] as const) {
      const take = speech([say(s, a), pause(15), say(s, a + 1)]);
      const first = `${s}:${a}:1`;
      const second = take.saidAt.get(`${s}:${a}:2`)!;
      for (const step of [1, 3]) {
        const now = shownAt(recite(take, { passage: passage(s, a, a + 1), checker: false, step, refinements: ALL }).positions).get(first);
        const rc1 = shownAt(recite(take, { passage: passage(s, a, a + 1), checker: false, step, refinements: without("declaredStart") }).positions).get(first);
        expect(now, `${first}, step ${step}`).toBeDefined();
        expect(now!, `${first}, step ${step}`).toBeLessThan(second);
        expect(now!, `${first}, step ${step}: no later than without it`).toBeLessThanOrEqual(rc1 ?? Infinity);
      }
    }
  }, 60_000);
});

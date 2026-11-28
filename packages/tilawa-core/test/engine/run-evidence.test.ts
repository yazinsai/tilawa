import { describe, expect, it } from "vitest";
import { CONSONANTS } from "../../src/engine/fuzzy.js";
import { ALL, EARLIER, ayah, ayahs, index, key, noise, passage, pause, recite, ref, say, shownAt, speech, spread, without } from "./takes.js";
import type { Fields, Piece, Speech } from "./takes.js";

/**
 * refinements.consonantConfirm and refinements.rivalHold: a later word of the tracker's run entered on weak evidence
 * that another place in the window explains as well shows on evidence of its own, not on a guess, while a word heard as
 * written shows when it did before, and every word said shows, one at a time. The takes are synthetic phoneme streams
 * built from the reference (`./takes.ts`); a planted skip leaves words out at a word boundary, as the bench's planted
 * takes do.
 */

const words = (s: number, a: number, b: number) => Array.from({ length: b - a + 1 }, (_, i) => ayah(s, a + i)).flat().map((wi) => key(ref(wi)));
const shown = (s: Speech, p: ReturnType<typeof passage> | null, refinements: Fields, step: number) =>
  shownAt(recite(s, { passage: p, refinements, checker: false, step }).positions);

/** Ayah `a`, then ayah `a + 1` with `d` words left out after its `k`th. */
const skip = (s: number, a: number, k: number, d: number) => ({
  take: speech([say(s, a), pause(15), say(s, a + 1, 1, k), say(s, a + 1, k + 1 + d)]),
  removed: ayah(s, a + 1).slice(k, k + d).map((wi) => key(ref(wi))),
});

describe("refinements.consonantConfirm: one syllable is not yet the word", () => {
  // The word after the cut opens with the same consonant and vowel as the word left out: on them alone, the word left
  // out would show.
  for (const [s, a, k] of [[2, 21, 20], [19, 9, 10], [20, 13, 2]] as const) {
    it(`${s}:${a + 1} with its word ${k + 1} left out: that word is not shown, and every word said is`, () => {
      const { take, removed } = skip(s, a, k, 1);
      for (const step of [1, 3]) {
        const now = shown(take, passage(s, a, a + 1), ALL, step);
        expect(removed.filter((w) => now.has(w)), `step ${step}`).toEqual([]);
        expect([...take.saidAt.keys()].filter((w) => !now.has(w)), `step ${step}: said, never shown`).toEqual([]);
        expect(removed.filter((w) => shown(take, passage(s, a, a + 1), without("consonantConfirm"), step).has(w)), `step ${step}`).toEqual(removed);
      }
    });
  }

  it("a word whose consonants are all misheard still shows, by the time the next word does", () => {
    for (const [s, a, w] of [[67, 2, 3], [36, 7, 2], [55, 5, 2]] as const) {
      const ws = ayah(s, a);
      const written = index.words[ws[w - 1]];
      const misheard = [...index.phonemes.slice(written.start, written.end)].map((p) => (CONSONANTS.has(p) ? (p === "ق" ? "ك" : "ق") : p)).join("");
      const take = speech([say(s, a - 1), pause(15), say(s, a, 1, w - 1), noise(misheard), say(s, a, w + 1), pause(15), say(s, a + 1)]);
      const [word, next] = [key(ref(ws[w - 1])), key(ref(ws[w]))];
      for (const step of [1, 3]) {
        const now = shown(take, passage(s, a - 1, a + 1), ALL, step);
        expect(now.get(word), `${word}, step ${step}`).toBeDefined();
        expect(now.get(word)!, `${word}, step ${step}`).toBeLessThanOrEqual(now.get(next)!);
        expect(now.get(next)!, `${next}, step ${step}`).toBeGreaterThanOrEqual(take.saidAt.get(next)!);
      }
    }
  });

  it("a word the reference gives no consonant (a nasal alone) is not held by it", () => {
    // 17:111:12 and 12:30:3 are each a ghunna, ںںں.
    for (const [s, a, w] of [[17, 111, 12], [12, 30, 3]] as const) {
      const word = `${s}:${a}:${w}`;
      expect([...index.phonemes.slice(index.words[ayah(s, a)[w - 1]].start, index.words[ayah(s, a)[w - 1]].end)].some((p) => CONSONANTS.has(p))).toBe(false);
      const take = speech(ayahs(s, a - 1, a));
      for (const step of [1, 3]) {
        expect(shown(take, passage(s, a - 1, a), ALL, step).get(word), `${word}, step ${step}`)
          .toBe(shown(take, passage(s, a - 1, a), without("consonantConfirm"), step).get(word));
      }
    }
  });
});

describe("refinements.rivalHold: a misheard opening that fits a word further on does not advance the run", () => {
  it("7:6 with its words 3 to 5 left out: none is shown, and every word said is", () => {
    // What is heard where 7:6:3 would begin is 7:6:6's opening, not 7:6:3's as written: without the hold, one of the
    // words left out shows.
    const { take, removed } = skip(7, 5, 2, 3);
    const p = passage(7, 5, 7);
    for (const step of [1, 3]) {
      const now = shown(take, p, ALL, step);
      expect(removed.filter((w) => now.has(w)), `step ${step}`).toEqual([]);
      expect([...take.saidAt.keys()].filter((w) => !now.has(w)), `step ${step}: said, never shown`).toEqual([]);
      expect(removed.filter((w) => shown(take, p, without("rivalHold"), step).has(w)), `step ${step}`).not.toEqual([]);
    }
  });

  it("the risk: a word left out whose next word opens with its sound shows on that sound, as in the kept version", () => {
    // 36:30 with مَا يَأْتِيهِم left out: مَاا and مِّن (مممِر) open on the same م. Heard as written, that sound
    // is ما's as well as من's, and the order the reciter reads in decides, as without the field. يأتيهم stays dark.
    const { take, removed } = skip(36, 29, 3, 2);
    for (const step of [1, 3]) {
      const [now, kept] = [shown(take, passage(36, 29, 30), ALL, step), shown(take, passage(36, 29, 30), EARLIER, step)];
      expect(now.get(removed[0]), `step ${step}`).toBe(kept.get(removed[0]));
      expect(now.has(removed[1]), `step ${step}`).toBe(false);
      expect([...take.saidAt.keys()].filter((w) => !now.has(w)), `step ${step}: said, never shown`).toEqual([]);
    }
  });

  it("the risk: a refrain repeated nearby shows every word, none early, a few phonemes later at most", () => {
    // 55:13 and 55:16, 77:15 and 77:19 repeat whole: each copy has a rival in the window. At 120 ms a phoneme.
    for (const [s, a, b] of [[55, 12, 18], [77, 14, 20]] as const) {
      const take = speech(ayahs(s, a, b));
      for (const step of [1, 3]) {
        const [now, before] = [shown(take, passage(s, a, b), ALL, step), shown(take, passage(s, a, b), without("rivalHold"), step)];
        for (const w of words(s, a, b)) {
          expect(now.get(w), `${w}, step ${step}`).toBeDefined();
          expect(now.get(w)!, `${w}, step ${step}: not before it is said`).toBeGreaterThanOrEqual(take.saidAt.get(w)!);
          expect(now.get(w)! - before.get(w)!, `${w}, step ${step}: at most four phonemes later`).toBeLessThanOrEqual(4 * 120);
        }
      }
    }
  });
});

describe("refinements.consonantConfirm and refinements.rivalHold: the flow", () => {
  /**
   * The words of each take shown with the next one (within 50 ms of it), and each ayah opening after the take's first
   * that shows later than the kept version (`EARLIER`) shows it.
   */
  const flow = (take: Speech, list: string[], openings: string[], p: ReturnType<typeof passage> | null, step: number, refinements: Fields) => {
    const now = shown(take, p, refinements, step);
    const kept = shown(take, p, EARLIER, step);
    const together = (sh: Map<string, number>) => list.filter((w, i) => i + 1 < list.length && sh.get(w)! > sh.get(list[i + 1])! - 50).length;
    return {
      late: openings.filter((w) => now.get(w) === undefined || now.get(w)! > kept.get(w)!),
      early: list.filter((w) => now.get(w)! < take.saidAt.get(w)!),
      dark: list.filter((w) => !now.has(w)),
      together: together(now),
      keptTogether: together(kept),
    };
  };

  // 60 to 150 ms a phoneme, 8 to 25 frames between ayahs, and every ayah's first phoneme heard at 0.4 to 0.7. 82:2–4
  // share their opening (وَإِذَا ٱلْ), as 55:13 and 55:16 do theirs, and 77:15 and 77:19 theirs. With every correction
  // on, the checker's included, the correct take stays silent at either push size.
  for (const [s, a, b] of [[82, 1, 5], [55, 10, 18], [77, 14, 20], [67, 1, 6], [36, 1, 9], [2, 1, 5]] as const) {
    it(`${s}:${a}–${b} at a varied pace, 1 and 3 phonemes a push: ayah openings in order show when the kept version shows them, and nothing is marked`, () => {
      for (const declared of [true, false]) for (const step of [1, 3]) {
        const [ms, gap, soft] = [spread(s * 1000 + a + step, 60, 150), spread(s + b + step, 8, 25), spread(a + b + step, 0.4, 0.7)];
        const pieces: Piece[] = [];
        for (let x = a; x <= b; x++) pieces.push(say(s, x), pause(Math.round(gap())));
        const take = speech(pieces, { ms, opening: soft });
        const openings = Array.from({ length: b - a }, (_, i) => key(ref(ayah(s, a + 1 + i)[0])));
        const p = declared ? passage(s, a, b) : null;
        const f = flow(take, words(s, a, b), openings, p, step, ALL);
        const mode = `${declared ? "declared" : "free"}, step ${step}`;
        expect(f.late, `${mode}: openings later than the kept version`).toEqual([]);
        expect(f.early, `${mode}: shown before said`).toEqual([]);
        expect(f.dark, `${mode}: never shown`).toEqual([]);
        expect(f.together, `${mode}: words shown with the next`).toBeLessThanOrEqual(f.keptTogether);
        expect(recite(take, { passage: p, refinements: ALL, step }).mistakes, `${mode}: marks`).toEqual([]);
      }
    });
  }
});

describe("refinements.consonantConfirm and refinements.rivalHold: a word heard as written shows when the kept version shows it", () => {
  // Both holds, in their first form, applied to every later word of the run, so on a correct recitation a
  // mid-ayah word showed three phonemes late. A hold now needs weak evidence that another place in the window explains
  // as well: none on speech heard as written. Every word but each ayah's first, of the take read correctly at 120 ms a
  // phoneme and at a varied pace, declared and free, at 1 and 3 phonemes a push, shows as with neither hold, and no
  // later than without the holds (sooner only in the take's first ayah, whose opening `declaredStart` now lights at
  // once); the median and the 90th percentile of the delay from its first sound are no more than without them.
  const q = (xs: number[], p: number) => [...xs].sort((u, v) => u - v)[Math.min(xs.length - 1, Math.floor(p * xs.length))];
  const neither: Fields = { ...ALL, consonantConfirm: false, rivalHold: false };
  for (const [s, a, b] of [[67, 1, 6], [36, 1, 9], [55, 1, 18], [2, 1, 5], [18, 1, 5], [112, 1, 4], [59, 22, 24], [3, 1, 6]] as const) {
    it(`${s}:${a}–${b}: every mid-ayah word shows at the kept version's time`, () => {
      const openings = new Set(Array.from({ length: b - a + 1 }, (_, i) => key(ref(ayah(s, a + i)[0]))));
      const [now, kept]: number[][] = [[], []];
      for (const pace of ["120 ms", "varied"] as const) {
        const take = pace === "120 ms" ? speech(ayahs(s, a, b)) : speech(ayahs(s, a, b), { ms: spread(s * 31 + a, 60, 150), opening: spread(s + b, 0.4, 0.7) });
        for (const p of [passage(s, a, b), null]) for (const step of [1, 3]) {
          const [x, y, z] = [shown(take, p, ALL, step), shown(take, p, EARLIER, step), shown(take, p, neither, step)];
          for (const [w, said] of take.saidAt) {
            if (openings.has(w)) continue;
            const mode = `${w}, ${pace}, ${p ? "declared" : "free"}, step ${step}`;
            expect(x.get(w), mode).toBe(z.get(w));
            expect(x.get(w)!, mode).toBeLessThanOrEqual(y.get(w)!);
            now.push(x.get(w)! - said);
            kept.push(y.get(w)! - said);
          }
        }
      }
      expect(q(now, 0.5)).toBeLessThanOrEqual(q(kept, 0.5));
      expect(q(now, 0.9)).toBeLessThanOrEqual(q(kept, 0.9));
    });
  }

  // The risk side: a word left out whose next word does not open as it does. Its first sound is not heard as written,
  // and the next word's opening explains it: either hold alone keeps the word left out dark, as without both it shows.
  for (const [s, a, k] of [[67, 5, 3], [36, 8, 3]] as const) {
    it(`${s}:${a + 1} with its word ${k + 1} left out: either hold alone keeps it dark, and every word said shows`, () => {
      const { take, removed } = skip(s, a, k, 1);
      const p = passage(s, a, a + 1);
      for (const step of [1, 3]) {
        for (const f of [ALL, without("rivalHold"), without("consonantConfirm")]) {
          const now = shown(take, p, f, step);
          expect(removed.filter((w) => now.has(w)), `step ${step}`).toEqual([]);
          expect([...take.saidAt.keys()].filter((w) => !now.has(w)), `step ${step}: said, never shown`).toEqual([]);
        }
        const neither = shown(take, p, { ...ALL, consonantConfirm: false, rivalHold: false }, step);
        expect(removed.filter((w) => neither.has(w)), `step ${step}`).toEqual(removed);
      }
    });
  }
});

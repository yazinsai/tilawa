import { describe, expect, it } from "vitest";
import type { WordRef } from "../../src/engine/events.js";
import { ALL, ayah, ayahs, guessed, key, noise, passage, pause, phrases, recite, ref, sameWord, say, shownAt, speech, spread, without } from "./takes.js";
import type { Piece, Speech } from "./takes.js";

/**
 * refinements.declaredStart: what the declared opening shows. While another ayah's opening fits the speech within half
 * an edit, the declared opening's words show only as far as every such rival shares them; a rival whose first word
 * differs keeps the opening dark until the speech tells them apart. Each family of takes, at 1 and 3 phonemes a push:
 * no word lights before it is said, or never said, but where an ayah said by then reads the same words (`guessed`);
 * every word said of the passage shows; and the declared first word shows as soon as the rule allows. The takes are
 * synthetic phoneme streams built from the reference (`./takes.ts`): a phoneme every 120 ms, an event at the end of the
 * phoneme heard, so a word shown on its own first sound shows 80 ms after it is said.
 */

const kw = (s: number, a: number, w = 1) => `${s}:${a}:${w}`;
/** The word after an ayah's first: its second word, or the next ayah's first. */
const next = (s: number, a: number) => (ayah(s, a).length > 1 ? kw(s, a, 2) : kw(s, a + 1));
const phrase = (name: string) => phrases.phrases.find((p) => p.name === name)!.p;
/** One phoneme, and the event at its end: the declared first word shows by its next word's second sound at the latest. */
const NEXT_SECOND_SOUND = 120 + 80;

interface Run {
  s: Speech;
  shown: Map<string, number>;
  /** When the declared first word showed, minus when it was said; undefined when it never showed. */
  first?: number;
}
const run = (pieces: Piece[], p: [WordRef, WordRef], step: number): Run => {
  const s = speech(pieces);
  const shown = shownAt(recite(s, { passage: p, checker: false, step, refinements: ALL }).positions);
  const first = shown.get(key(p[0]));
  return { s, shown, first: first === undefined ? undefined : first - s.saidAt.get(key(p[0]))! };
};
/** Nothing guessed, and every word said of the passage `p` shown. */
const honest = (r: Run, p: [WordRef, WordRef], label: string) => {
  expect(guessed(r.s, r.shown), `${label}: lit before said, or never said`).toEqual([]);
  const said = [...r.s.saidAt.keys()].filter((k) => {
    const [s, a] = k.split(":").map(Number);
    return s === p[0].surah && a >= p[0].ayah && a <= p[1].ayah;
  });
  expect(said.filter((k) => !r.shown.has(k)), `${label}: said and never shown`).toEqual([]);
};

describe("refinements.declaredStart: a correct start", () => {
  // Openings no other ayah's opening comes within half an edit of once the first word is heard, and openings that one
  // does: إِنَّ (إِنَّآ, a held length away; 26:67's refrain, a vowel after), الٓمٓ (الٓمٓر, الٓمٓصٓ), قُلْ (قُلْنَا),
  // وَٱلَّذِينَ, ٱلرَّحْمَٰنُ (1:3's), ٱللَّهُ (3:2's), the 1:1 basmala. The former show their first word before the next
  // is said; the latter by the next word's second sound.
  const apart: [number, number][] = [[67, 1], [36, 6], [18, 1], [54, 16], [37, 80], [55, 13], [1, 1], [9, 1], [19, 1], [20, 1], [42, 1], [50, 1],
    [68, 1], [38, 1], [96, 1], [108, 1], [103, 1], [101, 1], [69, 1]];
  const alike: [number, number][] = [[2, 4], [3, 1], [26, 67], [2, 255], [112, 1], [114, 1], [55, 1]];
  for (const [s, a] of [...apart, ...alike]) {
    it(`${s}:${a}–${a + 2}`, () => {
      const p = passage(s, a, a + 2);
      for (const step of [1, 3]) {
        const r = run(ayahs(s, a, a + 2), p, step);
        honest(r, p, `step ${step}`);
        const shown = r.shown.get(kw(s, a))!;
        if (apart.some(([x, y]) => x === s && y === a)) expect(shown, `step ${step}: before ${next(s, a)} is said`).toBeLessThan(r.s.saidAt.get(next(s, a))!);
        else expect(shown, `step ${step}: by ${next(s, a)}'s second sound`).toBeLessThanOrEqual(r.s.saidAt.get(next(s, a))! + NEXT_SECOND_SOUND);
      }
    });
  }

  it("the muqatta'at openings show their first word by the next word's second sound, and nothing unsaid", () => {
    for (const s of [2, 3, 7, 10, 11, 12, 13, 14, 15, 19, 20, 26, 27, 28, 29, 30, 31, 32, 36, 38, 40, 41, 42, 43, 44, 45, 46, 50, 68]) {
      const p = passage(s, 1, 3);
      for (const step of [1, 3]) {
        const r = run(ayahs(s, 1, 3), p, step);
        honest(r, p, `${s}:1, step ${step}`);
        expect(r.shown.get(kw(s, 1))!, `${s}:1, step ${step}`).toBeLessThanOrEqual(r.s.saidAt.get(next(s, 1))! + NEXT_SECOND_SOUND);
      }
    }
  }, 120_000);

  it("a passage of one word whose pausal form opens longer ayahs (55:1 alone) lights nothing, as without the field", () => {
    const p = passage(55, 1, 1);
    for (const step of [1, 3]) {
      const s = speech(ayahs(55, 1, 1));
      const shown = shownAt(recite(s, { passage: p, checker: false, step, refinements: ALL }).positions);
      expect(guessed(s, shown), `step ${step}`).toEqual([]);
      expect(shown, `step ${step}`).toEqual(shownAt(recite(s, { passage: p, checker: false, step, refinements: without("declaredStart") }).positions));
    }
  });
});

describe("refinements.declaredStart: what comes before the first ayah", () => {
  // The isti'adha, or the basmala, then a pause: the opening shows its first word as it does said alone. A pause after
  // the first word: shown before the pause, or, when a rival extends the word or differs from it in a held length, with
  // the next word's second sound after it.
  for (const [s, a] of [[67, 1], [36, 6], [18, 1], [2, 4], [3, 1], [55, 1], [112, 1]] as const) {
    it(`${s}:${a}–${a + 2} after the isti'adha, after the basmala, and with a pause after its first word`, () => {
      const p = passage(s, a, a + 2);
      for (const step of [1, 3]) {
        const alone = run(ayahs(s, a, a + 2), p, step);
        for (const [name, lead] of [["isti'adha", noise(phrase("istiadha"))], ["basmala", say(1, 1)]] as const) {
          const r = run([lead, pause(15), ...ayahs(s, a, a + 2)], p, step);
          honest(r, p, `${name}, step ${step}`);
          expect(r.first, `${name}, step ${step}`).toBe(alone.first);
        }
        const r = run([say(s, a, 1, 1), pause(15), say(s, a, 2), pause(15), ...ayahs(s, a + 1, a + 2)], p, step);
        honest(r, p, `a pause after the first word, step ${step}`);
        const shown = r.shown.get(kw(s, a))!;
        if (alone.shown.get(kw(s, a))! < alone.s.saidAt.get(next(s, a))!) expect(shown, `step ${step}`).toBe(alone.shown.get(kw(s, a))!);
        else expect(shown, `step ${step}`).toBeLessThanOrEqual(r.s.saidAt.get(next(s, a))! + NEXT_SECOND_SOUND);
      }
    });
  }
});

describe("refinements.declaredStart: a lead-in from the ayah before", () => {
  it("whose first word differs but sounds alike: nothing of the passage lights before it is said, and its first word shows when said", () => {
    // الذين before والذين, والشمس before والقمر, لا before ولا, oaths that open alike.
    for (const [s, x] of [[2, 4], [91, 2], [91, 3], [75, 2], [79, 2], [100, 3], [86, 2], [89, 2], [51, 2]] as const) {
      const p = passage(s, x, x + 2);
      for (const step of [1, 3]) {
        const r = run(ayahs(s, x - 1, x + 2), p, step);
        honest(r, p, `${s}:${x} from ${s}:${x - 1}, step ${step}`);
        expect(r.first!, `${s}:${x} from ${s}:${x - 1}, step ${step}`).toBeGreaterThanOrEqual(0);
        expect(r.shown.get(kw(s, x))!, `${s}:${x} from ${s}:${x - 1}, step ${step}`).toBeLessThan(r.s.saidAt.get(next(s, x))!);
      }
    }
  }, 120_000);

  it("whose first words are the same: the passage's words light only on the same words, and the page moves where they part", () => {
    // 2:149 and 2:150 share their first eight words, 2:69 and 2:70 their first seven, 82:3 and 82:4 their first.
    for (const [s, x] of [[2, 150], [2, 70], [82, 4]] as const) {
      const p = passage(s, x, x + 2);
      for (const step of [1, 3]) {
        const r = run(ayahs(s, x - 1, x + 2), p, step);
        honest(r, p, `${s}:${x} from ${s}:${x - 1}, step ${step}`);
        // Where the two part, the page moves to the ayah being said: its last word shows.
        const last = key(ref(ayah(s, x - 1).at(-1)!));
        expect(r.shown.has(last), `${last}, step ${step}`).toBe(true);
      }
    }
  }, 60_000);
});

describe("refinements.declaredStart: another surah whose opening sounds like the declared one", () => {
  // Declared one, said the other: يس and طسم, كهيعص and ق, the oaths of 37 and 51, 2:25's and 4:138's بَشِّرِ, 7:22's
  // and 7:107's فَ; 113:1 and 114:1, 82:1 and 84:1, which open with the same words. Nothing of the declared passage
  // lights that was not said, but the words an ayah said reads the same; and every word said that shows without the
  // field shows with it, or, said in the same words as the declared opening, as the declared opening's words.
  const takes: [[number, number, number], [number, number, number]][] = [
    [[36, 1, 3], [26, 1, 3]], [[36, 1, 3], [27, 1, 3]], [[36, 1, 3], [28, 1, 3]], [[19, 1, 3], [50, 1, 3]], [[37, 1, 3], [51, 1, 3]],
    [[4, 138, 139], [2, 25, 26]], [[7, 22, 23], [7, 107, 108]], [[113, 1, 3], [114, 1, 3]], [[82, 1, 3], [84, 1, 3]],
  ];
  for (const [declared, said] of takes) {
    it(`declared ${declared[0]}:${declared[1]}, said ${said[0]}:${said[1]}`, () => {
      const s = speech(ayahs(...said));
      for (const step of [1, 3]) {
        const shown = shownAt(recite(s, { passage: passage(...declared), checker: false, step, refinements: ALL }).positions);
        const off = shownAt(recite(s, { passage: passage(...declared), checker: false, step, refinements: without("declaredStart") }).positions);
        expect(guessed(s, shown), `step ${step}`).toEqual([]);
        const opening = ayah(declared[0], declared[1]);
        const asOpening = (k: string) => {
          const [su, a, w] = k.split(":").map(Number);
          const other = ayah(su, a);
          return w <= opening.length && other.slice(0, w).every((wi, i) => sameWord(wi, opening[i])) && shown.has(key(ref(opening[w - 1])));
        };
        expect([...s.saidAt.keys()].filter((k) => off.has(k) && !shown.has(k) && !asOpening(k)), `step ${step}: followed`).toEqual([]);
      }
    }, 60_000);
  }
});

describe("refinements.declaredStart: refrains", () => {
  // Refrains repeat the passage's opening nearby, word for word or a held length apart (لَءَايَةً and لَءَايَٰتٍ): each
  // word shows once no other ayah's word at its place is in play, so within about a second of being said, and the
  // in-order ayah is not held behind the first word.
  for (const [s, a, b] of [[26, 67, 70], [54, 16, 19], [37, 80, 83], [55, 13, 13]] as const) {
    it(`${s}:${a}–${b}`, () => {
      const p = passage(s, a, b);
      for (const step of [1, 3]) {
        const r = run(ayahs(s, a, b), p, step);
        honest(r, p, `step ${step}`);
        for (const [k, ms] of r.s.saidAt) expect(r.shown.get(k)! - ms, `step ${step}: ${k}`).toBeLessThanOrEqual(1200);
      }
    });
  }
});

describe("refinements.declaredStart: a correct recitation at a varied pace", () => {
  // 60 to 150 ms a phoneme, 8 to 25 frames between ayahs, and every ayah's first phoneme heard at 0.4 to 0.7.
  for (const [s, a, b] of [[67, 1, 4], [36, 1, 6], [2, 1, 4], [26, 67, 70], [82, 1, 5], [55, 10, 16], [112, 1, 4], [3, 1, 4]] as const) {
    it(`${s}:${a}–${b}: nothing lit before it is said, and the first word no later than without the field`, () => {
      const p = passage(s, a, b);
      for (const step of [1, 3]) {
        const [ms, gap, soft] = [spread(s * 1000 + a + step, 60, 150), spread(s + b + step, 8, 25), spread(a + b + step, 0.4, 0.7)];
        const pieces: Piece[] = [];
        for (let x = a; x <= b; x++) pieces.push(say(s, x), pause(Math.round(gap())));
        const take = speech(pieces, { ms, opening: soft });
        const shown = shownAt(recite(take, { passage: p, checker: false, step, refinements: ALL }).positions);
        const off = shownAt(recite(take, { passage: p, checker: false, step, refinements: without("declaredStart") }).positions);
        expect(guessed(take, shown), `step ${step}`).toEqual([]);
        expect([...take.saidAt.keys()].filter((k) => !shown.has(k)), `step ${step}: never shown`).toEqual([]);
        expect(shown.get(kw(s, a))!, `step ${step}`).toBeLessThanOrEqual(off.get(kw(s, a)) ?? Infinity);
      }
    });
  }
});

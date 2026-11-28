import { describe, expect, it } from "vitest";
import type { PositionEvent, WordRef } from "../../src/engine/events.js";
import {
  ALL, DEFAULT, at, ayah, ayahs, begin, breakAt, drawn, key, passage, pause, recite, say, sameWord, shownAt, speech, spread, without,
  withdrawals,
} from "./takes.js";
import type { Piece, Speech, Take } from "./takes.js";

/**
 * refinements.withdrawBegun: a word the reciter only begins and then leaves goes dark again once the run moves past
 * it (`withdrawn` on the locked position that first shows the new run), and lights again when a later run covers it.
 * A word said is never withdrawn. The rule changes nothing else: with the field off every event is the same but for
 * `withdrawn`. Each take runs at 1 and 3 phonemes a push. The takes are synthetic phoneme streams built from the
 * reference (`./takes.ts`): a word begun is its first one or two phonemes, said as written.
 */

const OFF = without("withdrawBegun");
const STEPS = [1, 3];
/** The positions without what they withdrew. */
const bare = (positions: PositionEvent[]) => positions.map(({ withdrawn: _, ...rest }) => rest);
/** A take with the field on and off. */
const both = (take: Speech, p: [WordRef, WordRef] | null, step: number): [Take, Take] =>
  [recite(take, { passage: p, step, refinements: ALL }), recite(take, { passage: p, step, refinements: OFF })];
/** The words withdrawn, in order. */
const withdrawnWords = (t: Take) => withdrawals(t.positions).map((w) => w.split("@")[0]);
/** The words said that the page shows dark at the take's end. */
const darkSaid = (take: Speech, t: Take) => [...take.saidAt.keys()].filter((k) => !drawn(t.positions).has(k));
/** The field adds `withdrawn` and changes nothing else: positions, checker events and marks. */
const onlyWithdrawn = (on: Take, off: Take, mode: string) => {
  expect(bare(on.positions), `${mode}: positions`).toEqual(off.positions);
  expect(on.checker, `${mode}: checker events`).toEqual(off.checker);
  expect(on.mistakes, `${mode}: marks`).toEqual(off.mistakes);
};
const word = (k: string): WordRef => {
  const [surah, ayah_, w] = k.split(":").map(Number);
  return { surah, ayah: ayah_, word: w };
};

describe("refinements.withdrawBegun: a word begun, then skipped", () => {
  // The word's first one or two phonemes (its first consonant, or that and its vowel), then a skip forward.
  const takes: [string, string, (n: number) => Piece[], [WordRef, WordRef]][] = [
    ["18:10:1 begun, then 18:11: an ayah left out", "18:10:1",
      (n) => [say(18, 9), pause(15), begin(18, 10, 1, n), pause(15), say(18, 11), pause(15), say(18, 12)], passage(18, 9, 12)],
    ["67:2:5 begun, then 67:2:7: two words left out", "67:2:5",
      (n) => [say(67, 1), pause(15), say(67, 2, 1, 4), begin(67, 2, 5, n), pause(10), say(67, 2, 7), pause(15), say(67, 3)], passage(67, 1, 3)],
    ["2:3:3 begun, then 2:4: an ayah's end left out", "2:3:3",
      (n) => [say(2, 2), pause(15), say(2, 3, 1, 2), begin(2, 3, 3, n), pause(15), say(2, 4), pause(15), say(2, 5)], passage(2, 2, 5)],
    ["2:4:1 begun, then 2:5: an ayah left out", "2:4:1",
      (n) => [...ayahs(2, 2, 3), begin(2, 4, 1, n), pause(15), say(2, 5), pause(15), say(2, 6)], passage(2, 2, 6)],
  ];
  for (const [name, begun, pieces, p] of takes) {
    for (const n of [1, 2]) {
      it(`${name}, ${n} phoneme${n > 1 ? "s" : ""} of it heard: it goes dark and stays dark, and nothing else changes`, () => {
        const take = speech(pieces(n));
        for (const step of STEPS) {
          const [on, off] = both(take, p, step);
          const mode = `step ${step}`;
          expect(drawn(off.positions).has(begun), `${mode}: shown with the field off`).toBe(true);
          expect(withdrawnWords(on), mode).toEqual([begun]);
          expect(drawn(on.positions).has(begun), `${mode}: dark at the end`).toBe(false);
          expect(darkSaid(take, on), `${mode}: said, dark`).toEqual([]);
          onlyWithdrawn(on, off, mode);
        }
      });
    }
  }

  it("is on by default: the default policy and the engine's own defaults withdraw the word begun", () => {
    for (const [name, begun, pieces, p] of takes) {
      const take = speech(pieces(1));
      for (const step of STEPS) {
        for (const refinements of [DEFAULT, undefined]) {
          const t = recite(take, { passage: p, step, refinements });
          expect(withdrawnWords(t), `${name}, step ${step}`).toEqual([begun]);
          expect(darkSaid(take, t), `${name}, step ${step}: said, dark`).toEqual([]);
        }
      }
    }
  });

  it("mark an ayah left out after its first word was begun as without the field", () => {
    for (const n of [1, 2]) {
      const take = speech([...ayahs(2, 2, 3), begin(2, 4, 1, n), pause(15), say(2, 5), pause(15), say(2, 6)]);
      for (const step of STEPS) {
        const [on, off] = both(take, passage(2, 2, 6), step);
        expect(on.mistakes, `${n} phonemes, step ${step}`).toContain("skipped_passage 2:4:1");
        expect(on.mistakes, `${n} phonemes, step ${step}`).toEqual(off.mistakes);
      }
    }
  });
});

describe("refinements.withdrawBegun: a word begun, skipped, then said", () => {
  it("lights again once a run covers it, when it is said", () => {
    for (const n of [1, 2]) {
      const take = speech([say(67, 1), pause(15), say(67, 2, 1, 4), begin(67, 2, 5, n), pause(10), say(67, 2, 7, 8), pause(15), say(67, 2, 5), pause(15), say(67, 3)]);
      for (const step of STEPS) {
        const [on, off] = both(take, passage(67, 1, 3), step);
        const mode = `${n} phonemes, step ${step}`;
        expect(withdrawnWords(on), mode).toEqual(["67:2:5"]);
        const out = on.positions.findIndex((e) => e.withdrawn);
        const back = on.positions.findIndex((e, i) => i > out && e.state === "locked" &&
          at(e.observed[0].span.start) <= at(word("67:2:5")) && at(word("67:2:5")) <= at(e.observed[0].word));
        expect(back, `${mode}: lit again`).toBeGreaterThan(out);
        expect(on.positions[back].timeMs, `${mode}: lit again once said`).toBeGreaterThanOrEqual(take.saidAt.get("67:2:5")!);
        expect(darkSaid(take, on), `${mode}: said, dark`).toEqual([]);
        onlyWithdrawn(on, off, mode);
      }
    }
  });
});

describe("refinements.withdrawBegun: a correct recitation withdraws nothing", () => {
  // With nothing withdrawn every event is the same as with the field off, so the page is too.
  it("at a varied pace, each ayah's first sound heard softly, declared and free", () => {
    // 60 to 150 ms a phoneme, 8 to 25 frames between ayahs, and every ayah's first phoneme heard at 0.4 to 0.7.
    for (const [s, a, b] of [[67, 1, 4], [36, 1, 6], [2, 1, 4], [26, 67, 70], [82, 1, 5], [55, 10, 16], [112, 1, 4], [3, 1, 4]] as const) {
      for (const step of STEPS) {
        const [ms, gap, soft] = [spread(s * 1000 + a + step, 60, 150), spread(s + b + step, 8, 25), spread(a + b + step, 0.4, 0.7)];
        const pieces: Piece[] = [];
        for (let x = a; x <= b; x++) pieces.push(say(s, x), pause(Math.round(gap())));
        const take = speech(pieces, { ms, opening: soft });
        for (const p of [passage(s, a, b), null]) {
          const t = recite(take, { passage: p, step, refinements: ALL });
          const mode = `${s}:${a}–${b}, ${p ? "declared" : "free"}, step ${step}`;
          expect(withdrawals(t.positions), mode).toEqual([]);
        }
      }
    }
  }, 300_000);

  it("with pauses inside words and between the words of an ayah", () => {
    // In each ayah: a pause between its second and third words, and inside its third word after its first phoneme
    // and its fifth after its first two (a breath, a hesitation), 15 to 30 frames.
    for (const [s, a, b] of [[67, 1, 3], [18, 9, 11], [2, 2, 4], [36, 1, 5], [1, 2, 7], [55, 1, 4], [112, 1, 4]] as const) {
      for (const step of STEPS) {
        const gap = spread(s * 7 + a + step, 15, 30);
        const pieces: Piece[] = [];
        for (let x = a; x <= b; x++) {
          if (ayah(s, x).length > 3) pieces.push(say(s, x, 1, 2), pause(Math.round(gap())), say(s, x, 3), pause(15));
          else pieces.push(say(s, x), pause(15));
        }
        let take = speech(pieces);
        for (let x = a; x <= b; x++) {
          const words = ayah(s, x).length;
          if (words >= 3) take = breakAt(take, `${s}:${x}:3`, 1, Math.round(gap()));
          if (words >= 5) take = breakAt(take, `${s}:${x}:5`, 2, Math.round(gap()));
        }
        for (const p of [passage(s, a, b), null]) {
          const t = recite(take, { passage: p, step, refinements: ALL });
          const mode = `${s}:${a}–${b}, ${p ? "declared" : "free"}, step ${step}`;
          expect(withdrawals(t.positions), mode).toEqual([]);
        }
      }
    }
  }, 300_000);
});

describe("refinements.withdrawBegun: an ayah's first word guessed from a restart", () => {
  it("goes dark when the page moves back to the restart, and lights again when it is said", () => {
    // 18:9 said, then said again: its first sound (ء) is 18:10's first word's, which shows on it in order, until the
    // run goes back to the restart.
    const take = speech([say(18, 8), pause(15), say(18, 9), pause(15), say(18, 9), pause(15), say(18, 10), pause(15), say(18, 11)]);
    for (const step of STEPS) {
      const [on, off] = both(take, passage(18, 8, 11), step);
      const mode = `step ${step}`;
      const out = on.positions.find((e) => e.withdrawn);
      expect(out?.withdrawn?.map(key), mode).toEqual(["18:10:1"]);
      expect(out!.timeMs, `${mode}: before 18:10:1 is said`).toBeLessThan(take.saidAt.get("18:10:1")!);
      expect(at(out!.observed[0].word), `${mode}: the page moved back`).toBeLessThan(at(word("18:10:1")));
      expect(withdrawnWords(on), mode).toEqual(["18:10:1"]);
      expect(darkSaid(take, on), `${mode}: said, dark`).toEqual([]);
      onlyWithdrawn(on, off, mode);
    }
  });
});

describe("refinements.withdrawBegun: a word said in an earlier pass", () => {
  // The reciter goes back over words already said and leaves one only begun this time: it was said, so it stays shown.
  const takes: [string, string, (n: number) => Piece[], [WordRef, WordRef]][] = [
    ["67:2 said, then again with 67:2:5 begun and 67:2:5-6 left out", "67:2:5",
      (n) => [say(67, 1), pause(15), say(67, 2), pause(15), say(67, 2, 1, 4), begin(67, 2, 5, n), pause(10), say(67, 2, 7), pause(15), say(67, 3)],
      passage(67, 1, 3)],
    ["18:12-13 said, then 18:12:6 on with 18:13:1 begun, and a restart at 18:12:5", "18:13:1",
      (n) => [...ayahs(18, 12, 13), pause(15), say(18, 12, 6), begin(18, 13, 1, n), pause(15), say(18, 12, 5), pause(15), say(18, 13), pause(15), say(18, 14)],
      passage(18, 12, 14)],
  ];
  for (const [name, again, pieces, p] of takes) {
    for (const n of [1, 2]) {
      it(`${name}, ${n} phoneme${n > 1 ? "s" : ""} of it heard: it stays shown`, () => {
        const take = speech(pieces(n));
        for (const step of STEPS) {
          const [on, off] = both(take, p, step);
          const mode = `step ${step}`;
          expect(withdrawnWords(on), mode).not.toContain(again);
          expect(darkSaid(take, on), `${mode}: said, dark`).toEqual([]);
          onlyWithdrawn(on, off, mode);
        }
      });
    }
  }
});

describe("refinements.withdrawBegun: ayahs that open with the same words", () => {
  // The ayah next in order lights as the shared words are said, and the page moves to the one said where they part.
  // The in-order ayah's words, read from the sounds of the ayah said, go dark then (leftOnNewAudio), and light again
  // when said there (67:21's first word, lit while 67:20 is said again); no word is withdrawn once said, and every word
  // first lights as without the field.
  const takes: [string, Piece[], [WordRef, WordRef], string][] = [
    ["67:15, then 67:17 (67:16 left out)", [say(67, 15), pause(15), say(67, 17), pause(15), say(67, 18)], passage(67, 15, 18), "67:16"],
    ["67:16, then 67:17", ayahs(67, 15, 18), passage(67, 15, 18), ""],
    ["67:22, then 67:24 (67:23 left out)", [say(67, 22), pause(15), say(67, 24), pause(15), say(67, 25)], passage(67, 22, 25), "67:23"],
    ["67:23, then 67:24", ayahs(67, 22, 25), passage(67, 22, 25), ""],
    ["67:20 said again before 67:21", [say(67, 19), pause(15), say(67, 20), pause(15), say(67, 20), pause(15), say(67, 21)], passage(67, 19, 21), ""],
  ];
  for (const [name, pieces, p, left] of takes) {
    it(`${name}: lit first as without the field, no word withdrawn once said${left ? `, and ${left}'s words dark at the end` : ""}`, () => {
      const take = speech(pieces);
      for (const step of STEPS) {
        const [on, off] = both(take, p, step);
        const mode = `step ${step}`;
        expect(shownAt(on.positions), `${mode}: lit`).toEqual(shownAt(off.positions));
        const once = withdrawals(on.positions).filter((w) => (take.saidAt.get(w.split("@")[0]) ?? Infinity) <= Number(w.split("@")[1]));
        expect(once, `${mode}: withdrawn once said`).toEqual([]);
        expect(darkSaid(take, on), `${mode}: said, dark`).toEqual([]);
        onlyWithdrawn(on, off, mode);
        if (!left) continue;
        const onPage = (t: Take) => [...drawn(t.positions).keys()].filter((k) => k.startsWith(`${left}:`));
        expect(onPage(off).length, `${mode}: ${left}'s words lit with the field off`).toBeGreaterThan(0);
        expect(onPage(on), `${mode}: ${left}'s words on the page at the end`).toEqual([]);
        const read = recite(take, { passage: p, step, refinements: { ...ALL, leftOnNewAudio: false } });
        expect(onPage(read).length, `${mode}: ${left}'s words lit with leftOnNewAudio off`).toBeGreaterThan(0);
      }
    });
  }
});

describe("refinements.withdrawBegun: a word left out with its onset kept, then the words after it said", () => {
  // The word left out shares sounds with the next word's start (فَكَيْفَ then كَانَ in 67:18, أَحْصَىٰ then لِمَا in
  // 18:12): the run can show the word left out on its kept onset and the next word on its first sound, then leave it
  // by a jump. The words said stay lit: by the jump every consonant of the word said was heard. With the defaults, at
  // the cut's first one or two phonemes kept, with and without a pause after them.
  const takes: [string, (n: number, gap: number) => Piece[], [WordRef, WordRef]][] = [
    ["67:18:6 فَكَيْفَ left out", (n, gap) => [say(67, 17), pause(15), say(67, 18, 1, 5), begin(67, 18, 6, n), pause(gap), say(67, 18, 7), pause(15), say(67, 19)],
      passage(67, 17, 19)],
    ["18:12:6 أَحْصَىٰ left out", (n, gap) => [say(18, 11), pause(15), say(18, 12, 1, 5), begin(18, 12, 6, n), pause(gap), say(18, 12, 7), pause(15), say(18, 13)],
      passage(18, 11, 13)],
  ];
  for (const [name, pieces, p] of takes) {
    it(`${name}, its onset kept: every word said is lit at the end`, () => {
      for (const n of [1, 2]) {
        for (const gap of [0, 6]) {
          const take = speech(pieces(n, gap));
          for (const step of STEPS) {
            for (const refinements of [DEFAULT, undefined]) {
              const t = recite(take, { passage: p, step, refinements });
              expect(darkSaid(take, t), `${n} phonemes kept, a ${gap}-frame gap, step ${step}: said, dark`).toEqual([]);
            }
          }
        }
      }
    });
  }
});

describe("refinements.begunByConsonants: a word cut after its first syllable", () => {
  // لِيَبْلُوَكُمْ (67:2:5) said to its first syllable and the next consonant (لِيَ), then a skip to 67:2:7: more than
  // its onset was heard, fewer than all its consonants. It goes dark; with begunByConsonants off, only a word's onset
  // counts as begun and it stays lit.
  const take = speech([say(67, 1), pause(15), say(67, 2, 1, 4), begin(67, 2, 5, 3), pause(10), say(67, 2, 7), pause(15), say(67, 3)]);
  it("goes dark when the run jumps past it", () => {
    for (const step of STEPS) {
      for (const refinements of [DEFAULT, undefined]) {
        const t = recite(take, { passage: passage(67, 1, 3), step, refinements });
        expect(drawn(t.positions).has("67:2:5"), `step ${step}: on the page at the end`).toBe(false);
        expect(darkSaid(take, t), `step ${step}: said, dark`).toEqual([]);
      }
      const onset = recite(take, { passage: passage(67, 1, 3), step, refinements: { ...ALL, begunByConsonants: false } });
      expect(drawn(onset.positions).has("67:2:5"), `step ${step}: on the page with begunByConsonants off`).toBe(true);
    }
  });
});

describe("refinements.withdrawBegun with declaredStart: a false start of the declared opening, then a restart late in the take", () => {
  // The take opens with a false start of the declared opening (ٱلْحَمْدُ لِلَّهِ ٱلَّذِى, the words 34:1 shares with 6:1
  // and 18:1), and the reciter starts again: the declared lock is let go, and global search finds the same first word
  // again. A run locked after a let-go is a run of its own, so when the reciter later goes back over 34:2's last words,
  // no word they said before is withdrawn. Read as the run before it, whose path was no longer kept, the restart
  // withdrew the 23 words said between, and they stayed dark.
  const take = speech([say(34, 1, 1, 3), pause(20), say(34, 1), pause(15), say(34, 2), pause(15), say(34, 2, 10), pause(15), say(34, 3)]);
  it("keeps every word said lit", () => {
    for (const step of STEPS) {
      for (const refinements of [DEFAULT, undefined]) {
        const t = recite(take, { passage: passage(34, 1, 3), step, refinements });
        expect(darkSaid(take, t), `step ${step}: said, dark`).toEqual([]);
        expect(darkSaid(take, { ...t, positions: t.shown }), `step ${step}: said, dark on the page shown`).toEqual([]);
      }
    }
  });
});

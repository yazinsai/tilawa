import { describe, expect, it } from "vitest";
import type { CheckerEvent, WordRef } from "../../src/engine/events.js";
import { ALL, DEFAULT, ayahs, key, noise, passage, pause, recite, say, speech, spread, without } from "./takes.js";
import type { Piece, Take } from "./takes.js";

/**
 * refinements.markWhenSure: in a declared passage, a suspected wrong or added word whose evidence has stood and is sure
 * is sent again with `due` ahead of its confirmation, and a suspicion still open when the take ends is decided on its
 * evidence: a slip on the last word before a stop is marked. Each take runs at 1 and 3 phonemes a push. The takes are
 * synthetic phoneme streams built from the reference (`./takes.ts`), with posteriors that favour what was said.
 */

const OFF = without("markWhenSure");
const STEPS = [1, 3];
/** The checker's events, as "state[+due] class word". */
const events = (t: Take) => t.checker.map((e) => `${e.state}${e.due ? "+due" : ""} ${e.errorClass} ${key(e.location)}`);
const dues = (t: Take) => t.checker.filter((e) => e.due);
/** The take with the field on and off. */
const both = (pieces: Piece[], p: [WordRef, WordRef] | null, step: number): [Take, Take] => {
  const take = speech(pieces);
  return [recite(take, { passage: p, step, refinements: ALL }), recite(take, { passage: p, step, refinements: OFF })];
};

describe("refinements.markWhenSure: a slip on the last word before a stop", () => {
  it("a wrong last word is marked at the stop, as it is with a word after it", () => {
    // 67:1 said to its last word, قَدِير said بَصِير (a word of the Quran, heard clearly), then the take ends.
    for (const step of STEPS) {
      const [on, off] = both([say(67, 1, 1, 8), noise("بَصِۦۦر")], passage(67, 1, 1), step);
      expect(on.mistakes, `step ${step}`).toEqual(["wrong_word 67:1:9"]);
      expect(off.mistakes, `step ${step}: off`).toEqual([]);
      expect(events(off), `step ${step}: off`).toEqual(["suspected wrong_word 67:1:9", "uncertain wrong_word 67:1:9"]);
      // The same slip with the next ayah after it is marked either way.
      const [mid, midOff] = both([say(67, 1, 1, 8), noise("بَصِۦۦر"), pause(15), say(67, 2)], passage(67, 1, 2), step);
      expect(mid.mistakes, `step ${step}: read on`).toEqual(["wrong_word 67:1:9"]);
      expect(midOff.mistakes, `step ${step}: read on, off`).toEqual(["wrong_word 67:1:9"]);
    }
  });

  it("an added word before the last word, then a stop, is marked", () => {
    for (const step of STEPS) {
      const [on, off] = both([say(67, 1), pause(15), say(67, 2, 1, 10), noise("شَدِۦۦد"), say(67, 2, 11, 11)], passage(67, 1, 2), step);
      expect(on.mistakes, `step ${step}`).toEqual(["added_word 67:2:11"]);
      expect(off.mistakes, `step ${step}: off`).toEqual([]);
    }
  });

  it("the passage's last word left out, then a stop, is marked; a stop further back is not", () => {
    // 67:1 declared: its last word, or its last two, left out; three left out is a stop. 67:1–2 declared: 67:2's last
    // word left out. 67:1–3 declared: a stop inside 67:2 leaves 67:3 to come.
    const takes: [Piece[], [WordRef, WordRef], string[]][] = [
      [[say(67, 1, 1, 8)], passage(67, 1, 1), ["omitted_word 67:1:9"]],
      [[say(67, 1, 1, 7)], passage(67, 1, 1), ["omitted_word 67:1:8"]],
      [[say(67, 1, 1, 6)], passage(67, 1, 1), []],
      [[say(67, 1), pause(15), say(67, 2, 1, 10)], passage(67, 1, 2), ["omitted_word 67:2:11"]],
      [[say(67, 1), pause(15), say(67, 2, 1, 10)], passage(67, 1, 3), []],
    ];
    for (const [pieces, p, marks] of takes) {
      for (const step of STEPS) {
        const [on, off] = both(pieces, p, step);
        const mode = `${key(p[0])}–${key(p[1])}, ${pieces.length} pieces, step ${step}`;
        expect(on.mistakes, mode).toEqual(marks);
        expect(off.mistakes, `${mode}: off`).toEqual([]);
      }
    }
  });

  it("a word changed by two consonants into no word near it is marked only as a confirmation would mark it", () => {
    // قَدِير said بَسِير: the posteriors favour it by more than a change that spells a nearby word needs, though by less
    // than a confirmation asks of one that spells none. Neither due nor decided at the stop: the field asks what a
    // confirmation asks of the same evidence, and off, it is lost at the stop and, with the next ayah after it, at its
    // confirmation.
    for (const step of STEPS) {
      const [on, off] = both([say(67, 1, 1, 8), noise("بَسِۦۦر")], passage(67, 1, 1), step);
      expect(on.mistakes, `step ${step}`).toEqual([]);
      expect(off.mistakes, `step ${step}: off`).toEqual([]);
      const [mid, midOff] = both([say(67, 1, 1, 8), noise("بَسِۦۦر"), pause(15), say(67, 2)], passage(67, 1, 2), step);
      expect(mid.mistakes, `step ${step}: read on`).toEqual([]);
      expect(dues(mid), `step ${step}: read on, due`).toEqual([]);
      expect(midOff.mistakes, `step ${step}: read on, off`).toEqual([]);
      // Heard below the confidence a mark needs: not marked, at the stop or read on.
      const [weak] = both([say(67, 1, 1, 8), noise("بَسِۦۦر", 0.5)], passage(67, 1, 1), step);
      expect(weak.mistakes, `step ${step}: unclear`).toEqual([]);
      const [weakMid] = both([say(67, 1, 1, 8), noise("بَسِۦۦر", 0.5), pause(15), say(67, 2)], passage(67, 1, 2), step);
      expect(weakMid.mistakes, `step ${step}: unclear, read on`).toEqual([]);
      expect(dues(weakMid), `step ${step}: unclear, read on, due`).toEqual([]);
    }
  });

  it("a correct last word is not marked", () => {
    const takes: [Piece[], [WordRef, WordRef]][] = [
      [[say(67, 1)], passage(67, 1, 1)],
      [ayahs(67, 1, 2), passage(67, 1, 2)],
      [[say(67, 1), pause(15), say(67, 2, 1, 6)], passage(67, 1, 3)],
      [ayahs(2, 1, 5), passage(2, 1, 5)],
      [ayahs(36, 1, 4), passage(36, 1, 4)],
      [ayahs(112, 1, 4), passage(112, 1, 4)],
    ];
    for (const [pieces, p] of takes) {
      for (const step of STEPS) {
        const [on, off] = both(pieces, p, step);
        expect(on.mistakes, `${key(p[0])}, step ${step}`).toEqual([]);
        expect(dues(on), `${key(p[0])}, step ${step}: due`).toEqual([]);
        expect(on.checker, `${key(p[0])}, step ${step}: events`).toEqual(off.checker);
      }
    }
  });
});

describe("refinements.markWhenSure: a mark before its confirmation", () => {
  it("a wrong word whose evidence has stood is due before it is confirmed", () => {
    // 67:1:9 said بَسِيل (three consonants changed), then 67:2.
    for (const step of STEPS) {
      const [on, off] = both([say(67, 1, 1, 8), noise("بَسِۦۦل"), pause(15), say(67, 2)], passage(67, 1, 2), step);
      const first = on.checker.find((e) => e.state === "suspected")!;
      const due = dues(on);
      const confirmed = on.checker.find((e) => e.state === "confirmed")!;
      expect(due.map((e) => `${e.errorClass} ${key(e.location)}`), `step ${step}`).toEqual(["wrong_word 67:1:9"]);
      expect(due[0].id, `step ${step}: the same mistake`).toBe(first.id);
      expect(due[0].timeMs - first.timeMs, `step ${step}: stood`).toBeGreaterThanOrEqual(800);
      expect(due[0].timeMs, `step ${step}: before its confirmation`).toBeLessThan(confirmed.timeMs);
      // The confirmation keeps rc.1's time.
      expect(confirmed.timeMs, `step ${step}`).toBe(off.checker.find((e) => e.state === "confirmed")!.timeMs);
      expect(on.mistakes).toEqual(off.mistakes);
      expect(dues(off), `step ${step}: off`).toEqual([]);
    }
  });

  it("free recitation keeps rc.1's rule", () => {
    for (const step of STEPS) {
      const [on, off] = both([say(67, 1, 1, 8), noise("بَسِۦۦل"), pause(15), say(67, 2)], null, step);
      expect(on.checker, `step ${step}`).toEqual(off.checker);
      const [end, endOff] = both([say(67, 1, 1, 8), noise("بَصِۦۦر")], null, step);
      expect(end.checker, `step ${step}: at the end`).toEqual(endOff.checker);
    }
  });

  it("a correct recitation at a varied pace sends no due mark", () => {
    for (const [s, a, b] of [[67, 1, 5], [18, 9, 12], [2, 1, 5], [36, 1, 6], [55, 10, 16]] as const) {
      for (const step of STEPS) {
        const [ms, gap] = [spread(s * 1000 + a + step, 60, 150), spread(s + b + step, 8, 25)];
        const pieces: Piece[] = [];
        for (let x = a; x <= b; x++) pieces.push(say(s, x), pause(Math.round(gap())));
        const take = speech(pieces, { ms, opening: () => 0.96 });
        const t = recite(take, { passage: passage(s, a, b), step, refinements: ALL });
        const mode = `${s}:${a}–${b}, step ${step}`;
        expect(t.checker.filter((e: CheckerEvent) => e.due), mode).toEqual([]);
        expect(t.mistakes, mode).toEqual([]);
      }
    }
  }, 300_000);

  it("is off by default", () => {
    expect(DEFAULT.markWhenSure).toBe(false);
    const take = speech([say(67, 1, 1, 8), noise("بَصِۦۦر")]);
    expect(recite(take, { passage: passage(67, 1, 1) }).mistakes).toEqual([]);
    expect(recite(take, { passage: passage(67, 1, 1), refinements: { ...DEFAULT, markWhenSure: true } }).mistakes).toEqual(["wrong_word 67:1:9"]);
  });
});

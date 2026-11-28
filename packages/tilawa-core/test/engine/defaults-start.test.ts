import { describe, expect, it } from "vitest";
import { DEFAULT, ayah, ayahs, drawn, guessed, passage, pause, recite, say, shownAt, speech } from "./takes.js";
import type { Fields, Piece, Speech, Take } from "./takes.js";

/**
 * A take's start with the default refinements (`declaredStart` on, with `declaredLeadIn` and `declaredMove`), in the
 * cases where the declared opening and the place the reciter starts at can be taken for each other: a lead-in from the
 * ayah before, whole or from inside it, a take that starts one ayah later than declared at an ayah that opens with the
 * declared opening's words, and another surah whose opening sounds like the declared one. A session passes its
 * resolved policy to the tracker and checker (`DEFAULT` here), and a tracker or checker given no refinements takes the
 * same defaults (no `refinements` here). Each take runs at 1 and 3 phonemes a push. The lead-in and later-start cases
 * also run with the switch that mends them off, to show the takes reach what it is for. The takes are synthetic phoneme
 * streams built from the reference (`./takes.ts`).
 */

/** The session's resolved refinements, and the tracker's and checker's own defaults. */
const defaults: [string, Fields | undefined][] = [["the default policy", DEFAULT], ["the engine's own defaults", undefined]];
const STEPS = [1, 3];

/** The words shown before they are said, or shown and never said, with how early. */
const lit = (take: Speech, shown: Map<string, number>) =>
  [...shown]
    .filter(([w, t]) => !take.saidAt.has(w) || t < take.saidAt.get(w)!)
    .map(([w, t]) => `${w} shown ${take.saidAt.has(w) ? `${take.saidAt.get(w)! - t} ms before it is said` : "but never said"}`);
/** The words of ayahs `a`–`b` of surah `s` said and dark on the page at the take's end. */
const dark = (take: Speech, t: Take, s: number, a: number, b: number) => {
  const page = drawn(t.shown);
  return [...take.saidAt.keys()].filter((k) => {
    const [x, y] = k.split(":").map(Number);
    return x === s && y >= a && y <= b && !page.has(k);
  });
};

/** A passage from ayah `x` to the next, or `x` alone when it is its surah's last. */
const from = (s: number, x: number) => passage(s, x, ayah(s, x + 1).length ? x + 1 : x);

describe("the default refinements, at a take's start", () => {
  it("show no word of the passage before it is said when the reciter leads in from inside the ayah before", () => {
    // The last words of the ayah before, or that ayah carried on after a breath inside it, whose sounds fit the declared
    // opening as a landing must. With declaredLeadIn off, the declared first word lights about 2 s early, and in 1:7,
    // 3:7 and 87:19 the page then steps back into the ayah before.
    const last = (s: number, x: number, n: number) => say(s, x, ayah(s, x).length - n + 1);
    const takes: [string, number, number, Piece[]][] = [
      ["1:7 after ٱهْدِنَا, a breath, ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ", 1, 7, [say(1, 6, 1, 1), pause(15), say(1, 6, 2), pause(15), say(1, 7)]],
      ["1:7 from ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ", 1, 7, [say(1, 6, 2), pause(15), say(1, 7)]],
      ["87:19 from ٱلصُّحُفِ ٱلْأُولَىٰ", 87, 19, [last(87, 18, 2), pause(15), say(87, 19)]],
      ["37:100 from رَبِّىٓ سَيَهْدِينِ", 37, 100, [last(37, 99, 2), pause(15), say(37, 100), pause(15), say(37, 101)]],
      ["3:7 from 3:6's last three words", 3, 7, [last(3, 6, 3), pause(15), say(3, 7), pause(15), say(3, 8), pause(15)]],
      ["59:14 from 59:13's last two words", 59, 14, [last(59, 13, 2), pause(15), say(59, 14), pause(15), say(59, 15), pause(15)]],
    ];
    let litOff = 0;
    for (const [name, s, x, pieces] of takes) {
      const take = speech(pieces);
      const p = from(s, x);
      for (const step of STEPS) {
        for (const [mode, refinements] of defaults) {
          const t = recite(take, { passage: p, checker: false, step, refinements });
          expect(lit(take, shownAt(t.positions)), `${name}, step ${step}, ${mode}`).toEqual([]);
          expect(dark(take, t, s, x, x + 1), `${name}, step ${step}, ${mode}: said, dark`).toEqual([]);
        }
        const off = recite(take, { passage: p, checker: false, step, refinements: { ...DEFAULT, declaredLeadIn: false } });
        if (lit(take, shownAt(off.positions)).length) litOff++;
      }
    }
    expect(litOff, "takes whose passage lights before it is said with declaredLeadIn off").toBeGreaterThan(0);
  }, 180_000);

  it("show no word of the passage before it is said when the reciter leads in with the ayah before, whose first word sounds alike", () => {
    // The ayah before opens with a word that differs from the declared ayah's first word in a sound or two:
    // ٱلَّذِينَ and وَٱلَّذِينَ (2:3, 2:4; 23:2, 23:3), فَأَمَّا and وَأَمَّا (3:56, 3:57), قَالَتْ and قَالَ (12:32, 12:33;
    // 19:19, 19:20), فَٱتَّقُوا۟ and وَٱتَّقُوا۟ (26:131, 26:132). It keeps the lock off until the speech tells them apart, and
    // the reciter is followed through the ayah before into the passage.
    for (const [s, x] of [[2, 4], [23, 3], [3, 57], [12, 33], [19, 20], [26, 132]] as const) {
      const take = speech(ayahs(s, x - 1, x + 1));
      for (const step of STEPS) {
        for (const [mode, refinements] of defaults) {
          const t = recite(take, { passage: from(s, x), checker: false, step, refinements });
          const name = `${s}:${x} after ${s}:${x - 1}, step ${step}, ${mode}`;
          expect(lit(take, shownAt(t.positions)).filter((w) => w.startsWith(`${s}:${x}:`)), name).toEqual([]);
          expect(guessed(take, drawn(t.shown)), `${name}: lit, never said`).toEqual([]);
          expect(dark(take, t, s, x, x + 1), `${name}: said, dark`).toEqual([]);
          expect(t.shown.filter((e) => e.state === "lost").map((e) => e.timeMs), `${name}: lost at`).toEqual([]);
        }
      }
    }
  }, 180_000);

  it("find a reciter who starts one ayah later than declared where they are", () => {
    // The passage is y−1..y+1 and the reciter starts at y, an ayah that opens with y−1's first word or words. The shared
    // words light as y−1's, the ayah next in order; where the two ayahs part, the lock moves to y on all the speech
    // since the onset, and y−1's words go dark. With declaredMove off, the reciter is found again only from what was
    // heard after the lock: 36:25 lights 4:4:14 and 4:5:1–2 and leaves 36:24 unmarked, 12:74 and 53:50 prompt
    // lost-place during correct recitation, 2:70 leaves its first six words dark.
    let wrongOff = 0;
    for (const [s, y] of [[36, 25], [12, 74], [53, 50], [2, 70]] as const) {
      const take = speech(ayahs(s, y, y + 1));
      const faults = (step: number, refinements: Fields | undefined) => {
        const t = recite(take, { passage: passage(s, y - 1, y + 1), step, refinements });
        const shown = shownAt(t.positions);
        return [
          ...t.positions.filter((e) => e.state === "lost").map((e) => `lost at ${e.timeMs} ms`),
          ...guessed(take, shown).map((w) => `${w} lit`),
          ...guessed(take, drawn(t.shown)).map((w) => `${w} on the page at the end`),
          ...[...take.saidAt.keys()].filter((w) => !shown.has(w)).map((w) => `${w} said, never shown`),
          ...dark(take, t, s, y, y + 1).map((w) => `${w} said, dark at the end`),
          ...(t.mistakes.includes(`skipped_passage ${s}:${y - 1}:1`) ? [] : [`${s}:${y - 1} left out, not marked`]),
        ];
      };
      for (const step of STEPS) {
        for (const [mode, refinements] of defaults) {
          expect(faults(step, refinements), `${s}:${y - 1}–${y + 1} started at ${s}:${y}, step ${step}, ${mode}`).toEqual([]);
        }
        if (faults(step, { ...DEFAULT, declaredMove: false }).length) wrongOff++;
      }
    }
    expect(wrongOff, "takes the reciter is not found in with declaredMove off").toBeGreaterThan(0);
  }, 300_000);

  it("show no word of the declared passage that was not said when the reciter says another surah whose opening sounds alike", () => {
    // 36:1–3 declared, 26:1, 27:1 or 28:1 said (طسٓمٓ, طسٓ against يسٓ): nothing of the declared passage shows. 82:1–2
    // declared, 84:1–3 said: the words they share (إِذَا ٱلسَّمَآءُ) light as the declared ayah's, by the rule for ayahs
    // that open alike, and where the words part, a rival far from the passage lets the lock go to global search.
    for (const [declared, said] of [[[36, 1, 3], 26], [[36, 1, 3], 27], [[36, 1, 3], 28], [[82, 1, 2], 84]] as const) {
      const take = speech([say(said, 1), pause(15), say(said, 2), pause(15), say(said, 3)]);
      for (const step of STEPS) {
        for (const [mode, refinements] of defaults) {
          const t = recite(take, { passage: passage(...declared), checker: false, step, refinements });
          const name = `declared ${declared[0]}:${declared[1]}–${declared[2]}, said ${said}:1–3, step ${step}, ${mode}`;
          expect(guessed(take, shownAt(t.positions)), name).toEqual([]);
          expect(guessed(take, drawn(t.shown)), `${name}: on the page at the end`).toEqual([]);
          if (declared[0] === 36) expect([...shownAt(t.positions).keys()].filter((k) => k.startsWith("36:")), `${name}: 36 lit`).toEqual([]);
        }
      }
    }
  }, 120_000);
});

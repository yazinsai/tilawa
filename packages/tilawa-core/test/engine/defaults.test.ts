import { describe, expect, it } from "vitest";
import { DEFAULTS } from "../../src/defaults.js";
import { resolvePolicy } from "../../src/session.js";
import { DEFAULT, RC1, ayah, ayahs, index, passage, pause, recite, say, shownAt, speech } from "./takes.js";
import type { Fields } from "./takes.js";

/**
 * The default refinements: every one on but `markWhenSure`, which stays off while it gets cases wrong that it cannot
 * yet tell from the ones it is for (its doc). The take's start with `declaredStart` on is in `./defaults-start.test.ts`
 * and `./defaults-later.test.ts`.
 * The cases earlier defaults got wrong, each against the defaults: a session passes its resolved policy to the tracker
 * and checker (`DEFAULT` here), and a tracker or checker given no refinements takes the same defaults (no `refinements`
 * here). The takes are synthetic phoneme streams built from the reference (`./takes.ts`).
 */

/** The session's resolved refinements, and the tracker's and checker's own defaults. */
const defaults: [string, Fields | undefined][] = [["the default policy", DEFAULT], ["the engine's own defaults", undefined]];

describe("the default refinements", () => {
  it("are every refinement on but markWhenSure", () => {
    const want = {
      readNotSkipped: true, reanchorBack: true, stutter: true, ledgerPartial: true, omissionDue: true,
      commitAfterJump: true, firstAyahOut: true, marksInPassage: true, repeatNotWrong: true, markWhenSure: false,
      confirmAdvance: true, onsetRun: true, declaredStart: true, declaredLeadIn: true, declaredMove: true, noLostBeforeLock: true,
      consonantConfirm: true, rivalHold: true, passageWindow: true, restartNoEntry: true, withdrawBegun: true, leftOnNewAudio: true,
      begunByConsonants: true, showSaid: true, pace: true,
    };
    expect(DEFAULTS.policy.refinements).toEqual(want);
    expect(resolvePolicy({}, 40).refinements).toEqual(want);
    expect(Object.isFrozen(DEFAULTS.policy.refinements)).toBe(true);
  });

  it("show no word of a declared passage when another surah, whose opening sounds like it, is said", () => {
    // 36:1 (يَااسِۦۦۦۦۦۦن) and 26:1, 27:1, 28:1 (طَااسِ…) differ in their first sound; 19:1 and 50:1 open on كاف and
    // قاف. The reciter declares one and says the other: no word of the declared passage was said.
    for (const [declared, said] of [[36, 26], [36, 27], [36, 28], [19, 50]]) {
      const s = speech([say(said, 1), pause(15), say(said, 2)]);
      for (const step of [1, 3]) {
        for (const [name, refinements] of defaults) {
          const t = recite(s, { passage: passage(declared, 1, 3), checker: false, step, refinements });
          const unsaid = [...shownAt(t.positions).keys()].filter((k) => !s.saidAt.has(k));
          expect(unsaid, `declared ${declared}:1–3, said ${said}:1–2, step ${step}, ${name}`).toEqual([]);
        }
      }
    }
  }, 60_000);

  it("show no word before it is said when the reciter leads in from the ayah before, whose first word differs", () => {
    // The ayah before the declared passage opens with a different word whose first sounds fit the declared opening
    // within an edit (الذين before والذين, والشمس before والقمر, لا before ولا, oaths that open alike). Reading it to
    // lead into the passage is common. The declared opening stays dark until it is said; its first word shows once it
    // is said, before its next word is.
    const leadIns = [[2, 4], [91, 2], [91, 3], [75, 2], [79, 2], [100, 3], [86, 2], [89, 2], [51, 2]] as const;
    const text = (wi: number) => Array.from(index.phonemes.slice(index.words[wi].start, index.words[wi].end)).join(" ");
    for (const [s, x] of leadIns) {
      expect(text(ayah(s, x - 1)[0]), `${s}:${x - 1} and ${s}:${x} open with different words`).not.toEqual(text(ayah(s, x)[0]));
      const take = speech(ayahs(s, x - 1, x + 2));
      const second = take.saidAt.get(ayah(s, x).length > 1 ? `${s}:${x}:2` : `${s}:${x + 1}:1`)!;
      for (const step of [1, 3]) {
        for (const [name, refinements] of defaults) {
          const mode = `declared ${s}:${x}–${x + 2}, read from ${s}:${x - 1}, step ${step}, ${name}`;
          const shown = shownAt(recite(take, { passage: passage(s, x, x + 2), checker: false, step, refinements }).positions);
          expect([...take.saidAt].filter(([k, at]) => shown.has(k) && shown.get(k)! < at).map(([k]) => k), mode).toEqual([]);
          expect(shown.get(`${s}:${x}:1`)!, `${mode}: the first word shows before the next is said`).toBeLessThan(second);
        }
      }
    }
  }, 180_000);

  it("mark an ayah left out of a short re-reading, as rc.1 does", () => {
    // 55:1–12, back to 55:8 (22 words, one pass), then 55:8–14 without 55:10.
    const s = speech([...ayahs(55, 1, 12), pause(30), ...ayahs(55, 8, 14, [10])]);
    for (const p of [passage(55, 1, 14), null]) {
      const mode = p ? "declared" : "free";
      const rc1 = recite(s, { passage: p, refinements: RC1 }).mistakes;
      expect(rc1, `${mode}, rc.1`).toContain("skipped_passage 55:10:1");
      for (const [name, refinements] of defaults) expect(recite(s, { passage: p, refinements }).mistakes, `${mode}, ${name}`).toEqual(rc1);
    }
  }, 60_000);
});

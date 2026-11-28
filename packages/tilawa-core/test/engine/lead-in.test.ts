import { describe, expect, it } from "vitest";
import { ALL, ayah, ayahs, guessed, key, passage, recite, ref, shownAt, speech, without } from "./takes.js";

/**
 * refinements.declaredStart: reciting the ayah before the passage, to lead into it, is common. When it opens with the
 * same words as the passage, the speech so far fits it as well as the declared ayah. Of ayahs that open with the same
 * words, the declared (in-order) one lights as they are said, but no word of the passage shows that the ayah before
 * does not read the same; where the two part, the tracker finds the reciter in the ayah before. The takes are synthetic
 * phoneme streams built from the reference (`./takes.ts`).
 */

describe("refinements.declaredStart: a reciter who starts one ayah early", () => {
  // 2:149 and 2:150 share their first eight words, 82:3 and 82:4 their first, 2:69 and 2:70 their first seven.
  for (const [s, a, b] of [[2, 150, 152], [82, 4, 8], [2, 70, 72]] as [number, number, number][]) {
    it(`${s}:${a}–${b} started at ${s}:${a - 1}: nothing of the passage shows before ${s}:${a}:1 is said but on the same words`, () => {
      const take = speech(ayahs(s, a - 1, b));
      const said = take.saidAt.get(`${s}:${a}:1`)!;
      for (const step of [1, 3]) {
        const before = (refinements = ALL) =>
          recite(take, { passage: passage(s, a, b), checker: false, step, refinements })
            .positions.filter((p) => p.state === "locked" && p.timeMs < said)
            .map((p) => key(p.observed[0].word));
        const shown = shownAt(recite(take, { passage: passage(s, a, b), checker: false, step, refinements: ALL }).positions);
        expect(guessed(take, shown), `step ${step}`).toEqual([]);
        // Where the two part, the ayah before is followed: its last word shows before the passage is said.
        const last = key(ref(ayah(s, a - 1).at(-1)!));
        expect(before(), `step ${step}: the ayah before is followed`).toContain(last);
        expect(before(without("declaredStart")).filter((w) => !w.startsWith(`${s}:${a - 1}:`)), `step ${step}`).toEqual([]);
      }
    }, 30_000);
  }
});

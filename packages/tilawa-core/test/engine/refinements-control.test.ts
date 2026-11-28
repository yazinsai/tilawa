import { describe, expect, it } from "vitest";
import type { WordRef } from "../../src/engine/events.js";
import { ALL, DEFAULT, at, ayah, ayahs, key, passage, recite, ref, shownAt, speech } from "./takes.js";
import type { Fields } from "./takes.js";

/**
 * Correct recitations with every refinement on, and with the defaults (markWhenSure off), declared and free: no
 * checker event, no lost place, every word shown, none before it was said, and the shown word never stepping back. The
 * passages are short ayahs and long ones, ayahs of one word, sun-letter onsets (a doubled first sound), al-Fatiha, and
 * refrains repeated through a surah. The takes are synthetic phoneme streams built from the reference (`./takes.ts`).
 */

const passages: [number, number, number][] = [
  [1, 1, 7], [2, 1, 5], [36, 6, 9], [55, 1, 8], [55, 10, 28], [77, 14, 30], [81, 1, 8], [82, 1, 5], [91, 1, 6], [112, 1, 4],
];

describe("refinements, a wider control", () => {
  for (const [s, a, b] of passages) {
    it(`${s}:${a}–${b} recited right is silent and shown in order, declared and free, every refinement on and the defaults`, () => {
      const said = speech(ayahs(s, a, b));
      const declared = passage(s, a, b);
      const runs: [[WordRef, WordRef] | null, Fields][] = [[declared, ALL], [declared, DEFAULT], [null, ALL], [null, DEFAULT]];
      for (const [p, refinements] of runs) {
        const mode = `${p ? "declared" : "free"}, ${refinements === ALL ? "every refinement on" : "the defaults"}`;
        const take = recite(said, { passage: p, refinements });
        expect(take.checker, mode).toEqual([]);
        expect(take.positions.filter((q) => q.state === "lost"), mode).toEqual([]);
        const shown = shownAt(take.positions);
        for (let x = a; x <= b; x++) for (const wi of ayah(s, x)) {
          const k = key(ref(wi));
          // A basmala is a phrase that opens a surah, al-Fatiha's too: rc.1 shows the surah from 1:2, in free recitation
          // and in a declared passage without declaredStart, which locks on a declared 1:1.
          if ((!p || !refinements.declaredStart) && s === 1 && x === 1) continue;
          expect(shown.get(k), `${mode} ${k} shown`).toBeDefined();
          expect(shown.get(k)!, `${mode} ${k} not before it is said`).toBeGreaterThanOrEqual(said.saidAt.get(k)!);
        }
        const locked = take.positions.filter((q) => q.state === "locked").map((q) => at(q.observed[0].word));
        for (let k = 1; k < locked.length; k++) expect(locked[k], `${mode} no step back`).toBeGreaterThanOrEqual(locked[k - 1]);
      }
    });
  }
});

import { describe, expect, it } from "vitest";
import type { WordRef } from "../../src/engine/events.js";
import { ALL, ayah, ayahs, key, passage, pause, recite, ref, say, shownAt, speech, without } from "./takes.js";
import type { Take } from "./takes.js";

/**
 * refinements.passageWindow: in a declared passage no jump of the tracker's local search lands past its end, so a
 * look-alike after it takes no lock on a sound or two; reading on out of the passage is followed as without it.
 * refinements.restartNoEntry: starting a surah again from inside it is a plain jump, not a surah entry. The takes are
 * synthetic phoneme streams built from the reference (`./takes.ts`).
 */

const order = (a: WordRef, b: WordRef) => a.surah - b.surah || a.ayah - b.ayah || a.word - b.word;
/** The locked positions outside passage `p`, as "word@ms". */
const outside = (t: Take, p: [WordRef, WordRef]) =>
  t.positions
    .filter((q) => q.state === "locked" && q.observed[0] && (order(q.observed[0].word, p[0]) < 0 || order(q.observed[0].word, p[1]) > 0))
    .map((q) => `${key(q.observed[0].word)}@${q.timeMs}`);

describe("refinements.passageWindow: no jump out of the declared passage", () => {
  // The reciter reads the passage and says its last ayah again from its middle. A phrase of the ayah after the passage
  // fits the words said again: without the bound, the local search jumps there on a few sounds and lights it.
  for (const [s, b, k] of [[7, 13, 7], [18, 9, 5]] as const) {
    it(`${s}:${b - 2}–${b}, then ${s}:${b} again from its word ${k}: nothing outside the passage is shown, nothing marked`, () => {
      const p = passage(s, b - 2, b);
      const take = speech([say(s, b - 2), pause(15), say(s, b - 1), pause(15), say(s, b), pause(15), say(s, b, k), pause(15)]);
      const now = recite(take, { passage: p, refinements: ALL, step: 3 });
      expect(outside(now, p)).toEqual([]);
      expect(now.mistakes).toEqual([]);
      const before = recite(take, { passage: p, refinements: without("passageWindow"), step: 3 });
      expect(outside(before, p)).not.toEqual([]);
      expect(before.mistakes).toEqual([]);
    });
  }

  /** The shown word's moves back, as "from->to@ms". */
  const backs = (t: Take) => {
    const locked = t.positions.filter((q) => q.state === "locked" && q.observed[0]);
    return locked.slice(1).flatMap((q, i) => {
      const [from, to] = [locked[i].observed[0].word, q.observed[0].word];
      return order(to, from) < 0 ? [`${key(from)}->${key(to)}@${q.timeMs}`] : [];
    });
  };
  // The risk: reading on past the end is common. The search follows the reciter out
  // of the passage by reading on, as without the field: the shown word never moves back into the passage, no word
  // lights before it is said, and every word said shows, the ones past the end too (outside the passage, where the
  // app says the reciter has left it).
  for (const [s, a, b, c] of [[67, 1, 3, 6], [2, 1, 3, 6], [18, 1, 3, 6], [36, 1, 5, 9], [20, 1, 5, 9], [55, 1, 4, 8], [56, 1, 6, 12], [78, 1, 5, 10]] as const) {
    it(`${s}:${a}–${b} declared, ${s}:${a}–${c} read in order: followed past the end, never back`, () => {
      const take = speech(ayahs(s, a, c));
      const p = passage(s, a, b);
      for (const step of [1, 3]) {
        const now = recite(take, { passage: p, refinements: ALL, step });
        const before = shownAt(recite(take, { passage: p, refinements: without("passageWindow"), checker: false, step }).positions);
        const shown = shownAt(now.positions);
        expect(backs(now), `step ${step}`).toEqual([]);
        expect(now.mistakes, `step ${step}`).toEqual([]);
        for (const [w, said] of take.saidAt) {
          expect(shown.get(w), `${w}, step ${step}`).toBeDefined();
          expect(shown.get(w)!, `${w}, step ${step}`).toBeGreaterThanOrEqual(said);
          expect(shown.get(w)!, `${w}, step ${step}`).toBeLessThanOrEqual(before.get(w)!);
        }
        expect(outside(now, p).length, `step ${step}`).toBeGreaterThan(0);
      }
    });
  }

  it("a take that starts before the declared passage is followed there, as without it", () => {
    const take = speech(ayahs(67, 1, 5));
    const p = passage(67, 3, 5);
    for (const step of [1, 3]) {
      const now = shownAt(recite(take, { passage: p, refinements: ALL, checker: false, step }).positions);
      const before = shownAt(recite(take, { passage: p, refinements: without("passageWindow"), checker: false, step }).positions);
      for (const wi of [...ayah(67, 1), ...ayah(67, 2)]) expect(now.get(key(ref(wi))), `${key(ref(wi))}, step ${step}`).toBe(before.get(key(ref(wi))));
    }
  });

  // The other case: 67:3–6 declared, the reciter reads 67:3–4, goes back to the ayah before the passage
  // and reads 67:2–6. The local search follows the go-back (a jump back, as without the field), so `expected` keeps up
  // with the word shown.
  it("a go-back to the ayah before the passage: expected keeps up with the word shown", () => {
    const take = speech([...ayahs(67, 3, 4), pause(30), ...ayahs(67, 2, 6)]);
    for (const step of [1, 3]) {
      const now = recite(take, { passage: passage(67, 3, 6), refinements: ALL, step });
      const locked = now.positions.filter((q) => q.state === "locked" && q.observed[0]);
      expect(locked.filter((q) => !q.expected || key(q.expected) !== key(q.observed[0].word)).map((q) => key(q.observed[0].word)), `step ${step}`).toEqual([]);
      expect(key(locked.at(-1)!.observed[0].word), `step ${step}`).toBe("67:6:7");
      expect(now.mistakes, `step ${step}`).toEqual([]);
    }
  });
});

describe("refinements.restartNoEntry: a restart is not a surah entry", () => {
  /** The first locked position on the surah's first ayah once the restart has begun (ms after it), or Infinity. */
  const back = (t: Take, s: number, from: number) => {
    const q = t.positions.find((x) => x.state === "locked" && x.timeMs >= from && x.observed[0]?.word.surah === s && x.observed[0].word.ayah === 1);
    return q ? q.timeMs - from : Infinity;
  };
  for (const [s, u] of [[106, 2], [109, 1], [111, 2], [112, 1], [113, 1]] as const) {
    it(`${s}:1–${u}, then ${s}:1–3 from the top: followed back to ${s}:1 sooner`, () => {
      const first = [...ayahs(s, 1, u), pause(20)];
      const from = speech(first).phonemes.length;
      const take = speech([...first, ...ayahs(s, 1, 3)]);
      const restart = take.phonemes[from].start * 40;
      for (const p of [null, passage(s, 1, 4)]) {
        const mode = p ? "declared" : "free";
        const [now, before] = [ALL, without("restartNoEntry")].map((f) => recite(take, { passage: p, refinements: f, checker: false }));
        expect(back(now, s, restart), mode).toBeLessThan(back(before, s, restart));
        expect(back(now, s, restart), mode).toBeLessThan(1000);
      }
    });
  }

  it("entering a surah in order costs what it did: free recitation reads on into the next surah as without it", () => {
    for (const [s, a] of [[105, 4], [112, 3], [108, 2]] as const) {
      const take = speech([...ayahs(s, a, a + 1), ...ayahs(s + 1, 1, 2)]);
      const run = (f: typeof ALL | ReturnType<typeof without>) => recite(take, { passage: null, refinements: f, checker: false }).positions;
      expect(run(ALL), `${s}:${a} into ${s + 1}:1`).toEqual(run(without("restartNoEntry")));
    }
  });
});

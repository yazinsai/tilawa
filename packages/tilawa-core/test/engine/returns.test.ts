import { describe, expect, it } from "vitest";
import { ALL, EARLIER_ON, ayahs, earlierWithout, passage, pause, recite, say, speech, without } from "./takes.js";

/**
 * refinements.readNotSkipped: where a pass begins after the reciter goes back. Going back starts a new pass only once the
 * ayah gone back to is read through; a few words to find the place, then on to where they had been, is still the same
 * pass. The takes are synthetic phoneme streams built from the reference (`./takes.ts`).
 */

describe("refinements.readNotSkipped: going back", () => {
  it("back to the passage's first ayah for a few words, a long pause, then on from a later ayah read before: nothing left out", () => {
    // The reciter goes back to find the place, says the first ayah's first five words, stops, and goes on from the
    // third ayah: the words between were read in the pass.
    for (const [s, a, b] of [[18, 1, 12], [36, 13, 30]] as [number, number, number][]) {
      const take = speech([...ayahs(s, a, a + 7), pause(30), say(s, a, 1, 5), pause(200), ...ayahs(s, a + 2, a + 3)]);
      for (const step of [1, 3]) expect(recite(take, { passage: passage(s, a, b), step, refinements: ALL }).mistakes, `${s}:${a}, step ${step}`).toEqual([]);
      expect(recite(take, { passage: passage(s, a, b), refinements: without("readNotSkipped") }).mistakes, `${s}:${a}`).toEqual([`skipped_passage ${s}:${a}:6`]);
    }
  }, 120_000);

  it("the same inside a short passage", () => {
    const s = speech([...ayahs(67, 1, 5), pause(30), say(67, 1, 1, 3), pause(30), ...ayahs(67, 2, 5)]);
    expect(recite(s, { passage: passage(67, 1, 5), refinements: ALL }).mistakes).toEqual([]);
    // The tracker's later corrections (`consonantConfirm`, `rivalHold`, `passageWindow`) each also mend this return, as
    // the checker follows it: with none of them and without this one, the words between are marked.
    expect(recite(s, { passage: passage(67, 1, 5), refinements: EARLIER_ON }).mistakes).toEqual([]);
    expect(recite(s, { passage: passage(67, 1, 5), refinements: earlierWithout("readNotSkipped") }).mistakes).toEqual(["skipped_passage 67:1:4"]);
  });

  it("back behind the window, the ayah read through, then a jump ahead: a new pass, and the words jumped are left out", () => {
    const s = speech([...ayahs(67, 1, 12), pause(30), ...ayahs(67, 3, 5), pause(15), ...ayahs(67, 8, 12)]);
    expect(recite(s, { passage: passage(67, 1, 12), refinements: ALL }).mistakes).toEqual(["skipped_passage 67:6:1"]);
  });
});

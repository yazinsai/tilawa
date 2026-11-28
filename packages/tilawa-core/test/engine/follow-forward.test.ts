import { describe, expect, it } from "vitest";
import { ALL, ayahs, key, passage, pause, recite, say, speech, without } from "./takes.js";

/**
 * refinements.reanchorBack, forward again: once the window has followed a reciter back behind it, a return to where
 * they had been is followed too. Coming back to the furthest ayah read is a rejoin and marks nothing; going on past it
 * leaves out what lies between. A slide into an earlier copy of the ayah just finished, which the reciter corrects
 * alone, is not an error, and a deliberate revision that goes back to just after such a copy cannot be told from it:
 * neither is marked. The takes are synthetic phoneme streams built from the reference (`./takes.ts`).
 */

/** Mistakes and every decided checker event, as "class word". */
const marks = (t: ReturnType<typeof recite>) => ({
  mistakes: t.mistakes,
  decided: t.checker.filter((e) => e.state === "confirmed").map((e) => `${e.errorClass} ${key(e.location)}`),
});

describe("refinements.reanchorBack: back behind the window, then forward again", () => {
  it("67:1–10, back to 67:3 read through, then on at 67:11: nothing marked on the words read right, nor on 67:11", () => {
    const s = speech([...ayahs(67, 1, 10), pause(30), ...ayahs(67, 3, 3), pause(15), ...ayahs(67, 11, 13)]);
    expect(marks(recite(s, { passage: passage(67, 1, 15), refinements: ALL }))).toEqual({ mistakes: [], decided: [] });
    // rc.1's rule: the window, followed back, reads the return as the ayahs after 67:3.
    expect(recite(s, { passage: passage(67, 1, 15), refinements: without("reanchorBack") }).mistakes.length).toBeGreaterThan(0);
  });

  it("67:1–10, back to 67:3 for five words (not read through), then on at 67:11: nothing marked", () => {
    const s = speech([...ayahs(67, 1, 10), pause(30), say(67, 3, 1, 5), pause(30), ...ayahs(67, 11, 13)]);
    expect(marks(recite(s, { passage: passage(67, 1, 15), refinements: ALL }))).toEqual({ mistakes: [], decided: [] });
  });

  it("36:1–14, back to 36:5–6, then on at 36:15: nothing marked on 36:7 or anywhere", () => {
    const s = speech([...ayahs(36, 1, 14), pause(30), ...ayahs(36, 5, 6), pause(15), ...ayahs(36, 15, 17)]);
    expect(marks(recite(s, { passage: passage(36, 1, 20), refinements: ALL }))).toEqual({ mistakes: [], decided: [] });
  });

  it("going on past where the reciter had been leaves out the ayah between", () => {
    // 67:1–10, back to 67:3 read through, then on at 67:12: 67:11 was never read.
    const s = speech([...ayahs(67, 1, 10), pause(30), ...ayahs(67, 3, 3), pause(15), ...ayahs(67, 12, 14)]);
    expect(recite(s, { passage: passage(67, 1, 15), refinements: ALL }).mistakes).toEqual(["skipped_passage 67:11:1"]);
  });
});

describe("refinements.reanchorBack: a slide into an earlier copy of the ayah just finished, corrected alone", () => {
  it("26:105–127, then 26:111–112 in place of 26:128, then back at 26:128: nothing marked on 26:113–114 or 26:128", () => {
    // 26:127 is 26:109's copy; the reciter went on as 26:109 goes on, into Nuh's people's reply (26:111–112), and came
    // back alone. Self-correction is no error.
    const s = speech([...ayahs(26, 105, 127), say(26, 111), pause(15), say(26, 112), pause(30), ...ayahs(26, 128, 131)]);
    for (const step of [1, 3, 8]) {
      const t = recite(s, { passage: passage(26, 105, 135), step, refinements: ALL });
      expect(t.checker.filter((e) => /^26:(11[34]|128):/.test(key(e.location)) && e.state === "confirmed"), `step ${step}`).toEqual([]);
      expect(t.mistakes, `step ${step}`).toEqual([]);
    }
  }, 120_000);

  it("55:14–25, back to 55:19–20 (just after the refrain's copy 55:18), then on at 55:26: a revision, nothing marked at any push size", () => {
    const s = speech([...ayahs(55, 14, 25), pause(30), ...ayahs(55, 19, 20), pause(30), ...ayahs(55, 26, 30)]);
    for (let step = 1; step <= 8; step++) expect(marks(recite(s, { passage: passage(55, 14, 30), step, refinements: ALL })), `step ${step}`).toEqual({ mistakes: [], decided: [] });
    expect(recite(s, { passage: null, refinements: ALL }).mistakes, "free").toEqual([]);
  }, 240_000);
});

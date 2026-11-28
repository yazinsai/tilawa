import { describe, expect, it } from "vitest";
import { ALL, ayahs, passage, pause, recite, say, speech, without } from "./takes.js";

/**
 * refinements.readNotSkipped: a pause alone excuses nothing. Forget, hesitate, skip is the most common omission in
 * hifz, in a re-reading too: the reciter goes back an ayah, stalls on a word and goes on inside the ayah without it.
 * Only resuming where reading on from the furthest word read goes, or at an ayah's first word after breaking off,
 * rejoins. The takes are synthetic phoneme streams built from the reference (`./takes.ts`).
 */

describe("refinements.readNotSkipped: a word left out after a hesitation in a re-reading", () => {
  // Forget, hesitate, skip: the reciter stalls on a word and goes on inside the ayah. A pause is not a rejoin; only
  // resuming where reading on from the furthest word goes, or at an ayah's first word, is. Free and declared.
  for (const gap of [8, 12, 30]) {
    it(`67:1–3, back to 67:2, then 67:3 with a hesitation (${gap} frames) and its third word left out: marked`, () => {
      const s = speech([...ayahs(67, 1, 3), pause(30), say(67, 2), pause(15), say(67, 3, 1, 2), pause(gap), say(67, 3, 4), pause(15), ...ayahs(67, 4, 5)]);
      for (const p of [passage(67, 1, 5), null]) {
        expect(recite(s, { passage: p, refinements: ALL }).mistakes, p ? "declared" : "free").toEqual(["omitted_word 67:3:3"]);
        expect(recite(s, { passage: p, refinements: without("readNotSkipped") }).mistakes, p ? "declared" : "free").toEqual(["omitted_word 67:3:3"]);
      }
    });

    it(`36:6–8, back to 36:7, then 36:8 with a hesitation (${gap} frames) and its second word left out: marked`, () => {
      const s = speech([...ayahs(36, 6, 8), pause(30), say(36, 7), pause(15), say(36, 8, 1, 1), pause(gap), say(36, 8, 3), pause(15), ...ayahs(36, 9, 10)]);
      for (const p of [passage(36, 6, 10), null]) {
        expect(recite(s, { passage: p, refinements: ALL }).mistakes, p ? "declared" : "free").toEqual(["omitted_word 36:8:2"]);
        expect(recite(s, { passage: p, refinements: without("readNotSkipped") }).mistakes, p ? "declared" : "free").toEqual(["omitted_word 36:8:2"]);
      }
    });
  }
});

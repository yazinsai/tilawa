import { describe, expect, it } from "vitest";
import { ALL, RC1, ayahs, noise, passage, pause, phrases, recite, say, speech, without } from "./takes.js";

/**
 * refinements.readNotSkipped: a passage read again in the same session is a new pass, and what it leaves out is left
 * out, whatever the first pass read. The takes are synthetic phoneme streams built from the reference (`./takes.ts`).
 */

describe("refinements.readNotSkipped: each pass has its own omissions", () => {
  it("the passage read again is a new pass: a word it leaves out is caught", () => {
    // 67:1–5 whole, then again from 67:1 with 67:4:3 left out.
    const s = speech([...ayahs(67, 1, 5), pause(30), ...ayahs(67, 1, 3), say(67, 4, 1, 2), say(67, 4, 4), pause(15), say(67, 5)]);
    for (const step of [1, 3]) expect(recite(s, { passage: passage(67, 1, 5), step, refinements: ALL }).mistakes, `step ${step}`).toEqual(["omitted_word 67:4:3"]);
    expect(recite(s, { passage: passage(67, 1, 5), refinements: without("readNotSkipped") }).mistakes).toEqual(["omitted_word 67:4:3"]);
  });

  it("the passage read again is a new pass: an ayah it leaves out is caught", () => {
    // 67:1–5 whole, then again with 67:3 left out.
    const s = speech([...ayahs(67, 1, 5), pause(30), ...ayahs(67, 1, 5, [3])]);
    for (const step of [1, 3]) expect(recite(s, { passage: passage(67, 1, 5), step, refinements: ALL }).mistakes, `step ${step}`).toEqual(["skipped_passage 67:3:1"]);
    expect(recite(s, { passage: passage(67, 1, 5), refinements: without("readNotSkipped") }).mistakes).toEqual(["skipped_passage 67:3:1"]);
  });

  // 67:1–5, a pause, 67:1–3 again, then on at 67:6, the ayah after the furthest one read to
  // its end. In free recitation the passage read so far slid with the window, so the return to 67:1 read as a jump and
  // 67:4–5 as left out.
  const revision = () => speech([...ayahs(67, 1, 5), pause(30), ...ayahs(67, 1, 3), pause(15), ...ayahs(67, 6, 7)]);
  for (const declared of [true, false]) {
    it(`a revision from the top that goes on where the reciter had read to marks nothing (${declared ? "declared" : "free recitation"})`, () => {
      const [s, p] = [revision(), declared ? passage(67, 1, 7) : null];
      for (const step of [1, 3]) expect(recite(s, { passage: p, step, refinements: ALL }).mistakes, `step ${step}`).toEqual([]);
      expect(recite(s, { passage: p, refinements: without("readNotSkipped") }).mistakes).toEqual(["skipped_passage 67:4:1"]);
    });
  }

  it("the passage read again in free recitation is a new pass too: an ayah it leaves out is caught", () => {
    const s = speech([...ayahs(67, 1, 5), pause(30), ...ayahs(67, 1, 5, [3])]);
    for (const step of [1, 3]) expect(recite(s, { passage: null, step, refinements: ALL }).mistakes, `step ${step}`).toEqual(["skipped_passage 67:3:1"]);
  });

  it("al-Fatiha in a second rak'ah, in free recitation, is a new pass: an ayah it leaves out is caught", () => {
    const takbir = phrases.phrases.find((p) => p.name === "takbir")!;
    const afterTakbir = speech([...ayahs(1, 1, 7), pause(60), noise(takbir.p), pause(30), ...ayahs(1, 1, 7, [5])]);
    for (const step of [1, 3]) expect(recite(afterTakbir, { passage: null, step, refinements: ALL }).mistakes, `step ${step}`).toEqual(["skipped_passage 1:5:1"]);
    expect(recite(afterTakbir, { passage: null, refinements: RC1 }).mistakes).toEqual(["skipped_passage 1:5:1"]);
    // No takbir heard: going back to the surah's first ayah starts the pass.
    const afterPause = speech([...ayahs(1, 1, 7), pause(60), ...ayahs(1, 1, 7, [5])]);
    expect(recite(afterPause, { passage: null, refinements: ALL }).mistakes).toEqual(["skipped_passage 1:5:1"]);
  });
});

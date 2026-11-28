import { describe, expect, it } from "vitest";
import { ANSWER_FRAMES, scriptedTake } from "./scripted.js";

/**
 * refinements.confirmAdvance in a session: a clean ayah opening after a pause shows when rc.1 shows it, wherever the
 * model's answers cut the audio, so also when the opening's first two sounds fall in different answers.
 */

describe("refinements.confirmAdvance: an ayah's first word in a session", () => {
  it("shows at rc.1's push whatever the answer boundary, at pushes of 16 and 160 ms", async () => {
    let split = 0;
    for (const pushMs of [16, 160]) {
      // Each silence length moves the next ayah's first sound by a frame against the answers' boundaries.
      for (let gap = 15; gap < 15 + ANSWER_FRAMES; gap++) {
        const take = (confirmAdvance: boolean) => scriptedTake({ pushMs, policy: { refinements: { confirmAdvance } }, surah: 112, ayahs: [1, 2, 3], gap });
        const [now, rc1] = [await take(true), await take(false)];
        for (const k of ["112:2:1", "112:3:1"]) {
          const said = rc1.words.get(k)!.said / 40;
          if (said % ANSWER_FRAMES === ANSWER_FRAMES - 1) split++;
          expect(now.words.get(k)?.shown, `${k}, ${pushMs} ms pushes, gap ${gap}`).toBe(rc1.words.get(k)!.shown);
        }
      }
    }
    expect(split, "some opening's first two sounds fell in different answers").toBeGreaterThan(0);
  }, 120_000);
});

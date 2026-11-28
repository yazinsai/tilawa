import { describe, expect, it } from "vitest";
import { ALL, RC1, ayah, ayahs, passage, pause, recite, say, speech, without } from "./takes.js";

/**
 * refinements.readNotSkipped: an omission is excused only where the reading after it rejoins. Going back an ayah or two
 * and reading again is the core move of hifz practice: a word that continuous speech runs straight over in the
 * re-reading is left out, even though the first reading said it. A pass is set by how far back the reciter went, the
 * same in free recitation and in a declared passage. The takes are synthetic phoneme streams built from the reference
 * (`./takes.ts`). A word left out after a hesitation is in `hesitations.test.ts`.
 */

describe("refinements.readNotSkipped: a word left out in a re-reading", () => {
  it("67:1–3, back to 67:2, then 67:3 without its third word: the word is left out, as rc.1 marks it", () => {
    const s = speech([...ayahs(67, 1, 3), pause(30), say(67, 2), pause(15), say(67, 3, 1, 2), say(67, 3, 4), pause(15), ...ayahs(67, 4, 5)]);
    expect(recite(s, { passage: passage(67, 1, 5), refinements: ALL }).mistakes).toEqual(["omitted_word 67:3:3"]);
    expect(recite(s, { passage: passage(67, 1, 5), refinements: without("readNotSkipped") }).mistakes).toEqual(["omitted_word 67:3:3"]);
  });

  it("36:6–8, back to 36:7, then 36:8 without its second word: the word is left out, as rc.1 marks it", () => {
    const s = speech([...ayahs(36, 6, 8), pause(30), say(36, 7), pause(15), say(36, 8, 1, 1), say(36, 8, 3), pause(15), ...ayahs(36, 9, 10)]);
    expect(recite(s, { passage: passage(36, 6, 10), refinements: ALL }).mistakes).toEqual(["omitted_word 36:8:2"]);
    expect(recite(s, { passage: passage(36, 6, 10), refinements: without("readNotSkipped") }).mistakes).toEqual(["omitted_word 36:8:2"]);
  });
});

describe("refinements.readNotSkipped: a pass is set by how far back the reciter went", () => {
  it("67:1–10, back two ayahs to 67:8 read through, then on at 67:11: one pass, nothing left out, free and declared", () => {
    const s = speech([...ayahs(67, 1, 10), pause(30), ...ayahs(67, 8, 8), pause(15), ...ayahs(67, 11, 13)]);
    for (const p of [null, passage(67, 1, 15)]) {
      const mode = p ? "declared" : "free";
      expect(recite(s, { passage: p, refinements: ALL }).mistakes, mode).toEqual([]);
      expect(recite(s, { passage: p, refinements: without("readNotSkipped") }).mistakes, mode).toEqual(["skipped_passage 67:9:1"]);
    }
  });

  it("the same back one ayah, to 67:9: nothing left out, free and declared", () => {
    const s = speech([...ayahs(67, 1, 10), pause(30), ...ayahs(67, 9, 9), pause(15), ...ayahs(67, 11, 13)]);
    for (const p of [null, passage(67, 1, 15)]) expect(recite(s, { passage: p, refinements: ALL }).mistakes, p ? "declared" : "free").toEqual([]);
  });

  it("back fewer than PASS_BACK_WORDS, an ayah read again, then on at the furthest ayah's first word: the ayah between is left out, as rc.1 marks it", () => {
    // 67:1–12, back to 67:10 (its first word 24 words behind the furthest read), 67:10 read through, a pause, then
    // 67:12 again from its first word. Reading on from where the reciter had read to goes to 67:13: going on at 67:12
    // leaves 67:11 out of the re-reading, though the first reading said it.
    expect(ayah(67, 10).length + ayah(67, 11).length + ayah(67, 12).length).toBeLessThan(30);
    const s = speech([...ayahs(67, 1, 12), pause(30), ...ayahs(67, 10, 10), pause(15), ...ayahs(67, 12, 14)]);
    for (const p of [null, passage(67, 1, 15)]) {
      const mode = p ? "declared" : "free";
      expect(recite(s, { passage: p, refinements: ALL }).mistakes, mode).toEqual(["skipped_passage 67:11:1"]);
      expect(recite(s, { passage: p, refinements: RC1 }).mistakes, mode).toEqual(["skipped_passage 67:11:1"]);
    }
  });

  it("55:1–12, back to 55:8, then 55:8–14 without 55:10: the ayah left out of the re-reading is marked, as rc.1 marks it", () => {
    // 22 words back: one pass. Going on at 55:11, short of 55:12 (the furthest ayah read), is no rejoin.
    const s = speech([...ayahs(55, 1, 12), pause(30), ...ayahs(55, 8, 14, [10])]);
    for (const p of [passage(55, 1, 14), null]) {
      const mode = p ? "declared" : "free";
      expect(recite(s, { passage: p, refinements: ALL }).mistakes, mode).toEqual(["skipped_passage 55:10:1"]);
      expect(recite(s, { passage: p, refinements: RC1 }).mistakes, mode).toEqual(["skipped_passage 55:10:1"]);
    }
  });

  it("67:1–3, then from the top again without 67:2, on through 67:5: the ayah left out of the passage read again is marked", () => {
    // The second reading goes on at 67:3, the furthest ayah read, read to its end the first time: reading on from there
    // goes to 67:4, so 67:2 was left out. (rc.1 marks a wrong word on 67:4:1 instead: its commit reads the restart
    // against the window; refinements.commitAfterJump.)
    const s = speech([...ayahs(67, 1, 3), pause(40), ...ayahs(67, 1, 5, [2])]);
    for (const p of [passage(67, 1, 5), null]) {
      const mode = p ? "declared" : "free";
      expect(recite(s, { passage: p, refinements: ALL }).mistakes, mode).toEqual(["skipped_passage 67:2:1"]);
      expect(recite(s, { passage: p, refinements: without("readNotSkipped") }).mistakes, mode).toEqual(["skipped_passage 67:2:1"]);
    }
  });

  it("the risk: broken off inside 67:12, back to 67:10, then 67:12 started again: the ayah the run-up left out is not marked", () => {
    // Going back to run up at an ayah the reciter stalled in and starting it again is where they had read to: it
    // cannot be told from a run-up that forgot 67:11, which rc.1 marks.
    const s = speech([...ayahs(67, 1, 11), say(67, 12, 1, 3), pause(30), ...ayahs(67, 10, 10), pause(15), ...ayahs(67, 12, 14)]);
    for (const p of [null, passage(67, 1, 15)]) {
      const mode = p ? "declared" : "free";
      expect(recite(s, { passage: p, refinements: ALL }).mistakes, mode).toEqual([]);
      expect(recite(s, { passage: p, refinements: without("readNotSkipped") }).mistakes, mode).toEqual(["skipped_passage 67:11:1"]);
    }
  });
});

describe("refinements.readNotSkipped: an ayah restarted twice", () => {
  it("18:9–10, 18:10's first seven words twice more, then on at 18:11: one reading, nothing left out", () => {
    // A correct take: the reciter reads 18:10 whole, starts it again twice for a few words, and goes on at the next
    // ayah. rc.1 marks the rest of 18:10, read the first time, as left out of the last restart.
    for (const joined of [false, true]) {
      const restart = say(18, 10, 1, 7);
      const s = speech([say(18, 9), pause(15), say(18, 10), pause(20), restart, pause(20), restart, ...(joined ? [] : [pause(15)]), ...ayahs(18, 11, 12)]);
      for (const p of [passage(18, 9, 13), null]) {
        const mode = `${p ? "declared" : "free"}, ${joined ? "no pause" : "a pause"} before 18:11`;
        expect(recite(s, { passage: p, refinements: ALL }).mistakes, mode).toEqual([]);
        expect(recite(s, { passage: p, refinements: without("readNotSkipped") }).mistakes, mode).toEqual(["skipped_passage 18:10:8"]);
      }
    }
  }, 60_000);
});

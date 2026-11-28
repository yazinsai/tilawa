import { describe, expect, it } from "vitest";
import { ALL, ayah, ayahs, noise, passage, pause, recite, say, speech, without } from "./takes.js";
import type { Fields, Piece } from "./takes.js";

/**
 * The checker's corrections for restarts and for the passage's edges: refinements.commitAfterJump,
 * refinements.repeatNotWrong, refinements.firstAyahOut and refinements.marksInPassage. A student who reads two or three
 * ayahs and starts again from the top, or says a word wrong and starts again at once, is the everyday shape of hifz
 * practice; a wrong word put right by starting again is a self-correction, and one said wrong again after the restart
 * is still a mistake. The takes are synthetic phoneme streams built from the reference (`./takes.ts`).
 */

/** 67:1–3 read through, each ayah followed by a short pause, and a longer pause after 67:3. */
const firstReading: Piece[] = [say(67, 1), pause(15), say(67, 2), pause(15), say(67, 3), pause(40)];
/** 67:1–3 then 67:4's first two words. */
const upToTheSlip: Piece[] = [say(67, 1), pause(15), say(67, 2), pause(15), say(67, 3), pause(15), say(67, 4, 1, 2)];
/** A word that is not 67:4:3, said in its place. */
const slip = noise("فَسَدَ");
/** 67:4 from its fourth word, then 67:5. */
const afterTheSlip: Piece[] = [say(67, 4, 4), pause(15), say(67, 5)];
const p67 = passage(67, 1, 5);

describe("refinements.commitAfterJump: a commit keeps nothing its new window cannot place", () => {
  it("67:1–3, a pause, then 67:1–5 from the top: nothing marked", () => {
    const s = speech([...firstReading, ...ayahs(67, 1, 5)]);
    for (const p of [p67, null]) {
      const mode = p ? "declared" : "free";
      expect(recite(s, { passage: p, refinements: ALL }).mistakes, mode).toEqual([]);
      // rc.1: the commit keeps the restart of 67:1, which its new window (from 67:2) reads as a wrong 67:4:1.
      expect(recite(s, { passage: p, refinements: without("commitAfterJump") }).mistakes, mode).toEqual(["wrong_word 67:4:1"]);
    }
  });

  it("a wrong word, then the passage again from the top, the word said right: a self-correction, nothing marked", () => {
    const s = speech([...upToTheSlip, slip, pause(20), ...ayahs(67, 1, 5)]);
    expect(recite(s, { passage: p67, refinements: ALL }).mistakes).toEqual([]);
    expect(recite(s, { passage: p67, refinements: without("commitAfterJump") }).mistakes).toEqual([
      "wrong_word 67:4:3", "wrong_word 67:4:5", "wrong_word 67:4:6", "wrong_word 67:4:7",
    ]);
  });

  it("a wrong word, the passage again from the top, and the word said wrong again: the wrong word is marked", () => {
    const s = speech([...upToTheSlip, slip, pause(20), ...upToTheSlip, slip, ...afterTheSlip]);
    expect(recite(s, { passage: p67, refinements: ALL }).mistakes).toEqual(["wrong_word 67:4:3"]);
  });

  it("a restart from the top, then a wrong word: the wrong word alone is marked", () => {
    const s = speech([...firstReading, ...upToTheSlip, slip, ...afterTheSlip]);
    expect(recite(s, { passage: p67, refinements: ALL }).mistakes).toEqual(["wrong_word 67:4:3"]);
    expect(recite(s, { passage: p67, refinements: without("commitAfterJump") }).mistakes).toEqual(["wrong_word 67:4:1", "wrong_word 67:4:3"]);
  });

  it("a wrong word with no restart, or after a restart one ayah back, is marked as before", () => {
    const straight = speech([...upToTheSlip, slip, ...afterTheSlip]);
    const back = speech([...upToTheSlip.slice(0, 6), say(67, 2), pause(15), say(67, 3), pause(15), say(67, 4, 1, 2), slip, ...afterTheSlip]);
    for (const s of [straight, back]) {
      expect(recite(s, { passage: p67, refinements: ALL }).mistakes).toEqual(["wrong_word 67:4:3"]);
      expect(recite(s, { passage: p67, refinements: without("commitAfterJump") }).mistakes).toEqual(["wrong_word 67:4:3"]);
    }
  });
});

describe("refinements.firstAyahOut: starting later than the declared passage", () => {
  const starts: [string, [number, number, number], Piece[], string][] = [
    ["82:2–5 started at 82:3", [82, 2, 5], ayahs(82, 3, 5), "skipped_passage 82:2:1"],
    ["67:1–5 started at 67:2", [67, 1, 5], ayahs(67, 2, 5), "skipped_passage 67:1:1"],
    ["36:6–9 started at 36:7", [36, 6, 9], ayahs(36, 7, 9), "skipped_passage 36:6:1"],
  ];
  for (const [name, [s, a, b], pieces, mark] of starts) {
    it(`${name}: the first ayah is marked left out`, () => {
      const take = speech(pieces);
      expect(recite(take, { passage: passage(s, a, b), refinements: ALL }).mistakes).toEqual([mark]);
      expect(recite(take, { passage: passage(s, a, b), refinements: without("firstAyahOut") }).mistakes).toEqual([]);
    });
  }

  // at the take's start the alignment window reaches two ayahs into the passage, so a start
  // further in was read against the passage's first words, as wrong words and skips the student never reached.
  for (const [s, a, b, from, to] of [[67, 1, 30, 20, 24], [36, 1, 20, 9, 12], [55, 1, 30, 14, 17]] as const) {
    it(`${s}:${a}–${b} started at ${s}:${from}, read to ${s}:${to}: one skip, from the passage's first word to the start`, () => {
      const take = speech(ayahs(s, from, to));
      for (const step of [1, 3]) {
        expect(recite(take, { passage: passage(s, a, b), step, refinements: ALL }).mistakes, `step ${step}`).toEqual([`skipped_passage ${s}:${a}:1`]);
        expect(recite(take, { passage: passage(s, a, b), step, refinements: without("firstAyahOut") }).mistakes, `step ${step}`).not.toEqual([`skipped_passage ${s}:${a}:1`]);
      }
    });
  }

  it("a different surah said, after the declared one in the mushaf or before it, is not a late start", () => {
    // 114 after 113 (they open alike), 2 before 36.
    const takes: [[number, number, number], [number, number, number]][] = [[[113, 1, 5], [114, 1, 6]], [[114, 1, 6], [113, 1, 5]], [[36, 1, 5], [2, 1, 5]], [[2, 1, 5], [36, 1, 5]]];
    for (const [declared, said] of takes) {
      const take = speech(ayahs(...said));
      for (const step of [1, 3]) {
        const now = recite(take, { passage: passage(...declared), step, refinements: ALL }).mistakes;
        expect(now.filter((m) => m.startsWith("skipped_passage")), `declared ${declared[0]}, said ${said[0]}, step ${step}`).toEqual([]);
        expect(now, `declared ${declared[0]}, said ${said[0]}, step ${step}`).toEqual(recite(take, { passage: passage(...declared), step, refinements: without("firstAyahOut") }).mistakes);
      }
    }
  }, 60_000);

  it("a take whose first word is cut off (a microphone that opened late) marks nothing", () => {
    for (const [s, a, b] of [[67, 1, 5], [82, 1, 5], [36, 6, 9]] as const) {
      for (const from of [2, 3]) {
        const take = speech([say(s, a, from), pause(15), ...ayahs(s, a + 1, b)]);
        expect(recite(take, { passage: passage(s, a, b), refinements: ALL }).mistakes, `${s}:${a} from word ${from}`).toEqual([]);
      }
    }
  });

  it("a first ayah of fewer than three words, left out, is not marked: it cannot be told from a late microphone", () => {
    // 55:1 is one word.
    expect(ayah(55, 1).length).toBeLessThan(3);
    expect(recite(speech(ayahs(55, 2, 5)), { passage: passage(55, 1, 5), refinements: ALL }).mistakes).toEqual([]);
  });

  it("speech before the passage that is not the Quran's is set aside, as before", () => {
    const take = speech([noise("زظغذثخطضصق"), pause(30), ...ayahs(67, 1, 5)]);
    expect(recite(take, { passage: p67, refinements: ALL }).mistakes).toEqual([]);
  });

  it("in free recitation nothing was declared, so nothing is left out", () => {
    expect(recite(speech(ayahs(67, 2, 5)), { passage: null, refinements: ALL }).mistakes).toEqual([]);
  });
});

describe("refinements.marksInPassage: no word mark outside the declared passage", () => {
  it("a restart whose commit reads it against the words after the passage marks nothing there", () => {
    // With rc.1's commit (commitAfterJump off) the restart is charged to words of the window, which reaches two ayahs
    // past the furthest word read: one of them, 67:6:3, is outside the passage.
    const s = speech([...upToTheSlip, slip, pause(20), ...ayahs(67, 1, 5)]);
    const commit: Fields = { ...ALL, commitAfterJump: false };
    expect(recite(s, { passage: p67, refinements: { ...commit, marksInPassage: false } }).mistakes).toContain("wrong_word 67:6:3");
    expect(recite(s, { passage: p67, refinements: commit }).mistakes).not.toContain("wrong_word 67:6:3");
  });

  it("the risk: a reciter who reads on past the passage's end with a wrong word there is not marked for it", () => {
    const s = speech([...ayahs(67, 1, 3), say(67, 4, 1, 2), slip, say(67, 4, 4)]);
    const p = passage(67, 1, 3);
    expect(recite(s, { passage: p, refinements: ALL }).mistakes).toEqual([]);
    expect(recite(s, { passage: p, refinements: without("marksInPassage") }).mistakes).toEqual(["wrong_word 67:4:3"]);
  });

  it("free recitation keeps its marks", () => {
    const s = speech([...upToTheSlip, slip, ...afterTheSlip]);
    expect(recite(s, { passage: null, refinements: ALL }).mistakes).toEqual(["wrong_word 67:4:3"]);
  });
});

describe("refinements.repeatNotWrong: a restart is not a wrong word, and nothing else is let through", () => {
  /**
   * An ayah read whole, then started again (its first `k[0]` words, its first `k[1]`, its first `k[2]`) before reading
   * on, the take opening on `lead`. A checker commit falls inside the restarts and takes the whole reading out of the
   * window, so the alignment charges the last restart to the ayah's next word as one long wrong word. The field also
   * removes T's two strays of this kind, restarts in 18:10 on DFD908-03 and -04 (`wrong_word 18:10:11`).
   */
  const restarted = (s: number, a: number, lead: Piece[], k: [number, number, number], after: Piece[]): Piece[] =>
    [...lead, say(s, a), pause(25), say(s, a, 1, k[0]), say(s, a, 1, k[1]), say(s, a, 1, k[2]), pause(20), ...after];
  const shapes: [string, number, number, Piece[], [number, number, number], string][] = [
    ["67:8 read whole, then 67:8:1–10, 1–4 and 1–6", 67, 8, [say(67, 7, 1), pause(15)], [10, 4, 6], "wrong_word 67:8:11"],
    ["18:10 read whole, then 18:10:1–13, 1–4 and 1–9", 18, 10, [say(18, 9, 5), pause(15)], [13, 4, 9], "wrong_word 18:10:14"],
  ];
  const modes = (s: number, a: number) => [[passage(s, a - 1, a + 2), "declared"], [null, "free"]] as const;

  for (const [name, s, a, lead, k, restart] of shapes) {
    it(`${name}, then on: nothing marked, at 1, 2, 3 and 5 phonemes a push`, () => {
      const take = speech(restarted(s, a, lead, k, [say(s, a + 1), pause(15), say(s, a + 2)]));
      for (const step of [1, 2, 3, 5]) for (const [p, mode] of modes(s, a)) {
        expect(recite(take, { passage: p, step, refinements: ALL }).mistakes, `${mode}, step ${step}`).toEqual([]);
        // rc.1 and the kept version: the last restart is charged to the ayah's next word as a wrong word.
        expect(recite(take, { passage: p, step, refinements: without("repeatNotWrong") }).mistakes, `${mode}, step ${step}`).toEqual([restart]);
      }
    }, 120_000);
  }

  it("the same ayahs read straight through mark nothing, with the field or without it", () => {
    for (const [, s, a, lead] of shapes) {
      const take = speech([...lead, say(s, a), pause(20), say(s, a + 1), pause(15), say(s, a + 2)]);
      for (const step of [1, 3]) for (const [p, mode] of modes(s, a)) for (const fields of [ALL, without("repeatNotWrong")]) {
        expect(recite(take, { passage: p, step, refinements: fields }).mistakes, `${s}:${a}, ${mode}, step ${step}`).toEqual([]);
      }
    }
  }, 120_000);

  it("the risk: earlier text said in place of words read right, in a restart, then on where they had read to, is not marked", () => {
    // The field's documented risk. 67:8 read whole and right, then started again, the last restart saying 67:7's first
    // three words in place of 67:8:5–6, then 67:9. Without the field the only mark is the restart's own, on 67:8:11 (a
    // word said right each time), as in the kept version: neither marks the words replaced.
    const [, s, a, lead] = shapes[0];
    const take = speech([...lead, say(s, a), pause(25), say(s, a, 1, 10), say(s, a, 1, 4), say(s, a, 1, 4), say(s, a - 1, 1, 3), pause(20),
      say(s, a + 1), pause(15), say(s, a + 2)]);
    for (const step of [1, 3]) for (const [p, mode] of modes(s, a)) {
      expect(recite(take, { passage: p, step, refinements: ALL }).mistakes, `${mode}, step ${step}`).toEqual([]);
      expect(recite(take, { passage: p, step, refinements: without("repeatNotWrong") }).mistakes, `${mode}, step ${step}`).toEqual(["wrong_word 67:8:11"]);
    }
  }, 120_000);

  it("a wrong word after the restarts, in the next ayah or the one after, stays marked, alone", () => {
    const [, s, a, lead, k, restart] = shapes[0];
    const takes: [string, Piece[], string][] = [
      ["67:9:6", [say(s, a + 1, 1, 5), slip, say(s, a + 1, 7), pause(15), say(s, a + 2)], "wrong_word 67:9:6"],
      ["67:10:3", [say(s, a + 1), pause(15), say(s, a + 2, 1, 2), slip, say(s, a + 2, 4)], "wrong_word 67:10:3"],
    ];
    for (const [name, after, mark] of takes) {
      const take = speech(restarted(s, a, lead, k, after));
      for (const step of [1, 3]) for (const [p, mode] of modes(s, a)) {
        expect(recite(take, { passage: p, step, refinements: ALL }).mistakes, `${name}, ${mode}, step ${step}`).toEqual([mark]);
        expect(recite(take, { passage: p, step, refinements: without("repeatNotWrong") }).mistakes, `${name}, ${mode}, step ${step}`).toEqual([restart, mark]);
      }
    }
    const [, s2, a2, lead2, k2, restart2] = shapes[1];
    const take = speech(restarted(s2, a2, lead2, k2, [say(s2, a2 + 1, 1, 2), slip, say(s2, a2 + 1, 4), pause(15), say(s2, a2 + 2)]));
    for (const [p, mode] of modes(s2, a2)) {
      expect(recite(take, { passage: p, refinements: ALL }).mistakes, mode).toEqual(["omitted_word 18:11:3"]);
      expect(recite(take, { passage: p, refinements: without("repeatNotWrong") }).mistakes, mode).toEqual([restart2, "omitted_word 18:11:3"]);
    }
  }, 120_000);

  it("a wrong word in the ayah's first reading, then the restarts: the field takes away only the restart's own mark", () => {
    // Both ways, on the restart shape. Said wrong again in the restart: the wrong word stays. Said right
    // in the restarts: the first reading's word is judged as without the field (commitAfterJump's and readNotSkipped's
    // rules decide it), and only the restart's own mark goes.
    const [, s, a, lead] = shapes[0];
    const wrongAgain = speech([...lead, say(s, a, 1, 4), slip, say(s, a, 6), pause(25), say(s, a, 1, 4), slip, say(s, a, 6), pause(20), say(s, a + 1), pause(15), say(s, a + 2)]);
    const putRight = speech([...lead, say(s, a, 1, 4), slip, say(s, a, 6), pause(25), say(s, a, 1, 10), say(s, a, 1, 4), say(s, a, 1, 6), pause(20), say(s, a + 1), pause(15), say(s, a + 2)]);
    for (const step of [1, 3]) for (const [p, mode] of modes(s, a)) {
      const label = `${mode}, step ${step}`;
      for (const fields of [ALL, without("repeatNotWrong")]) {
        expect(recite(wrongAgain, { passage: p, step, refinements: fields }).mistakes, label).toEqual(["wrong_word 67:8:5"]);
      }
      const on = recite(putRight, { passage: p, step, refinements: ALL }).mistakes;
      const off = recite(putRight, { passage: p, step, refinements: without("repeatNotWrong") }).mistakes;
      expect(off, label).toContain("wrong_word 67:8:11");
      expect(on, label).toEqual(off.filter((m) => m !== "wrong_word 67:8:11"));
    }
    // From the top of the passage: put right, nothing marked; said wrong again, marked; the same without the field.
    for (const fields of [ALL, without("repeatNotWrong")]) {
      expect(recite(speech([...upToTheSlip, slip, pause(20), ...ayahs(67, 1, 5)]), { passage: p67, refinements: fields }).mistakes).toEqual([]);
      expect(recite(speech([...upToTheSlip, slip, pause(20), ...upToTheSlip, slip, ...afterTheSlip]), { passage: p67, refinements: fields }).mistakes)
        .toEqual(["wrong_word 67:4:3"]);
    }
  }, 120_000);

  it("earlier text said in place of a word never read, then reading on: the word is marked wrong", () => {
    for (const [name, earlier] of [["67:3:1–3", say(67, 3, 1, 3)], ["67:3:1–4", say(67, 3, 1, 4)], ["67:2:1–3", say(67, 2, 1, 3)]] as const) {
      const s = speech([...upToTheSlip, earlier, ...afterTheSlip]);
      for (const p of [p67, null]) {
        expect(recite(s, { passage: p, refinements: ALL }).mistakes, `${name}, ${p ? "declared" : "free"}`).toEqual(["wrong_word 67:4:3"]);
      }
    }
  });

  it("a wrong word with no restart, or after a restart, is marked as without it", () => {
    for (const s of [speech([...upToTheSlip, slip, ...afterTheSlip]), speech([...firstReading, ...upToTheSlip, slip, ...afterTheSlip]),
      speech([...upToTheSlip, slip, pause(20), ...upToTheSlip, slip, ...afterTheSlip])]) {
      expect(recite(s, { passage: p67, refinements: ALL }).mistakes).toEqual(["wrong_word 67:4:3"]);
      expect(recite(s, { passage: p67, refinements: without("repeatNotWrong") }).mistakes).toEqual(["wrong_word 67:4:3"]);
    }
  });

  it("an ayah's opening said again before reading on marks nothing", () => {
    const s = speech([...upToTheSlip, say(67, 4, 1, 2), say(67, 4, 3), pause(15), say(67, 5)]);
    for (const p of [p67, null]) expect(recite(s, { passage: p, refinements: ALL }).mistakes, p ? "declared" : "free").toEqual([]);
  });
});

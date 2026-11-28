import { describe, expect, it } from "vitest";
import type { EngineEvent, PositionEvent, WordRef } from "../../src/engine/events.js";
import { ALL, EARLIER_ON, FRAME_MS, RC1, at, ayah, ayahs, decided, earlierWithout, guessed, index, key, noise, passage, pause, recite, ref, say, shownAt, speech, without } from "./takes.js";
import type { Fields, Piece, Take } from "./takes.js";

/**
 * The risk cases of the refinements, each a take they must still get right.
 * The takes are synthetic phoneme streams built from the reference (`./takes.ts`).
 */

const firstLock = (t: Take) => t.positions.find((p) => p.state === "locked")!;
/** Positions and mistakes as plain text, to compare takes pushed in different chunks. */
const summary = (t: Take) => ({
  positions: t.positions.map((p) => `${p.state}:${p.observed[0] ? `${key(p.observed[0].span.start)}..${key(p.observed[0].word)}` : "-"}@${p.timeMs}`),
  mistakes: t.mistakes,
});

describe("refinements.declaredStart: only the declared opening, and not held back by stray sounds", () => {
  it("a reciter who starts at a later ayah sharing the opening is shown at most the words they share there, then followed", () => {
    // 82:2, 82:3 and 82:4 all open وَإِذَا; the passage is 82:2–5 and the reciter starts at 82:3. The declared
    // opening's first word, the word they share, may show as it is said; 82:2's other words never show.
    const s = speech([say(82, 3), pause(15), say(82, 4), pause(15), say(82, 5)]);
    for (const step of [1, 3]) {
      const now = recite(s, { passage: passage(82, 2, 5), step, refinements: ALL });
      const shown = shownAt(now.positions);
      expect(guessed(s, shown), `step ${step}`).toEqual([]);
      expect([...shown.keys()].filter((w) => !s.saidAt.has(w) && w !== "82:2:1"), `step ${step}`).toEqual([]);
      for (const w of ["82:3:2", "82:4:1", "82:5:1", "82:5:4"]) expect(shown.has(w), `${w}, step ${step}`).toBe(true);
      expect(now.mistakes, `step ${step}`).toEqual(recite(s, { passage: passage(82, 2, 5), step, refinements: without("declaredStart") }).mistakes);
    }
  });

  it("a passage that opens like its next ayahs locks on its opening by its second word's second sound", () => {
    // وَإِذَا, which 82:3 and 82:4 share, is told apart from the openings whose first word only sounds like it (وَإِذْ,
    // وَإِذًا) by the sounds after it.
    const s = speech([say(82, 2), pause(15), say(82, 3), pause(15), say(82, 4)]);
    const now = firstLock(recite(s, { passage: passage(82, 2, 5), checker: false, step: 1, refinements: ALL }));
    expect(key(now.observed[0].span.start)).toBe("82:2:1");
    expect(now.timeMs).toBeLessThanOrEqual(s.saidAt.get("82:2:2")! + 5 * FRAME_MS);
    expect(now.timeMs).toBeLessThan(firstLock(recite(s, { passage: passage(82, 2, 5), checker: false, step: 1, refinements: without("declaredStart") })).timeMs);
  });

  it("sounds that are no phrase, then a pause, do not hold the lock back", () => {
    const junk = "زظغذثخطضصقزظغذثخ";
    const clean = speech([say(36, 6), pause(15), say(36, 7)]);
    const after = (t: Take, s: typeof clean) => firstLock(t).timeMs - s.saidAt.get("36:6:1")!;
    const base = after(recite(clean, { passage: passage(36, 6, 7), checker: false, step: 1, refinements: ALL }), clean);
    for (const n of [2, 6, 16]) {
      const s = speech([noise(junk.slice(0, n)), pause(30), say(36, 6), pause(15), say(36, 7)]);
      const t = recite(s, { passage: passage(36, 6, 7), checker: false, step: 1, refinements: ALL });
      expect(key(firstLock(t).observed[0].word), `${n} sounds`).toBe("36:6:1");
      expect(after(t, s), `${n} sounds`).toBe(base);
    }
  });

  it("checks the opening in time order: the same take pushed 1, 3 or 8 phonemes at a time gives the same events", () => {
    const basmala = index.phonemes.slice(index.words[ayah(1, 1)[0]].start, index.words[ayah(1, 1).at(-1)!].end);
    const takes: [Piece[], [WordRef, WordRef]][] = [
      [[noise("زظغذثخ"), pause(30), say(36, 6), pause(15), say(36, 7)], passage(36, 6, 7)],
      [[noise(basmala), pause(20), say(67, 1), pause(15), say(67, 2)], passage(67, 1, 2)],
    ];
    for (const [pieces, p] of takes) {
      const s = speech(pieces);
      const one = summary(recite(s, { passage: p, step: 1, refinements: ALL }));
      for (const step of [3, 8]) expect(summary(recite(s, { passage: p, step, refinements: ALL })), `step ${step}`).toEqual(one);
    }
  });
});

describe("refinements.declaredStart: a look-alike opening elsewhere in the mushaf", () => {
  // Starting the wrong one of two openings that share their first words is a classic memorization slip. Of ayahs that
  // open with the same words, the in-order one (here, the declared opening) lights as they are said, and the page moves
  // if the reciter was saying the other. So the declared opening's words show on the shared words, but no other word of
  // the declared passage shows; where the two part, the tracker finds the reciter where they are, and nothing is
  // marked that is not marked without the field.
  const takes: [string, [number, number, number], [number, number, number]][] = [
    ["declared 113, said 114", [113, 1, 5], [114, 1, 6]],
    ["declared 114, said 113", [114, 1, 6], [113, 1, 5]],
    ["declared 2:255, said 3:2–4", [2, 255, 255], [3, 2, 4]],
    ["declared 3:2–4, said 2:255", [3, 2, 4], [2, 255, 255]],
    ["declared 82:1–5, said 84:1–5", [82, 1, 5], [84, 1, 5]],
    ["declared 18:1–5, said 6:1–3", [18, 1, 5], [6, 1, 3]],
  ];
  for (const [name, declared, said] of takes) {
    it(`${name}: of the declared passage only the shared words show, then the reciter is followed, no mark added`, () => {
      const s = speech(ayahs(...said));
      const opening = `${declared[0]}:${declared[1]}:`;
      for (const step of [1, 3]) {
        const t = recite(s, { passage: passage(...declared), step, refinements: ALL });
        const shown = shownAt(t.positions);
        expect(guessed(s, shown), `step ${step}`).toEqual([]);
        expect([...shown.keys()].filter((w) => !s.saidAt.has(w) && !w.startsWith(opening)), `step ${step}`).toEqual([]);
        // The last words said show: the reciter is followed.
        for (const word of [...s.saidAt.keys()].slice(-3)) expect(shown.has(word), `step ${step}: ${word}`).toBe(true);
        // Once the page has moved to where the reciter is, it does not step back.
        const path = t.positions.filter((p) => p.state === "locked" && !key(p.observed[0].word).startsWith(opening)).map((p) => at(p.observed[0].word));
        for (let k = 1; k < path.length; k++) expect(path[k], `step ${step}: a step back`).toBeGreaterThanOrEqual(path[k - 1]);
        const before = recite(s, { passage: passage(...declared), step, refinements: without("declaredStart") }).mistakes;
        for (const mark of t.mistakes) expect(before, `step ${step}: ${mark}`).toContain(mark);
      }
    }, 60_000);
  }
});

describe("refinements.confirmAdvance: only where a step is risky", () => {
  it("a clean recitation, ayah openings after a pause included, shows every word when rc.1 shows it, at any push size", () => {
    for (const [s0, a, b] of [[67, 1, 3], [36, 6, 8], [55, 1, 4]] as [number, number, number][]) {
      const s = speech(ayahs(s0, a, b));
      for (const step of [1, 3, 8]) {
        const now = shownAt(recite(s, { passage: passage(s0, a, b), checker: false, step, refinements: ALL }).positions);
        const rc1 = shownAt(recite(s, { passage: passage(s0, a, b), checker: false, step, refinements: without("confirmAdvance") }).positions);
        let words = 0;
        for (let x = a; x <= b; x++) words += ayah(s0, x).length;
        expect([now.size, rc1.size], `${s0}:${a}–${b}, step ${step}: every word shown`).toEqual([words, words]);
        for (const [k, t] of rc1) expect(now.get(k), `${k}, step ${step}`).toBe(t);
      }
    }
  });
});

describe("refinements.omissionDue: confirmed stays final", () => {
  it("a word left out, said due once the reciter read on, then said: resolved, never confirmed, and not counted", () => {
    // 36:6:4 left out; 36:6:5…, 36:7:1–2 said; a 1.2 s pause; back to 36:6:4 and on.
    const s = speech([say(36, 6, 1, 3), say(36, 6, 5), pause(15), say(36, 7, 1, 2), pause(30), say(36, 6, 4), say(36, 6, 5), pause(15),
      say(36, 7), pause(15), say(36, 8)]);
    const now = recite(s, { passage: passage(36, 6, 8), refinements: ALL });
    const events = now.checker.filter((e) => key(e.location) === "36:6:4");
    expect(events.map((e) => `${e.state}${e.due ? " due" : ""}`)).toEqual(["suspected", "suspected due", "resolved"]);
    expect(new Set(events.map((e) => e.id)).size).toBe(1);
    expect(events[1].timeMs).toBeLessThan(s.saidAt.get("36:6:4")!);
    expect(now.mistakes).toEqual([]);
    const rc1 = recite(s, { passage: passage(36, 6, 8), refinements: without("omissionDue") });
    expect(rc1.checker.filter((e) => key(e.location) === "36:6:4").map((e) => e.state)).toEqual(["suspected", "resolved"]);
  });

  it("a word left out and never said stays confirmed", () => {
    const s = speech([say(36, 6, 1, 3), say(36, 6, 5), pause(15), say(36, 7), pause(15), say(36, 8), pause(15), say(36, 9)]);
    expect(recite(s, { passage: passage(36, 6, 9), refinements: ALL }).mistakes).toEqual(["omitted_word 36:6:4"]);
  });
});

describe("refinements.reanchorBack in free recitation", () => {
  it("a reciter who starts again from an earlier ayah is followed there", () => {
    const s = speech([...ayahs(67, 1, 4), pause(30), ...ayahs(67, 1, 5)]);
    // `commitAfterJump` also mends this restart: without both, it is marked.
    expect(recite(s, { passage: null, refinements: earlierWithout("reanchorBack") }).mistakes.length).toBeGreaterThan(0);
    expect(recite(s, { passage: null, refinements: EARLIER_ON }).mistakes).toEqual([]);
    expect(recite(s, { passage: null, refinements: ALL }).mistakes).toEqual([]);
  });

  it("a lock on an earlier copy of a refrain the window also holds moves nothing", () => {
    // 55:13–21 said right; after 55:18 (a copy of the refrain), the tracker locks on 55:13's copy.
    const s = speech(ayahs(55, 13, 21));
    const lockAt = speech(ayahs(55, 13, 17)).phonemes.length + speech([say(55, 18)]).phonemes.length;
    const refrain = ayah(55, 13);
    for (const p of [null, passage(55, 13, 21)]) {
      let injected = false;
      const t = recite(s, {
        passage: p,
        refinements: ALL,
        inject: (pushed, events) => {
          if (injected || pushed < lockAt) return events;
          injected = true;
          const end = ref(refrain.at(-1)!);
          const lock: PositionEvent = {
            type: "position", v: 1, timeMs: s.phonemes[pushed - 1].end * FRAME_MS, referenceVersion: index.version, expected: end,
            observed: [{ word: end, span: { start: ref(refrain[0]), end }, confidence: 0.8, cost: 0.1, per: 0.1 }], state: "locked",
          };
          return [...events, lock];
        },
      });
      expect(injected).toBe(true);
      expect(t.checker, p ? "declared" : "free").toEqual([]);
    }
  });
});

describe("refinements.reanchorBack: a skip while the tracker holds behind it", () => {
  it("36:6:4 left out while the tracker stays on 36:6:3 (a hold, not a return): confirmed at its usual time", () => {
    const s = speech([say(36, 6, 1, 3), say(36, 6, 5), pause(15), say(36, 7), pause(15), say(36, 8), pause(15), say(36, 9), pause(15), say(36, 10)]);
    const ws = ayah(36, 6);
    const from = speech([say(36, 6, 1, 3)]).phonemes.length;
    const until = speech([say(36, 6, 1, 3), say(36, 6, 5), pause(15), say(36, 7), pause(15), say(36, 8), pause(15), say(36, 9)]).phonemes.length;
    const hold = (pushed: number): PositionEvent => {
      const end = ref(ws[2]);
      return {
        type: "position", v: 1, timeMs: s.phonemes[pushed - 1].end * FRAME_MS, referenceVersion: index.version, expected: end,
        observed: [{ word: end, span: { start: ref(ws[0]), end }, confidence: 0.8, cost: 0.1, per: 0.1 }], state: "locked",
      };
    };
    let held = 0;
    // The tracker's place stays on 36:6:3 from the skip until 36:9 is said through.
    const inject = (pushed: number, events: EngineEvent[]) => {
      if (pushed <= from || pushed > until) return events;
      held++;
      return [...events.filter((e) => e.type !== "position"), hold(pushed)];
    };
    const usual = decided(recite(s, { passage: passage(36, 6, 10), refinements: ALL }), "omitted_word", "36:6:4")!;
    expect(usual.state).toBe("confirmed");
    const t = recite(s, { passage: passage(36, 6, 10), inject, refinements: ALL });
    expect(held).toBeGreaterThan(10);
    expect(decided(t, "omitted_word", "36:6:4")).toMatchObject({ state: "confirmed", timeMs: usual.timeMs });
    // The usual time falls inside the hold.
    expect(usual.timeMs).toBeLessThan(s.phonemes[until - 1].end * FRAME_MS);
  });
});

describe("refinements.stutter: a restarted word with a changed vowel", () => {
  const take = (word: string, refinements: Fields = ALL) =>
    recite(speech([say(18, 12), pause(15), say(18, 13, 1, 2), noise(word), say(18, 13, 4), pause(15), say(18, 14)]), { passage: passage(18, 12, 14), refinements });

  it("is let through, as rc.1 lets the same vowel change through without the restart", () => {
    // عَلَيكَ said عَلِيكَيكَ and عُلَيكَيكَ: the consonants are the written word's, a short vowel is not.
    for (const [restarted, changed] of [["عَلِيكَيكَ", "عَلِيكَ"], ["عُلَيكَيكَ", "عُلَيكَ"]]) {
      expect(take(restarted).mistakes, restarted).toEqual([]);
      expect(take(restarted, without("stutter")).mistakes, restarted).toEqual(["wrong_word 18:13:3"]);
      expect(take(changed, RC1).mistakes, changed).toEqual([]);
    }
  });
});

describe("refinements.ledgerPartial: an ayah left out whose words another ayah shares", () => {
  it("is still a whole-ayah skip, as in rc.1, where it opens like the ayah after it or ends like the one before", () => {
    // 2:68 and 23:3 open as the next ayah opens; 2:82 and 16:97 end as the one before ends. The alignment matches none
    // of the left-out ayah's words by chance on the shared ones.
    for (const [s, a] of [[2, 68], [23, 3], [2, 82], [16, 97]] as const) {
      const take = (refinements: Fields = ALL) => recite(speech(ayahs(s, a - 2, a + 2, [a])), { passage: passage(s, a - 2, a + 2), refinements }).mistakes;
      expect(take(), `${s}:${a}`).toEqual([`skipped_passage ${s}:${a}:1`]);
      expect(take(RC1), `${s}:${a}`).toEqual([`skipped_passage ${s}:${a}:1`]);
    }
  });
});

describe("refinements.noLostBeforeLock: the end of a take", () => {
  it("a memorization take that never finds its place ends lost, and reports no lost before", () => {
    const junk = "زظغذثخطضصقزظغذثخطضصق";
    const t = recite(speech([noise(junk), pause(30), noise(junk)]), { passage: passage(36, 6, 7), checker: false, refinements: ALL });
    expect(t.positions.at(-1)!.state).toBe("lost");
    expect(t.positions.slice(0, -1).map((p) => p.state)).not.toContain("lost");
  });
});

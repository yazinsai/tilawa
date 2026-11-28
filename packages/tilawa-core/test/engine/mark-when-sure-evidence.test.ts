import { describe, expect, it } from "vitest";
import { createChecker } from "../../src/engine/checker.js";
import { createTracker } from "../../src/engine/tracker.js";
import type { CheckerEvent, EngineEvent } from "../../src/engine/events.js";
import type { PhonemeEvidence } from "../../src/engine/evidence.js";
import { DEFAULT, FRAME_MS, ayah, ayahs, index, key, noise, passage, pause, phrases, recite, say, speech } from "./takes.js";
import type { Piece, Take } from "./takes.js";

/**
 * refinements.markWhenSure asks of a due mark, and of a decision at the take's end, all a confirmation asks of the same
 * evidence: its evidence standing replaces only the confirmation's wait. And at the take's end it decides only a wrong
 * word where the reading stopped: speech after the words recited right that the alignment could not place on the
 * passage is not decided as wrong words.
 */
const ids = [...index.tokens];
const id = (t: string) => ids.indexOf(t) + 1;
/**
 * A take of `pieces` through the tracker and the checker, three phonemes a push, with the sounds `degrade` names heard
 * at confidence 0.7 and their posteriors giving the heard sound 0.45, blank 0.4 and the written sound only 0.02 (a
 * confusable sound on degraded audio, the case `UNSURE` doubles the margin for).
 */
function degraded(pieces: Piece[], degrade: { from: number; written: string }[], p: ReturnType<typeof passage>, refinements?: object) {
  const s = speech(pieces);
  const written = new Map<number, string>();
  for (const d of degrade) {
    s.phonemes[d.from].conf = 0.7;
    written.set(s.phonemes[d.from].start, d.written);
  }
  const declared = { from: p[0], to: p[1] };
  const tracker = createTracker(index, { mode: "memorization", phrases, acquireMs: 1000, passage: declared, refinements });
  const checker = createChecker(index, { phrases, passage: declared, policy: { refinements } });
  const out: CheckerEvent[] = [];
  let frame = 0;
  for (let i = 0; i < s.phonemes.length; i += 3) {
    const part = s.phonemes.slice(i, i + 3);
    const to = Math.min(part.at(-1)!.end + 2, s.phonemes[i + 3]?.start ?? Infinity);
    const rows: [number, number][][] = Array.from({ length: to - frame }, () => [[0, 0.95]]);
    for (const ph of part) {
      for (let f = ph.start; f <= ph.end; f++) {
        const w = written.get(ph.start);
        rows[f - frame] = w ? [[id(ph.p), 0.45], [0, 0.4], [id(w), 0.02]] : [[id(ph.p), 0.9], [0, 0.05]];
      }
    }
    const flatIds: number[] = [];
    const flatProbs: number[] = [];
    for (const row of rows) {
      for (let t = 1; row.length < 5; t++) if (!row.some(([x]) => x === t)) row.push([t, 0]);
      for (const [x, q] of row) {
        flatIds.push(x);
        flatProbs.push(q);
      }
    }
    const e: PhonemeEvidence = {
      inventory: index.inventoryId,
      frameMs: FRAME_MS,
      phonemes: part,
      decodedTo: to,
      posteriors: { k: 5, startFrame: frame, nFrames: rows.length, ids: flatIds, probs: flatProbs },
    };
    frame = to;
    const events: EngineEvent[] = tracker.push(e);
    out.push(...checker.push(e, events));
  }
  out.push(...checker.end(tracker.end()));
  return {
    events: out.map((e) => `${e.state}${e.due ? "+due" : ""} ${e.errorClass} ${key(e.location)}`),
    mistakes: checker.mistakes().map((e) => `${e.errorClass} ${key(e.location)}`),
  };
}
/** The field on (it is off by default), every other field at its default. */
const ON = { ...DEFAULT, markWhenSure: true };
const sounds = (wi: number) => index.phonemes.slice(index.words[wi].start, index.words[wi].end);
const fmt = (t: Take) => t.checker.map((e) => `${e.state}${e.due ? "+due" : ""} ${e.errorClass} ${key(e.location)} @${e.timeMs}`).join(" | ");

describe("refinements.markWhenSure: a correct recitation heard on degraded audio", () => {
  // [surah, ayah, word number, the heard text, the written sounds behind its first three changed sounds]
  const cases: [number, number, number, string, string][] = [
    [36, 2, 2, "نهَفِۦۦم", "لحك"],
    [1, 2, 4, "نءَاانَمِۦۦن", "لعل"],
    [67, 2, 3, "ننَودَ", "لمت"],
  ];
  for (const [s, a, n, heard, behind] of cases) {
    it(`${s}:${a}:${n} heard "${heard}" marks nothing, due or confirmed, as with the field off`, () => {
      const before: Piece[] = [...ayahs(s, a - 1 >= 1 ? a - 1 : a, a - 1), { words: ayah(s, a).slice(0, n - 1) }];
      const pieces: Piece[] = [...before, { said: heard }, { words: ayah(s, a).slice(n) }, pause(15), ...ayahs(s, a + 1, a + 1)];
      const k0 = speech(before).phonemes.length;
      const written = sounds(ayah(s, a)[n - 1]);
      const changed = [...heard].flatMap((x, i) => (x !== written[i] ? [i] : []));
      const degrade = changed.map((i, j) => ({ from: k0 + i, written: behind[j] }));
      const p = passage(s, a - 1 >= 1 ? a - 1 : a, a + 1);
      const off = degraded(pieces, degrade, p, { markWhenSure: false });
      const on = degraded(pieces, degrade, p, ON);
      expect(off.mistakes, `off: ${off.events.join(" | ")}`).toEqual([]);
      expect(on.mistakes, `on: ${on.events.join(" | ")}`).toEqual([]);
      expect(on.events.filter((e) => e.includes("+due")), `on: ${on.events.join(" | ")}`).toEqual([]);
    });
  }
});

describe("refinements.markWhenSure: speech the alignment cannot place, then the take's end", () => {
  // 67:1, then 67:2 to its eighth word with its fourth said as 67:3's third word (a wrong word, read on past).
  const head: Piece[] = [say(67, 1), pause(15), { words: ayah(67, 2).slice(0, 3) }, { said: sounds(ayah(67, 3)[2]) }, { words: ayah(67, 2).slice(4, 8) }];
  const other = noise("زظغذثخطضصقزظغذثخطضصقزظغذثخطضصق");
  it("other speech after a stop adds no wrong words", () => {
    const sp = speech([...head, other]);
    const on = recite(sp, { passage: passage(67, 1, 3), refinements: ON });
    const off = recite(sp, { passage: passage(67, 1, 3), refinements: { markWhenSure: false } });
    expect(off.mistakes).toEqual(["wrong_word 67:2:4"]);
    expect(on.mistakes, `on: ${fmt(on)}`).toEqual(off.mistakes);
  });
  it("other speech, then a later ayah (a hold), marks what the field off marks, and nothing due that it does not confirm", () => {
    const sp = speech([...head, other, pause(15), say(67, 3)]);
    const on = recite(sp, { passage: passage(67, 1, 3), refinements: ON });
    const off = recite(sp, { passage: passage(67, 1, 3), refinements: { markWhenSure: false } });
    expect(on.mistakes, `on: ${fmt(on)}`).toEqual(off.mistakes);
    const confirmed = new Set(off.checker.filter((e) => e.state === "confirmed").map((e) => key(e.location)));
    expect(on.checker.filter((e) => e.due && !confirmed.has(key(e.location))).map((e) => key(e.location)), `on: ${fmt(on)}`).toEqual([]);
  });
});

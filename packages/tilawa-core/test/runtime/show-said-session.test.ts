import { describe, expect, it } from "vitest";
import { createEngine } from "../../src/runtime/session.js";
import { loadNodeModel } from "../../src/runtime/node.js";
import type { EngineEvent, PositionEvent } from "../../src/engine/events.js";
import type { PhonemeEvidence } from "../../src/engine/evidence.js";
import { FRAME_MS, ayah, index, key, pause, phrases, ref, say, speech } from "../engine/takes.js";
import type { Piece } from "../engine/takes.js";
import { scriptedTake } from "./scripted.js";

/**
 * refinements.showSaid through a session as an app runs it: a word the reciter pauses after shows about a syllable
 * after its end, while the pause goes on, not once the reciter speaks again. A session cuts no evidence cell while
 * nothing is heard, so the time that passes in silence reaches the tracker as `Tracker.silence`.
 */
const tiny = new URL("../fixtures/tiny", import.meta.url).pathname;
const ids = [...index.tokens];
const id = (t: string) => ids.indexOf(t) + 1;
/** Evidence as a listener delivers it: every 320 ms, the phonemes decoded in that window, `decodedTo` through silence. */
function chunks(phonemes: ReturnType<typeof speech>["phonemes"]): PhonemeEvidence[] {
  const out: PhonemeEvidence[] = [];
  for (let t = 0; t < phonemes.at(-1)!.end + 20; t += 8) {
    const rows: [number, number][][] = Array.from({ length: 8 }, () => [[0, 0.95]]);
    for (const p of phonemes) for (let f = p.start; f <= p.end; f++) if (f >= t && f < t + 8) rows[f - t] = [[id(p.p), 0.9], [0, 0.05]];
    const flatIds: number[] = [];
    const flatProbs: number[] = [];
    for (const row of rows) {
      for (let k = 1; row.length < 5; k++) if (!row.some(([x]) => x === k)) row.push([k, 0]);
      for (const [x, q] of row) {
        flatIds.push(x);
        flatProbs.push(q);
      }
    }
    out.push({
      inventory: index.inventoryId,
      frameMs: FRAME_MS,
      phonemes: phonemes.filter((p) => p.start >= t && p.start < t + 8),
      decodedTo: t + 8,
      posteriors: { k: 5, startFrame: t, nFrames: 8, ids: flatIds, probs: flatProbs },
    });
  }
  return out;
}
const wordIdx = new Map(index.words.map((x, i) => [key(x), i]));
/** When word `k` first shows: the evidence time of the first locked position whose run reaches it. */
const litAt = (events: EngineEvent[], k: string) => {
  for (const e of events) {
    if (e.type !== "position" || e.state !== "locked" || !e.observed[0]) continue;
    const c = (e as PositionEvent).observed[0];
    if (wordIdx.get(key(c.span.start))! <= wordIdx.get(k)! && wordIdx.get(k)! <= wordIdx.get(key(c.word))!) return e.timeMs;
  }
  return Infinity;
};

describe("refinements.showSaid through a session: a word followed by a pause", () => {
  // [surah, first ayah, last ayah, the ayah with the pause, the word number the pause follows]
  const cases: [number, number, number, number, number][] = [[67, 1, 3, 2, 1], [67, 1, 3, 2, 4], [2, 255, 255, 255, 12], [18, 9, 11, 10, 6]];
  for (const [s, a, b, pa, pw] of cases) {
    it(`pushEvidence: ${s}:${pa}:${pw}, then a 2 s pause, shows within 600 ms of its end`, async () => {
      const engine = createEngine({ reference: index, phrases, model: await loadNodeModel(tiny) });
      const pieces: Piece[] = [];
      for (let x = a; x <= b; x++) pieces.push(...(x === pa ? [say(s, x, 1, pw), pause(50), say(s, x, pw + 1)] : [say(s, x)]), pause(15));
      const sp = speech(pieces);
      const wi = ayah(s, pa)[pw - 1];
      const k = key(ref(wi));
      const first = sp.phonemes.findIndex((p) => p.start * FRAME_MS >= sp.saidAt.get(k)!);
      const endMs = (sp.phonemes[first + index.words[wi].end - index.words[wi].start - 1].end + 1) * FRAME_MS;
      const lit: Record<string, number> = {};
      for (const [label, refinements] of [["on", undefined], ["off", { showSaid: false }]] as const) {
        const session = engine.startMemorization({ surah: s, fromAyah: a, toAyah: b }, { refinements });
        const events: EngineEvent[] = [];
        for (const c of chunks(sp.phonemes)) events.push(...session.pushEvidence(c));
        events.push(...session.endEvidence());
        lit[label] = litAt(events, k) - endMs;
      }
      // Off, the position reaches the word as it is said.
      expect(lit.off, "off: lit by its end").toBeLessThanOrEqual(0);
      expect(lit.on, `on: lit ${lit.on} ms after its end`).toBeLessThanOrEqual(600);
    });
  }

  it("pushAudio, at any push size: the word shows during the pause, once the listener has decoded a pause's quiet", async () => {
    // The listener decodes a model window (320 ms of audio, with 130 ms more heard ahead) at a time, so what it decodes
    // reaches a session up to 450 ms after it is said, the field off too; a pause is 480 ms of quiet, known at the end
    // of the window that decodes it, or of the push that brings it when pushes are longer (up to 320 ms or a push more).
    for (const pushMs of [20, 100, 320, 1000]) {
      for (const k of ["67:2:1", "67:2:4"]) {
        const t = await scriptedTake({ pushMs, surah: 67, ayahs: [1, 2, 3], pauseAfter: { [k]: 50 } });
        const w = t.words.get(k)!;
        const x = index.words[wordIdx.get(k)!];
        const endMs = w.said + (x.end - x.start) * 40;
        const next = t.words.get(k.replace(/:(\d+)$/, (_, n) => `:${+n + 1}`))!;
        const mode = `${k}, pushes of ${pushMs} ms`;
        expect(w.shown - endMs, `${mode}: shown ${w.shown - endMs} ms after its end`).toBeLessThanOrEqual(450 + 480 + Math.max(320, pushMs));
        expect(w.shown, `${mode}: shown before the reciter goes on`).toBeLessThan(next.said);
      }
    }
  }, 120_000);
});

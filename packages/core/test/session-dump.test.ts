import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/reference.js";
import type { CheckerEvent, PositionEvent, WordRef } from "../src/events.js";
import { scoreSessionEvents } from "../src/bench/sessions.js";
import type { BenchSession } from "../src/bench/sessions.js";
import { quranText, sessionDumpContext, sessionDumpRecord } from "../src/bench/session-dump.js";
const asset = JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8"));
const index = loadReferenceIndex(asset);
const at: WordRef = { surah: 94, ayah: 5, word: 1 };
const far: WordRef = { surah: 2, ayah: 255, word: 1 };
const session = (): BenchSession => ({ version: 1, id: "two-visits", track: "memorization", source: "recorded", speaker: "one", audio: "private.wav", durationS: 12,
  context: { kind: "assignment", from: at, to: at },
  spans: [{ id: "first", kind: "recitation", from: at, to: at, startS: 1, endS: 3 }, { id: "again", kind: "repeat", from: at, to: at, startS: 5, endS: 7 }],
  words: [{ span: "first", at, startS: 1, endS: 3 }, { span: "again", at, startS: 5, endS: 7 }],
  events: [{ span: "first", location: at, errorClass: "wrong_word", state: "confirmed", startS: 2, endS: 3 }],
  nonErrors: [{ span: "again", at, kind: "repeat", startS: 5, endS: 7 }] });
const warning = (id: string, timeMs: number, state: CheckerEvent["state"]): CheckerEvent => ({ type: "checker", v: 1, id, timeMs, state, location: at, errorClass: "wrong_word", confidence: 1 });
const position = (timeMs: number, word: WordRef): PositionEvent => ({ type: "position", v: 1, timeMs, state: "locked", referenceVersion: index.version, expected: null,
  observed: [{ word, confidence: 1, span: { start: word, end: word } }] } as PositionEvent);

describe("session review dump", () => {
  it("carries Quran text, every warning state with its assignment, and wrong positions", () => {
    const s = session();
    const events = [position(1500, at), warning("a", 3000, "suspected"), warning("a", 4000, "confirmed"), warning("b", 6000, "confirmed"), position(6500, far)];
    const score = scoreSessionEvents(s, events, index);
    const rec = sessionDumpRecord(s, "sessions/x/private.wav", quranText(asset));
    const ctx = sessionDumpContext(s, events, score, index);
    expect(rec.audio).toBe("sessions/x/private.wav");
    expect(rec.words[0]).toMatchObject({ at: "94:5:1", text: asset.ayahs.find((a: any) => a.s === 94 && a.a === 5).w[0], startS: 1, endS: 3 });
    expect(rec.events).toHaveLength(1); expect(rec.nonErrors).toHaveLength(1);
    expect(ctx.warnings.map(w => [w.id, w.at, w.states.map(x => `${x.state}@${x.timeMs}`), w.eventIndex, w.confirmed, w.nonError])).toEqual([
      ["a", "94:5:1", ["suspected@3000", "confirmed@4000"], 0, true, false],
      ["b", "94:5:1", ["confirmed@6000"], null, true, true]]);
    expect(ctx.wrongPositions).toEqual([{ timeMs: 6500, observed: "2:255:1", recited: ["94:5:1"] }]);
    expect(ctx.score.falseWarnings).toBe(1); expect(ctx.score.wrongPositionEvents).toBe(1);
  });
});

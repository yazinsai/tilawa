import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "../src/reference.js";
import type { CheckerEvent, PositionEvent, WordRef } from "../src/events.js";
import { contextAdapter, scoreSessionEvents, summarizeSessions, validateSession, weakerContexts, qudSession, qudAudit } from "../src/bench/sessions.js";
import type { BenchSession } from "../src/bench/sessions.js";
const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const at: WordRef = { surah: 94, ayah: 5, word: 1 };
const session = (): BenchSession => ({ version: 1, id: "two-visits", track: "memorization", source: "recorded", speaker: "one", audio: "private.wav", durationS: 12,
  context: { kind: "assignment", from: at, to: at },
  spans: [{ id: "first", kind: "recitation", from: at, to: at, startS: 1, endS: 3 }, { id: "again", kind: "repeat", from: at, to: at, startS: 5, endS: 7 }],
  words: [{ span: "first", at, startS: 1, endS: 3 }, { span: "again", at, startS: 5, endS: 7 }],
  events: [{ span: "first", location: at, errorClass: "wrong_word", state: "confirmed", startS: 2, endS: 3 }],
  nonErrors: [{ span: "again", at, kind: "repeat", startS: 5, endS: 7 }] });
const warning = (id: string, timeMs: number, state: CheckerEvent["state"]): CheckerEvent => ({ type: "checker", v: 1, id, timeMs, state, location: at, errorClass: "wrong_word", confidence: 1 });
const position = (timeMs: number, word = at): PositionEvent => ({ type: "position", v: 1, timeMs, state: "locked", referenceVersion: index.version, expected: null,
  observed: [{ word, confidence: 1, span: { start: word, end: word } }] } as PositionEvent);
describe("session truth and scoring", () => {
  it("rejects mixed voices, train data and uncontained truth", () => {
    const s = session(); s.source = "stitched"; s.segments = [{ id: "clip", speaker: "other", audio: "clip.wav", split: "eval", timing: "alignment", startS: 0, endS: 12 }];
    expect(() => validateSession(s, index)).toThrow(/mixed speaker/);
    s.segments[0].speaker = s.speaker; s.events[0].endS = 4;
    expect(() => validateSession(s, index)).toThrow(/outside recitation/);
  });
  it("rejects duplicate spans, missing word times and NaN", () => {
    const s = session(); s.spans[1].id = "first"; expect(() => validateSession(s, index)).toThrow();
    s.spans[1].id = "again"; s.words = []; expect(() => validateSession(s, index)).toThrow();
    s.durationS = NaN; expect(() => validateSession(s, index)).toThrow();
  });
  it("counts an ever-confirmed warning once even when later resolved", () => {
    const r = scoreSessionEvents(session(), [warning("a", 3000, "suspected"), warning("a", 4000, "confirmed"), warning("a", 6000, "resolved")], index);
    expect(r.hits).toBe(1); expect(r.warnings).toBe(1); expect(r.confirmedDelayS).toEqual([1]);
  });
  it("does not match a repeat warning or a second id to the old occurrence", () => {
    const r = scoreSessionEvents(session(), [warning("a", 3000, "confirmed"), warning("b", 6000, "confirmed")], index);
    expect(r.hits).toBe(1); expect(r.falseWarnings).toBe(1); expect(r.nonErrorWarnings).toBe(1);
    const late = scoreSessionEvents(session(), [warning("b", 6000, "confirmed")], index);
    expect(late.hits).toBe(0); expect(late.misses).toBe(1);
  });
  it("does not rematch later states after a repeat starts", () => {
    const r = scoreSessionEvents(session(), [warning("a", 3000, "suspected"), warning("a", 6000, "confirmed")], index);
    expect(r.hits).toBe(1); expect(r.nonErrorWarnings).toBe(0);
  });
  it("keeps abstentions in recall denominator", () => {
    const r = scoreSessionEvents(session(), [warning("a", 3000, "suspected"), warning("a", 4000, "uncertain")], index);
    expect(r.recall).toBe(0); expect(r.uncertainErrors).toBe(1); expect(r.warnings).toBe(0);
  });
  it("reports recall per class and the mutashabihat slice with abstentions as misses", () => {
    const s = session(); s.events[0].forms_other_ayah = true;
    s.events.push({ span: "first", location: at, errorClass: "omitted_word", state: "confirmed", startS: 1, endS: 2 });
    const r = scoreSessionEvents(s, [warning("a", 3000, "confirmed"), { ...warning("b", 2500, "uncertain"), errorClass: "omitted_word" }], index);
    expect(r.byClass.wrong_word).toEqual({ errors: 1, hits: 1, recall: 1 });
    expect(r.byClass.omitted_word).toEqual({ errors: 1, hits: 0, recall: 0 });
    expect(r.byClass.skipped_passage).toEqual({ errors: 0, hits: 0, recall: null });
    expect(r.mutashabihat).toEqual({ errors: 1, hits: 1, recall: 1 });
  });
  it("splits missed errors by abstention cause: checker uncertain, twin ambiguity, general ambiguity", () => {
    const ambiguous = (timeMs: number, a: number, b: number): PositionEvent => ({ ...position(timeMs), state: "ambiguous",
      observed: [a, b].map(ayah => ({ word: { surah: 94, ayah, word: 1 }, confidence: 0.5, span: { start: at, end: at } })) } as PositionEvent);
    expect(scoreSessionEvents(session(), [warning("a", 3000, "uncertain"), ambiguous(1500, 5, 6)], index).abstentions)
      .toEqual({ uncertain: 1, twinAmbiguous: 0, ambiguous: 0 });
    expect(scoreSessionEvents(session(), [ambiguous(1500, 5, 6)], index).abstentions).toEqual({ uncertain: 0, twinAmbiguous: 1, ambiguous: 0 });
    expect(scoreSessionEvents(session(), [ambiguous(2500, 5, 1)], index).abstentions).toEqual({ uncertain: 0, twinAmbiguous: 0, ambiguous: 1 });
    expect(scoreSessionEvents(session(), [warning("a", 3000, "confirmed"), ambiguous(1500, 5, 6)], index).abstentions)
      .toEqual({ uncertain: 0, twinAmbiguous: 0, ambiguous: 0 });
    const clean = { ...session(), events: [] };
    const r = scoreSessionEvents(clean, [ambiguous(1500, 5, 6), position(2000), ambiguous(5500, 5, 1), ambiguous(5600, 5, 1), warning("u", 6000, "uncertain")], index);
    expect(r.ambiguousEpisodes).toEqual({ twin: 1, general: 1 }); expect(r.uncertainWarnings).toBe(1);
  });
  it("summarizes delays with counts and censoring, abstention shares and per-clean-session rates", () => {
    const ambiguous = (timeMs: number, a: number, b: number): PositionEvent => ({ ...position(timeMs), state: "ambiguous",
      observed: [a, b].map(ayah => ({ word: { surah: 94, ayah, word: 1 }, confidence: 0.5, span: { start: at, end: at } })) } as PositionEvent);
    const rows = [
      scoreSessionEvents(session(), [warning("a", 3500, "suspected"), warning("a", 4000, "confirmed")], index),
      scoreSessionEvents(session(), [warning("a", 4000, "suspected"), warning("a", 6000, "confirmed")], index),
      scoreSessionEvents(session(), [warning("a", 3000, "uncertain")], index),
      scoreSessionEvents(session(), [ambiguous(1500, 5, 6)], index),
      scoreSessionEvents({ ...session(), events: [] }, [ambiguous(1500, 5, 6), warning("u", 6000, "uncertain"), warning("v", 9000, "uncertain")], index),
    ];
    const [g] = summarizeSessions(rows);
    expect(g.key).toBe("memorization.assignment.recorded");
    expect(g.errors).toBe(4); expect(g.recall).toBe(0.5);
    expect(g.byClass.wrong_word).toEqual({ errors: 4, hits: 2, recall: 0.5 });
    expect(g.suspectedDelay).toEqual({ n: 2, p50: 0.5, p90: 1, censoredByMiss: 2 });
    expect(g.confirmedDelay).toEqual({ n: 2, p50: 1, p90: 3, censoredByMiss: 2 });
    expect(g.abstentionShare).toEqual({ uncertain: 0.25, twinAmbiguous: 0.25, ambiguous: 0 });
    expect(g.cleanSessions).toBe(1);
    expect(g.perCleanSession).toEqual({ uncertain: 2, twinAmbiguous: 1, ambiguous: 0 });
  });
  it("holds displayed positions, counts wrong locks and ignores long silence", () => {
    const r = scoreSessionEvents(session(), [position(1000), position(6000, { surah: 1, ayah: 1, word: 1 })], index);
    expect(r.within5S).toBe(true); expect(r.wrongPositionEvents).toBe(1);
    expect(r.locatedShare).toBeGreaterThan(0); expect(r.locatedShare).toBeLessThan(1);
    expect(r.positionSamples).toBeLessThan(110);
  });
  it("marks unsupported contexts and never derives true start", () => {
    const c = { kind: "resume", at: { surah: 2, ayah: 5, word: 1 } } as const;
    expect(contextAdapter(c).tracker.prior?.ayah).toBe(5);
    expect(weakerContexts(c).map(x => x.kind)).toEqual(["resume", "surah", "none"]);
  });
  it("maps surah and page context to a range prior", () => {
    const surah = contextAdapter({ kind: "surah", surah: 3 });
    expect(surah).toMatchObject({ supported: true, effective: "surah" });
    expect(surah.tracker.priorRange?.ranges).toEqual([{ from: { surah: 3, ayah: 1 }, to: { surah: 3, ayah: Infinity } }]);
    const page = contextAdapter({ kind: "page", page: 50 });
    expect(page).toMatchObject({ supported: true, effective: "page" });
    expect(page.tracker.priorRange?.ranges).toEqual([{ from: { surah: 3, ayah: 1 }, to: { surah: 3, ayah: 9 } }]);
  });
  it("converts QUD repeated visits without collapsing them", () => {
    const q = qudSession({ id: "q", meta: { reciter: "r" }, frameMs: 40, nFrames: 300, truth: { words: [
      { word: "94:5:1", start_s: 1, end_s: 2 }, { word: "94:5:2", start_s: 2, end_s: 3 }, { word: "94:5:1", start_s: 5, end_s: 6 }], non_quran: [] } } as any, "q.wav");
    validateSession(q, index); expect(q.spans).toHaveLength(2); expect(q.spans[1].kind).toBe("repeat");
  });
  it("retains the last word when the evidence clock rounds down", () => {
    const q = qudSession({ id: "q", meta: { reciter: "r", duration_s: 2.015 }, frameMs: 40, nFrames: 50,
      truth: { words: [{ word: "94:5:1", start_s: 1, end_s: 2.015 }], non_quran: [] } } as any, "q.wav");
    validateSession(q, index); expect(q.durationS).toBe(2.015);
  });
  const qud = (words: [string, number, number][], non_quran: [number, number][] = [], duration_s = 30) => ({ id: "q", meta: { reciter: "r", duration_s },
    frameMs: 40, nFrames: Math.round(duration_s * 25), truth: { words: words.map(([word, start_s, end_s]) => ({ word, start_s, end_s })),
      non_quran: non_quran.map(([start_s, end_s]) => ({ start_s, end_s })) } }) as any;
  it("clips truth that overruns the audio clock by annotation rounding, rejects larger overruns", () => {
    const q = qudSession(qud([["94:5:1", 1, 2], ["94:5:2", 2, 10.04]], [], 10), "q.wav");
    validateSession(q, index); expect(q.words[1].endS).toBe(10); expect(q.spans.at(-1)!.endS).toBe(10);
    const n = qudSession(qud([["94:5:1", 1, 2]], [[9.5, 10.08]], 10), "q.wav");
    validateSession(n, index); expect(n.spans.at(-1)!.endS).toBe(10);
    expect(() => qudSession(qud([["94:5:1", 1, 10.5]], [], 10), "q.wav")).toThrow(/overruns/);
  });
  it("makes a return to an earlier ayah its own restart or repeat occurrence, even across special words", () => {
    const q = qudSession(qud([["1:1:1", 1, 2], ["1:1:2", 2, 3], ["112:1:1", 4, 5], ["Basmala:1", 6, 7], ["1:1:1", 8, 9], ["1:1:2", 9, 10],
      ["94:5:1", 12, 13], ["94:6:1", 13, 14], ["94:5:2", 14, 15]]), "q.wav");
    validateSession(q, index);
    expect(q.spans.map(s => s.kind)).toEqual(["recitation", "recitation", "non_quran", "restart", "recitation", "recitation", "repeat"]);
    expect(q.nonErrors.map(n => [n.kind, n.startS])).toEqual([["restart", 8], ["restart", 9], ["repeat", 14]]);
  });
  it("clips non-Quran truth to the gap, splits the occurrence there, and rejects non-Quran covering a word", () => {
    const q = qudSession(qud([["94:5:1", 1, 2], ["94:5:2", 3, 4]], [[1.5, 2.8]]), "q.wav");
    validateSession(q, index);
    expect(q.spans.map(s => [s.kind, s.startS, s.endS])).toEqual([["recitation", 1, 2], ["non_quran", 2, 2.8], ["recitation", 3, 4]]);
    expect(() => qudSession(qud([["94:5:1", 1, 2]], [[0.5, 2.5]]), "q.wav")).toThrow(/contradict/);
    expect(() => qudSession(qud([["94:5:1", 1, 2]], [[1.2, 1.8]]), "q.wav")).toThrow(/contradict/);
  });
  it("audits each recording as converted with counts, or excluded with the reason", () => {
    const ok = qudAudit(qud([["1:1:1", 1, 2], ["112:1:1", 4, 5], ["Basmala:1", 6, 7], ["1:1:1", 8, 9], ["94:5:1", 12, 13], ["94:5:1", 14, 15]], [[20, 21]]), index);
    expect(ok.audit).toEqual({ id: "q", status: "converted", reason: null, spans: 7, repeats: 1, restarts: 1, nonQuran: 2, surahJumps: 3 });
    expect(ok.session?.id).toBe("q");
    const bad = qudAudit(qud([["94:5:1", 1, 2]], [[0.5, 2.5]]), index);
    expect(bad.session).toBeNull(); expect(bad.audit.status).toBe("excluded"); expect(bad.audit.reason).toMatch(/contradicts/);
  });
});

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { disambiguationWords } from "../src/disambiguation.js";
import type { CheckerEvent, EngineEvent, PositionEvent, WordRef } from "../src/events.js";
import { loadReferenceIndex } from "../src/reference.js";
import { COMPONENTS, VARIANTS, editDistance, exclusionsFor, labReport, listenerErrors, markDataChecks, mergeReports, oracleEvidence, recitedIntervals,
  refIndex, saidItems, scoreLabSession, scoreTracker, shardEntries, totals, trackerEntries } from "../src/bench/lab.js";
import type { LabArms, LabExclusions, LabMeta, SessionResult } from "../src/bench/lab.js";
import { buildSession, wordPhonemes } from "../src/bench/lab-fixture.js";
import type { FixturePart } from "../src/bench/lab-fixture.js";
import type { BenchSession } from "../src/bench/sessions.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const disamb = disambiguationWords(index);
const ctx = { index, disamb, phrases };
const W = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const phon = (r: WordRef) => wordPhonemes(index, r);
const r3 = (x: number) => Math.round(x * 1000) / 1000;
const build = (id: string, parts: FixturePart[]) => buildSession(index, id, parts);
const word = (s: BenchSession, r: WordRef, nth = 0) => s.words.filter(w => key(w.at) === key(r))[nth];
const confirmed = (id: string, errorClass: CheckerEvent["errorClass"], location: WordRef, timeS: number): CheckerEvent =>
  ({ type: "checker", v: 1, id, timeMs: Math.round(timeS * 1000), state: "confirmed", errorClass, location, confidence: 1 });
const locked = (timeS: number, at: WordRef, start = at): PositionEvent => ({ type: "position", v: 1, timeMs: Math.round(timeS * 1000), referenceVersion: index.version,
  expected: at, state: "locked", observed: [{ word: at, span: { start, end: at }, confidence: 1, cost: 0, per: 0 }] });
/** A tracker that is always on the word being recited. */
const follow = (s: BenchSession): PositionEvent[] => s.words.map(w => locked(w.startS + 0.01, w.at));
const sumParts = (a: Record<string, number>) => COMPONENTS.reduce((n, c) => n + a[c], 0);

// Al-Sharh with a wrong, an omitted and an added word, then a repeat of its last ayah.
const sharh = () => build("sharh", [
  { from: W(94, 1, 1), to: W(94, 8, 3), wrong: [{ at: W(94, 2, 2), said: W(114, 1, 3), mutashabihat: true }], omit: [W(94, 4, 2)], add: [{ before: W(94, 6, 2), said: W(112, 1, 2) }] },
  { kind: "repeat", from: W(94, 8, 1), to: W(94, 8, 3) },
]);

describe("lab truth", () => {
  it("says the planted words and spreads oracle evidence over the session clock", () => {
    const s = sharh();
    const said = saidItems(s, index).map(x => x.phonemes).join("");
    const ev = oracleEvidence(s, index);
    expect(ev.phonemes.map(p => p.p).join("")).toBe(said);
    expect(said).toContain(phon(W(114, 1, 3)));
    expect(said).toContain(phon(W(112, 1, 2)));
    expect(ev.nFrames * ev.frameMs / 1000).toBeCloseTo(s.durationS, 1);
    expect(ev.posteriors!.ids.length).toBe(ev.nFrames * 5);
    const w = word(s, W(94, 2, 2)), first = ev.phonemes.find(p => p.start * 20 / 1000 >= w.startS)!;
    expect(first.start * 20 / 1000).toBeLessThan(w.endS);
    expect(listenerErrors(s, ev, index)).toEqual({ edits: 0, phonemes: said.length });
  });
  it("rejects a planted wrong word without said phonemes", () => {
    const s = sharh(); delete s.events[0].said;
    expect(() => saidItems(s, index)).toThrow(/lacks said/);
  });
  it("counts recited time over words and mistakes, bridging pauses up to 2 s, minus non-Quran", () => {
    const s = build("gaps", [{ from: W(94, 1, 1), to: W(94, 1, 4), pauseAfter: 3 }, { from: W(94, 2, 1), to: W(94, 2, 3) }]);
    const iv = recitedIntervals(s);
    expect(iv).toHaveLength(2);
    expect(iv[0]).toEqual({ startS: s.words[0].startS, endS: s.words[3].endS });
    s.spans.push({ id: "noise", kind: "non_quran", label: "cough", startS: r3(s.words[3].endS + 0.5), endS: r3(s.words[3].endS + 1) });
    s.spans.sort((a, b) => a.startS - b.startS);
    expect(recitedIntervals(s)).toHaveLength(2);
    const t = build("t", [{ from: W(94, 1, 1), to: W(94, 1, 4), pauseAfter: 1 }, { from: W(94, 2, 1), to: W(94, 2, 3) }]);
    t.spans.push({ id: "noise", kind: "non_quran", label: "cough", startS: r3(t.words[3].endS + 0.2), endS: r3(t.words[3].endS + 0.8) });
    t.spans.sort((a, b) => a.startS - b.startS);
    const parts = recitedIntervals(t);
    expect(parts).toHaveLength(2);
    expect(parts[0].endS).toBe(r3(t.words[3].endS + 0.2));
    expect(parts[1].startS).toBe(r3(t.words[3].endS + 0.8));
  });
  it("measures edit distance", () => {
    expect(editDistance("kitten", "sitting")).toBe(3);
    expect(editDistance("", "abc")).toBe(3);
  });
});

describe("attribution", () => {
  const s = sharh();
  const [wrong, omitted, added] = s.events;
  const hit = (e: BenchSession["events"][number], id: string) => confirmed(id, e.errorClass as CheckerEvent["errorClass"], e.location, e.endS + 0.5);
  const tA = word(s, W(94, 3, 2)).endS, tB = word(s, W(94, 5, 3)).endS, tC = word(s, W(94, 7, 2)).endS, tR = word(s, W(94, 8, 2), 1).endS;
  const arms = (): LabArms => ({
    oracle: [hit(wrong, "o1"), hit(omitted, "o2"), confirmed("o3", "wrong_word", W(94, 3, 2), tA)],
    trackerOracle: follow(s),
    variants: { phone: { evidence: oracleEvidence(s, index),
      truePosition: [hit(wrong, "t1"), confirmed("t2", "wrong_word", W(94, 3, 2), tA + 1), confirmed("t3", "added_word", W(94, 5, 3), tB)],
      engine: [...follow(s), confirmed("e1", "wrong_word", W(94, 3, 2), tA + 2), confirmed("e2", "wrong_word", W(94, 5, 3), tB + 0.5),
        confirmed("e3", "omitted_word", W(94, 7, 2), tC), confirmed("e4", "wrong_word", W(94, 8, 2), tR)] } },
  });
  it("assigns each wrong call to the first arm that makes it: checker, then listener, then tracker", () => {
    const r = scoreLabSession(s, "practice", undefined, arms(), ctx);
    const v = r.variants.phone!;
    expect(v.missed).toBe(3); expect(v.falseWarnings).toBe(4); expect(v.nonErrorFalseWarnings).toBe(1);
    expect(v.attribution).toEqual({ listener: { missed: 1, falseWarnings: 1 }, tracker: { missed: 1, falseWarnings: 2 }, checker: { missed: 1, falseWarnings: 1 } });
    const by = Object.fromEntries(r.cases.map(c => [c.kind === "mistake" ? `${c.errorClass}` : `fw ${c.word}`, c.component]));
    expect(by).toEqual({ wrong_word: "tracker", omitted_word: "listener", added_word: "checker",
      "fw 94:3:2": "checker", "fw 94:5:3": "listener", "fw 94:7:2": "tracker", "fw 94:8:2": "tracker" });
    expect(r.cases.find(c => c.word === "94:8:2")).toMatchObject({ kind: "false_warning", nonError: "repeat" });
    expect(v.byClass.wrong_word).toEqual({ mistakes: 1, missed: 1, listener: 0, tracker: 1, checker: 0 });
    expect(v.mutashabihat).toEqual({ mistakes: 1, missed: 1, listener: 0, tracker: 1, checker: 0 });
    expect(r.checker).toEqual({ missed: 1, falseWarnings: 1, nonErrorFalseWarnings: 0 });
  });
  it("drops mistakes and false warnings inside a flagged window and keeps the parts adding up", () => {
    const windows = [{ startS: wrong.startS - 0.01, endS: wrong.endS }, { startS: tA + 1.9, endS: tA + 2.1 }];
    const r = scoreLabSession(s, "practice", undefined, arms(), ctx, windows);
    const v = r.variants.phone!;
    expect(v.mistakes).toBe(2); expect(v.missed).toBe(2); expect(v.falseWarnings).toBe(3);
    expect(v.attribution).toEqual({ listener: { missed: 1, falseWarnings: 1 }, tracker: { missed: 0, falseWarnings: 2 }, checker: { missed: 1, falseWarnings: 0 } });
    expect(r.cases.some(c => c.word === "94:2:2" || c.word === "94:3:2")).toBe(false);
    expect(r.checker).toEqual({ missed: 1, falseWarnings: 1, nonErrorFalseWarnings: 0 });
  });
  it("keys cases stably by session, variant, type, word and time", () => {
    const a = scoreLabSession(s, "practice", undefined, arms(), ctx), b = scoreLabSession(s, "practice", undefined, arms(), ctx);
    expect(a.cases.map(c => c.key)).toEqual(b.cases.map(c => c.key));
    expect(new Set(a.cases.map(c => c.key)).size).toBe(a.cases.length);
    expect(a.cases[0].key).toBe(`sharh/phone/mistake:wrong_word/94:2:2/${wrong.startS.toFixed(2)}`);
  });
  it("marks cases on data that failed the automated checks and leaves the counts alone", () => {
    const r = scoreLabSession(s, "practice", undefined, arms(), ctx), before = JSON.stringify(r.variants);
    const fail = { id: "sharh-x1", session: "sharh", type: wrong.errorClass, location: wrong.location, startS: wrong.startS, endS: wrong.endS, reasons: ["llr"] };
    markDataChecks(r, { failed: [fail, { ...fail, session: "other" }], failedClean: [{ id: "sharh-clean-3", session: "sharh", startS: tC - 0.5, endS: tC - 0.2, reasons: ["per"] }] });
    const marked = r.cases.filter(c => c.dataCheck).map(c => [c.kind === "mistake" ? c.errorClass : `fw ${c.word}`, c.dataCheck!.check]);
    expect(marked).toEqual([["wrong_word", "sharh-x1"], ["fw 94:7:2", "sharh-clean-3"]]);
    expect(JSON.stringify(r.variants)).toBe(before);
  });
  it("adds up for any arms: listener + tracker + checker = missed + false warnings", () => {
    let seed = 11;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const randomArm = (tag: string): CheckerEvent[] => [
      ...s.events.flatMap((e, i) => rnd() < 0.5 ? [hit(e, `${tag}h${i}`)] : []),
      ...Array.from({ length: Math.floor(rnd() * 6) }, (_, j) => {
        const w = s.words[Math.floor(rnd() * s.words.length)];
        return confirmed(`${tag}f${j}`, (["wrong_word", "omitted_word", "added_word"] as const)[Math.floor(rnd() * 3)], w.at, w.endS + rnd() * 3);
      }),
    ];
    for (let trial = 0; trial < 12; trial++) {
      const r = scoreLabSession(s, trial % 2 ? "exam" : "practice", undefined, { oracle: randomArm("o"), trackerOracle: null,
        variants: Object.fromEntries(VARIANTS.map(v => [v, { evidence: oracleEvidence(s, index), truePosition: randomArm("t"), engine: [...follow(s), ...randomArm("e")] }])) }, ctx);
      for (const v of VARIANTS) {
        const x = r.variants[v]!;
        expect(COMPONENTS.reduce((n, c) => n + x.attribution[c].missed, 0)).toBe(x.missed);
        expect(COMPONENTS.reduce((n, c) => n + x.attribution[c].falseWarnings, 0)).toBe(x.falseWarnings);
        expect(r.cases.filter(c => c.variant === v && c.component).length).toBe(x.missed + x.falseWarnings);
        const t = totals([r], v);
        expect(sumParts(t.attribution)).toBe(t.overall.wrongCalls);
        expect(t.attribution.total).toBe(t.overall.missed + t.overall.falseWarnings);
      }
    }
  }, 120_000);
});

describe("tracker score from the disambiguation point", () => {
  // A strict start inside the Al-Rahman refrain: 55:13 is word for word 55:16, 55:18, ... until 55:14 begins.
  const refrain = () => build("refrain", [{ from: W(55, 13, 2), to: W(55, 17, 4) }, { kind: "restart", from: W(55, 13, 1), to: W(55, 15, 4) }]);
  it("finds the point after the refrain, one word of tolerance, and a re-entry at a restart", () => {
    const s = refrain();
    const [e, again] = trackerEntries(s, index, disamb);
    const k = disamb[refIndex(index)(W(55, 13, 2))];
    expect(k).toBeGreaterThan(0);
    expect(s.words[k].at.ayah).toBe(14);
    expect(e.pointS).toBe(s.words[k].endS);
    expect(e.scoredFromS).toBe(s.words[k + 1].endS);
    expect(again.word).toEqual(W(55, 13, 1));
    expect(again.startS).toBe(s.words.find(w => w.span === "p1")!.startS);
  });
  it("counts a confident lock on the twin refrain before the point as a false jump, and the lock delay from the point", () => {
    const s = refrain();
    const [e] = trackerEntries(s, index, disamb);
    const twin = W(55, 16, 3), lateLock = r3(e.pointS! + 0.35);
    const events: EngineEvent[] = [locked(s.words[1].endS, twin, W(55, 16, 2)), locked(lateLock, W(55, 14, 2), W(55, 13, 2)),
      ...s.words.filter(w => w.startS > lateLock).map(w => locked(w.startS + 0.01, w.at))];
    const t = scoreTracker(s, events, index, disamb);
    expect(t.entries).toBe(2); expect(t.reentries).toBe(1);
    expect(t.falseJumps).toBe(1); expect(t.falseJumpsBeforePoint).toBe(1);
    expect(t.delaysS[0]).toBeGreaterThanOrEqual(0.35); expect(t.delaysS[0]).toBeLessThan(0.46);
    expect(t.delaysS).toHaveLength(2); expect(t.censored).toBe(0);
    expect(t.unscoredS).toBeGreaterThan(0);
    expect(t.onPlaceS / t.scoredS).toBeGreaterThan(0.9);
  });
  it("censors an entry the tracker never locks on and scores no time on the wrong place", () => {
    const s = refrain();
    const t = scoreTracker(s, [locked(1.2, W(2, 255, 1))], index, disamb);
    expect(t.onPlaceS).toBe(0); expect(t.censored).toBe(2); expect(t.falseJumps).toBe(1);
  });
});

describe("report and merge", () => {
  const meta: LabMeta = { testSet: "fixture", listener: "student", teacher: null, sha: "abc1234", run: null, reference: index.version, manifest: "m.json" };
  const s = sharh();
  const result = (id: string, split: "practice" | "exam"): SessionResult => {
    const t = { ...s, id };
    return scoreLabSession(t, split, id.endsWith("strict"), { oracle: [], trackerOracle: null,
      variants: { phone: { evidence: oracleEvidence(t, index), truePosition: [], engine: follow(t) }, studio: { evidence: oracleEvidence(t, index), truePosition: [], engine: follow(t) } } }, ctx);
  };
  it("headlines the exam split on phone audio and shows practice cases only", () => {
    const rs = [result("a", "practice"), result("b-strict", "exam")];
    const r = labReport(rs, meta);
    expect(r.headline).toEqual({ split: "exam", variant: "phone" });
    expect(r.overall).toEqual(r.splits.exam.overall);
    expect(r.overall.missed).toBe(3);
    expect(r.splits.all.overall.missed).toBe(6);
    expect(r.attribution).toMatchObject({ checker: 3, listener: 0, tracker: 0, total: 3 });
    expect(r.slices.strict.all.sessions).toBe(1);
    expect(r.slices.studio.practice.sessions).toBe(1);
    expect(r.cases.every(c => c.split === "practice")).toBe(true);
    expect(r.results.find(x => x.split === "exam")!.cases).toEqual([]);
    expect(r.overall.per5Min).toBeCloseTo(3 * 300 / rs[1].recitedS, 2);
  });
  it("merges shards into the same report as one run and refuses a missing shard", () => {
    const rs = [result("a", "practice"), result("b", "practice"), result("c", "exam")];
    const whole = labReport(rs, meta, { recordedAt: "x" });
    const shards = [0, 1].map(i => labReport(rs.filter((_, j) => j % 2 === i), meta, { shard: { index: i, count: 2 }, recordedAt: "x" }));
    const merged = mergeReports(JSON.parse(JSON.stringify(shards)), { recordedAt: "x" });
    expect({ ...merged, shard: null }).toEqual({ ...whole, sessions: { ...whole.sessions } });
    expect(() => mergeReports([shards[0]])).toThrow(/missing shard 1/);
  });
  it("shards one split's sessions only and reads an exclusion list by session", () => {
    const es = ["p1", "e1", "p2", "e2", "p3"].map(id => ({ id, split: id[0] === "p" ? "practice" : "exam" }));
    const ids = (shard: number, shards: number, split?: "practice" | "exam") => shardEntries(es, { split, shard, shards }).map(e => e.id);
    expect([0, 1, 2].map(i => ids(i, 3, "practice"))).toEqual([["p1"], ["p2"], ["p3"]]);
    expect(ids(0, 2)).toEqual(["p1", "p2", "p3"]);
    expect(ids(1, 2, "exam")).toEqual(["e2"]);
    const x: LabExclusions = { version: 1, kind: "lab-flags", testSet: "t.v2", base: "t",
      excluded: [{ session: "p1", startS: 3, endS: 9, item: "p1-x1" }, { session: "p2", item: "p2-phone-5" }] };
    expect(exclusionsFor(x, "p1")).toEqual({ session: false, windows: [{ startS: 3, endS: 9 }] });
    expect(exclusionsFor(x, "p2")).toEqual({ session: true, windows: [] });
    expect(exclusionsFor(null, "p3")).toEqual({ session: false, windows: [] });
  });
});

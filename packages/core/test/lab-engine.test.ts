import { readFileSync } from "node:fs";
import { beforeEach, describe, expect, it } from "vitest";
import { disambiguationWords } from "../src/disambiguation.js";
import type { WordRef } from "../src/events.js";
import { loadReferenceIndex } from "../src/reference.js";
import { COMPONENTS, labReport, labSession, oracleEvidence, totals } from "../src/bench/lab.js";
import { buildSession, deafToWrongWords, donorWord, lastWordOf } from "../src/bench/lab-fixture.js";
import type { FixturePart } from "../src/bench/lab-fixture.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const phrases = JSON.parse(readFileSync(new URL("../assets/phrases-v1.json", import.meta.url), "utf8"));
const ctx = { index, disamb: disambiguationWords(index), phrases };
const W = (surah: number, ayah: number, word: number): WordRef => ({ surah, ayah, word });
const lastWord = (surah: number, ayah: number) => lastWordOf(index, surah, ayah);
const donor = (at: WordRef) => donorWord(index, at);
const build = (id: string, parts: FixturePart[]) => buildSession(index, id, parts);
const sumParts = (a: Record<string, number>) => COMPONENTS.reduce((n, c) => n + a[c], 0);

// Each test runs the engine synchronously for many seconds; without a macrotask between tests the worker can't answer vitest's RPC in time.
beforeEach(() => new Promise<void>(r => setImmediate(r)));

describe("engine arms on synthetic sessions", () => {
  // Al-Mulk 1-11: wrong (a mutashabihat-tagged swap), omitted and added words, a skipped ayah, then a repeat.
  const mulk = () => build("mulk", [
    { from: W(67, 1, 1), to: lastWord(67, 11), wrong: [{ at: W(67, 3, 4), said: donor(W(67, 3, 4)), mutashabihat: true }], omit: [W(67, 5, 3)],
      add: [{ before: W(67, 7, 2), said: W(112, 1, 2) }], skip: [[W(67, 9, 1), lastWord(67, 9)]] },
    { kind: "repeat", from: W(67, 11, 1), to: lastWord(67, 11) },
  ]);
  it("gives a perfect listener no losses and PER 0, with the tracker on place from the disambiguation point", () => {
    const s = mulk();
    const r = labSession({ session: s, split: "practice", strict: false, evidence: { studio: oracleEvidence(s, index) } }, ctx);
    const v = r.variants.studio!;
    expect(v.listener.edits).toBe(0);
    expect(v.attribution.listener).toEqual({ missed: 0, falseWarnings: 0 });
    expect(v.mistakes).toBe(4);
    expect(COMPONENTS.reduce((n, c) => n + v.attribution[c].missed + v.attribution[c].falseWarnings, 0)).toBe(v.missed + v.falseWarnings);
    expect(r.checker.missed).toBe(0);
    expect(v.tracker.entries).toBe(2);
    expect(v.tracker.onPlaceS / v.tracker.scoredS).toBeGreaterThan(0.8);
    expect(r.trackerOracle!.onPlaceS / r.trackerOracle!.scoredS).toBeGreaterThan(0.8);
  }, 300_000);
  it("blames the listener for a wrong word it decodes as written", () => {
    const s = mulk();
    const r = labSession({ session: s, split: "practice", strict: false, evidence: { phone: oracleEvidence(deafToWrongWords(s, index), index) }, trackerOracle: false }, ctx);
    const v = r.variants.phone!;
    expect(v.listener.edits).toBeGreaterThan(0);
    const swap = r.cases.find(c => c.kind === "mistake" && c.errorClass === "wrong_word")!;
    expect(swap).toMatchObject({ oracle: "hit", truePosition: "miss", outcome: "miss", component: "listener", mutashabihat: true });
    expect(v.mutashabihat).toEqual({ mistakes: 1, missed: 1, listener: 1, tracker: 0, checker: 0 });
    const t = totals([r], "phone");
    expect(sumParts(t.attribution)).toBe(t.overall.wrongCalls);
  }, 300_000);
  it("scores a strict start inside the Al-Rahman refrain with a late planted mistake", () => {
    const s = build("rahman", [{ from: W(55, 13, 2), to: lastWord(55, 25), wrong: [{ at: W(55, 22, 2), said: W(114, 1, 3) }] }]);
    const r = labSession({ session: s, split: "exam", strict: true, evidence: { phone: oracleEvidence(s, index) } }, ctx);
    const v = r.variants.phone!;
    expect(r.strict).toBe(true);
    expect(v.tracker.entries).toBe(1);
    expect(v.tracker.delaysS.length + v.tracker.censored).toBe(1);
    expect(v.missed + v.falseWarnings).toBe(COMPONENTS.reduce((n, c) => n + v.attribution[c].missed + v.attribution[c].falseWarnings, 0));
    const report = labReport([r], { testSet: "fixture", listener: "oracle", teacher: null, sha: "x", run: null, reference: index.version, manifest: "-" });
    expect(report.slices.strict.exam.sessions).toBe(1);
    expect(report.cases).toEqual([]);
  }, 300_000);
  it("blames the checker when even oracle evidence and the true position give the wrong call", () => {
    // A short donor in a long word's slot: the checker hears an omission, not a wrong word, whatever its inputs.
    const s = build("short-donor", [{ from: W(67, 1, 1), to: lastWord(67, 4), wrong: [{ at: W(67, 3, 4), said: W(114, 1, 3) }] }]);
    const r = labSession({ session: s, split: "practice", evidence: { studio: oracleEvidence(s, index) }, trackerOracle: false }, ctx);
    const v = r.variants.studio!;
    expect(r.cases.find(c => c.kind === "mistake")).toMatchObject({ outcome: "miss", oracle: "miss", component: "checker" });
    expect(r.cases.filter(c => c.kind === "false_warning" && c.word === "67:3:4").every(c => c.component === "checker")).toBe(true);
    expect(v.attribution.listener).toEqual({ missed: 0, falseWarnings: 0 });
    expect(r.checker.missed).toBe(1);
  }, 300_000);
});

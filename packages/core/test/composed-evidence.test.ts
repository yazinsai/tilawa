import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { composedReversal, createChecker, createTracker, loadReferenceIndex, trackerInput } from "../src/index.js";
import type { CheckerEvent, CheckerOptions, PhonemeEvidence, WrongWordCandidate, WrongWordEvidence } from "../src/index.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import { parallelPaths } from "../src/parallel-phrases.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const word = (s: number, a: number, w: number) => index.words.find(x => x.surah === s && x.ayah === a && x.word === w)!;
const from = word(2, 255, 1), target = word(2, 255, 23), end = word(2, 256, 10), alien = word(108, 1, 3);
const said = index.phonemes.slice(alien.start, alien.end);
const fixture = syntheticFixture(index, { id: "channel-certificate", from, to: end, chunkSize: 8, edits: [
  { kind: "del", at: target.start - from.start, count: target.end - target.start },
  { kind: "ins", at: target.start - from.start, text: said },
] });
const chunks = fixture.evidence.map(e => withPosteriors(index, e));
const certificate = (c: WrongWordCandidate): WrongWordEvidence => ({ ...c,
  greedyWritten: -5, greedyAccused: -4, composedWritten: -3, composedAccused: -4 });
function replay(evidence = chunks, options: CheckerOptions = {}) {
  const tracker = createTracker(index), checker = createChecker(index, options), positions = [];
  const events: CheckerEvent[] = [];
  for (const e of evidence) { const t = tracker.push(trackerInput(e)); positions.push(...t); events.push(...checker.push(e, t)); }
  const t = tracker.end(); positions.push(...t); events.push(...checker.end(t));
  return { positions, events, mistakes: checker.mistakes() };
}

describe("composed listener certificates", () => {
  it("requires finite strict preferences in opposite directions", () => {
    const c = certificate({ word: target, written: "بت", accused: "بث", fromFrame: 0, toFrame: 3 });
    expect(composedReversal(c)).toBe(true);
    expect(composedReversal({ ...c, greedyWritten: c.greedyAccused })).toBe(false);
    expect(composedReversal({ ...c, composedWritten: c.composedAccused })).toBe(false);
    expect(composedReversal({ ...c, composedWritten: Infinity })).toBe(false);
  });

  it("only vetoes an already supported string-path warning and preserves the tracker", () => {
    const baseline = replay(), seen: WrongWordCandidate[] = [];
    const judged = replay(chunks, { wrongWordEvidence(c) { seen.push(c); return certificate(c); } });
    expect(baseline.mistakes.some(e => e.errorClass === "wrong_word")).toBe(true);
    expect(seen.length).toBeGreaterThan(0);
    expect(judged.mistakes.filter(e => e.errorClass === "wrong_word")).toEqual([]);
    expect(judged.positions).toEqual(baseline.positions);
    expect(judged.events).toEqual(baseline.events.map(e => e.state === "confirmed" && e.errorClass === "wrong_word"
      ? { ...e, state: "uncertain" } : e));
    const off = replay(chunks, { wrongWordEvidence(c) { return { ...certificate(c), greedyWritten: -3 }; } });
    expect(off).toEqual(baseline);
    const wrongWindow = replay(chunks, { wrongWordEvidence(c) { return { ...certificate(c), toFrame: c.toFrame + 1 }; } });
    expect(wrongWindow).toEqual(baseline);
  });

  it("replays serialized certificates only for their exact complete hypotheses and interval", () => {
    const found: WrongWordEvidence[] = [];
    const baseline = replay(chunks, { wrongWordEvidence(c) { found.push(certificate(c)); return null; } });
    const carry = (metadata: WrongWordEvidence[]): PhonemeEvidence[] => chunks.map((e, i) => i ? e : { ...e,
      phonemes: e.phonemes.map((p, j) => j ? p : { ...p, wrongWordEvidence: metadata }) });
    const judged = replay(JSON.parse(JSON.stringify(carry(found))));
    expect(found.length).toBeGreaterThan(0);
    expect(judged.mistakes.filter(e => e.errorClass === "wrong_word")).toEqual([]);
    expect(judged.positions).toEqual(baseline.positions);
    expect(replay(carry(found.map(c => ({ ...c, accused: c.accused + "ب" }))))).toEqual(baseline);
  });

  it("never consults the comparison on additions", () => {
    const f = syntheticFixture(index, { id: "insert-control", from, to: end, chunkSize: 8,
      edits: [{ kind: "ins", at: target.start - from.start, text: said }] });
    let queries = 0;
    const result = replay(f.evidence.map(e => withPosteriors(index, e)), {
      wrongWordEvidence(c) { queries++; return certificate(c); },
    });
    expect(result.mistakes.some(e => e.errorClass === "added_word")).toBe(true);
    expect(queries).toBe(0);
  });

  it("preserves every other decision when a swap and an addition share the session", () => {
    const extra = word(2, 255, 40);
    const f = syntheticFixture(index, { id: "swap-and-insert-control", from, to: end, chunkSize: 3, edits: [
      { kind: "del", at: target.start - from.start, count: target.end - target.start },
      { kind: "ins", at: target.start - from.start, text: said },
      { kind: "ins", at: extra.start - from.start, text: said },
    ] });
    const evidence = f.evidence.map(e => withPosteriors(index, e));
    const baseline = replay(evidence), judged = replay(evidence, { wrongWordEvidence: certificate });
    expect(baseline.mistakes.some(e => e.errorClass === "added_word")).toBe(true);
    expect(judged.events).toEqual(baseline.events.map(e => e.state === "confirmed" && e.errorClass === "wrong_word"
      ? { ...e, state: "uncertain" } : e));
    expect(judged.positions).toEqual(baseline.positions);
  });

  it("preserves a posterior-confirmed parallel-graph swap", () => {
    // The shorter parallel is certified by the graph. The longer direction is
    // already caught by a word-local string accusation before the graph closes.
    const at = index.words.indexOf(word(37, 119, 2));
    const shorter = word(37, 108, 2);
    const path = parallelPaths(index).get(at)!.find(p => p.altWord === index.phonemes.slice(shorter.start, shorter.end))!;
    const start = index.words[path.start], through = index.words[path.start + 5], target = index.words[path.word];
    const f = syntheticFixture(index, { id: "graph-control", from: start, to: through, chunkSize: 6, edits: [
      { kind: "del", at: target.start - start.start, count: target.end - target.start },
      { kind: "ins", at: target.start - start.start, text: path.altWord },
    ] });
    let queries = 0;
    const run = (veto: boolean) => {
      const checker = createChecker(index, { passage: { from: start, to: through }, policy: { confirm: { after: "next-word" } },
        ...(veto && { wrongWordEvidence(c: WrongWordCandidate) { queries++; return certificate(c); } }) });
      for (const e of f.evidence) checker.push(withPosteriors(index, e), []);
      checker.end();
      return checker.mistakes();
    };
    expect(run(false).some(e => e.errorClass === "wrong_word")).toBe(true);
    expect(run(true)).toEqual(run(false));
    expect(queries).toBe(0);
  });
});

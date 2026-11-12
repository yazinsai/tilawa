import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, type SequenceEvidenceCandidate } from "../src/checker.js";
import { loadReferenceIndex } from "../src/reference.js";
import { parallelPaths } from "../src/parallel-phrases.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { CheckerEvent } from "../src/events.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 2, ayah: 255, word: 1 }, to = { surah: 2, ayah: 256, word: 10 };
const start = index.words.find(w => w.surah === from.surah && w.ayah === from.ayah && w.word === from.word)!;
const target = index.words.find(w => w.surah === 2 && w.ayah === 255 && w.word === 23)!;
const fixture = () => syntheticFixture(index, { id: "reconstruction-hook", from, to, chunkSize: 8, edits: [
  { kind: "del", at: target.start-start.start, count: target.end-target.start },
  { kind: "ins", at: target.start-start.start, text: "ءَعطَۦۦنَاا" },
] }).evidence.map(e => withPosteriors(index, e));
function run(sequenceEvidence?: (candidate: SequenceEvidenceCandidate) => boolean, posteriors = true) {
  const checker = createChecker(index, { passage: { from, to }, sequenceEvidence, policy: { confirm: { after: "next-word" } } });
  const events: CheckerEvent[] = [];
  for (const chunk of fixture()) {
    if (!posteriors) delete chunk.posteriors;
    events.push(...checker.push(chunk, []));
  }
  events.push(...checker.end());
  return { events, mistakes: checker.mistakes() };
}
describe("listener reconstruction hook", () => {
  it("reports exact complete hypotheses without altering the baseline", () => {
    const candidates: SequenceEvidenceCandidate[] = [];
    expect(run(c => { candidates.push(c); return true; })).toEqual(run());
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.every(c => c.written.length && c.rival.length && c.toFrame > c.fromFrame)).toBe(true);
  });
  it("vetoes only final wrong-word verdicts and excludes them from mistakes", () => {
    const before = run(), after = run(() => false);
    expect(before.mistakes.some(e => e.errorClass === "wrong_word")).toBe(true);
    expect(after.events).toEqual(before.events.map(e => e.state === "confirmed" && e.errorClass === "wrong_word" ? { ...e, state: "uncertain" } : e));
    expect(after.mistakes).toEqual(before.mistakes.filter(e => e.errorClass !== "wrong_word"));
  });
  it("does not recheck accusations without posterior evidence", () => {
    const candidates: SequenceEvidenceCandidate[] = [];
    expect(run(c => { candidates.push(c); return false; }, false)).toEqual(run(undefined, false));
    expect(candidates).toEqual([]);
  });
  it("preserves graph confirmations even when every string comparison is rejected", () => {
    const paths = parallelPaths(index);
    const wi = index.words.findIndex(w => w.surah === 84 && w.ayah === 25 && w.word === 1);
    const path = paths.get(wi)!.find(p => p.altWord === "ءِننننَ")!;
    const first = index.words[path.start], last = index.words[path.start+5], changed = index.words[path.word];
    const chunks = syntheticFixture(index, { id: "graph-preservation", from: first, to: last, chunkSize: 6, edits: [
      { kind: "del", at: changed.start-first.start, count: changed.end-changed.start },
      { kind: "ins", at: changed.start-first.start, text: path.altWord },
    ] }).evidence.map(e => withPosteriors(index, e, j => e.phonemes[j].p === "ن" ? [["ن", .6], ["ل", .05], ["", .3]] : undefined));
    const candidates: SequenceEvidenceCandidate[] = [];
    const checker = createChecker(index, { passage: { from: first, to: last }, policy: { confirm: { after: "delay", ms: 60000 } },
      sequenceEvidence(c) { candidates.push(c); return false; } });
    for (const e of chunks) checker.push(e, []);
    checker.end();
    expect(checker.mistakes().some(e => e.errorClass === "wrong_word")).toBe(true);
    expect(candidates).toEqual([]);
  });
});

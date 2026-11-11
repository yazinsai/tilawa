import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker } from "../src/checker.js";
import type { WrongWordRecheckCandidate } from "../src/checker.js";
import type { AntialiasEvidence, PhonemeEvidence } from "../src/evidence.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import { loadReferenceIndex } from "../src/reference.js";
import { parallelPaths } from "../src/parallel-phrases.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 2, ayah: 255, word: 1 }, to = { surah: 2, ayah: 256, word: 10 };
const target = index.words.find(w => w.surah === 2 && w.ayah === 255 && w.word === 23)!;
const first = index.words.find(w => w.surah === from.surah && w.ayah === from.ayah && w.word === from.word)!;
const alien = index.words.find(w => w.surah === 108 && w.ayah === 1 && w.word === 3)!;
const fixture = syntheticFixture(index, { id: "antialias-unit", from, to, edits: [
  { kind: "del", at: target.start - first.start, count: target.end - target.start },
  { kind: "ins", at: target.start - first.start, text: index.phonemes.slice(alien.start, alien.end) },
] });
const primary = fixture.evidence.map(e => withPosteriors(index, e));

function run(chunks: PhonemeEvidence[], candidates?: WrongWordRecheckCandidate[]) {
  const checker = createChecker(index, { passage: { from, to }, wrongWordRecheck: candidates ? c => {
    candidates.push(c); return true;
  } : undefined });
  for (const e of chunks) checker.push(e, []);
  checker.end();
  return checker.mistakes();
}
function certified(change: (c: AntialiasEvidence) => AntialiasEvidence = c => c) {
  const candidates: WrongWordRecheckCandidate[] = [];
  expect(run(primary, candidates).some(e => e.errorClass === "wrong_word")).toBe(true);
  expect(candidates.length).toBeGreaterThan(0);
  const metadata = candidates.map(c => change({ ...c, recovered: c.wordWritten, writtenNll: 1, accusedNll: 30 }));
  return primary.map((e, i) => i ? e : { ...e,
    phonemes: e.phonemes.map((p, j) => j ? p : { ...p, antialiasRechecks: metadata }) });
}

describe("antialias certificates on ordinary evidence", () => {
  it("survives JSON transport and vetoes only a positive complete written reading", () => {
    const enriched = certified();
    expect(run(JSON.parse(JSON.stringify(enriched))).some(e => e.errorClass === "wrong_word")).toBe(false);
    expect(enriched.map(e => ({ ...e, phonemes: e.phonemes.map(({ antialiasRechecks: _, ...p }) => p) })))
      .toEqual(primary);
  });
  it("keeps the accusation for an incomplete reading, a mismatched interval, or retained rival support", () => {
    for (const modify of [
      (c: AntialiasEvidence) => ({ ...c, recovered: "" }),
      (c: AntialiasEvidence) => ({ ...c, toFrame: c.toFrame + 1 }),
      (c: AntialiasEvidence) => ({ ...c, writtenNll: 1000 }),
    ]) expect(run(certified(modify)).some(e => e.errorClass === "wrong_word")).toBe(true);
  });
  it("preserves a parallel-phrase confirmation even when every string recheck vetoes", () => {
    const wi = index.words.findIndex(w => w.surah === 37 && w.ayah === 108 && w.word === 2);
    const path = parallelPaths(index).get(wi)!.find(p => p.altWord.length >
      index.words[wi].end - index.words[wi].start)!;
    const from = index.words[path.start], to = index.words[path.start + 5], target = index.words[wi];
    const fx = syntheticFixture(index, { id: "antialias-graph-unit", from, to, chunkSize: 6, edits: [
      { kind: "del", at: target.start - from.start, count: target.end - target.start },
      { kind: "ins", at: target.start - from.start, text: path.altWord },
    ] });
    const graphRun = (veto: boolean) => {
      const checker = createChecker(index, { passage: { from, to },
        policy: { confirm: { after: "next-word" } }, wrongWordRecheck: veto ? () => false : undefined });
      for (const e of fx.evidence) checker.push(withPosteriors(index, e), []);
      checker.end();
      return checker.mistakes();
    };
    const baseline = graphRun(false);
    expect(baseline.some(e => e.errorClass === "wrong_word" && e.location.word === target.word)).toBe(true);
    expect(graphRun(true)).toEqual(baseline);
  });
});

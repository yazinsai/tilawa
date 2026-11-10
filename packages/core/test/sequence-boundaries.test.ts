import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, createTracker, loadReferenceIndex, trackerInput } from "../src/index.js";
import type { SequenceBoundaryEvidence, SequenceEvidenceCandidate } from "../src/index.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 2, ayah: 255, word: 1 }, target = { ...from, word: 23 }, to = { surah: 2, ayah: 256, word: 10 };
const word = (r: typeof from) => index.words.find(w => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word)!;
const written = word(target), alien = word({ surah: 108, ayah: 1, word: 3 });
const fixture = syntheticFixture(index, { id: "boundary-certificate", from, to, chunkSize: 8,
  edits: [{ kind: "del", at: written.start - word(from).start, count: written.end - written.start },
    { kind: "ins", at: written.start - word(from).start, text: index.phonemes.slice(alien.start, alien.end) }] });
const evidence = fixture.evidence.map(e => withPosteriors(index, e));
function run(metadata: SequenceBoundaryEvidence[] = [], hook?: (c: SequenceEvidenceCandidate) => boolean) {
  const tracker = createTracker(index), checker = createChecker(index, { sequenceEvidence: hook });
  for (const [i, e] of evidence.entries()) {
    const enriched = i ? e : { ...e, phonemes: e.phonemes.map((p, j) => j ? p : { ...p, sequenceBoundaries: metadata }) };
    checker.push(enriched, tracker.push(trackerInput(enriched)));
  }
  checker.end(tracker.end());
  return checker.mistakes().filter(e => e.errorClass === "wrong_word" && e.location.word === target.word);
}
const certificate = (c: SequenceEvidenceCandidate): SequenceBoundaryEvidence => ({ word: c.word, written: c.written, accused: c.accused,
  fromFrame: c.fromFrame, toFrame: c.toFrame, resetMargin: c.margin, conditionalMargin: -1 });

describe("listener boundary certificate consumption", () => {
  it("only vetoes the exact complete comparison, after ordinary support clears", () => {
    const candidates: SequenceEvidenceCandidate[] = [];
    expect(run([], c => { candidates.push(c); return true; })).toHaveLength(1);
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.every(c => c.margin >= c.need)).toBe(true);
    const metadata = candidates.map(certificate);
    expect(run(JSON.parse(JSON.stringify(metadata)))).toEqual([]);
    expect(run(metadata.map(c => ({ ...c, fromFrame: c.fromFrame + 1 })))).toHaveLength(1);
    expect(run(metadata.map(c => ({ ...c, accused: c.accused + "ب" })))).toHaveLength(1);
    expect(run(metadata.map(c => ({ ...c, conditionalMargin: 1 })))).toHaveLength(1);
  });
  it("allows an online listener veto without changing the primary stream", () => {
    expect(run([], () => false)).toEqual([]);
    expect(run()).toHaveLength(1);
  });
});

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, createTracker, loadReferenceIndex } from "../src/index.js";
import type { CheckerEvent, HeldBandEvidence, WordEvidenceRequest } from "../src/index.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 2, ayah: 255, word: 1 }, to = { surah: 2, ayah: 256, word: 10 };
const word = (surah: number, ayah: number, word: number) => index.words.find(w => w.surah === surah && w.ayah === ayah && w.word === word)!;
const target = word(2, 255, 23), alien = word(108, 1, 3);
const observed = index.phonemes.slice(alien.start, alien.end);
const fixture = () => {
  const f = syntheticFixture(index, { id: "held-band-unit", from, to, chunkSize: 100000, edits: [
    { kind: "del", at: target.start - word(2, 255, 1).start, count: target.end - target.start },
    { kind: "ins", at: target.start - word(2, 255, 1).start, text: observed },
  ] });
  return f.evidence.map(e => withPosteriors(index, e));
};

function run(metadata: HeldBandEvidence[] = [], callback?: (c: WordEvidenceRequest) => boolean) {
  const evidence = fixture();
  evidence[0].phonemes[0].heldBands = metadata;
  const tracker = createTracker(index);
  const checker = createChecker(index, { wordEvidence: callback });
  const events: CheckerEvent[] = [], positions = [];
  for (const e of evidence) {
    const p = tracker.push(e); positions.push(...p); events.push(...checker.push(e, p));
  }
  const p = tracker.end(); positions.push(...p); events.push(...checker.end(p));
  return { events, positions, mistakes: checker.mistakes() };
}

describe("held-band listener certificate", () => {
  it("vetoes only the published substitution and carries the exact full hypotheses and window", () => {
    const requests: WordEvidenceRequest[] = [];
    const baseline = run([], c => { requests.push(c); return true; });
    expect(baseline.events.some(e => e.errorClass === "wrong_word" && e.state === "confirmed")).toBe(true);
    expect(requests.length).toBeGreaterThan(0);
    expect(requests.every(c => c.written && c.observed && c.toFrame > c.fromFrame)).toBe(true);
    const judged = run(requests.map(c => ({ ...c, margins: [1, 2] })));
    expect(judged.events.filter(e => e.errorClass === "wrong_word" && e.state === "confirmed")).toEqual([]);
    expect(judged.events.some(e => e.state === "uncertain")).toBe(true);
    expect(judged.mistakes.filter(e => e.errorClass === "wrong_word")).toEqual([]);
    expect(judged.positions).toEqual(baseline.positions);
    expect(judged.events.filter(e => e.errorClass !== "wrong_word")).toEqual(baseline.events.filter(e => e.errorClass !== "wrong_word"));
  });

  it("ignores one-partition disagreement, nonfinite scores and mismatched complete hypotheses or frames", () => {
    const requests: WordEvidenceRequest[] = [];
    const baseline = run([], c => { requests.push(c); return true; });
    for (const change of [
      (c: WordEvidenceRequest): HeldBandEvidence => ({ ...c, margins: [1, -1] }),
      (c: WordEvidenceRequest): HeldBandEvidence => ({ ...c, margins: [1, NaN] }),
      (c: WordEvidenceRequest): HeldBandEvidence => ({ ...c, written: c.written + "ب", margins: [1, 1] }),
      (c: WordEvidenceRequest): HeldBandEvidence => ({ ...c, fromFrame: c.fromFrame + 1, margins: [1, 1] }),
    ]) expect(run(requests.map(change)).events).toEqual(baseline.events);
  });
});

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { createChecker, type CheckerOptions, type PendingEditWindow } from "../src/checker.js";
import type { PendingEditEvidence, PhonemeEvidence } from "../src/evidence.js";
import type { CheckerEvent, WordRef } from "../src/events.js";
import { loadReferenceIndex } from "../src/reference.js";
import { syntheticFixture, withPosteriors } from "../src/fixtures/index.js";
import type { Edit } from "../src/fixtures/index.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../assets/ref-v2.json", import.meta.url), "utf8")));
const from: WordRef = { surah: 2, ayah: 255, word: 18 };
const target: WordRef = { surah: 2, ayah: 255, word: 23 };
const wi = (r: WordRef) => index.words.findIndex(w => w.surah === r.surah && w.ayah === r.ayah && w.word === r.word);
const text = (r: WordRef) => { const w = index.words[wi(r)]; return index.phonemes.slice(w.start, w.end); };
const offset = index.words[wi(target)].start - index.words[wi(from)].start;
const wrong: Edit[] = [
  { kind: "sub", at: offset + text(target).indexOf("ش"), to: "غ" },
  { kind: "sub", at: offset + text(target).indexOf("ف"), to: "ق" },
];
const passage = { from, to: { surah: 2, ayah: 257, word: 8 } };

function evidence(edits = wrong): PhonemeEvidence[] {
  const fixture = syntheticFixture(index, { id: "pending-edits", ...passage, edits, chunkSize: 4 });
  const phonemes = fixture.evidence.flatMap(e => e.phonemes);
  // This test needs an unambiguous baseline accusation, above the nonlexical
  // support gate; it tests transport and timing rather than acoustic ambiguity.
  const whole = withPosteriors(index, { ...fixture.evidence[0], phonemes }, j => [[phonemes[j].p, .999], ["", .0005]]);
  const p = whole.posteriors!;
  let a = p.startFrame;
  return fixture.evidence.map((e, i) => {
    const b = fixture.evidence[i + 1]?.phonemes[0].start ?? p.startFrame + p.nFrames;
    const out = { ...e, posteriors: { k: p.k, startFrame: a, nFrames: b - a,
      ids: Array.from(p.ids).slice((a - p.startFrame) * p.k, (b - p.startFrame) * p.k),
      probs: Array.from(p.probs).slice((a - p.startFrame) * p.k, (b - p.startFrame) * p.k) } };
    a = b;
    return out;
  });
}
function run(chunks: PhonemeEvidence[], options: CheckerOptions = {}) {
  const checker = createChecker(index, { passage, ...options });
  const events: CheckerEvent[] = [];
  for (const e of chunks) events.push(...checker.push(e, []));
  events.push(...checker.end());
  return { events, mistakes: checker.mistakes() };
}
const accused = (es: CheckerEvent[]) => es.filter(e => e.location.surah === target.surah &&
  e.location.ayah === target.ayah && e.location.word === target.word && e.errorClass === "wrong_word");

describe("joint listener decisions at the horizon", { timeout: 60_000 }, () => {
  it("retains the complete original accusation and its timestamp while leaving primary decisions intact", () => {
    const chunks = evidence(), baseline = run(chunks);
    const comparisons: { window: PendingEditWindow; available: number }[] = [];
    const control = run(chunks, { jointPendingEdit(window, frames, available) {
      expect(frames.startFrame).toBeLessThanOrEqual(window.fromFrame);
      expect(frames.endFrame).toBeGreaterThanOrEqual(window.toFrame);
      comparisons.push({ window, available });
      return false;
    } });
    expect(accused(baseline.events).some(e => e.state === "confirmed")).toBe(true);
    expect(comparisons.some(c => c.window.target.word === wi(target))).toBe(true);
    expect(control.events).toEqual(baseline.events);
    expect(control.mistakes).toEqual(baseline.mistakes);
    const veto = run(chunks, { jointPendingEdit: () => true });
    expect(accused(veto.events).at(-1)).toEqual({ ...accused(baseline.events).at(-1), state: "uncertain" });
    expect(accused(veto.mistakes)).toHaveLength(0);
  });

  it("transports only a matching positive certificate and waits until its available frame", () => {
    const chunks = evidence();
    let certificate: PendingEditEvidence | undefined;
    run(chunks, { jointPendingEdit(window, _frames, availableFrame) {
      if (window.target.word === wi(target)) certificate = { word: target, observed: window.target.observed,
        fromFrame: window.target.fromFrame, toFrame: window.target.toFrame, availableFrame,
        replacement: .2, correctWithNeighbour: .7, independentReplacement: .9 };
      return false;
    } });
    expect(certificate).toBeDefined();
    const carry = (c: PendingEditEvidence) => chunks.map((e, i) => i ? e : { ...e,
      phonemes: e.phonemes.map((p, j) => j ? p : { ...p, pendingEdits: [c] }) });
    const baseline = run(chunks);
    const valid = run(carry(certificate!));
    expect(accused(valid.events).at(-1)).toEqual({ ...accused(baseline.events).at(-1), state: "uncertain" });
    const checker = createChecker(index, { passage });
    for (const e of carry(certificate!)) {
      const out = accused(checker.push(e, []));
      if ((e.phonemes.at(-1)!.end + 1) < certificate!.availableFrame)
        expect(out.filter(event => event.state !== "suspected")).toHaveLength(0);
    }
    for (const invalid of [
      { ...certificate!, independentReplacement: .5 },
      { ...certificate!, correctWithNeighbour: .1 },
      { ...certificate!, replacement: .8, correctWithNeighbour: .9 },
      { ...certificate!, fromFrame: certificate!.fromFrame + 1 },
      { ...certificate!, observed: certificate!.observed + "ب" },
      { ...certificate!, availableFrame: chunks.at(-1)!.phonemes.at(-1)!.end + 100 },
    ]) expect(run(carry(invalid))).toEqual(baseline);
  });

  it("keeps decoded-only evidence and lexical substitutions on their existing confirmation paths", () => {
    const decoded = evidence().map(({ posteriors, ...e }) => e);
    let called = false;
    expect(run(decoded, { jointPendingEdit: () => { called = true; return true; } })).toEqual(run(decoded));
    expect(called).toBe(false);
    const alien = text({ surah: 108, ayah: 1, word: 3 });
    const chunks = evidence([{ kind: "del", at: offset, count: text(target).length },
      { kind: "ins", at: offset, text: alien }]);
    expect(accused(run(chunks).events).some(e => e.state === "confirmed")).toBe(true);
    expect(run(chunks, { jointPendingEdit: () => true })).toEqual(run(chunks));
  });

  it("retains a later lexical hypothesis across commits without changing its confirmation", () => {
    const neighbour = { surah: 2, ayah: 256, word: 10 };
    const at = index.words[wi(neighbour)].start - index.words[wi(from)].start;
    const alien = text({ surah: 108, ayah: 1, word: 3 });
    const chunks = evidence([...wrong, { kind: "del", at, count: text(neighbour).length },
      { kind: "ins", at, text: alien }]);
    const baseline = run(chunks);
    let comparison: PendingEditWindow | undefined;
    const veto = run(chunks, { jointPendingEdit(window) { comparison = window; return true; } });
    expect(comparison).toBeDefined();
    expect(comparison!.neighbours.some(e => e.word === wi(neighbour))).toBe(true);
    expect(comparison!.wordTo).toBeGreaterThan(wi(neighbour));
    const others = (events: CheckerEvent[]) => events.filter(e => e.location.ayah !== target.ayah || e.location.word !== target.word);
    expect(others(baseline.events).some(e => e.state === "confirmed")).toBe(true);
    expect(others(veto.events)).toEqual(others(baseline.events));
  });
});

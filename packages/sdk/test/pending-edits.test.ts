import { describe, expect, it } from "vitest";
import { PHRASE_SWITCH, PosteriorFrames } from "@tilawa/core";
import type { PendingEditWindow, ReferenceIndex } from "@tilawa/core";
import { jointPendingEditPosterior } from "../src/pending-edits.js";

const index = { tokens: new Set("بتج"), phonemes: "بت", words: [
  { surah: 1, ayah: 1, word: 1, start: 0, end: 1 },
  { surah: 1, ayah: 1, word: 2, start: 1, end: 2 },
] } as unknown as ReferenceIndex;
const window: PendingEditWindow = { wordFrom: 0, wordTo: 2, fromFrame: 0, toFrame: 3,
  target: { kind: "replace", word: 0, observed: "ج", fromFrame: 0, toFrame: 1 },
  neighbours: [{ kind: "replace", word: 1, observed: "ج", fromFrame: 1, toFrame: 3 }] };
function framesOf(rows: number[][]) {
  const frames = new PosteriorFrames(3);
  frames.append({ startFrame: 0, nFrames: rows.length, k: 4,
    ids: rows.flatMap(() => [0, 1, 2, 3]), probs: rows.flat() });
  return frames;
}

describe("joint pending edit posterior", () => {
  it("reverses replacement only when a neighbouring edit explains a complete correct reading", () => {
    // Independent scoring forces the final ت into the accused word's evidence.
    // The neighbouring replacement allows the observed بتج to read the first word correctly.
    const contextual = { ...index, phonemes: "بتت", words: [
      { surah: 1, ayah: 1, word: 1, start: 0, end: 2 },
      { surah: 1, ayah: 1, word: 2, start: 2, end: 3 },
    ] } as ReferenceIndex;
    const frames = framesOf([[.001,.997,.001,.001], [.001,.001,.997,.001], [.001,.001,.001,.997]]);
    const result = jointPendingEditPosterior(contextual, frames, window)!;
    expect(result.independentReplacement).toBe(1);
    expect(result.correctWithNeighbour).toBeGreaterThan(result.replacement);
    expect(result.veto).toBe(true);
    expect(jointPendingEditPosterior(contextual, frames, { ...window, neighbours: [] })!.veto).toBe(false);
  });

  it("matches exhaustive collapsed acoustic paths with each event prior applied once", () => {
    // Three frames cannot traverse any word/ayah repeat: the exhaustive
    // calculation independently checks marginalization and CTC boundaries.
    const rows = [[.1,.3,.2,.4], [.3,.1,.2,.4], [.2,.1,.4,.3]];
    const mass = new Map<string, number>();
    function visit(path: number[], probability: number) {
      if (path.length === rows.length) {
        const text = path.filter((p, i) => p && (i === 0 || p !== path[i-1])).join("");
        mass.set(text, (mass.get(text) ?? 0) + probability);
      } else rows[path.length].forEach((p, id) => visit([...path, id], probability * p));
    }
    visit([], 1);
    const prior = Math.exp(-PHRASE_SWITCH);
    const unchanged = mass.get("12")!, targetOnly = mass.get("32")! * prior;
    const neighbourOnly = mass.get("13")! * prior, both = mass.get("33")! * prior ** 2;
    const total = unchanged + targetOnly + neighbourOnly + both;
    const result = jointPendingEditPosterior(index, framesOf(rows), window)!;
    expect(result.replacement).toBeCloseTo((targetOnly + both) / total, 12);
    expect(result.correctWithNeighbour).toBeCloseTo(neighbourOnly / total, 12);
    expect(result.unchanged).toBeCloseTo(unchanged / total, 12);
    expect(result.independentReplacement).toBeCloseTo(targetOnly / (targetOnly + unchanged), 12);
    expect(result.paths).toBe(4);
  });

  it("deduplicates hypotheses and excludes incompatible alternatives for one event", () => {
    const frames = framesOf(Array.from({ length: 3 }, () => [.25,.25,.25,.25]));
    const plain = jointPendingEditPosterior(index, frames, window)!;
    expect(jointPendingEditPosterior(index, frames, { ...window, neighbours: [...window.neighbours, ...window.neighbours] }))
      .toEqual(plain);
    const alternatives = [...window.neighbours, { ...window.neighbours[0], observed: "ب" }];
    expect(jointPendingEditPosterior(index, frames, { ...window, neighbours: alternatives })!.paths).toBe(6);
  });

  it("keeps an insertion and replacement compatible despite greedy span overlap", () => {
    const frames = framesOf(Array.from({ length: 6 }, () => [.25,.25,.25,.25]));
    const neighbours = [{ ...window.neighbours[0], kind: "insert" as const, word: 0 }];
    expect(jointPendingEditPosterior(index, frames, { ...window, toFrame: 6, neighbours })!.paths).toBe(4);
  });

  it("cannot certify correctness by distributing mass among rival mistakes", () => {
    const frames = framesOf(Array.from({ length: 3 }, () => [.25,.25,.25,.25]));
    const result = jointPendingEditPosterior(index, frames, { ...window, neighbours: [] })!;
    expect(result.correctWithNeighbour).toBe(0);
    expect(result.veto).toBe(false);
  });

  it("abstains on unavailable evidence, invalid phones and overflow without pruning", () => {
    const frames = framesOf(Array.from({ length: 3 }, () => [.25,.25,.25,.25]));
    expect(jointPendingEditPosterior(index, frames, { ...window, fromFrame: -1 })).toBeNull();
    expect(jointPendingEditPosterior(index, frames, { ...window, toFrame: 4 })).toBeNull();
    expect(jointPendingEditPosterior(index, frames, { ...window, target: { ...window.target, observed: "ز" } })).toBeNull();
    expect(jointPendingEditPosterior(index, frames, { ...window, target: { ...window.target, toFrame: 4 } })).toBeNull();
    const tooWide = { ...index, words: [index.words[0], { ...index.words[1], ayah: 3 }] } as ReferenceIndex;
    expect(jointPendingEditPosterior(tooWide, frames, window)).toBeNull();
    expect(jointPendingEditPosterior(index, frames, window, 3)).toBeNull();
  });
});

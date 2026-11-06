import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PhonemeEvidence, ReferenceIndex } from "@tilawa/core";
import { wordEarSpans } from "../src/word-ear.js";

const plan = vi.hoisted(() => ({ steps: [1, 11, 20], order: [0, 1, 2] }));
vi.mock("@tilawa/core", async importOriginal => {
  const actual = await importOriginal<typeof import("@tilawa/core")>();
  return { ...actual, createTracker: () => ({
    push: (e: PhonemeEvidence) => {
      const i = plan.steps.indexOf(e.phonemes[0]?.start ?? -1);
      if (i < 0) return [];
      const w = index.words[plan.order[i]];
      return [{ type: "position", state: "locked", timeMs: (e.phonemes[0].end+1)*40,
        observed: [{ word: { surah: w.surah, ayah: w.ayah, word: w.word } }] }];
    },
  }), waqfMarked: () => false };
});
const index = { inventoryId: "toy", tokens: new Set("بت"), words: [
  { surah: 1, ayah: 1, word: 1, start: 0, end: 1 },
  { surah: 1, ayah: 2, word: 1, start: 1, end: 2 },
  { surah: 1, ayah: 2, word: 2, start: 2, end: 3 },
], phonemes: "بتب" } as ReferenceIndex;
const input = (): PhonemeEvidence[] => [{ inventory: "toy", frameMs: 40, decodedTo: 25,
  phonemes: [1, 11, 20].map(start => ({ p: "ب", start, end: start, conf: .9 })),
  posteriors: { startFrame: 0, nFrames: 25, k: 1,
    ids: Array.from({ length: 25 }, (_, i) => [1, 11, 20].includes(i) ? 1 : 0), probs: Array(25).fill(1) } }];

describe("word ear seam clock", () => {
  beforeEach(() => { plan.steps = [1, 11, 20]; plan.order = [0, 1, 2]; });
  it("uses a labelled pause and the step onto word two with existing padding", () => {
    const chunks = input(), saved = structuredClone(chunks);
    expect(wordEarSpans(chunks, index)).toEqual([{ word: 1, fromFrame: 2, toFrame: 29,
      seamFromFrame: 2, seamToFrame: 21, prefixFrame: undefined }]);
    expect(chunks).toEqual(saved);
  });
  it("does not call speech inside the gap a labelled pause", () => {
    const chunks = input();
    (chunks[0].posteriors!.ids as number[])[5] = 2;
    expect(wordEarSpans(chunks, index)).toEqual([]);
  });
  it("does not create a seam from a relocation or absent word-two step", () => {
    plan.order = [0, 2, 1];
    expect(wordEarSpans(input(), index)).toEqual([]);
    plan.steps = [1, 11]; plan.order = [0, 1];
    expect(wordEarSpans(input(), index)).toEqual([]);
  });
  it("abstains without stored frame evidence", () => {
    expect(wordEarSpans(input().map(({ posteriors: _, ...e }) => e), index)).toEqual([]);
  });
});

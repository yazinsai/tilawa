import { describe, expect, it } from "vitest";
import type { PhonemeEvidence, SequenceBoundaryEvidence } from "@tilawa/core";
import { carryBoundaryEvidence } from "../src/sequence-boundaries.js";
import { insertionCells } from "../src/seam-insertion.js";

describe("boundary certificate transport", () => {
  it("retains primary values through JSON serialization and native cells", () => {
    const chunks: PhonemeEvidence[] = [{ inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 1, end: 2, conf: .9 }],
      posteriors: { startFrame: 0, nFrames: 4, k: 1, ids: [0,1,1,0], probs: [1,1,1,1] } }];
    const original = structuredClone(chunks);
    const metadata: SequenceBoundaryEvidence[] = [{ word: { surah: 1, ayah: 1, word: 1 }, written: "ب", accused: "ت",
      fromFrame: 1, toFrame: 3, resetMargin: 11, conditionalMargin: -2 }];
    const enriched = carryBoundaryEvidence(chunks, metadata);
    expect(chunks).toEqual(original);
    expect(enriched[0].posteriors).toBe(chunks[0].posteriors);
    const cells = insertionCells(JSON.parse(JSON.stringify(enriched)));
    expect(cells.flatMap(e => e.phonemes)[0].sequenceBoundaries).toEqual(metadata);
    expect(cells.flatMap(e => e.phonemes).map(({ sequenceBoundaries: _, ...p }) => p)).toEqual(chunks[0].phonemes);
    expect(cells.flatMap(e => Array.from(e.posteriors!.probs))).toEqual([1,1,1,1]);
  });
});

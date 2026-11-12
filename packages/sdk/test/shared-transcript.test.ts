import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "@tilawa/core";
import type { PhonemeEvidence, SharedTranscriptEvidence } from "@tilawa/core";
import { carryTranscriptEvidence, poolTranscriptScores } from "../src/shared-transcript.js";
import { insertionCells } from "../src/seam-insertion.js";

const frames = (rows: number[][]) => {
  const f = new PosteriorFrames(2);
  f.append({ k: 3, startFrame: 0, nFrames: rows.length, ids: rows.flatMap(() => [0,1,2]), probs: rows.flat() });
  return f;
};

describe("shared transcript sequence pool", () => {
  it("normalizes an equal geometric pool and vetoes only a reversal toward the written transcript", () => {
    const r = poolTranscriptScores([12, 2], [1, 15])!;
    expect(r.pooledMargin).toBe(-2);
    expect(r.writtenProbability).toBeCloseTo(1 / (1 + Math.exp(-2)));
    expect(r.veto).toBe(true);
    expect(poolTranscriptScores([112,102], [301,315])).toEqual(r);
    expect(poolTranscriptScores([12,2], [1,5])!.veto).toBe(false);
    expect(poolTranscriptScores([2,12], [1,15])!.veto).toBe(false);
    expect(poolTranscriptScores([12,2], [1,11])!.veto).toBe(false);
  });

  it("sums alignments before pooling when the two views emit at different times", () => {
    const rows = [[.01,.79,.2], [.79,.01,.2]], other = [...rows].reverse();
    const a = frames(rows), b = frames(other);
    const pooled = poolTranscriptScores([a.nll([1],0,2),a.nll([2],0,2)], [b.nll([1],0,2),b.nll([2],0,2)])!;
    expect(pooled.writtenProbability).toBeGreaterThan(.5);
    const products = rows.map((row,i) => row.map((p,j) => p * other[i][j]));
    const fused = frames(products.map(row => row.map(p => p / row.reduce((s,x) => s+x,0))));
    expect(fused.nll([1],0,2)).toBeGreaterThan(fused.nll([2],0,2));
  });

  it("abstains on any impossible or missing complete sequence score", () => {
    expect(poolTranscriptScores([Infinity, 1], [1,2])).toBeNull();
    expect(poolTranscriptScores([1,2], [NaN,2])).toBeNull();
    const f = frames([[.1,.8,.1]]);
    expect(poolTranscriptScores([2,1], [f.nll([1,1],0,1),f.nll([2],0,1)])).toBeNull();
  });

  it("carries the exact veto through JSON and frame cells without modifying primary evidence", () => {
    const evidence: PhonemeEvidence[] = [{ inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 0, end: 0, conf: .9 }],
      posteriors: { k: 1, startFrame: 0, nFrames: 2, ids: [1,0], probs: [.9,.9] } }];
    const certificate: SharedTranscriptEvidence = { word: { surah: 1, ayah: 1, word: 1 }, written: "ب", rival: "ت",
      fromFrame: 0, toFrame: 2, primaryMargin: 10, cropMargin: -12 };
    const original = structuredClone(evidence), enriched = carryTranscriptEvidence(evidence, [certificate]);
    expect(evidence).toEqual(original);
    expect(enriched[0].posteriors).toBe(evidence[0].posteriors);
    expect(enriched[0].phonemes.map(({ sharedTranscripts: _, ...p }) => p)).toEqual(evidence[0].phonemes);
    expect(insertionCells(JSON.parse(JSON.stringify(enriched))).flatMap(c => c.phonemes)[0].sharedTranscripts).toEqual([certificate]);
  });
});

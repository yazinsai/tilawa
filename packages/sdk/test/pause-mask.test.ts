import { describe, expect, it } from "vitest";
import { pauseFloor, pauseMaskSpans } from "../src/pause-mask.js";
import type { PhonemeEvidence, ReferenceIndex } from "@tilawa/core";

const index = { words: [
  { surah: 1, ayah: 1, word: 1, start: 0, end: 1 },
  { surah: 1, ayah: 2, word: 1, start: 1, end: 2 },
  { surah: 1, ayah: 2, word: 2, start: 2, end: 3 },
] } as ReferenceIndex;
const locks = [{ wi: 0, timeMs: 0 }, { wi: 1, timeMs: 480 }, { wi: 2, timeMs: 960 }];
const evidence = (blank: number): PhonemeEvidence[] => [{ inventory: "test", frameMs: 40, phonemes: [],
  posteriors: { startFrame: 0, nFrames: 24, k: 2,
    ids: Array.from({ length: 24 }, () => [0,1]).flat(),
    probs: Array.from({ length: 24 }, (_, t) => t >= 3 && t < 11 ? [blank,1-blank] : [.1,.9]).flat() } }];

describe("pause-blocked auxiliary streaming", () => {
  it("uses the labelled pause onset and ends at the primary step onto word 2", () => {
    expect(pauseMaskSpans(evidence(.9),locks,index)).toEqual([{ fromFrame: 3, toFrame: 24 }]);
    expect(pauseMaskSpans(evidence(.49),locks,index)).toEqual([]);
  });
  it("requires contiguous in-order primary steps across the seam", () => {
    expect(pauseMaskSpans(evidence(.9),[locks[0],locks[2],locks[1]],index)).toEqual([]);
  });
  it("keeps original positions and masks only query frames inside the half-open span", () => {
    const spans = [{ fromFrame: 3, toFrame: 8 }];
    expect(Array.from(pauseFloor(spans,0,16))).toEqual([-1n,-1n,-1n,-1n,-1n,-1n,6n,6n,6n,6n,6n,6n,6n,6n,6n,6n]);
    expect(Array.from(pauseFloor(spans,16,16))).toEqual(Array(16).fill(-1n));
  });
});

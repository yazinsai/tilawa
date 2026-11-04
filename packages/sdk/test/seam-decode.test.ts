import { describe, expect, it } from "vitest";
import type { PhonemeEvidence, ReferenceIndex } from "@tilawa/core";
import { attachSeamEvidence, seamSpans } from "../src/seam-decode.js";
import type { LoadedModel } from "../src/model.js";

const words = [
  { surah: 1, ayah: 1, word: 1, start: 0, end: 1 },
  { surah: 1, ayah: 1, word: 2, start: 1, end: 2 },
  { surah: 1, ayah: 2, word: 1, start: 2, end: 3 },
  { surah: 1, ayah: 2, word: 2, start: 3, end: 4 },
];
const index = { words, inventoryId: "test" } as ReferenceIndex;
describe("tracker-bounded seam evidence", () => {
  it("needs neither word-1's step nor a greedy visit, and rejects a lock drop between boundaries", () => {
    const steps = [{ wi: 1, timeMs: 4000 }, { wi: 3, timeMs: 6000 }];
    expect(seamSpans(index, steps)).toHaveLength(1);
    expect(seamSpans(index, [steps[0], { wi: -1, timeMs: 5000 }, steps[1]])).toEqual([]);
  });

  it("carries only bounded ear tokens and leaves the main string and rows unchanged", async () => {
    const chunks: PhonemeEvidence[] = [{ inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 0, end: 0, conf: .9 }],
      posteriors: { k: 1, startFrame: 0, nFrames: 175, ids: Array(175).fill(0), probs: Array(175).fill(1) } }];
    const before = structuredClone(chunks);
    const model = { manifest: { sampleRate: 16000 }, seamEar: { decode: async () => ({ inventory: "test", frameMs: 40,
      phonemes: [90, 120, 170].map(start => ({ p: "ت", start, end: start, conf: .9 })),
      posteriors: { k: 1, startFrame: 0, nFrames: 175, ids: Array(175).fill(1), probs: Array(175).fill(.9) } }) } } as LoadedModel;
    const out = await attachSeamEvidence(model, new Float32Array(16000 * 7), chunks, [{ wi: 1, timeMs: 4000 }, { wi: 3, timeMs: 6000 }], index);
    expect(chunks).toEqual(before);
    expect(out[0].posteriors).toBe(chunks[0].posteriors);
    expect(out[0].phonemes[0].p).toBe("ب");
    expect(out[0].phonemes[0].seams?.[0].phonemes.map(p => p.start)).toEqual([120]);
  });
});

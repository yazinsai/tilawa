import { describe, expect, it } from "vitest";
import type { PhonemeEvidence, ReferenceIndex, WrongWordRecheckCandidate } from "@tilawa/core";
import { robustNormVerdict } from "../src/robust-norm-recheck.js";

const index = { inventoryId: "test", tokens: new Set("نتك") } as ReferenceIndex;
const candidate: WrongWordRecheckCandidate = {
  word: { surah: 1, ayah: 1, word: 1 }, written: "نت", accused: "نك", wordWritten: "نت", wordAccused: "نك",
  fromFrame: 0, toFrame: 4, slotFromFrame: 0, slotToFrame: 4, margin: 20, need: 10,
};
const evidence: PhonemeEvidence = { inventory: "test", frameMs: 40,
  phonemes: [{ p: "ن", start: 0, end: 0, conf: .9 }, { p: "ت", start: 2, end: 2, conf: .9 }],
  posteriors: { k: 4, startFrame: 0, nFrames: 4,
    ids: [1, 0, 2, 3, 0, 1, 2, 3, 2, 0, 1, 3, 0, 1, 2, 3],
    probs: [.9, .08, .01, .01, .9, .04, .04, .02, .9, .08, .01, .01, .9, .04, .04, .02] } };

const control = { ...evidence, posteriors: { ...evidence.posteriors!, probs: evidence.posteriors!.probs.map(() => .25) } };

describe("robust-norm written-reading witness", () => {
  it("vetoes a complete recovered written word when the unchanged margin no longer confirms the rival", () => {
    const v = robustNormVerdict(candidate, evidence, index, control)!;
    expect(v.recovered).toBe("نت");
    expect(v.writtenNll).toBeLessThan(v.accusedNll);
    expect(v.veto).toBe(true);
  });
  it("requires an improvement beyond gain alone", () => {
    expect(robustNormVerdict(candidate, evidence, index, evidence)?.veto).toBe(false);
  });
  it("requires complete identity and loses no warning merely because its margin became weak", () => {
    expect(robustNormVerdict(candidate, { ...evidence, phonemes: evidence.phonemes.slice(0, 1) }, index, control)?.veto).toBe(false);
    expect(robustNormVerdict(candidate, { ...evidence, phonemes: [...evidence.phonemes,
      { p: "ك", start: 3, end: 3, conf: .9 }] }, index, control)?.veto).toBe(false);
  });
  it("abstains on absent rows and incompatible inventory or intervals", () => {
    expect(robustNormVerdict({ ...candidate, toFrame: 5 }, evidence, index, control)).toBeNull();
    expect(robustNormVerdict(candidate, { ...evidence, inventory: "other" }, index, control)).toBeNull();
    expect(robustNormVerdict(candidate, { ...evidence, posteriors: undefined }, index, control)).toBeNull();
  });
});

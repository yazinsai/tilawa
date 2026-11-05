import { describe, expect, it } from "vitest";
import { pauseSpellingSpans, preserveColdSpellings } from "../src/seam-excision.js";
import type { DecodedPhoneme, PhonemeEvidence } from "../src/evidence.js";

const tokens = "اوِمنب";
const phones: DecodedPhoneme[] = [
  { p: "ا", start: 0, end: 0, conf: 0.95 },
  { p: "و", start: 10, end: 10, conf: 0.95 },
  { p: "م", start: 12, end: 12, conf: 0.95 },
  { p: "ِ", start: 13, end: 13, conf: 0.95 },
  { p: "ن", start: 15, end: 16, conf: 0.95 },
  { p: "ب", start: 30, end: 30, conf: 0.95 },
];
function evidence(from = 0, to = 31, tailUntil = 20, blank = 0.99): PhonemeEvidence {
  const ids: number[] = [], probs: number[] = [];
  for (let f = from; f < to; f++) {
    const ph = phones.find(p => p.start <= f && f <= p.end);
    const id = ph ? tokens.indexOf(ph.p) + 1 : f > 16 && f <= tailUntil ? tokens.indexOf("ن") + 1 : 0;
    ids.push(id, id ? 0 : 1);
    probs.push(id ? 0.95 : blank, id ? 0.04 : 1 - blank);
  }
  return { inventory: "test", frameMs: 40, phonemes: phones.filter(p => p.start >= from && p.start < to).map(p => ({ ...p })),
    posteriors: { k: 2, startFrame: from, nFrames: to - from, ids, probs } };
}

describe("cold-spelling side evidence", () => {
  it("bounds exactly the parent utterance despite the final token's held argmax tail", () => {
    const spans = pauseSpellingSpans([...tokens]).push(evidence());
    expect(spans.map(({ carrier, ...span }) => span)).toEqual([{ fromFrame: 10, toFrame: 17, raw: "ومِن" }]);
    expect(spans[0].carrier.p).toBe("ب");
  });
  it("finds the same span across evidence chunks while retaining only the unprocessed gap rows", () => {
    const decoder = pauseSpellingSpans([...tokens]);
    expect(decoder.push(evidence(0, 17))).toEqual([]);
    expect(decoder.push(evidence(17, 31)).map(({ carrier, ...span }) => span))
      .toEqual([{ fromFrame: 10, toFrame: 17, raw: "ومِن" }]);
  });
  it("requires the unchanged run-end pause and blank ownership", () => {
    expect(pauseSpellingSpans([...tokens]).push(evidence(0, 31, 25))).toEqual([]);
    expect(pauseSpellingSpans([...tokens]).push(evidence(0, 31, 20, 0.4))).toEqual([]);
    expect(pauseSpellingSpans([...tokens]).push({ ...evidence(), posteriors: undefined })).toEqual([]);
  });
  it("preserves proofs when the existing gap splice removes their carrier, without changing warm phones", () => {
    const proof = { fromFrame: 10, toFrame: 17, phonemes: "ومِن" };
    const before = [{ ...phones[1] }, { ...phones.at(-1)!, seamCold: [proof] }];
    const after = [{ ...phones[1] }, { ...phones.at(-1)!, start: 35, end: 35 }];
    const result = preserveColdSpellings(before, after);
    expect(result[1].seamCold).toEqual([proof]);
    expect(result.map(({ seamCold, ...p }) => p)).toEqual(after);
    expect(after[1].seamCold).toBeUndefined();
    expect(JSON.parse(JSON.stringify(result))[1].seamCold).toEqual([proof]);
  });
});

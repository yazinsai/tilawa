import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadReferenceIndex, sameWrongWord, trackerInput, createTracker } from "@tilawa/core";
import { syntheticFixture, withPosteriors } from "../../core/src/fixtures/index.js";
import { carryWrongWordAgreements, rejudgeWrongWords, assertAgreementOnlyVetoes } from "../src/wrong-word-agreement.js";
import { insertionCells } from "../src/seam-insertion.js";
import { pauseFloor } from "../src/pause-mask.js";

const index = loadReferenceIndex(JSON.parse(readFileSync(new URL("../../core/assets/ref-v2.json", import.meta.url), "utf8")));
const from = { surah: 26, ayah: 106, word: 1 }, to = { surah: 26, ayah: 107, word: 3 };
const at = index.words.find(w => w.surah === 26 && w.ayah === 106 && w.word === 5)!;
const start = index.words.find(w => w.surah === 26 && w.ayah === 106 && w.word === 1)!.start;
const rival = index.words.find(w => w.surah === 26 && w.ayah === 142 && w.word === 5)!;
const make = (swap: boolean) => syntheticFixture(index, { id: "side-ear", from, to, chunkSize: 4, edits: swap ? [
  { kind: "del", at: at.start - start, count: at.end - at.start },
  { kind: "ins", at: at.start - start, text: index.phonemes.slice(rival.start, rival.end) },
] : [] }).evidence.map(e => withPosteriors(index, e));
const options = { passage: { from, to }, policy: { confirm: { after: "next-word" as const } } };

describe("pause-ear wrong-word rejudgment", () => {
  it("keeps identical-ear accusations and vetoes a side ear that reads the text", () => {
    const primary = make(true);
    const same = rejudgeWrongWords(primary, primary, index, options, "memorization");
    expect(same).toHaveLength(1);
    expect(same[0].rejudged.some(c => sameWrongWord(same[0].primary, c))).toBe(true);
    const correct = rejudgeWrongWords(primary, make(false), index, options, "memorization");
    expect(correct).toHaveLength(1);
    expect(correct[0].rejudged.some(c => sameWrongWord(correct[0].primary, c))).toBe(false);
    expect(() => assertAgreementOnlyVetoes(primary, carryWrongWordAgreements(primary, correct), index, options, "memorization")).not.toThrow();
  });
  it("preserves primary rows, teacher certificates and tracker inputs through JSON and rechunking", () => {
    const primary = make(true);
    primary[0].phonemes[0].seamInsertions = [{ word: from, observed: "بت", fromFrame: 1, toFrame: 2, marginal: .9 }];
    const before = structuredClone(primary);
    const agreements = rejudgeWrongWords(primary, primary, index, options, "memorization");
    const enriched = carryWrongWordAgreements(primary, agreements);
    expect(primary).toEqual(before);
    expect(enriched.map(e => e.posteriors)).toEqual(primary.map(e => e.posteriors));
    const strip = enriched.map(e => ({ ...e, phonemes: e.phonemes.map(({ wrongWordAgreements: _, ...p }) => p) }));
    expect(JSON.stringify(strip)).toBe(JSON.stringify(primary));
    const tracker = createTracker(index, { mode: "free" }), other = createTracker(index, { mode: "free" });
    expect(enriched.flatMap(e => tracker.push(trackerInput(e)))).toEqual(primary.flatMap(e => other.push(trackerInput(e))));
    expect(tracker.end()).toEqual(other.end());
    const cells = insertionCells(JSON.parse(JSON.stringify(enriched)));
    expect(cells.flatMap(e => e.phonemes)[0].wrongWordAgreements).toEqual(agreements);
    expect(cells.flatMap(e => e.phonemes)[0].seamInsertions).toEqual(primary[0].phonemes[0].seamInsertions);
  });
  it("applies the trained cutoff on the embedding clock and leaves other queries unmasked", () => {
    expect(Array.from(pauseFloor([{ fromFrame: 2, toFrame: 4 }], 0, 10)))
      .toEqual([-1n,-1n,-1n,-1n,4n,4n,4n,4n,-1n,-1n]);
  });
});

describe("primary evidence invariant", () => {
  it("rejects changes to primary rows and folded teacher certificates", async () => {
    const { assertPrimaryEvidence } = await import("../src/primary-invariant.js");
    const primary = make(true);
    const changed = structuredClone(primary);
    changed[0].posteriors!.probs = Array.from(changed[0].posteriors!.probs, p => p * .9);
    expect(() => assertPrimaryEvidence(primary, changed)).toThrow(/primary evidence/);
    const teacher = structuredClone(primary);
    teacher[0].phonemes[0].seamInsertions = [{ word: from, observed: "بت", fromFrame: 1, toFrame: 2, marginal: .9 }];
    expect(() => assertPrimaryEvidence(primary, teacher)).toThrow(/teacher certificates/);
  });
});

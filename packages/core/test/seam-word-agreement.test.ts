import { expect, it } from "vitest";
import { seamWordAgreement, seamWordCompetitors, seamWordFit } from "../src/seam-word-agreement.js";
import type { ReferenceIndex } from "../src/reference.js";

it("counts hard edits separately from existing vowel and consonant soft costs", () => {
  expect(seamWordFit("سَلِم", "تَلُم")).toEqual({ cost: 15, hard: 1 });
  expect(seamWordFit("قَبل", "كَبل")).toEqual({ cost: 5, hard: 0 });
  expect(seamWordFit("سَلِم", "تَلِب").hard).toBe(2);
  expect(seamWordFit("سَلِم", "تَلِب", 1).cost).toBe(Infinity);
});

it("includes local pronunciation prefixes and complete contiguous back-runs", () => {
  const forms = ["بَت", "ثِج", "دَذ", "رَز", "سَش", "صَض"];
  let offset = 0;
  const index = { phonemes: forms.join(""), words: forms.map(f => {
    const start = offset; offset += f.length;
    return { start, end: offset };
  }), wordForms: forms.map(f => [f]), isolatedWordForms: ["", "", "دَذُ"] } as unknown as ReferenceIndex;
  const competitors = seamWordCompetitors(index, 3);
  for (const form of ["رَز", "رَ", "سَش", "دَذُ", "ثِجدَذ", "بَتثِجدَذ"]) expect(competitors).toContain(form);
});

it("requires the complete proposed identity and rejects more than one hard edit", () => {
  expect(seamWordAgreement("سَلِم", "تَلِم", ["وَقَل"]).agrees).toBe(true);
  expect(seamWordAgreement("سَلِم", "تَلِب", ["وَقَل"]).agrees).toBe(false);
  expect(seamWordAgreement("سَلِم", "", ["وَقَل"]).agrees).toBe(false);
});

it("rejects a tie, a better neighbour and a better prefix or back-run", () => {
  expect(seamWordAgreement("سَلِم", "تَلِم", ["بَلِم"]).agrees).toBe(false);
  expect(seamWordAgreement("سَلِم", "تَلِم", ["تَلِم"]).agrees).toBe(false);
  expect(seamWordAgreement("قَبل", "كَبل", ["كَبل"]).agrees).toBe(false);
});

it("collapses held sounds and removes mushaf pause marks without changing identity", () => {
  expect(seamWordFit("قَبڇل", "قَبل")).toEqual({ cost: 0, hard: 0 });
  expect(seamWordFit("سَلِۦۦم", "سَلِۦم")).toEqual({ cost: 0, hard: 0 });
});

import { describe, expect, it } from "vitest";
import { channelContext, channelLogLikelihood, fitMorphemeChannel, initialChannel, morphemeUnits, selectMorphemeChannels } from "../src/morpheme-channel.js";
import type { ChannelReference, MorphemeRole } from "../src/morpheme-channel.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";

const ref = (phones: string, roles?: MorphemeRole[]): ChannelReference => ({ phones, roles: roles ?? Array(phones.length).fill("stem") });
const localIndex = () => loadReferenceIndex({ version: "unit", text: "unit", profile: {}, inventory: { id: "unit", tokens: "بتث" },
  ayahs: [{ s: 1, a: 1, w: ["بتبتبت", "ثثثتتت", "بثتثبث", "تبتثتث"], p: "بتبتبتثثثتتتبثتثبثتبتثتث", o: [0, 6, 12, 18], iso: ["بتبتبت", "ثثثتتت", "بثتثبث", "تبتثتث"] }] });

describe("surface-preserving morphological units", () => {
  it("keeps attached prepositions, assimilated articles, and pronouns in the complete surface string", () => {
    const units = morphemeUnits("وَبِٱلْكِتَابِهِ", "وَبِلكِتاابِهِ");
    expect(units.map(u => u.role)).toEqual(["prefix", "prefix", "prefix", "stem", "suffix"]);
    expect(units.map(u => u.phones).join("")).toBe("وَبِلكِتاابِهِ");
    const assimilated = morphemeUnits("بِٱلنُّذُرِ", "بِننننُذُرِ");
    expect(assimilated).toEqual([{ role: "prefix", phones: "بِ" }, { role: "prefix", phones: "ننن" }, { role: "stem", phones: "نُذُرِ" }]);
  });
  it("leaves unresolved lexical onsets and missing orthography in the stem", () => {
    for (const [text, phones] of [["وَرَدَ", "وَرَدَ"], ["فَتَحَ", "فَتَحَ"], ["", "بِننننُذُرِ"]]) {
      expect(morphemeUnits(text, phones)).toEqual([{ role: "stem", phones }]);
    }
    expect(morphemeUnits("وَجَدْنَا", "وَجَدناا", new Set(["وجدنا"]))).toEqual([
      { role: "stem", phones: "وَجَد" }, { role: "suffix", phones: "ناا" }]);
  });
  it("does not discard a surface phone when orthographic boundaries cannot be realized", () => {
    const units = morphemeUnits("بِٱلْكِتَابِهِ", "تُرااب");
    expect(units.map(u => u.phones).join("")).toBe("تُرااب");
  });
});

describe("normalized joint edit channel", () => {
  it("sums to one over observed strings, including the charged empty realization", () => {
    const model = initialChannel("بت", "morphology", 16);
    const reference = ref("بت", ["prefix", "stem"]);
    let sum = 0;
    let level = [""];
    for (let length = 0; length <= 9; length++) {
      for (const heard of level) sum += Math.exp(channelLogLikelihood(reference, heard, model));
      level = level.flatMap(s => [...model.phones].map(p => s + p));
    }
    expect(sum).toBeCloseTo(1, 10);
    expect(channelLogLikelihood(reference, "", model)).toBeLessThan(channelLogLikelihood(reference, "بت", model));
  });
  it("makes the context control invariant to morphology and jointly changes errors across the role boundary", () => {
    const phones = "بت";
    const a = ref(phones, ["prefix", "stem"]), b = ref(phones);
    const control = initialChannel(phones, "phonetic", 16);
    expect(channelLogLikelihood(a, "تب", control)).toBe(channelLogLikelihood(b, "تب", control));
    const joint = initialChannel(phones, "morphology", 16);
    joint.contexts[channelContext(a, 1, "morphology")] = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]];
    expect(channelLogLikelihood(a, "تب", joint)).toBeGreaterThan(channelLogLikelihood(b, "تب", joint));
  });
  it("shares sparse contexts, marginalizes edit placement, and fits deterministic probabilities", () => {
    const takes = ["بت", "بت", "تب", "ب", "ت", "بت"].map((heard, i) => ({ heard, reference: ref("بت", ["prefix", "stem"]), speaker: String(i % 3) }));
    const trained = fitMorphemeChannel(takes, "بت", "morphology", 16, 3);
    expect(trained).toEqual(fitMorphemeChannel(takes, "بت", "morphology", 16, 3));
    expect(trained.insertion).toBeGreaterThan(0);
    expect(trained.insertion).toBeLessThan(1);
    for (const table of [trained.pooled, ...Object.values(trained.contexts)]) for (const row of table) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      expect(row.every(p => p > 0)).toBe(true);
    }
    expect(Number.isFinite(channelLogLikelihood(ref("تبت"), "بت", trained))).toBe(true);
    expect(selectMorphemeChannels(takes, "بت").selection).toHaveLength(6);
  });
  it("rejects missing role ownership and unknown phones instead of fabricating observations", () => {
    const model = initialChannel("بت", "morphology", 16);
    expect(() => channelLogLikelihood(ref("بت", ["stem"]), "بت", model)).toThrow("length mismatch");
    expect(() => channelLogLikelihood(ref("بت"), "ث", model)).toThrow("Unknown");
  });
  it("keeps the local recurrence and rewinds unchanged under an identical channel control", () => {
    const index = localIndex();
    const start = { offset: 6, origin: 0, edits: 0 };
    const baseline = createLocalSearch(index, start);
    const observations: unknown[] = [];
    const controlled = createLocalSearch(index, start, { channels: { morphology: initialChannel("بتث", "morphology", 16), phonetic: initialChannel("بتث", "phonetic", 16) },
      audit: c => { expect(c.continuationCorrection).toBe(0); expect(c.restartCorrection).toBe(0); observations.push(c); } });
    const heard = "بثتثبثتبتثتث";
    for (const p of heard) { baseline.step(p); controlled.step(p); expect(controlled.position()).toEqual(baseline.position()); }
    expect(observations.length).toBeGreaterThan(0);
    const saved = controlled.save();
    controlled.step("ث");
    const after = controlled.position();
    controlled.restore(saved);
    controlled.step("ث");
    expect(controlled.position()).toEqual(after);
  });
  it("can reverse a competing restart on the same observations without removing its movement charge", () => {
    const index = localIndex();
    const start = { offset: 6, origin: 0, edits: 0 };
    const baseline = createLocalSearch(index, start);
    const morphology = initialChannel("بتث", "morphology", 16);
    const phonetic = initialChannel("بتث", "phonetic", 16);
    const restart = ref("بثتثب");
    for (const at of [0, 4]) morphology.contexts[channelContext(restart, at, "morphology")] = Array.from({ length: 3 }, () => [1e-6, 0.9, 0.099999]);
    let comparisons = 0;
    const updated = createLocalSearch(index, start, { channels: { morphology, phonetic }, audit: c => {
      expect(c.heard).toBe("بثتثب");
      expect(c.restart.movementCost).toBeGreaterThan(20);
      expect(c.continuation.movementCost).toBe(0);
      expect(c.continuationCorrection).toBeLessThan(c.restartCorrection!);
      comparisons++;
    } });
    for (const p of "بثتثب") { baseline.step(p); updated.step(p); }
    expect(baseline.position().origin).toBe(12);
    expect(updated.position().origin).toBe(0);
    expect(comparisons).toBe(1);
  });
});

import { describe, expect, it } from "vitest";
import { coalescenceCosts, coalescenceEligible, coalescenceLogP, fitCoalescence } from "../src/coalescence.js";
import type { CoalescenceModel } from "../src/coalescence.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";

const phones = ["ب", "ل", "ت", "د", "ر"];
const disabled = (): CoalescenceModel => ({ phones, pairs: {}, classes: {}, pooled: [1, ...phones.map(() => 0)], mass: 8 });
const merged = (): CoalescenceModel => ({ ...disabled(), pairs: { بل: [0.2, 0.8, 0, 0, 0, 0] } });
const index = () => loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: phones.join("") },
  ayahs: [{ s: 1, a: 1, w: ["first", "second"], p: "بلتدر", o: [0, 3], iso: ["بلت", "در"] }] });

describe("joint phone coalescence", () => {
  it("partitions probability across ordinary and merged events, with normalized complete-string likelihood", () => {
    const model = { ...disabled(), phones: ["ب", "ل"], pooled: [0.7, 0.2, 0.1] };
    let probability = 0;
    for (let length = 0; length <= 8; length++) for (let bits = 0; bits < 2 ** length; bits++) {
      const heard = Array.from({ length }, (_, i) => model.phones[(bits >> i) & 1]).join("");
      probability += Math.exp(coalescenceLogP(model, { written: "بل", heard, starts: [0] }));
    }
    expect(probability).toBeCloseTo(1, 5);
  });

  it("excludes length repetitions, identical sounds, and word boundaries", () => {
    const starts = new Set([0]);
    expect(coalescenceEligible("بلت", 0, starts)).toBe(true);
    expect(coalescenceEligible("نننبل", 2, starts)).toBe(false);
    expect(coalescenceEligible("بللل", 0, starts)).toBe(false);
    expect(coalescenceEligible("بب", 0, starts)).toBe(false);
    expect(coalescenceEligible("بل", 0, new Set([0, 1]))).toBe(false);
    expect(coalescenceEligible("ب", 0, starts)).toBe(false);
    const costs = coalescenceCosts(merged(), "بل", new Set([0, 1]));
    expect(costs.ordinary(0)).toBe(0);
    expect(costs.merged(0, "ب")).toBe(Infinity);
  });

  it("consumes two reference phones and one observed token once, and restores the complete state", () => {
    const search = createLocalSearch(index(), { offset: 0, origin: 0, edits: 0 }, merged());
    search.step("ب");
    expect(search.position().offset).toBe(2);
    expect(search.position().edits).toBe(1);
    const saved = search.save();
    search.step("ت");
    const after = search.position();
    expect(after.offset).toBe(3);
    search.restore(saved);
    expect(search.position().offset).toBe(2);
    search.step("ت");
    expect(search.position()).toEqual(after);
  });

  it("retains ordinary local paths and movement when the trained model selects no coalescence", () => {
    const base = createLocalSearch(index(), { offset: 0, origin: 0, edits: 0 });
    const off = createLocalSearch(index(), { offset: 0, origin: 0, edits: 0 }, disabled());
    for (const p of "بلتدربلتدر") {
      base.step(p); off.step(p);
      expect(off.position()).toEqual(base.position());
    }
  });

  it("uses the same recognition event at a restart and preserves the existing movement edge", () => {
    const ref = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: phones.join("") },
      ayahs: [{ s: 1, a: 1, w: ["source", "destination"], p: "درتبلت", o: [0, 3], iso: ["درت", "بلت"] }] });
    const search = createLocalSearch(ref, { offset: 0, origin: 0, edits: 0 }, merged());
    search.step("ب"); search.step("ت");
    expect(search.position().offset).toBe(6);
    expect(search.position().origin).toBe(3);
    expect(search.position().edits).toBe(1);
  });

  it("keeps a genuinely omitted word as a movement jump rather than a free coalescence", () => {
    const ref = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: phones.join("") },
      ayahs: [{ s: 1, a: 1, w: ["source", "omitted", "destination"], p: "بلتللللللللدرت", o: [0, 3, 11], iso: ["بلت", "لللللللل", "درت"] }] });
    const search = createLocalSearch(ref, { offset: 3, origin: 0, edits: 0 }, merged());
    for (const p of "درت") search.step(p);
    expect(search.position().offset).toBe(14);
    expect(search.position().origin).toBe(11);
    expect(search.position().recentCost).toBeGreaterThan(0);
  });

  it("marginalizes latent paths and shares sparse pairs by phonetic class", () => {
    const records = Array.from({ length: 20 }, () => ({ written: "بل", heard: "ب", starts: [0] }));
    const model = fitCoalescence(records, phones, 8);
    for (const row of [model.pooled, ...Object.values(model.classes), ...Object.values(model.pairs)]) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
      expect(row.every(p => Number.isFinite(p) && p > 0)).toBe(true);
    }
    expect(model.pairs["بل"][1]).toBeGreaterThan(model.pooled[1]);
    expect(coalescenceLogP(model, records[0])).toBeGreaterThan(coalescenceLogP(disabled(), records[0]));
    const costs = coalescenceCosts(model, "در", new Set([0]));
    expect(costs.merged(0, "ب")).toBeLessThan(Infinity);
    expect(costs.merged(0, "?")).toBe(Infinity);
    expect(() => fitCoalescence([{ written: "بل", heard: "?", starts: [0] }], phones, 8)).toThrow();
  });
});

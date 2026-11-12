import { describe, expect, it } from "vitest";
import { articulation, articulatoryCosts, articulatoryLogLikelihood, bundleObservation, fitArticulatoryModel, posteriorObservation } from "../src/articulatory-channel.js";
import type { ArticulatoryModel } from "../src/articulatory-channel.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { UNIT } from "../src/fuzzy.js";
import { trackerInput } from "../src/tracker.js";

const phones = ["ب", "ت", "د", "ن", "ز"];
const records = phones.map((p, i) => ({ written: p.repeat(4), observed: Array.from({ length: 4 }, () => phones.map((_, j) => i === j ? 0.96 : 0.01)) }));
const fitted = () => fitArticulatoryModel(records, phones, "joint", 4);

describe("joint articulation evidence", () => {
  it("does not manufacture a voiced alveolar stop from separate plausible traits", () => {
    const model = fitted();
    const q = bundleObservation(model, [0.5, 0.5, 0, 0, 0])!;
    expect(q[model.states.indexOf(articulation("د"))]).toBe(0);
    expect(q[model.states.indexOf(articulation("ب"))]).toBe(0.5);
    expect(q[model.states.indexOf(articulation("ت"))]).toBe(0.5);
  });

  it("preserves posterior uncertainty, residual mass and absolute run coordinates", () => {
    const rows = { k: 2, startFrame: 8, nFrames: 2, ids: [0, 1, 0, 2], probs: [0.4, 0.4, 0.2, 0.6] };
    const q = posteriorObservation(rows, ["ب", "ت"], { start: 8, end: 9 })!;
    expect(q[0]).toBeCloseTo(0.6 / 1.4);
    expect(q[1]).toBeCloseTo(0.8 / 1.4);
    expect(posteriorObservation(rows, ["ب", "ت"], { start: 7, end: 8 })).toBeUndefined();
  });

  it("fits normalized emissions with fractional, ambiguous placement", () => {
    const ambiguous = [{ written: "بت", observed: [[0.5, 0.5, 0, 0, 0]] }];
    for (const kind of ["joint", "identity"] as const) {
      const model = fitArticulatoryModel([...records, ...ambiguous], phones, kind, 4);
      for (const row of model.emission) {
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
        expect(row.every(p => p > 0)).toBe(true);
      }
      expect(Number.isFinite(articulatoryLogLikelihood(model, ambiguous[0]))).toBe(true);
      expect(model).toEqual(fitArticulatoryModel([...records, ...ambiguous], phones, kind, 4));
    }
  });

  it("shares bundles while retaining a phone identity control on identical data", () => {
    const inventory = ["ت", "ط", "د"];
    const record = [{ written: "تطد", observed: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }];
    const joint = fitArticulatoryModel(record, inventory, "joint", 4);
    const identity = fitArticulatoryModel(record, inventory, "identity", 4);
    expect(joint.phoneState[0]).toBe(joint.phoneState[1]);
    expect(identity.phoneState[0]).not.toBe(identity.phoneState[1]);
    expect(articulatoryCosts(joint, [0.2, 0.6, 0.2]).get("ت")).toBe(articulatoryCosts(joint, [0.2, 0.6, 0.2]).get("ط"));
  });

  it("pooled, absent and invalid observations contribute no preference", () => {
    const model = fitted();
    const pooled: ArticulatoryModel = { ...model, emission: model.emission.map(() => model.background) };
    expect([...articulatoryCosts(pooled, [0.1, 0.2, 0.3, 0.2, 0.2]).values()].every(x => x === 0)).toBe(true);
    expect(articulatoryCosts(model, [NaN]).size).toBe(0);
    expect(articulatoryCosts(model, phones.map(() => 0)).size).toBe(0);
  });

  it("retains local posterior metadata without mutating listener evidence or changing acquisition input", () => {
    const model = { ...fitted(), inventory: "test" };
    const input = { inventory: "test", frameMs: 40, phonemes: [{ p: "ب", start: 4, end: 4, conf: 0.6 }],
      posteriors: { startFrame: 4, nFrames: 1, k: 3, ids: [0, 1, 2], probs: [0.1, 0.6, 0.3] } };
    const before = JSON.stringify(input);
    const local = trackerInput(input, model);
    expect(local.posteriors).toBeUndefined();
    expect(local.phonemes[0].phonePosterior?.[0]).toBeCloseTo(2 / 3);
    expect(local.phonemes[0].p).toBe("ب");
    expect(JSON.stringify(input)).toBe(before);
    expect(trackerInput({ ...input, inventory: "other" }, model).phonemes).toBe(input.phonemes);
  });
});

const index = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "بنتز" },
  ayahs: [{ s: 1, a: 1, w: ["بننن", "تز"], p: "بنننتز", o: [0, 4], iso: ["بننن", "تز"] }] });
const search = () => createLocalSearch(index, { offset: 1, origin: 0, edits: 0 });

describe("local observation lattice", () => {
  it("retains the existing duration deletion charge and movement edges", () => {
    const plain = search(), acoustic = search();
    const zero = new Map<string | null, number>([[null, 0], ...[...index.tokens].map(p => [p, 0] as [string, number])]);
    for (const p of "نتز") {
      plain.step(p); acoustic.step(p, zero);
      expect(acoustic.position()).toEqual(plain.position());
    }
    // Matching noon then crossing the shortened length obligation still pays two 0.2-edit charges.
    expect(plain.position().recentCost * 3).toBeCloseTo(0.4);
  });

  it("uses the same acoustic correction for continuation and a restarted diagonal", () => {
    const a = search(), b = search();
    a.step("ن");
    b.step("ن", new Map([["ن", -UNIT / 3], [null, -UNIT / 3], ["ب", -UNIT / 3], ["ت", -UNIT / 3], ["ز", -UNIT / 3]]));
    expect(b.position().offset).toBe(a.position().offset);
    expect(b.position().origin).toBe(a.position().origin);
    expect(b.position().stepCost).toBeCloseTo(a.position().stepCost - 1 / 3);
    const state = b.save();
    b.step("ت"); b.restore(state);
    expect(b.position().offset).toBe(a.position().offset);
  });
});

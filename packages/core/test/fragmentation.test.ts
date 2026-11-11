import { describe, expect, it } from "vitest";
import { fitFragmentation, fragmentationCosts, fragmentationEligible, fragmentationLogP, fragmentationProbability, heterogeneous } from "../src/fragmentation.js";
import type { FragmentationModel } from "../src/fragmentation.js";
import { createLocalSearch } from "../src/local.js";
import { deletionCost } from "../src/fuzzy.js";
import { loadReferenceIndex } from "../src/reference.js";

const phones = ["ب", "د", "ت", "ل", "ر", "ن", "م"];
const disabled = (): FragmentationModel => ({ phones, mass: 8, phonesRows: {}, classes: {}, pooled: { ordinary: 1, sequences: {}, residual: 0 } });
const model = (): FragmentationModel => ({ ...disabled(), phonesRows: { ب: { ordinary: 0.1, sequences: { بد: 0.9 }, residual: 0 } } });
const index = (p = "بتلر", starts = [0, 2]) => loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: phones.join("") },
  ayahs: [{ s: 1, a: 1, w: starts.map((_, i) => `word${i}`), p, o: starts, iso: starts.map((s, i) => p.slice(s, starts[i + 1] ?? p.length)) }] });

describe("heterogeneous observation fragments", () => {
  it("normalizes full sequence support and complete-string probability", () => {
    const m = { ...disabled(), phones: ["ب", "د"], pooled: { ordinary: 0.6, sequences: { بد: 0.2 }, residual: 0.2 } };
    let eventMass = m.pooled.ordinary;
    let stringMass = 0;
    for (let length = 0; length <= 8; length++) for (let bits = 0; bits < 2 ** length; bits++) {
      const heard = Array.from({ length }, (_, i) => m.phones[(bits >> i) & 1]).join("");
      if (heterogeneous(heard)) eventMass += fragmentationProbability(m, "ب", heard);
      stringMass += Math.exp(fragmentationLogP(m, { written: "ب", heard, starts: [0] }));
    }
    expect(eventMass).toBeCloseTo(1, 12);
    expect(stringMass).toBeCloseTo(1, 5);
  });

  it("consumes two different observations once with one reference owner, then continues", () => {
    const search = createLocalSearch(index(), { offset: 0, origin: 0, edits: 0 }, model());
    search.step("ب"); search.step("د");
    expect(search.position().offset).toBe(1);
    expect(search.position().edits).toBe(1);
    expect(search.position().fragment).toEqual({ referenceOffset: 0, heard: "بد", fromObservation: 0, toObservation: 2 });
    search.step("ت");
    expect(search.position().offset).toBe(2);
    expect(search.position().origin).toBe(0);
    expect(search.position().edits).toBe(1);
  });

  it("retains ordered three-token emissions and restores their entire observation history", () => {
    const m = { ...disabled(), phonesRows: { ب: { ordinary: 0.1, sequences: { بدب: 0.9 }, residual: 0 } } };
    const search = createLocalSearch(index(), { offset: 0, origin: 0, edits: 0 }, m);
    search.step("ب"); search.step("د");
    const saved = search.save();
    search.step("ب");
    expect(search.position().offset).toBe(1);
    expect(search.position().fragment?.heard).toBe("بدب");
    const after = search.position();
    search.step("ت"); search.step("ل");
    search.restore(saved); search.step("ب");
    expect(search.position()).toEqual(after);
    expect(fragmentationProbability(m, "ب", "ببد")).toBe(0);
  });

  it("charges the same fragment event on a word-start restart, retaining the movement price", () => {
    const search = createLocalSearch(index("لللللللللبت", [0, 9]), { offset: 0, origin: 0, edits: 0 }, model());
    for (const p of "بدت") search.step(p);
    expect(search.position().offset).toBe(11);
    expect(search.position().origin).toBe(9);
    expect(search.position().fragment?.referenceOffset).toBe(9);
    expect(search.position().recentCost).toBeGreaterThan(0);
  });

  it("preserves ordinary paths exactly when training selects no fragmentation", () => {
    const a = createLocalSearch(index(), { offset: 0, origin: 0, edits: 0 });
    const b = createLocalSearch(index(), { offset: 0, origin: 0, edits: 0 }, disabled());
    for (const p of "بتلربدلر") {
      a.step(p); b.step(p);
      expect(b.position()).toEqual(a.position());
    }
  });

  it("keeps common-interval costs and fragment ownership through window recentering", () => {
    const texts = ["تل", "در", "بت", "لم", "رنم", "تل"];
    const ref = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: phones.join("") },
      ayahs: texts.map((p, i) => ({ s: 1, a: i + 1, w: ["word"], p, o: [0], iso: [p] })) });
    const search = createLocalSearch(ref, { offset: 0, origin: 0, edits: 0 }, model());
    for (const p of "تلدربد") search.step(p);
    const saved = search.save();
    const finish = () => { for (const p of "تلمرنمتل") search.step(p); return search.position(); };
    const after = finish();
    expect(after.offset).toBe(ref.phonemes.length);
    expect(after.origin).toBe(0);
    expect(after.fragment?.referenceOffset).toBe(4);
    const state = search.save() as { cost: Float64Array; lo: number; total: number };
    const charge = fragmentationCosts(model(), ref.phonemes, new Set(ref.words.map(w => w.start))).fragmented(4, "بد");
    expect(state.cost[after.offset - state.lo] + state.total).toBeCloseTo(charge, 10);
    search.restore(saved);
    expect(finish()).toEqual(after);
  });

  it("keeps duration symbols chargeable and prevents another fragmented event inside their run", () => {
    const ref = "ننننت";
    const m = { ...disabled(), phonesRows: { ن: { ordinary: 0.1, sequences: { نم: 0.9 }, residual: 0 } } };
    const costs = fragmentationCosts(m, ref, new Set([0]));
    expect(fragmentationEligible(ref, 0, new Set([0]))).toBe(true);
    expect(fragmentationEligible(ref, 1, new Set([0]))).toBe(false);
    expect(costs.fragmented(1, "نم")).toBe(Infinity);
    expect(fragmentationEligible(ref, 1, new Set([0, 1]))).toBe(true);
    const search = createLocalSearch(index(ref, [0]), { offset: 0, origin: 0, edits: 0 }, m);
    for (const p of "نمت") search.step(p);
    expect(search.position().offset).toBe(5);
    const state = search.save() as { cost: Float64Array; lo: number; total: number };
    expect(state.cost[5 - state.lo] + state.total).toBeCloseTo(costs.fragmented(0, "نم")
      + [1, 2, 3].reduce((sum, at) => sum + deletionCost(ref, at), 0), 10);
  });

  it("does not turn a genuine missing word into a free reference-consuming event", () => {
    const search = createLocalSearch(index("بتللللللللدرت", [0, 2, 10]), { offset: 2, origin: 0, edits: 0 }, model());
    for (const p of "درت") search.step(p);
    expect(search.position().origin).toBe(10);
    expect(search.position().recentCost).toBeGreaterThan(0);
  });

  it("fits latent fragmentation sequences with normalized class backoff and rejects unknown phones", () => {
    const recordings = Array.from({ length: 20 }, () => ({ written: "بت", heard: "بدت", starts: [0] }));
    const fitted = fitFragmentation(recordings, phones, 8);
    for (const row of [fitted.pooled, ...Object.values(fitted.classes), ...Object.values(fitted.phonesRows)]) {
      expect(row.ordinary + row.residual + Object.values(row.sequences).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
      expect(row.ordinary).toBeGreaterThan(0);
      expect(row.residual).toBeGreaterThan(0);
    }
    expect(fragmentationLogP(fitted, recordings[0])).toBeGreaterThan(fragmentationLogP(disabled(), recordings[0]));
    expect(fragmentationProbability(fitted, "ب", "بد")).toBeGreaterThan(fragmentationProbability(fitted, "م", "بد"));
    expect(fragmentationProbability(fitted, "د", "بد")).toBeGreaterThan(0);
    expect(fragmentationProbability(fitted, "ب", "بب")).toBe(0);
    expect(fragmentationProbability(fitted, "ب", "ب?")).toBe(0);
    expect(() => fitFragmentation([{ written: "ب", heard: "?", starts: [0] }], phones, 8)).toThrow();
  });
});

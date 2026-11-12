import { describe, expect, it } from "vitest";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";
import { composeRunTransducers, createTeacherObservationScorer, createTeacherObservationComparison, fitTeacherObservationModels,
  observationCost, validateRunTransducer } from "../src/teacher-observation.js";
import type { RunTransducer, TeacherObservationModel } from "../src/teacher-observation.js";

const phones = [..."بت"];
const channel = (rows: number[][]): RunTransducer => ({ phones, fragments: ["", "ب", "ت", "بت"], rows });
const model: TeacherObservationModel = {
  referenceToTeacher: channel([[.1, .5, .2, .2], [.2, .1, .5, .2]]),
  teacherToStudent: channel([[.1, .4, .2, .3], [.2, .1, .5, .2]]),
};
const strings = (max: number): string[] => max ? ["", ...strings(max - 1).flatMap(s => phones.map(p => p + s))] : [""];
const index = loadReferenceIndex({ version: "synthetic", text: "synthetic", profile: {}, inventory: { id: "synthetic", tokens: "بت" },
  ayahs: [{ s: 1, a: 1, w: ["first", "second", "third"], p: "ببتتببت", o: [0, 2, 4], iso: ["بب", "تت", "ببت"] }] });

describe("teacher-mediated observations", () => {
  it("marginalizes complete teacher runs and all splits, without multiplying unnormalized alternatives", () => {
    const composed = composeRunTransducers(model), scorer = createTeacherObservationScorer(model);
    for (let r = 0; r < phones.length; r++) {
      expect(composed.rows[r].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      for (const output of strings(4)) expect(scorer.probability(phones[r], output))
        .toBeCloseTo(composed.rows[r][composed.fragments.indexOf(output)] ?? 0, 12);
    }
    // Empty teacher output and a deleted nonempty teacher run are distinct, normalized explanations.
    expect(scorer.probability("ب", "")).toBeCloseTo(.1 + .5 * .1 + .2 * .2 + .2 * .1 * .2, 12);
  });

  it("sums alternative token-to-run segmentations and preserves the full-string probability mass", () => {
    const scorer = createTeacherObservationScorer(model);
    let total = 0;
    for (const output of strings(8)) total += Math.exp(scorer.logP("بت", output));
    expect(total).toBeCloseTo(1, 10);
    const p = scorer.probability;
    expect(Math.exp(scorer.logP("بت", "بت"))).toBeCloseTo(
      p("ب", "") * p("ت", "بت") + p("ب", "ب") * p("ت", "ت") + p("ب", "بت") * p("ت", ""), 12);
  });

  it("rejects invalid simplexes and inventory mismatches", () => {
    expect(() => validateRunTransducer(channel([[1, 1, 0, 0], [1, 0, 0, 0]]))).toThrow();
    expect(() => createTeacherObservationScorer({ ...model, teacherToStudent: { ...model.teacherToStudent, phones: [..."تب"] } })).toThrow();
    expect(() => createTeacherObservationScorer(model).logP("ب", "ز")).toThrow();
  });

  it("fits both normalized channels with exactly the same capacity from paired fallible decodes", () => {
    const records = Array.from({ length: 8 }, (_, i) => ({ written: i % 2 ? "تبتب" : "بتبت",
      teacher: i % 2 ? "تبتب" : "بتبت", student: "بببب" }));
    const fitted = fitTeacherObservationModels(records, phones, 2, 3);
    expect(fitted.parameters).toBe(2 * phones.length * (fitted.mediated.referenceToTeacher.fragments.length - 1));
    for (const m of [fitted.mediated, fitted.direct]) for (const t of [m.referenceToTeacher, m.teacherToStudent]) validateRunTransducer(t);
    expect(fitted.direct.referenceToTeacher.fragments).toEqual(fitted.mediated.referenceToTeacher.fragments);
    expect(Number.isFinite(createTeacherObservationScorer(fitted.mediated).logP("تبتب", "بببب"))).toBe(true);
  });

  it("retains ordinary local decisions, jump charges and duration charges with an identical-channel control", () => {
    let comparisons = 0;
    for (const heard of ["ببتببتببت", ...strings(5).filter(s => s.length === 5)]) {
      const ordinary = createLocalSearch(index, { offset: 0, origin: 0, edits: 0 });
      const controlled = createLocalSearch(index, { offset: 0, origin: 0, edits: 0 }, { mediated: model, direct: model }, fit => {
        comparisons++;
        expect(fit.n).toBe(fit.heard.length);
        expect(Object.isFrozen(fit)).toBe(true);
        expect(Object.isFrozen(fit.continuation)).toBe(true);
        expect(Object.isFrozen(fit.restart)).toBe(true);
      });
      for (const p of heard) {
        ordinary.step(p); controlled.step(p);
        expect(controlled.position()).toEqual(ordinary.position());
      }
    }
    expect(comparisons).toBeGreaterThan(0);
  });

  it("rewinds observations and path witnesses together, including after a restart", () => {
    const local = createLocalSearch(index, { offset: 0, origin: 0, edits: 0 }, { mediated: model, direct: model });
    for (const p of "ببت") local.step(p);
    const saved = local.save();
    for (const p of "ببت") local.step(p);
    const first = local.position();
    local.restore(saved);
    for (const p of "ببت") local.step(p);
    expect(local.position()).toEqual(first);
    expect(observationCost(-3)).toBe(10);
  });

  it("can reverse a restart's raw ordering through a learned observation residual over the same tokens", () => {
    const identity = channel([[0, 1, 0, 0], [0, 0, 1, 0]]);
    const direct: TeacherObservationModel = { referenceToTeacher: identity,
      teacherToStudent: channel([[.01, .98, .01, 0], [.01, .01, .98, 0]]) };
    const mediated: TeacherObservationModel = { referenceToTeacher: identity,
      teacherToStudent: channel([[.01, .01, .98, 0], [.01, .98, .01, 0]]) };
    const compare = createTeacherObservationComparison({ mediated, direct });
    const fit = compare({ written: "ب", cost: 24 }, { written: "ت", cost: 20 }, "ت")!;
    expect(fit.n).toBe(1);
    expect(fit.winner).toBe("continuation");
    expect(fit.continuation.cost).toBe(24); // the original deletion/duration/movement charges survive
    expect(fit.restart.cost).toBe(20);
    const control = createTeacherObservationComparison({ mediated: direct, direct });
    expect(control({ written: "ب", cost: 24 }, { written: "ت", cost: 20 }, "ت")!.winner).toBe("restart");
    expect(compare({ written: "", cost: 0 }, { written: "ت", cost: 0 }, "ت")).toBe(null);
  });

  it("uses the observation comparison to keep a systematically confused word on its continuation", () => {
    const make = (rows: number[][]): RunTransducer => ({ phones: [..."بتد"], fragments: ["", "ب", "ت", "د"], rows });
    const identity = make([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
    const direct: TeacherObservationModel = { referenceToTeacher: identity, teacherToStudent: make([
      [.01, .98, .005, .005], [.01, .005, .98, .005], [.01, .005, .005, .98],
    ]) };
    const mediated: TeacherObservationModel = { referenceToTeacher: identity, teacherToStudent: make([
      [.01, .005, .98, .005], [.01, .98, .005, .005], [.01, .005, .005, .98],
    ]) };
    const reference = loadReferenceIndex({ version: "synthetic", text: "synthetic", profile: {}, inventory: { id: "synthetic", tokens: "بتد" },
      ayahs: [{ s: 1, a: 1, w: ["source", "look-alike"], p: "بببدددتتتددد", o: [0, 6], iso: ["بببددد", "تتتددد"] }] });
    const control = createLocalSearch(reference, { offset: 0, origin: 0, edits: 0 }, { mediated: direct, direct });
    const local = createLocalSearch(reference, { offset: 0, origin: 0, edits: 0 }, { mediated, direct });
    for (const p of "تتتددد") { control.step(p); local.step(p); }
    expect(control.position().origin).toBe(6);
    expect(local.position().origin).toBe(0);
    expect(local.position().offset).toBe(6);
  });
});

import { describe, expect, it } from "vitest";
import { teacherObservationFolds, teacherObservationTrainingLabels } from "../src/teacher-observation-training.js";
import type { TeacherObservationLabel } from "../src/teacher-observation-training.js";

const label = (id: string, overrides: Partial<TeacherObservationLabel> = {}): TeacherObservationLabel => ({ id, speaker: id,
  split: "train", kind: "none", scored: true, ayahs: ["1:1"], wav: `${id}.wav`, events: [], non_errors: [], ...overrides });

describe("teacher observation training isolation", () => {
  it("excludes lexical/pronunciation scripts, repeat and additional mistakes, and entire held-out speakers", () => {
    const all = [label("eligible"), label("script", { kind: "vowel" }), label("unscored", { scored: false }),
      label("excluded", { excluded: "review" }), label("extra", { events: [{}] }),
      label("unsupported", { unsupported: [{}] }), label("repeat", { non_errors: [{}] }),
      label("held", { split: "eval" }), label("mixed-train", { speaker: "mixed" }),
      label("mixed-eval", { speaker: "mixed", split: "eval" })];
    expect(teacherObservationTrainingLabels(all).map(l => l.id)).toEqual(["eligible"]);
  });
  it("keeps speakers together and partitions independently of input order", () => {
    const all = [label("a"), label("b"), label("c"), label("second-a", { speaker: "a" })];
    expect([...teacherObservationFolds(all)]).toEqual([...teacherObservationFolds([...all].reverse())]);
    expect(teacherObservationFolds(all).size).toBe(3);
    expect(() => teacherObservationFolds([label("a")])).toThrow();
  });
});

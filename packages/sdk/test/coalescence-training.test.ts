import { describe, expect, it } from "vitest";
import { coalescenceSpeakerFolds, coalescenceTrainingLabels } from "../src/coalescence-training.js";
import type { CoalescenceLabel } from "../src/coalescence-training.js";

const label = (id: string, speaker: string, extra: Partial<CoalescenceLabel> = {}): CoalescenceLabel => ({
  id, speaker, kind: "none", split: "train", scored: true, ayahs: ["1:1"], wav: "unused",
  events: [], non_errors: [], ...extra,
});

describe("coalescence training isolation", () => {
  it("excludes all non-training speakers, labelled mistakes and repetitions before decoding", () => {
    const rows = [label("clean", "a"), label("mistake", "b", { kind: "skipped_word" }),
      label("repeat", "c", { non_errors: [{}] }), label("event", "d", { events: [{}] }),
      label("unscored", "e", { scored: false }), label("excluded", "f", { excluded: "audit" }),
      label("held", "g", { split: "eval" }), label("mixed", "g"), label("other", "h")];
    expect(coalescenceTrainingLabels(rows).map(l => l.id)).toEqual(["clean", "other"]);
  });

  it("keeps every recording of a speaker in one deterministic training partition", () => {
    const labels = [label("a1", "a"), label("a2", "a"), label("b1", "b"), label("c1", "c")];
    const folds = coalescenceSpeakerFolds(labels);
    expect([...folds]).toEqual([...coalescenceSpeakerFolds([...labels].reverse())]);
    expect(folds.size).toBe(3);
    expect(new Set(folds.values()).size).toBe(3);
    expect(() => coalescenceSpeakerFolds(labels.slice(0, 2))).toThrow();
  });
});

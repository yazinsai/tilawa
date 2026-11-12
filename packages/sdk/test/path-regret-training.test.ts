import { describe, expect, it } from "vitest";
import { regretSpeakerPartitions, regretTrainingLabels } from "../src/path-regret-training.js";
import type { RegretTrainingLabel } from "../src/path-regret-training.js";

const label = (id: string, changes: Partial<RegretTrainingLabel> = {}): RegretTrainingLabel => ({
  id, speaker: id, split: "train", kind: "none", scored: true, ayahs: [], wav: "", events: [], non_errors: [], ...changes,
});

describe("authorized speaker-separated uncertainty training", () => {
  it("excludes labelled mistakes, repeats, exclusions and unscored takes", () => {
    const all = [label("clean"), label("script", { kind: "omission" }), label("events", { events: [{}] }),
      label("repeat", { non_errors: [{}] }), label("excluded", { excluded: true }), label("unscored", { scored: false })];
    expect(regretTrainingLabels(all).map(l => l.id)).toEqual(["clean"]);
  });
  it("excludes every recording of any speaker with a nontraining row", () => {
    expect(regretTrainingLabels([label("train", { speaker: "shared" }),
      label("held", { speaker: "shared", split: "test" }), label("safe")]).map(l => l.id)).toEqual(["safe"]);
  });
  it("keeps each speaker in one deterministic disjoint partition", () => {
    const labels = [label("c"), label("b"), label("a"), label("also", { speaker: "a" })];
    expect([...regretSpeakerPartitions(labels)]).toEqual([...regretSpeakerPartitions([...labels].reverse())]);
    expect(regretSpeakerPartitions(labels).size).toBe(3);
    expect(() => regretSpeakerPartitions([label("one")])).toThrow();
  });
});

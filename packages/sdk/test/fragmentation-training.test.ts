import { describe, expect, it } from "vitest";
import { fragmentationSpeakerFolds, fragmentationTrainingLabels } from "../src/fragmentation-training.js";
import type { FragmentationLabel } from "../src/fragmentation-training.js";

const label = (id: string, speaker: string, extra: Partial<FragmentationLabel> = {}): FragmentationLabel => ({
  id, speaker, kind: "none", split: "train", scored: true, ayahs: ["1:1"], wav: `${id}.wav`, events: [], non_errors: [], ...extra,
});
describe("fragmentation training isolation", () => {
  it("excludes planted edits, repeats, held-out speakers and mixed-split speakers", () => {
    const all = [label("a", "s1"), label("b", "s1"), label("c", "s2"), label("d", "s3", { split: "eval" }),
      label("e", "s4"), label("f", "s4", { split: "eval" }), label("g", "s5", { kind: "skip" }),
      label("h", "s5", { events: [{}] }), label("i", "s5", { non_errors: [{}] }), label("j", "s5", { scored: false }),
      label("k", "s5", { excluded: true })];
    expect(fragmentationTrainingLabels(all).map(l => l.id)).toEqual(["a", "b", "c"]);
  });
  it("keeps speakers together in deterministic training partitions", () => {
    const labels = [label("a", "s1"), label("b", "s1"), label("c", "s2"), label("d", "s3")];
    expect(fragmentationSpeakerFolds(labels)).toEqual(fragmentationSpeakerFolds([...labels].reverse()));
    expect(fragmentationSpeakerFolds(labels).size).toBe(3);
    expect(() => fragmentationSpeakerFolds([label("a", "s1")])).toThrow();
  });
});

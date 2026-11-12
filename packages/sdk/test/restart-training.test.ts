import { describe, expect, it } from "vitest";
import { restartSpeakerFolds, restartTrainingLabels } from "../src/restart-training.js";
import type { RestartTrainingLabel } from "../src/restart-training.js";
const label = (id: string, speaker = id): RestartTrainingLabel => ({ id, speaker, split: "train", kind: "none", scored: true,
  ayahs: ["1:1"], wav: `${id}.wav`, events: [], non_errors: [] });
describe("restart training eligibility", () => {
  it("excludes every held-out speaker, even when another recording says train", () => {
    const clean = label("clean"), leaked = label("leaked", "held");
    const held = { ...label("held", "held"), split: "eval" };
    expect(restartTrainingLabels([clean, leaked, held])).toEqual([clean]);
  });
  it("excludes scripted mistakes, repeats, unscored takes and excluded recordings", () => {
    const clean = label("clean");
    expect(restartTrainingLabels([clean, { ...label("mistake"), events: [{}] }, { ...label("repeat"), non_errors: [{}] },
      { ...label("script"), kind: "omitted_word" }, { ...label("unscored"), scored: false }, { ...label("excluded"), excluded: true }])).toEqual([clean]);
  });
  it("keeps speakers together and makes partitions independent of recording order", () => {
    const labels = [label("a"), label("b"), label("a2", "a")];
    expect([...restartSpeakerFolds(labels)]).toEqual([...restartSpeakerFolds([...labels].reverse())]);
    expect(() => restartSpeakerFolds([label("a")])).toThrow("two");
  });
});

import { describe, expect, it } from "vitest";
import { editOrderSpeakerFolds, editOrderTrainingLabels } from "../src/edit-order-training.js";
import type { EditOrderLabel } from "../src/edit-order-training.js";

const clean = (id: string, speaker = id): EditOrderLabel => ({ id, speaker, split: "train", kind: "none", scored: true, ayahs: ["1:1"], wav: "unused", events: [], non_errors: [] });
describe("edit-order training isolation", () => {
  it("excludes non-training speakers across all their recordings, mistakes, repeats and excluded clips", () => {
    const labels = [clean("a"), clean("b"), { ...clean("held", "b"), split: "eval" },
      { ...clean("c"), events: [{}] }, { ...clean("d"), kind: "omitted_word" },
      { ...clean("e"), non_errors: [{}] }, { ...clean("f"), excluded: true }, { ...clean("g"), scored: false }];
    expect(editOrderTrainingLabels(labels).map(l => l.id)).toEqual(["a"]);
  });
  it("keeps recordings from one speaker together, reproducibly without using recording identity", () => {
    const labels = [clean("z", "a"), clean("x", "a"), clean("y", "b"), clean("w", "c")];
    expect(editOrderSpeakerFolds(labels)).toEqual(editOrderSpeakerFolds([...labels].reverse()));
    expect(editOrderSpeakerFolds(labels).size).toBe(3);
    expect(() => editOrderSpeakerFolds(labels.slice(0, 2))).toThrow();
  });
});

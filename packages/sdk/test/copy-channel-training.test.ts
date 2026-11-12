import { describe, expect, it } from "vitest";
import { copySpeakerFolds, copyTrainingLabels } from "../src/copy-channel-training.js";
import type { CopyTrainingLabel } from "../src/copy-channel-training.js";

const label = (speaker: string, patch: Partial<CopyTrainingLabel> = {}): CopyTrainingLabel => ({
  id: speaker, speaker, split: "train", kind: "none", scored: true, ayahs: ["1:1"], wav: "recording.wav", events: [], non_errors: [], ...patch,
});
describe("copy-channel training isolation", () => {
  it("excludes held-out speakers entirely, mistakes, repeats and unscored or excluded takes", () => {
    const all = [label("eligible"), label("held", { split: "eval" }), label("mixed"), label("mixed", { split: "eval" }),
      label("mistake", { kind: "skip_word", events: [{}] }), label("repeat", { non_errors: [{}] }),
      label("unscored", { scored: false }), label("excluded", { excluded: true })];
    expect(copyTrainingLabels(all).map(l => l.speaker)).toEqual(["eligible"]);
  });
  it("groups all takes of a speaker into one deterministic training-validation partition", () => {
    const all = [label("one"), label("two"), label("three"), label("one", { id: "another" })];
    expect([...copySpeakerFolds(all)]).toEqual([...copySpeakerFolds([...all].reverse())]);
    expect(copySpeakerFolds(all).size).toBe(3);
    expect(() => copySpeakerFolds([label("only")])).toThrow("at least two");
  });
});

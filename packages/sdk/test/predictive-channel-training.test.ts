import { expect, it } from "vitest";
import { predictiveTrainingLabels, predictiveSpeakerFolds } from "../src/predictive-channel-training.js";
import type { PredictiveTrainingLabel } from "../src/predictive-channel-training.js";

const label = (speaker: string, change = {}): PredictiveTrainingLabel => ({ id: speaker, speaker, split: "train", kind: "none",
  scored: true, ayahs: ["1:1"], wav: "wav/clip.wav", events: [], non_errors: [], ...change });
it("excludes held-out and mixed-split speakers, planted edits, repeats and unscored takes", () => {
  const all = [label("a"), label("b"), label("c"), label("c", { split: "eval" }), label("d", { kind: "vowel" }),
    label("e", { non_errors: [{}] }), label("f", { events: [{}] }), label("g", { scored: false })];
  expect(predictiveTrainingLabels(all).map(l => l.speaker)).toEqual(["a", "b"]);
  expect([...predictiveSpeakerFolds(predictiveTrainingLabels(all))]).toEqual([["a", 0], ["b", 1]]);
});

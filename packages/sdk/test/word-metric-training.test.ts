import { expect, it } from "vitest";
import { wordSpeakerFolds, wordTrainingLabels } from "../src/word-metric-training.js";
import type { WordTrainingLabel } from "../src/word-metric-training.js";

const label = (speaker: string, extra: Partial<WordTrainingLabel> = {}): WordTrainingLabel => ({
  id: speaker, speaker, split: "train", kind: "none", scored: true, ayahs: [], wav: "sample.wav", events: [], non_errors: [], ...extra,
});

it("excludes scripted mistakes, repeats, excluded clips, and every held-out speaker", () => {
  const rows = [label("allowed"), label("heldout"), label("heldout", { split: "exam" }),
    label("script", { kind: "omitted_word" }), label("mistake", { events: [{}] }),
    label("repeat", { non_errors: [{}] }), label("excluded", { excluded: "audit" }),
    label("unscored", { scored: false }), label("unknown", { split: "" })];
  expect(wordTrainingLabels(rows).map(l => l.speaker)).toEqual(["allowed"]);
});

it("keeps all takes of a speaker together regardless of input order", () => {
  const rows = [label("c"), label("a"), label("b"), label("a", { id: "another" })];
  expect([...wordSpeakerFolds(rows)]).toEqual([...wordSpeakerFolds([...rows].reverse())]);
  expect(wordSpeakerFolds(rows).size).toBe(3);
  expect(() => wordSpeakerFolds([label("one")])).toThrow();
});

import { expect, it } from "vitest";
import { editSpeakerFolds, editTrainingLabels } from "../src/edit-parameter-training.js";
import type { EditTrainingLabel } from "../src/edit-parameter-training.js";

const row = (id: string, speaker: string, changes: Partial<EditTrainingLabel> = {}): EditTrainingLabel =>
  ({ id, speaker, split: "train", kind: "none", scored: true, ayahs: [], wav: "", events: [], non_errors: [], ...changes });

it("excludes scripted mistakes, repeats, excluded clips and every held-out speaker's recordings", () => {
  const labels = [row("a", "A"), row("b", "B"), row("c", "C", { split: "eval" }), row("d", "C"),
    row("e", "A", { kind: "skipped_word" }), row("f", "B", { non_errors: [{}] }),
    row("g", "A", { events: [{}] }), row("h", "B", { scored: false }), row("i", "A", { excluded: "audit" })];
  expect(editTrainingLabels(labels).map(l => l.id)).toEqual(["a", "b"]);
});

it("keeps each speaker in one reproducible validation fold regardless of recording order", () => {
  const labels = [row("a", "A"), row("b", "B"), row("c", "A"), row("d", "D")];
  expect(editSpeakerFolds(labels)).toEqual(editSpeakerFolds([...labels].reverse()));
  expect(new Set(editSpeakerFolds(labels).values()).size).toBe(3);
  expect(() => editSpeakerFolds([row("a", "A")])).toThrow();
});

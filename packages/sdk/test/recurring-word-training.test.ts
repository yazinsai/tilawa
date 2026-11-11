import { expect, it } from "vitest";
import { recurringWordTrainingLabels } from "../src/recurring-word-training.js";
import type { RecurringWordLabel } from "../src/recurring-word-training.js";

const label = (extra: Partial<RecurringWordLabel> = {}): RecurringWordLabel => ({ id: "take", speaker: "training",
  split: "train", kind: "none", scored: true, wav: "take.wav", events: [], non_errors: [], ...extra });
it("excludes all takes of a held-out speaker and every accused take", () => {
  const clean = label();
  expect(recurringWordTrainingLabels([clean, label({ speaker: "held-out" }), label({ speaker: "held-out", split: "test" }),
    label({ kind: "wrong_word" }), label({ events: [{}] }), label({ non_errors: [{}] }), label({ scored: false }), label({ excluded: true })])).toEqual([clean]);
});
it("allows explicitly benign repeats for earlier acoustic exemplars", () => {
  const repeat = label({ kind: "repeat", non_errors: [{ kind: "repeat" }] });
  expect(recurringWordTrainingLabels([repeat])).toEqual([repeat]);
});
it("requires explicit complete training authorization", () => {
  expect(recurringWordTrainingLabels([label({ split: "" }), label({ speaker: "" }), label({ wav: "" }),
    label({ events: undefined as unknown as unknown[] })])).toEqual([]);
});

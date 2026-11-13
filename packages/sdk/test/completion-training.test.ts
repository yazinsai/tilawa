import { describe, expect, it } from "vitest";
import { completionExamples, completionTrainingLabels } from "../src/completion-training.js";
import type { CompletionLabel } from "../src/completion-training.js";
import type { LocalSnapshot } from "../../core/dist/local.js";
import type { WordPrefix } from "../../core/dist/completion-survival.js";

describe("completion training isolation and censoring", () => {
  const label: CompletionLabel = { id: "take", speaker: "train-a", split: "train", kind: "none", scored: true,
    wav: "take.wav", ayahs: ["1:1"], events: [], non_errors: [] };
  it("excludes non-training, mixed-split, edited, and repeat takes", () => {
    const labels = [label, { ...label, id: "eval", speaker: "mixed", split: "eval" }, { ...label, id: "mixed", speaker: "mixed" },
      { ...label, id: "edit", events: [{}] }, { ...label, id: "repeat", non_errors: [{}] }, { ...label, id: "unscored", scored: false }];
    expect(completionTrainingLabels(labels)).toEqual([label]);
  });
  const p: WordPrefix = { word: 0, entry: 1, progress: 2, length: 5, observations: 2, matches: 2, editCost: 0, lastCost: 0, stalled: 0 };
  const snapshot = (token: number, prefix: WordPrefix, origin = 0): LocalSnapshot => ({ token, selected: prefix,
    candidates: [{ offset: prefix.progress, origin, cost: 0, prefix }] });
  it("requires later in-order progress and censors a terminal unfinished prefix", () => {
    const first = snapshot(1, p), next = snapshot(2, { ...p, word: 1, progress: 1 });
    expect(completionExamples([first], new Set([0, 1]))[0].outcome).toBe("censored");
    expect(completionExamples([first, next], new Set([0, 1]))[0].outcome).toBe("complete");
    expect(completionExamples([first, snapshot(2, next.selected!, 6)], new Set([0, 1]))[0].outcome).toBe("abandon");
  });
  it("does not call a wrong-location prefix completed merely because the cursor advances", () => {
    const rows = [snapshot(1, p), snapshot(2, { ...p, word: 1, progress: 1 })];
    expect(completionExamples(rows, new Set([1]))[0].outcome).toBe("abandon");
  });
  it("retires an earlier generation before a new local search can appear to complete it", () => {
    const rows = [snapshot(3, p), snapshot(1, { ...p, word: 1, progress: 1 })];
    expect(completionExamples(rows, new Set([0, 1]))[0].outcome).toBe("abandon");
  });
});

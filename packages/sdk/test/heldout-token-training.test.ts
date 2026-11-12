import { describe, expect, it } from "vitest";
import { tokenSpeakerFolds, tokenTrainingLabels } from "../src/heldout-token-training.js";
import type { TokenTrainingLabel } from "../src/heldout-token-training.js";

const label = (id: string, extra: Partial<TokenTrainingLabel> = {}): TokenTrainingLabel => ({
  id, speaker: id, split: "train", kind: "none", scored: true, ayahs: [], wav: "", events: [], non_errors: [], ...extra,
});

describe("training-speaker isolation", () => {
  it("excludes mistakes, repeats, unscored clips and every speaker with a non-train row", () => {
    const rows = [label("clean"), label("mistake", { kind: "omission" }), label("event", { events: [{}] }),
      label("repeat", { non_errors: [{}] }), label("unscored", { scored: false }), label("excluded", { excluded: "review" }),
      label("mixed"), label("held-out", { speaker: "mixed", split: "eval" })];
    expect(tokenTrainingLabels(rows).map(l => l.id)).toEqual(["clean"]);
  });

  it("keeps a speaker's recordings together independently of label ordering", () => {
    const rows = [label("a"), label("b"), label("a2", { speaker: "a" }), label("c")];
    expect([...tokenSpeakerFolds(rows)]).toEqual([...tokenSpeakerFolds([...rows].reverse())]);
    expect(tokenSpeakerFolds(rows).size).toBe(3);
    expect(() => tokenSpeakerFolds([label("a")])).toThrow();
  });
});

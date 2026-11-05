import { describe, it, expect } from "vitest";
import { correctSlot, eligibleLearner, minedAccusations } from "../src/training/accusations.js";
import type { LearnerLabel } from "../src/training/accusations.js";
import type { CheckerEvent } from "@tilawa/core";

const w = (word: number) => ({ surah: 1, ayah: 2, word });
const label = (): LearnerLabel => ({ id: "training", speaker: "learner", split: "train", device: "phone", scored: true,
  kind: "substitution", wav: "wav/training.wav", ayahs: ["1:2"], events: [{ errorClass: "wrong_word", state: "confirmed", location: w(2), observed: "بَاب" }], non_errors: [] });
const check = (id: string, word: number, state: CheckerEvent["state"]): CheckerEvent => ({ type: "checker", v: 1, id, timeMs: 4000,
  state, errorClass: "wrong_word", location: w(word), confidence: 0.9 });
describe("training accusations", () => {
  it("requires training phones and complete labels", () => {
    expect(eligibleLearner(label())).toBe(true);
    for (const patch of [{ split: "eval" }, { scored: false }, { excluded: "audit" }, { unsupported: ["tajweed"] }])
      expect(eligibleLearner({ ...label(), ...patch })).toBe(false);
    expect(eligibleLearner({ ...label(), device: "laptop", events: [] })).toBe(false);
    expect(eligibleLearner({ ...label(), device: "laptop" })).toBe(true);
    expect(minedAccusations({ ...label(), device: "laptop" }, [check("false", 1, "confirmed")])).toEqual([]);
    const l = label(); delete l.events[0].observed;
    expect(eligibleLearner(l)).toBe(false);
  });
  it("mines final confirmed false accusations across all confirmation paths", () => {
    const events = [check("false", 1, "suspected"), check("real", 2, "confirmed"), check("resolved", 3, "confirmed"),
      check("false", 1, "confirmed"), check("resolved", 3, "resolved"), check("uncertain", 4, "uncertain")];
    expect(minedAccusations(label(), events).map(e => e.id)).toEqual(["false"]);
  });
  it("does not call skipped words, repeated slots or a different ayah correct", () => {
    const l = label(); l.events.push({ errorClass: "skipped_passage", state: "suspected", location: w(3), words: 3 });
    l.non_errors.push({ kind: "repeat", location: w(6) });
    expect([1, 2, 3, 4, 5, 6, 7].map(i => correctSlot(l, w(i)))).toEqual([true, false, false, false, false, false, true]);
    expect(correctSlot(l, { ...w(1), ayah: 3 })).toBe(false);
  });
});

import { describe, expect, it } from "vitest";
import { loadReferenceIndex } from "@tilawa/core";
import { spokenPhoneRecording, spokenPhoneTrainingLabels } from "../src/spoken-phone-training.js";
import type { SpokenPhoneLabel } from "../src/spoken-phone-training.js";

const label = (overrides: Partial<SpokenPhoneLabel> = {}): SpokenPhoneLabel => ({ id: "clip", speaker: "learner", split: "train", kind: "none", scored: true,
  ayahs: ["1:1"], wav: "wav/clip.wav", events: [], non_errors: [], ...overrides });
const site = { surah: 1, ayah: 1, word: 2 };
describe("spoken-phone training isolation", () => {
  it("excludes held-out and mixed-split speakers, lexical mistakes, repeats and extra mistakes", () => {
    const labels = [label(), label({ id: "vowel", kind: "vowel", events: [{ errorClass: "short_vowel", location: site }] }),
      label({ speaker: "mixed" }), label({ speaker: "mixed", split: "eval" }), label({ speaker: "held", split: "eval" }),
      label({ kind: "substitution", events: [{ errorClass: "wrong_word", location: site }] }), label({ scored: false }),
      label({ excluded: true }), label({ non_errors: [{ kind: "repeat" }] }), label({ kind: "tajweed", events: [{ errorClass: "omitted_word", location: site }] })];
    expect(spokenPhoneTrainingLabels(labels).map(l => l.id)).toEqual(["clip", "vowel"]);
  });

  it("weakly supervises the labelled pronunciation within its original word", () => {
    const index = loadReferenceIndex({ version: "test", text: "test", profile: {}, inventory: { id: "test", tokens: "بَُمن" },
      ayahs: [{ s: 1, a: 1, w: ["first", "second"], p: "بَمنُ", o: [0, 2], iso: ["بَ", "منُ"] }] });
    const vowel = spokenPhoneRecording(index, label({ kind: "vowel", events: [{ errorClass: "short_vowel", location: site }] }), "بَمنَ");
    expect(vowel.written).toBe("بَمنُ");
    expect(vowel.variable).toEqual([false, false, false, false, true]);
    const tajweed = spokenPhoneRecording(index, label({ kind: "tajweed", unsupported: [{ location: site }] }), "بَمُ");
    expect(tajweed.variable).toEqual([false, false, true, true, true]);
    expect(spokenPhoneRecording(index, label(), "بَمنُ").variable.some(Boolean)).toBe(false);
    expect(() => spokenPhoneRecording(index, label({ kind: "vowel", events: [{ errorClass: "short_vowel", location: { ...site, word: 9 } }] }), "بَ")).toThrow();
  });
});

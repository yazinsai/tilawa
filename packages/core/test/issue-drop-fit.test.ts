import { describe, expect, it } from "vitest";
import { issueDropFit } from "../src/tracker.js";

describe("issue drop fit", () => {
  const consonants = "بتثجحخدذرس";
  it("keeps an edit whose continuation fits and beats the commit margin", () => {
    const fit = issueDropFit("صصص" + consonants, consonants, consonants);
    expect(fit).toMatchObject({ n: consonants.length, edited: 0, explains: true });
    expect(fit!.gain).toBeGreaterThanOrEqual(fit!.margin);
  });

  it("drops an edit that does not clear the commit margin", () => {
    // Two phones: the margin is 15 edits, more than deleting a three-consonant word.
    const short = "بت";
    const fit = issueDropFit("صصص" + short, short, short)!;
    expect(fit).toMatchObject({ n: 2, explains: false });
    expect(fit.gain).toBeLessThan(fit.margin);
  });

  it("drops an edit that leaves the continuation a misfit", () => {
    const fit = issueDropFit("صصص" + consonants, "غغغغغغغغغغ", consonants);
    expect(fit!.explains).toBe(false);
    expect(fit!.edited).toBeGreaterThan(0.35);
  });

  it("has nothing to score without heard phonemes", () => {
    expect(issueDropFit("بتث", "بتث", "")).toBeNull();
  });
});

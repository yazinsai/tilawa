import { describe, expect, it } from "vitest";
import { insertionDropFit, issueDropFit, relocationSupport } from "../src/tracker.js";

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

describe("insertion drop fit", () => {
  const tail = "بتثجحخدذ";
  const inserted = "رزسشصضطظع";

  it("keeps an added word when deleting its heard phones restores the anchor", () => {
    const fit = insertionDropFit(tail, inserted + tail, tail);
    expect(fit).toMatchObject({ n: inserted.length + tail.length, edited: 0, explains: true });
    expect(fit!.gain).toBeGreaterThanOrEqual(fit!.margin);
  });

  it("uses the full span's margin, not the tail's", () => {
    const fit = insertionDropFit(tail, inserted + tail, tail)!;
    expect(fit.n).toBe(inserted.length + tail.length);
    expect(fit.margin).toBeLessThan(30 / tail.length);
  });

  it("drops an edit that leaves the continuation a misfit", () => {
    const fit = insertionDropFit(tail, inserted + "غغغغغغغغغغ", "غغغغغغغغغغ");
    expect(fit!.explains).toBe(false);
    expect(fit!.edited).toBeGreaterThan(0.35);
  });

  it("drops an edit that does not clear the commit margin", () => {
    const fit = insertionDropFit("ب", "تب", "ب")!;
    expect(fit.n).toBe(2);
    expect(fit.explains).toBe(false);
    expect(fit.gain).toBeLessThan(fit.margin);
  });

  it("has nothing to score without the kept phones", () => {
    expect(insertionDropFit(tail, inserted, "")).toBeNull();
  });
});

describe("relocation support", () => {
  const shared = "ءننننهمم";
  const unique = "وقفوهممم";
  const anchor = [shared, "ءلفوواا"];

  it("does not count a heard word the anchor explains once one word is inserted", () => {
    const fit = relocationSupport(anchor, [
      { destination: unique, heard: unique },
      { destination: shared, heard: shared },
    ]);
    expect(fit).toMatchObject({ count: 1 });
  });

  it("counts a short word the destination reads exactly", () => {
    const fit = relocationSupport(["غغغغ", "صصصص"], [
      { destination: "بتثج", heard: "بتثج" },
      { destination: "خدذر", heard: "خدذر" },
    ]);
    expect(fit!.count).toBe(2);
  });

  it("does not count a short imperfect reading that misses the wide margin", () => {
    const fit = relocationSupport(["غغغغغغ", "عععععع"], [
      { destination: "رزس", heard: "رزش" },
      { destination: "صصصصصص", heard: "صصصصصص" },
    ]);
    // One substitution, three phones: destination cost is not zero and the gain cannot clear a margin of 10.
    expect(fit!.count).toBe(1);
  });

  it("counts several words that fit only the destination", () => {
    const fit = relocationSupport(["غغغغغغغغ", "صصصصصصصص"], [
      { destination: "بتثجحخدذ", heard: "بتثجحخدذ" },
      { destination: "رزسشصضطظ", heard: "رزسشصضطظ" },
    ]);
    expect(fit!.count).toBe(2);
  });

  it("has nothing to score without heard words", () => {
    expect(relocationSupport(anchor, [])).toBeNull();
  });
});

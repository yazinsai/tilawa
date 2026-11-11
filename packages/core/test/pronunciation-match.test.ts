import { describe, expect, it } from "vitest";
import { completePronunciationMatch as matches } from "../src/pronunciation-match.js";

describe("complete recheck pronunciation", () => {
  it("retains pause, held-vowel and direct soft-substitution rules", () => {
    expect(matches("ذَات", "ۜزَااااتۦ")).toBe(true);
    expect(matches("نَام", "نَام")).toBe(true);
  });
  it("requires every phone and rejects surplus, a nasal swap and a shortened geminate", () => {
    expect(matches("ذَات", "َات")).toBe(false);
    expect(matches("ذَات", "ذَاتف")).toBe(false);
    expect(matches("نَام", "مَام")).toBe(false);
    expect(matches("ننَ", "نَ")).toBe(false);
    expect(matches("", "")).toBe(false);
  });
});

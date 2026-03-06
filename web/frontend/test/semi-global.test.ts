import { describe, it, expect } from "vitest";
import { semiGlobalDistance, fragmentScore } from "../src/lib/levenshtein";

describe("semiGlobalDistance", () => {
  it("returns 0 for exact substring match", () => {
    expect(semiGlobalDistance("bcd", "abcde")).toBe(0);
  });

  it("returns 0 for prefix match", () => {
    expect(semiGlobalDistance("abc", "abcde")).toBe(0);
  });

  it("returns 0 for suffix match", () => {
    expect(semiGlobalDistance("cde", "abcde")).toBe(0);
  });

  it("returns edit distance for 1-char substitution in substring", () => {
    expect(semiGlobalDistance("bxd", "abcde")).toBe(1);
  });

  it("returns query length when ref is empty", () => {
    expect(semiGlobalDistance("abc", "")).toBe(3);
  });

  it("returns 0 when query is empty", () => {
    expect(semiGlobalDistance("", "abcde")).toBe(0);
  });

  it("returns full distance when query is longer than ref", () => {
    expect(semiGlobalDistance("abcdef", "bcd")).toBe(3);
  });

  it("handles identical strings", () => {
    expect(semiGlobalDistance("abc", "abc")).toBe(0);
  });

  it("handles realistic phoneme fragment", () => {
    const transcript = "bismi allahi";
    const verse = "bismi allahi arraHmaani arraHiimi";
    expect(semiGlobalDistance(transcript, verse)).toBe(0);
  });

  it("handles no-space matching", () => {
    const transcript = "bismiallahi";
    const verse = "bismiallahiarraHmaaniarraHiimi";
    expect(semiGlobalDistance(transcript, verse)).toBe(0);
  });
});

describe("fragmentScore", () => {
  it("returns 1.0 for exact substring", () => {
    expect(fragmentScore("bcd", "abcde")).toBe(1.0);
  });

  it("returns high score for near-match substring", () => {
    expect(fragmentScore("bxd", "abcde")).toBeCloseTo(0.667, 2);
  });

  it("returns 0 for completely different strings", () => {
    const score = fragmentScore("zzzzz", "abcde");
    expect(score).toBeLessThan(0.2);
  });

  it("returns 1.0 for identical strings", () => {
    expect(fragmentScore("abc", "abc")).toBe(1.0);
  });

  it("returns 1.0 for empty query", () => {
    expect(fragmentScore("", "abc")).toBe(1.0);
  });

  it("penalizes when query is longer than ref", () => {
    expect(fragmentScore("abcdef", "bcd")).toBeCloseTo(0.5, 2);
  });

  it("handles realistic partial transcript vs full verse", () => {
    const transcript = "bismiallahi";
    const verse = "bismiallahiarraHmaaniarraHiimi";
    expect(fragmentScore(transcript, verse)).toBe(1.0);
  });
});

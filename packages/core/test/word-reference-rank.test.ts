import { describe, expect, it } from "vitest";
import { candidateRanksFirst, referencePrefixEnds, subsequenceDtw } from "../src/word-reference-rank.js";

describe("posteriorgram word rank", () => {
  it("allows free reference edges while requiring the entire query", () => {
    expect(subsequenceDtw([[1, 0], [0, 1]], [[0, 1], [1, 0], [0, 1], [1, 0]])).toBe(0);
    expect(subsequenceDtw([[1, 0], [0, 1]], [[1, 0]])).toBe(Infinity);
  });
  it("warps held sounds and retains blank frames", () => {
    expect(subsequenceDtw([[1, 0], [1, 0], [0, 1]], [[1, 0], [0, 1]])).toBe(0);
    expect(subsequenceDtw([[0.5, 0.5]], [[0.5, 0.5]])).toBeCloseTo(Math.log(2));
  });
  it("rejects absent evidence, ties and incomplete competitor coverage", () => {
    expect(subsequenceDtw([], [[1]])).toBe(Infinity);
    expect(subsequenceDtw([[1]], [[1, 0]])).toBe(Infinity);
    expect(candidateRanksFirst([0.2], [[0.3], [0.4]])).toBe(true);
    expect(candidateRanksFirst([0.2], [[0.2]])).toBe(false);
    expect(candidateRanksFirst([0.2], [[]])).toBe(false);
    expect(candidateRanksFirst([Infinity], [[Infinity]])).toBe(false);
    expect(candidateRanksFirst([0.2], [[Infinity]])).toBe(false);
  });
  it("cuts a reference prefix before the next label, retaining intervening blanks", () => {
    expect(referencePrefixEnds([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]], [1, 2])).toEqual([3]);
    expect(referencePrefixEnds([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], [1, 1])).toEqual([3]);
    expect(referencePrefixEnds([[1, 0, 0]], [1, 2])).toEqual([]);
  });
});

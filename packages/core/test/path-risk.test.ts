import { describe, expect, it } from "vitest";
import { fitPathChannel, pathChannelLogs, positionSequenceDistance, positionSequenceMedoid } from "../src/path-risk.js";
import { createLocalSearch } from "../src/local.js";
import { loadReferenceIndex } from "../src/reference.js";

describe("position sequence risk", () => {
  it("selects a representative retained route rather than the most probable outlier", () => {
    const paths = [
      { words: [1, 2, 20, 21], logProbability: Math.log(0.4) },
      { words: [1, 2, 3, 4], logProbability: Math.log(0.3) },
      { words: [1, 2, 3, 5], logProbability: Math.log(0.3) },
    ];
    const result = positionSequenceMedoid(paths)!;
    expect(result.representative).toBe(1);
    expect(result.risk).toBeCloseTo(1.1);
    expect(result.trajectories[0].risk).toBeCloseTo(1.2);
  });

  it("merges identical complete routes before calculating risk", () => {
    const paths = [
      { words: [1, 2], logProbability: Math.log(0.25) },
      { words: [1, 2], logProbability: Math.log(0.35) },
      { words: [1, 9], logProbability: Math.log(0.4) },
    ];
    const result = positionSequenceMedoid(paths)!;
    expect(result.trajectories).toHaveLength(2);
    expect(result.trajectories[0].probability).toBeCloseTo(0.6);
    expect(result.representative).toBe(1);
    expect(result.risk).toBeCloseTo(0.4);
  });

  it("retains repeat occurrences rather than comparing sets of visited positions", () => {
    expect(positionSequenceDistance([1, 2, 2, 3], [1, 2, 3])).toBe(1);
    expect(positionSequenceDistance([1, 2, 1, 2, 3], [1, 2, 3])).toBe(2);
    const result = positionSequenceMedoid([
      { words: [1, 2, 2, 3], logProbability: Math.log(0.8) },
      { words: [1, 2, 3], logProbability: Math.log(0.2) },
    ])!;
    expect(result.trajectories).toHaveLength(2);
    expect(result.representative).toBe(0);
  });

  it("keeps a concentrated genuine jump and is stable to common log scaling", () => {
    const paths = [
      { words: [1, 2, 20, 21], logProbability: Math.log(0.98) - 10000 },
      { words: [1, 2, 3, 4], logProbability: Math.log(0.01) - 10000 },
      { words: [1, 2, 3, 5], logProbability: Math.log(0.01) - 10000 },
    ];
    const result = positionSequenceMedoid(paths)!;
    expect(result.representative).toBe(0);
    expect(result.trajectories.reduce((sum, t) => sum + t.probability, 0)).toBeCloseTo(1);
  });
});

describe("training token channel", () => {
  it("normalizes operations and conditional emission probabilities", () => {
    const channel = fitPathChannel([{ written: "abc", heard: "adc" }, { written: "abc", heard: "ab" }], ["a", "b", "c", "d"]);
    expect(channel.operations.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    for (const row of [...channel.identity, channel.insertion]) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    const logs = pathChannelLogs(channel);
    expect(logs.match("a", "a")).toBeGreaterThan(logs.match("a", "d"));
    expect(logs.insert("a")).toBeLessThan(0);
    expect(logs.deletion).toBeLessThan(0);
  });

  it("rejects empty training and phones outside the supplied inventory", () => {
    expect(() => fitPathChannel([], ["a"])).toThrow();
    expect(() => fitPathChannel([{ written: "a", heard: "b" }], ["a"])).toThrow();
  });
});

describe("complete retained local trajectories", () => {
  const index = loadReferenceIndex({ version: "tiny", text: "tiny", profile: {}, inventory: { id: "tiny", tokens: "abc" },
    ayahs: [{ s: 1, a: 1, w: ["a", "b", "c"], p: "abc", o: [0, 1, 2], iso: ["a", "b", "c"] }] });
  const channel = fitPathChannel([{ written: "abc", heard: "abc" }], [...index.tokens]);

  it("retains acoustic-cost-dominated histories at the same endpoint and explicit repeats", () => {
    const local = createLocalSearch(index, { offset: 1, origin: 0, edits: 0 }, channel);
    local.step("b");
    const paths = local.trajectories().filter(p => p.offset === 2);
    expect(paths.some(p => JSON.stringify(p.words) === "[1]")).toBe(true);
    expect(paths.some(p => JSON.stringify(p.words) === "[0]")).toBe(true);
    expect(paths.some(p => p.words.length === 0)).toBe(true);
    local.step("a");
    expect(local.trajectories().some(p => JSON.stringify(p.words) === "[0,0]")).toBe(true);
  });

  it("restores the whole common interval without mutating saved histories", () => {
    const local = createLocalSearch(index, { offset: 1, origin: 0, edits: 0 }, channel);
    local.step("b");
    const before = local.trajectories();
    const saved = local.save();
    local.step("a");
    const after = local.trajectories();
    local.restore(saved);
    expect(local.trajectories()).toEqual(before);
    local.step("a");
    expect(local.trajectories()).toEqual(after);
  });

  it("leaves the acoustic search identical with no fitted channel", () => {
    const plain = createLocalSearch(index, { offset: 1, origin: 0, edits: 0 });
    const inactive = createLocalSearch(index, { offset: 1, origin: 0, edits: 0 }, null);
    for (const p of "bacb") {
      plain.step(p);
      inactive.step(p);
      expect(inactive.position()).toEqual(plain.position());
      expect(inactive.trajectories()).toEqual([]);
    }
  });
});

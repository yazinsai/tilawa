import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "../src/ctc.js";
import { acousticPrefixes } from "../src/ctc-prefix.js";

const frames = (rows: number[][], startFrame = 0) => {
  const f = new PosteriorFrames(rows[0].length - 1);
  f.append({ startFrame, nFrames: rows.length, k: rows[0].length,
    ids: rows.flatMap((r) => r.map((_, i) => i)), probs: rows.flat() });
  return f;
};

describe("acoustic prefix boundaries", () => {
  it("sums each raw CTC path exactly once, including distinct boundaries for the same string", () => {
    const rows = [[0.3, 0.4, 0.3], [0.2, 0.5, 0.3], [0.4, 0.2, 0.4]];
    const sums = new Map<string, number>();
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) for (let c = 0; c < 3; c++) {
      let text = "", previous = -1;
      for (const id of [a, b, c]) {
        if (id && id !== previous) text += ["", "ن", "ر"][id];
        previous = id;
      }
      sums.set(text, (sums.get(text) ?? 0) + rows[0][a] * rows[1][b] * rows[2][c]);
    }
    const prefixes = acousticPrefixes(frames(rows), ["ن", "ر"], 0, 3, 64);
    for (const p of prefixes) expect(Math.exp(p.logProbability)).toBeCloseTo(sums.get(p.phones) ?? 0, 12);
    expect(prefixes.filter((p) => Number.isFinite(p.logProbability))).toHaveLength(sums.size);
    expect(prefixes.reduce((s, p) => s + Math.exp(p.logProbability), 0)).toBeCloseTo(1, 12);
    expect(new Set(prefixes.map((p) => p.phones)).size).toBe(prefixes.length);
  });

  it("retains a blank-separated repeat even when greedy decoding gives one held run", () => {
    const prefixes = acousticPrefixes(frames([[0.1, 0.9], [0.49, 0.51], [0.1, 0.9]], 10), ["ن"], 10, 13);
    const repeated = prefixes.find((p) => p.phones === "نن")!;
    expect(Math.exp(repeated.logProbability)).toBeCloseTo(0.9 * 0.49 * 0.9, 12);
    expect(repeated.runs).toEqual([{ p: "ن", start: 10, end: 10 }, { p: "ن", start: 12, end: 12 }]);
    expect(prefixes.some((p) => p.phones === "ن")).toBe(true);
  });

  it("restores the exact mass of retained strings after prefix pruning", () => {
    const f = frames([[0.4, 0.6], [0.6, 0.4], [0.4, 0.6]]);
    const retained = acousticPrefixes(f, ["ن"], 0, 3, 1);
    expect(retained).toHaveLength(1);
    expect(retained[0].logProbability).toBeCloseTo(-f.nll([...retained[0].phones].map(() => 1), 0, 3), 12);
  });

  it("creates no phones absent from the listed acoustic support", () => {
    const f = new PosteriorFrames(2);
    f.append({ startFrame: 0, nFrames: 2, k: 2, ids: [1, 0, 1, 0], probs: [0.6, 0.3, 0.6, 0.3] });
    expect(acousticPrefixes(f, ["ن", "ر"], 0, 2).every((p) => !p.phones.includes("ر"))).toBe(true);
  });
});

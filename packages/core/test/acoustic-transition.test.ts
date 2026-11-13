import { describe, expect, it } from "vitest";
import { acousticPathSuffix, appendAcousticPath, acousticRows, TransitionFrames, createAcousticTransitionScorer } from "../src/acoustic-transition.js";
import type { AcousticTransitionModel } from "../src/acoustic-transition.js";
import { PosteriorFrames } from "../src/ctc.js";
import { trackerInput } from "../src/tracker.js";

function model(): AcousticTransitionModel {
  return { version: 1, inventory: "toy", tokens: "ab", frames: 64, mean: [0, 0, 0], scale: [1, 1, 1],
    acoustic: [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
    acousticBias: [0, 0], text: [[[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 0, 1]]],
    textBias: [0, 0], joint: [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], jointBias: [0, 0],
    output: [1, 1], positionWeight: 20, temperature: 1 };
}

describe("acoustic pronunciation comparison", () => {
  it("preserves posterior rows for local use and keeps startup decoded-only", () => {
    const p = { k: 3, startFrame: 0, nFrames: 1, ids: [0, 1, 2], probs: [0.6, 0.3, 0.1] };
    const features = { version: 1 as const, width: 80, startFrame: 0, nFrames: 1, values: new Array(80).fill(0) };
    const e = { inventory: "toy", frameMs: 40, phonemes: [], posteriors: p, acousticTransitions: features };
    const input = trackerInput(e);
    expect(input.posteriors).toBeUndefined();
    expect(input.acousticTransitions).toBe(features);
    expect(input.localPosteriors).toBe(p);
    expect(trackerInput(input).localPosteriors).toBe(p);
    expect(e.posteriors).toBe(p);
  });

  it("distinguishes sub-blank distributions with identical greedy observations", () => {
    const score = createAcousticTransitionScorer(model(), "toy");
    expect(score([[0.6, 0.3, 0.1]], "a")!).toBeLessThan(score([[0.6, 0.3, 0.1]], "b")!);
    expect(score([[0.6, 0.1, 0.3]], "b")!).toBeLessThan(score([[0.6, 0.1, 0.3]], "a")!);
  });

  it("uses temporal order rather than pooled class counts", () => {
    const score = createAcousticTransitionScorer(model(), "toy");
    const rows = [[0.6, 0.35, 0.05], [0.6, 0.05, 0.35]];
    expect(score(rows, "ab")!).toBeLessThan(score(rows, "ba")!);
    expect(score([...rows].reverse(), "ba")!).toBeLessThan(score([...rows].reverse(), "ab")!);
  });

  it("averages complete posterior distributions while retaining intervening blank frames", () => {
    const frames = new PosteriorFrames(2);
    frames.append({ k: 3, startFrame: 5, nFrames: 4, ids: [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
      probs: [0.6, 0.3, 0.1, 0.8, 0.1, 0.1, 0.6, 0.1, 0.3, 0.8, 0.1, 0.1] });
    const rows = acousticRows(frames, 5, 9, 2)!;
    expect(rows[0][0]).toBeCloseTo(0.7);
    expect(rows[0][1]).toBeCloseTo(0.2);
    expect(rows[1][2]).toBeCloseTo(0.2);
    expect(acousticRows(frames, 4, 9)).toBeNull();
    expect(acousticRows(frames, 5, 10)).toBeNull();
  });

  it("retains movement costs and only removes acoustic charges for the common suffix", () => {
    const first = { previous: null, reference: "a", acousticCost: 0, consumes: true };
    const move = { previous: first, reference: "", acousticCost: 0, consumes: false };
    const deletion = { previous: move, reference: "b", acousticCost: 5, consumes: false };
    const second = { previous: deletion, reference: "a", acousticCost: 10, consumes: true };
    expect(acousticPathSuffix(second)).toEqual({ pronunciation: "aba", acousticCost: 15, observations: 2 });
    expect(acousticPathSuffix(second, 1)).toEqual({ pronunciation: "a", acousticCost: 10, observations: 1 });
  });

  it("bounds retained history without changing the scored observation suffix", () => {
    let path = null as import("../src/acoustic-transition.js").AcousticPath | null;
    for (let i = 0; i < 100; i++) path = appendAcousticPath(path, "a", 2, true);
    expect(path!.retained).toBeLessThanOrEqual(32);
    expect(acousticPathSuffix(path)).toEqual({ pronunciation: "a".repeat(16), acousticCost: 32, observations: 16 });
  });

  it("rejects incompatible models and abstains on unavailable pronunciation symbols", () => {
    expect(() => createAcousticTransitionScorer(model(), "other")).toThrow();
    expect(() => createAcousticTransitionScorer({ ...model(), temperature: 0 }, "toy")).toThrow();
    expect(createAcousticTransitionScorer(model(), "toy")([[0.6, 0.3, 0.1]], "z")).toBeNull();
  });
});

it("retains exactly the original acoustic interval through append and retirement", () => {
  const f = new TransitionFrames();
  f.append({ version: 1, width: 80, startFrame: 5, nFrames: 2, values: [...new Array(80).fill(1), ...new Array(80).fill(3)] });
  expect(f.rows(5, 7, 1)).toEqual([new Array(80).fill(2)]);
  f.dropBefore(6);
  expect(f.rows(5, 7)).toBeNull();
  expect(f.rows(6, 7)).toEqual([new Array(80).fill(3)]);
  expect(() => f.append({ version: 1, width: 80, startFrame: 8, nFrames: 1, values: new Array(80).fill(0) })).toThrow();
});

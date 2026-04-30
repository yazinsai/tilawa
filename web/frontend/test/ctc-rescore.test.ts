import { describe, expect, it } from "vitest";
import {
  alignCtcSequence,
  scoreCtcSequence,
  type AcousticEvidence,
} from "../src/lib/ctc-rescore";

const BLANK = 0;
const VOCAB = 4;

function evidenceFor(bestIds: number[]): AcousticEvidence {
  const logprobs = new Float32Array(bestIds.length * VOCAB);
  for (let t = 0; t < bestIds.length; t++) {
    for (let id = 0; id < VOCAB; id++) {
      logprobs[t * VOCAB + id] = id === bestIds[t] ? Math.log(0.97) : Math.log(0.01);
    }
  }
  return {
    logprobs,
    timeSteps: bestIds.length,
    vocabSize: VOCAB,
    blankId: BLANK,
  };
}

describe("alignCtcSequence", () => {
  it("aligns an exact target to the expected frame range", () => {
    const evidence = evidenceFor([BLANK, 1, 2, BLANK]);
    const alignment = alignCtcSequence(evidence, [1, 2]);

    expect(alignment.feasible).toBe(true);
    expect(alignment.startFrame).toBe(1);
    expect(alignment.endFrame).toBe(2);
    expect(alignment.tokenFrames).toEqual([1, 2]);
  });

  it("does not let leading or trailing blanks steal token span", () => {
    const evidence = evidenceFor([BLANK, BLANK, 1, 2, BLANK, BLANK]);
    const alignment = alignCtcSequence(evidence, [1, 2]);

    expect(alignment.feasible).toBe(true);
    expect(alignment.startFrame).toBe(2);
    expect(alignment.endFrame).toBe(3);
    expect(alignment.tokenFrames).toEqual([2, 3]);
  });

  it("returns infeasible when repeated target lacks a separating blank frame", () => {
    const evidence = evidenceFor([1, 1]);
    const alignment = alignCtcSequence(evidence, [1, 1]);

    expect(alignment.feasible).toBe(false);
    expect(alignment.startFrame).toBe(-1);
    expect(alignment.endFrame).toBe(-1);
    expect(alignment.tokenFrames).toEqual([]);
  });

  it("handles repeated tokens with an intervening blank", () => {
    const evidence = evidenceFor([BLANK, 1, BLANK, 1, BLANK]);
    const alignment = alignCtcSequence(evidence, [1, 1]);

    expect(alignment.feasible).toBe(true);
    expect(alignment.startFrame).toBe(1);
    expect(alignment.endFrame).toBe(3);
    expect(alignment.tokenFrames).toEqual([1, 3]);
  });

  it("returns a finite Viterbi score consistent with CTC scoring", () => {
    const evidence = evidenceFor([BLANK, 1, 2, BLANK]);
    const alignment = alignCtcSequence(evidence, [1, 2]);
    const forwardScore = scoreCtcSequence(evidence, [1, 2]);

    expect(alignment.feasible).toBe(true);
    expect(Number.isFinite(alignment.score)).toBe(true);
    expect(alignment.score).toBeGreaterThanOrEqual(0);
    expect(alignment.score).toBeGreaterThanOrEqual(forwardScore);
  });
});

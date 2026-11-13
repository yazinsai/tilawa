import { expect, it } from "vitest";
import { summarizeTransition, carryAcousticTransitions } from "../src/acoustic-transition-features.js";
import { trackerInput } from "@tilawa/core";

it("retains the direction of a transition even when the average spectrum is identical", () => {
  const frames = [new Float32Array(80).fill(1), new Float32Array(80).fill(3)];
  const forward = summarizeTransition(frames, 86);
  const backward = summarizeTransition([...frames].reverse(), 86);
  expect(forward.slice(0, 40)).toEqual(backward.slice(0, 40));
  expect(forward.slice(40, 80)).toEqual(new Array(40).fill(2));
  expect(backward.slice(40, 80)).toEqual(new Array(40).fill(-2));
  expect(forward.slice(80)).toEqual(new Array(6).fill(0));
});

it("rejects incomplete feature shapes instead of fabricating acoustic detail", () => {
  expect(() => summarizeTransition([], 86)).toThrow();
  expect(() => summarizeTransition([new Float32Array(79)], 86)).toThrow();
});

it("preserves original rows through token-only serialization, including blank intervals", () => {
  const features = { version: 1 as const, width: 86, startFrame: 0, nFrames: 5,
    values: Array.from({ length: 5 * 86 }, (_, i) => i) };
  const phones = [{ p: "a", start: 1, end: 1, conf: 0.8 }, { p: "b", start: 4, end: 4, conf: 0.8 }];
  const packed = carryAcousticTransitions([{ inventory: "toy", frameMs: 40, phonemes: phones }], [features]);
  const serialized = JSON.parse(JSON.stringify(packed[0].phonemes));
  const input = trackerInput({ inventory: "toy", frameMs: 40, phonemes: serialized });
  expect(input.acousticTransitionSegments!.flatMap(f => Array.from(f.values))).toEqual(features.values);
  expect(input.acousticTransitionSegments!.map(f => [f.startFrame, f.nFrames])).toEqual([[0, 2], [2, 3]]);
  expect(phones[0]).not.toHaveProperty("acousticTransitions");
});

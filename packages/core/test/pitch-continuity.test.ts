import { describe, expect, it } from "vitest";
import { pitchContinuity, pitchFits } from "../src/pitch-continuity.js";
const phones = (pitch: number[]) => [{ p: "م", start: 0, end: pitch.length - 1, conf: 1, pitch }];
describe("session-relative pitch continuity", () => {
  it("measures pitch reset across an internal unvoiced gap, excluding edge silence", () => {
    expect(pitchContinuity(phones([0, 100, 0, 0, 200, 0]), 0, 6)).toEqual({ jump: 1, gap: 2 });
  });
  it("accepts equality and rejects either kind of discontinuity", () => {
    const envelope = { jump: 0.5, gap: 2 };
    expect(pitchFits({ ...envelope }, envelope)).toBe(true);
    expect(pitchFits({ jump: 0.6, gap: 0 }, envelope)).toBe(false);
    expect(pitchFits({ jump: 0, gap: 3 }, envelope)).toBe(false);
  });
  it("abstains without a voiced span, full coverage, or a session envelope", () => {
    expect(pitchContinuity(phones([0, 0]), 0, 2)).toBeNull();
    expect(pitchContinuity(phones([100]), 0, 2)).toBeNull();
    expect(pitchFits({ jump: 0, gap: 0 }, null)).toBe(false);
  });
});

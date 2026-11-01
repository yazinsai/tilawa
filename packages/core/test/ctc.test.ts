import { describe, expect, it } from "vitest";
import { PosteriorFrames, ctcForward } from "../src/ctc.js";

describe("ctc forward", () => {
  it("scores one frame as the label's negative log-probability", () => {
    const row = new Float64Array([Math.log(0.2), Math.log(0.8)]);
    expect(ctcForward(() => row, [1], 0, 1)).toBeCloseTo(-Math.log(0.8), 10);
    expect(ctcForward(() => row, [], 0, 1)).toBeCloseTo(-Math.log(0.2), 10);
  });

  it("puts a blank between a repeated label", () => {
    // Frame 0 is token 1, frame 1 is blank, frame 2 is token 1. One label cannot use both spikes; two labels can.
    const rows = [
      new Float64Array([Math.log(0.1), Math.log(0.9)]),
      new Float64Array([Math.log(0.8), Math.log(0.2)]),
      new Float64Array([Math.log(0.1), Math.log(0.9)]),
    ];
    const one = ctcForward((f) => rows[f], [1], 0, 3);
    const two = ctcForward((f) => rows[f], [1, 1], 0, 3);
    expect(two).toBeLessThan(one);
  });

  it("matches the top-k posterior forward", () => {
    const frames = new PosteriorFrames(1);
    frames.append({ k: 2, startFrame: 0, nFrames: 2, ids: [1, 0, 0, 1], probs: [0.7, 0.3, 0.6, 0.4] });
    const logs = [frames.row(0), frames.row(1)];
    expect(ctcForward((f) => logs[f], [1], 0, 2)).toBeCloseTo(frames.nll([1], 0, 2), 10);
  });
});

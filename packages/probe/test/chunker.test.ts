import { describe, expect, it } from "vitest";
import { FrameChunker } from "../src/chunker.js";

const frames = (from: number, n: number) => Array.from({ length: n }, (_, i) => Float32Array.of(from + i));

describe("FrameChunker", () => {
  it("emits overlapping windows of `size` frames advancing by `shift`", () => {
    const c = new FrameChunker(5, 3);
    expect(c.push(frames(0, 4))).toEqual([]);
    const out = c.push(frames(4, 7));
    expect(out.map((w) => w.map((f) => f[0]))).toEqual([
      [0, 1, 2, 3, 4],
      [3, 4, 5, 6, 7],
      [6, 7, 8, 9, 10],
    ]);
  });
  it("pads the tail on flush so the last frames still reach the encoder", () => {
    const c = new FrameChunker(4, 2);
    c.push(frames(0, 5));
    const tail = c.flush();
    expect(tail.map((w) => w.map((f) => f[0]))).toEqual([[2, 3, 4, 0]]);
  });
});

describe("FrameChunker.flush", () => {
  it("emits nothing when every frame was already in a window", () => {
    const c = new FrameChunker(4, 2);
    c.push(frames(0, 4));
    expect(c.flush()).toEqual([]);
  });
});

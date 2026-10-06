import { describe, expect, it } from "vitest";
import { percentile, summarise } from "../src/stats.js";

describe("percentile", () => {
  it("uses nearest-rank so p95 of 1..20 is 19", () => {
    expect(percentile(Array.from({ length: 20 }, (_, i) => i + 1), 95)).toBe(19);
  });
  it("is order-independent", () => {
    expect(percentile([5, 1, 3, 2, 4], 50)).toBe(3);
  });
});

describe("summarise", () => {
  it("reports RTF as compute time over audio time, with p95 chunk latency", () => {
    const s = summarise({ audioSec: 10, chunkMs: [100, 200, 300, 400], loadMs: 1234 });
    expect(s.rtf).toBe(0.1);
    expect(s.realTime).toBe(true);
    expect(s.p95ChunkMs).toBe(400);
    expect(s.loadMs).toBe(1234);
    expect(s.chunks).toBe(4);
  });
  it("flags slower than real time", () => {
    expect(summarise({ audioSec: 1, chunkMs: [600, 600], loadMs: 0 }).realTime).toBe(false);
  });
});

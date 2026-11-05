import { describe, expect, it } from "vitest";
import { PosteriorFrames } from "../src/ctc.js";
import { referenceRows, type PackedWordReference } from "../src/word-reference-bank.js";

const packed: PackedWordReference = {
  from: 10, to: 12, k: 2,
  ids: Buffer.from([0, 2, 1, 0]).toString("base64"),
  probs: Buffer.from([0x40, 0x1f, 0xdc, 0x05, 0x38, 0x18, 0xb8, 0x0b]).toString("base64"),
};

describe("word reference storage", () => {
  it("preserves the live top-k expansion exactly, including omitted mass", () => {
    const live = new PosteriorFrames(4);
    live.append({ startFrame: 10, nFrames: 2, k: 2, ids: [0, 2, 1, 0], probs: [0.8, 0.15, 0.62, 0.3] });
    const stored = referenceRows(packed, 4, 10, 12);
    expect(stored).toEqual([10, 11].map(f => Float64Array.from(live.row(f), Math.exp)));
    expect(referenceRows(packed, 4, 11, 12)).toEqual([stored[1]]);
  });
  it("rejects a partial asset or a span outside its decode", () => {
    expect(() => referenceRows({ ...packed, probs: "" }, 4, 10, 12)).toThrow("Malformed");
    expect(() => referenceRows(packed, 4, 9, 11)).toThrow("outside");
    expect(() => referenceRows(packed, 4, 11, 13)).toThrow("outside");
  });
});
